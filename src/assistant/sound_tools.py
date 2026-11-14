"""Chat tools for sound effects: add a sound from the library (or find it on the web / a YouTube link), at a time or
wherever a phrase is spoken, and change or remove placed sounds. Registered in tools.REGISTRY when this module is
imported (tools.py does that at its end)."""
from __future__ import annotations

from src import workflow
from src.assistant import phrases
from src.assistant.tools import ToolContext, ToolError, _int, _opt_num, _p, _str, tool
from src.sfx import fetch, library
from src.sfx.library import SfxError

CLOSE_CALL = 0.05  # other sounds within this score of the best match are mentioned as alternatives

_WHEN = {
    "at_time": _p("number", "play at this many seconds on the video timeline"),
    "when_phrase": _p("string", "OR play wherever the speaker says these words (e.g. 'bhai'); every occurrence by "
                                "default"),
    "occurrence": _p("string", "with when_phrase: all (default), first, last, or nth (then give `which`)",
                     phrases.OCCURRENCES),
    "which": _p("integer", "for occurrence=nth: 1 = the first match"),
    "align": _p("string", "with when_phrase: play at the start of the words (default) or right after them",
                ("start", "end")),
    "offset": _p("number", "seconds to shift the start (negative = earlier), e.g. -0.1"),
    "volume": _p("number", "0 to 2; 1 = as recorded, default 0.8 (a bit under the voice)"),
}


def _times(ctx: ToolContext, args: dict) -> tuple[list[tuple[float, dict | None]], int | None]:
    """(start, match) pairs for where the sound plays, and how many matches the phrase had (None for at_time)."""
    offset = _opt_num(args, "offset") or 0.0
    phrase, at = args.get("when_phrase"), _opt_num(args, "at_time")
    if phrase and str(phrase).strip():
        found = phrases.search(ctx.job, str(phrase).strip())
        end_align = args.get("align") == "end"
        return [(max(0.0, (h["end"] if end_align else h["start"]) + offset), h)
                for h in phrases.select(found, args)], len(found)
    if at is not None:
        return [(max(0.0, at + offset), None)], None
    raise ToolError("Say when the sound plays: at_time (seconds) or when_phrase (the words it plays at).")


def _place(ctx: ToolContext, entry: dict, args: dict, how: str) -> dict:
    times, total = _times(ctx, args)
    placed = []
    for start, hit in times:
        s = workflow.add_sound(ctx.job, entry["id"], start, args.get("volume"))
        placed.append({"id": s["id"], "at": s["start"], **({"words": hit["heard_as"]} if hit else {})})
    ctx.cards.append({"type": "sound", "lib_id": entry["id"], "caption": entry["name"]})
    out = {"ok": True, "sound": entry["name"], "from": how, "length_s": entry.get("duration"), "placed": placed}
    if total is not None:
        out["matches_found"] = total
    return out


def _resolve(ctx: ToolContext, name: str) -> tuple[dict, str, list[str]]:
    """The library sound for `name`: an id or fuzzy name match, else fetched from YouTube and saved for next time."""
    exact = library.get(name)
    if exact is not None and library.available(exact):
        return exact, "library", []
    hits = library.find(name, limit=4)
    if hits:
        return hits[0], "library", [h["name"] for h in hits[1:] if h["score"] >= hits[0]["score"] - CLOSE_CALL]
    ctx.log(f"'{name}' is not in the library, searching YouTube...")
    try:
        return fetch.from_name(name, log=ctx.log), "youtube search (saved to the library)", []
    except SfxError as exc:
        raise ToolError(str(exc)) from None


@tool("list_sounds", "The sound effects in the library (meme sounds like fah, vine boom, bruh, airhorn, plus "
      "utility ones). Give `query` to see the closest matches. Anything not in the library can still be added: "
      "add_sound fetches it from YouTube by name.",
      {"query": _p("string", "a name to look for (optional)")}, label=lambda a: "Looking up sounds")
def _list_sounds(ctx: ToolContext, args: dict) -> dict:
    q = (args.get("query") or "").strip()
    if q:
        return {"sounds": [{"id": h["id"], "name": h["name"], "seconds": h.get("duration")}
                           for h in library.find(q, limit=8, minimum=0.5)]}
    sounds = [{"id": s["id"], "name": s["name"]} for s in library.load() if library.available(s)]
    return {"count": len(sounds), "sounds": sounds}


@tool("add_sound", "Play a sound effect in the video. `sound` is a name or id from the library (fah, vine boom, "
      "bruh, sad violin...); if it is not in the library it is searched on YouTube and saved for next time. Say "
      "when: at_time (seconds) or when_phrase (the words the speaker says). With when_phrase it is added at EVERY "
      "occurrence unless `occurrence` says otherwise, so tell the user how many matches there were and where.",
      {"sound": _p("string", "the sound's name or id, e.g. 'fah'"), **_WHEN}, required=("sound",), mutating=True,
      label=lambda a: f"Adding the '{a.get('sound')}' sound")
def _add_sound(ctx: ToolContext, args: dict) -> dict:
    _times(ctx, args)  # validate when/where BEFORE any slow download
    entry, how, also = _resolve(ctx, _str(args, "sound"))
    out = _place(ctx, entry, args, how)
    if also:
        out["also_matches"] = also
        out["note"] = "Other close library matches are listed in also_matches; mention them in case the user meant one."
    return out


@tool("add_sound_from_youtube", "Use a sound from a YouTube link the user pasted. Give from_time (like 0:42) if they "
      "said where the sound is in the video; otherwise describe it in `about` and the AI listens to the video and "
      "picks the moment. The sound is saved in the library, then placed like add_sound (at_time or when_phrase). "
      "Takes 10-30 seconds.",
      {"url": _p("string", "the YouTube link"),
       "from_time": _p("string", "where the sound starts inside the YouTube video, like 0:42 or 42 (optional)"),
       "about": _p("string", "which sound to find, when from_time is not given (e.g. 'the fah sound')"),
       "length": _p("number", "how many seconds of it to keep, default 3, up to 15"),
       "name": _p("string", "a name to save it under (optional)"), **_WHEN},
      required=("url",), mutating=True, label=lambda a: "Getting the sound from YouTube")
def _add_sound_from_youtube(ctx: ToolContext, args: dict) -> dict:
    _times(ctx, args)
    try:
        entry = fetch.from_youtube(_str(args, "url"), about=args.get("about") or "", from_time=args.get("from_time"),
                                   length=_opt_num(args, "length"), name=args.get("name"), log=ctx.log)
    except SfxError as exc:
        raise ToolError(str(exc)) from None
    out = _place(ctx, entry, args, "the YouTube link (saved to the library)")
    if entry.get("why"):
        out["picked_because"] = entry["why"]
    out["starts_at_in_video"] = entry.get("window_start")
    return out


@tool("update_sound", "Move a placed sound effect and/or change its volume (ids are in get_plan under sounds).",
      {"sound_id": _p("integer", "the placed sound's id"), "start": _p("number", "new start, seconds"),
       "volume": _p("number", "0 to 2; 1 = as recorded")}, required=("sound_id",), mutating=True,
      label=lambda a: "Changing a sound")
def _update_sound(ctx: ToolContext, args: dict) -> dict:
    if args.get("start") is None and args.get("volume") is None:
        raise ToolError("Give a new start and/or a new volume.")
    return {"ok": True, "sound": workflow.update_sound(ctx.job, _int(args, "sound_id"), _opt_num(args, "start"),
                                                       _opt_num(args, "volume"))}


@tool("remove_sound", "Remove one placed sound effect by id, or all of them when no id is given.",
      {"sound_id": _p("integer", "the placed sound's id; leave out to remove all")}, mutating=True,
      label=lambda a: "Removing sounds")
def _remove_sound(ctx: ToolContext, args: dict) -> dict:
    sid = None if args.get("sound_id") is None else _int(args, "sound_id")
    return {"ok": True, "removed": workflow.remove_sound(ctx.job, sid)}
