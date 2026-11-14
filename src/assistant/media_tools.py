"""Chat tools that bring in pictures and clips: web image search, AI image generation, the user's attached pictures
and videos, and pasted YouTube links. They are registered in tools.REGISTRY when this module is imported (tools.py
does that at its end)."""
from __future__ import annotations

import uuid
from pathlib import Path

from src import workflow
from src.assistant import assets, media, ytclip
from src.assistant.tools import (ToolContext, ToolError, _int, _num, _opt_num, _p, _str, slot_view, tool)
from src.planning.layouts import LAYOUTS
from src.rendering.compose import PIP_POSITIONS
from src.sourcing import stills

DEFAULT_SLOT_S = 3.0
MAX_CLIP_S = 12.0
_PLACE = {
    "slot_id": _p("integer", "REPLACE the clip of this existing slot (keeps its time and layout). Leave out to add "
                             "a new slot at start..end."),
    "start": _p("number", "where the new slot begins on the video timeline, in seconds"),
    "end": _p("number", "where it ends (default: start + 3 s)"),
    "layout": _p("string", "overlay = fills the screen, split = top half with the speaker below, pip = small box. "
                           "Default overlay for a new slot.", LAYOUTS),
    "position": _p("string", "for pip: where the box sits", PIP_POSITIONS),
    "size": _p("number", "for pip: box width as a fraction of the screen, 0.3 to 0.95"),
}


def _slot_ids(args: dict) -> list[int]:
    v = args.get("slot_id")
    return [int(v)] if isinstance(v, (int, float)) and not isinstance(v, bool) else []


def _with_gaps_hint(ctx: ToolContext, exc: Exception) -> ToolError:
    gaps = ", ".join(f"{a:g}-{b:g}s" for a, b in workflow.free_gaps(ctx.job)) or "none"
    return ToolError(f"{exc} Free time ranges for a new slot: {gaps}.")


def _target(ctx: ToolContext, args: dict, default_len: float) -> tuple[int | None, float, float]:
    """Where the new media goes: (slot_id or None, start, end). Checks the room before any slow work starts."""
    if args.get("slot_id") is not None:
        sid = _int(args, "slot_id")
        plan = workflow.load_plan(ctx.job)
        item = next((i for i in plan["items"] if i["id"] == sid), None)
        if item is None:
            raise ToolError(f"No B-roll with id {sid}.")
        return sid, item["timestamp_start"], item["timestamp_end"]
    start = _num(args, "start")
    end = _opt_num(args, "end")
    end = start + default_len if end is None else end
    try:
        workflow.check_room(ctx.job, start, end)
    except workflow.WorkflowError as exc:
        raise _with_gaps_hint(ctx, exc) from None
    return None, start, end


def _spoken(job: Path, start: float, end: float) -> str:
    try:
        words = workflow.read_transcript_words(job)
    except (OSError, ValueError):
        return ""
    return " ".join(w["word"] for w in words if start - 0.05 <= w["start"] < end)[:200]


def _place(ctx: ToolContext, args: dict, slot_id: int | None, start: float, end: float, clip: Path, asset: dict,
           asset_type: str) -> dict:
    layout = args.get("layout")
    if layout is not None and layout not in LAYOUTS:
        raise ToolError(f"Unknown layout '{layout}'. Choose one of: {', '.join(LAYOUTS)}.")
    wants_pip = layout == "pip" or (layout is None and (args.get("position") or args.get("size")))
    pip = {"pos": args.get("position"), "scale": _opt_num(args, "size")} if wants_pip else None
    try:
        if slot_id is not None:
            item = workflow.set_asset(ctx.job, slot_id, clip, asset, asset_type)
            if wants_pip:
                item = workflow.set_pip(ctx.job, slot_id, pos=pip["pos"], scale=pip["scale"])
            elif layout:
                item = workflow.set_layout(ctx.job, slot_id, layout)
        else:
            item = workflow.add_slot(
                ctx.job, start, end, clip, asset, asset_type, layout="pip" if wants_pip else (layout or "overlay"),
                pip={k: v for k, v in (pip or {}).items() if v is not None} or None,
                spoken_text=_spoken(ctx.job, start, end), meaning=asset.get("title", "")[:90])
    except workflow.WorkflowError as exc:
        raise _with_gaps_hint(ctx, exc) from None
    return {"ok": True, "slot": slot_view(item)}


def _tmp_clip(job: Path, suffix: str = ".mp4") -> Path:
    d = assets.assets_dir(job) / "_work"
    d.mkdir(exist_ok=True)
    return d / f"clip_{uuid.uuid4().hex[:8]}{suffix}"


# ----------------------------------------------------------------------------- pictures
@tool("search_web_images", "Search the web for pictures (people, places, memes, logos...). Shows small thumbnails "
      "to the user, who can pick one. Returns ids like res_3 that place_image accepts.",
      {"query": _p("string", "what to search for, e.g. 'Virat Kohli playing football'"),
       "count": _p("integer", "how many results, 3 to 8 (default 6)")},
      required=("query",), label=lambda a: "Searching the web for pictures")
def _search_web_images(ctx: ToolContext, args: dict) -> dict:
    try:
        results = media.search_images(ctx.job, _str(args, "query"), int(_opt_num(args, "count") or media.SEARCH_COUNT))
    except media.MediaError as exc:
        raise ToolError(str(exc)) from None
    ctx.cards.append({"type": "choices", "items": [
        {"id": r["id"], "path": r["thumb"], "caption": f"{r['id']} · {r['domain']}"} for r in results]})
    return {"results": [{k: r[k] for k in ("id", "title", "domain", "width", "height")} for r in results],
            "note": "The user can see the thumbnails. If they said where to put one, pick the best match by title "
                    "and place it; otherwise ask which one they like."}


@tool("generate_image", "Create a new picture with AI from a description (vertical 9:16). The user sees it in the "
      "chat; then use place_image with the returned id once they like it and say where. May be refused for real "
      "people or unavailable on this account; then offer search_web_images.",
      {"prompt": _p("string", "a detailed English description of the picture")},
      required=("prompt",), label=lambda a: "Generating a picture")
def _generate_image(ctx: ToolContext, args: dict) -> dict:
    try:
        out = media.generate_image(ctx.job, _str(args, "prompt"))
    except media.MediaError as exc:
        raise ToolError(str(exc)) from None
    ctx.cards.append({"type": "image", "path": out["path"], "caption": out["id"]})
    return {"image_id": out["id"], "note": "Shown to the user. Ask if they like it, or place it if they already said where."}


@tool("place_image", "Put a picture on the video: either as a NEW slot (give start, optional end) or into an "
      "EXISTING slot (give slot_id). `image` is att_N (the user's attached picture), res_N (a search result), gen_N "
      "(a generated picture) or an image link. Fills the screen by default; layout pip makes it a small box.",
      {"image": _p("string", "att_1, res_2, gen_3 or an http(s) image link"), **_PLACE},
      required=("image",), mutating=True, clip_ids=_slot_ids, label=lambda a: "Placing the picture")
def _place_image(ctx: ToolContext, args: dict) -> dict:
    slot_id, start, end = _target(ctx, args, DEFAULT_SLOT_S)
    image, extra = media.resolve_image(ctx.job, ctx, _str(args, "image"))
    clip = _tmp_clip(ctx.job)
    try:
        info = stills.image_to_clip(image, clip)
    except stills.StillError as exc:
        raise ToolError(str(exc)) from None
    asset = {"source": "image", "clip_duration": info["clip_duration"], "query_used": extra.get("title", ""), **extra}
    return _place(ctx, args, slot_id, start, end, clip, asset, "image")


# ----------------------------------------------------------------------------- attached video
@tool("place_video_file", "Use part of a video the user attached (att_N) as B-roll: cuts the seconds you choose "
      "and places them in a new slot (start, optional end) or an existing one (slot_id).",
      {"video": _p("string", "the attachment id, e.g. att_2"),
       "from_second": _p("number", "where in the attached video to start cutting (default 0)"), **_PLACE},
      required=("video",), mutating=True, clip_ids=_slot_ids, label=lambda a: "Cutting the attached video")
def _place_video_file(ctx: ToolContext, args: dict) -> dict:
    ref = _str(args, "video")
    att = ctx.attachments.get(ref) or assets.load_attachment(ctx.job, ref)
    if att is None or att.kind != "video":
        raise ToolError(f"'{ref}' is not an attached video.")
    slot_id, start, end = _target(ctx, args, 5.0)
    length = end - start
    clip = _tmp_clip(ctx.job)
    try:
        got = media.trim_video(att.path, clip, _opt_num(args, "from_second") or 0.0, length + 1.0)
    except media.MediaError as exc:
        raise ToolError(str(exc)) from None
    if got < length - 0.05:
        clip.unlink(missing_ok=True)
        raise ToolError(f"The attached video only has {got:.1f}s from that point, but the slot is {length:.1f}s. "
                        "Start earlier or make the slot shorter.")
    asset = {"source": "upload", "title": att.name, "clip_duration": round(got, 2), "query_used": att.name}
    return _place(ctx, args, slot_id, start, end, clip, asset, "video")


# ----------------------------------------------------------------------------- YouTube link
@tool("place_youtube_clip", "Cut a short clip out of a YouTube link the user gave and place it as B-roll (new slot "
      "with start/end, or an existing slot_id). Give `from_time` if the user said when in the YouTube video the "
      "moment is; otherwise describe what to find in `about` and the AI will watch the video and pick the moment. "
      "Takes 10-30 seconds.",
      {"url": _p("string", "the YouTube link"),
       "from_time": _p("string", "where the clip starts inside the YouTube video, like 2:15 or 135 (optional)"),
       "about": _p("string", "what the clip should show, when from_time is not given (e.g. 'Kohli playing football')"),
       **_PLACE}, required=("url",), mutating=True, clip_ids=_slot_ids,
      label=lambda a: "Cutting a clip from YouTube")
def _place_youtube_clip(ctx: ToolContext, args: dict) -> dict:
    slot_id, start, end = _target(ctx, args, 5.0)
    length = end - start
    if length > MAX_CLIP_S:
        raise ToolError(f"Keep B-roll clips under {MAX_CLIP_S:g} seconds.")
    try:
        info = ytclip.video_info(_str(args, "url"))
        src_start = ytclip.parse_time(args.get("from_time"))
        reason = ""
        if src_start is None:
            about = _str(args, "about", "")
            if not about:
                raise ToolError("Tell me what to look for in that video (about), or when it happens (from_time).")
            ctx.log("Watching the video to find the moment...")
            src_start, reason = ytclip.find_window(info, about, length)
        stem = _tmp_clip(ctx.job, "").with_suffix("")
        path, got = ytclip.cut_clip(info, src_start, length, stem)
    except ytclip.YtError as exc:
        raise ToolError(str(exc)) from None
    asset = {"source": "youtube", "video_id": info["id"], "url": info["url"], "title": info["title"],
             "channel": info["channel"], "window_start": src_start, "window_end": round(src_start + got, 2),
             "clip_duration": got, "query_used": args.get("about") or info["title"], "tried_ids": [info["id"]]}
    result = _place(ctx, args, slot_id, start, end, path, asset, "video")
    result["from_video"] = {"title": info["title"], "channel": info["channel"], "starts_at": src_start,
                            "why": reason or "the time you gave"}
    return result
