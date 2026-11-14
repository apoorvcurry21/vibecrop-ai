"""The tools the chat assistant can call. Each is a thin, validated wrapper over `workflow` (the same functions the
UI buttons use). The model only ever passes ids and numbers, never file paths.

Every change takes an undo snapshot first; if a tool fails, the snapshot is restored, so a failed call never leaves
the project half-edited."""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src import workflow
from src.assistant import snapshots
from src.planning.layouts import LAYOUTS, STYLES
from src.rendering import framing as framing_mod
from src.rendering.ass import TEXT_COLORS, TEXT_FONTS, TEXT_POSITIONS, TEXT_SIZES
from src.rendering.compose import PIP_POSITIONS

MAX_TRANSCRIPT_CHARS = 9000
FRAMING_PRESETS = {  # (zoom at the start, zoom at the end), always centred
    "zoom_in": (1.0, 1.6), "zoom_out": (1.6, 1.0), "hold": (1.0, 1.0), "hold_zoomed": (1.4, 1.4),
}


class ToolError(ValueError):
    """Bad arguments from the model: reported back to it so it can correct itself."""


@dataclass
class Attachment:
    id: str  # att_1, att_2... what the model calls it
    path: Path
    kind: str  # "image" or "video"
    name: str  # the user's original file name (shown to the model as a hint)


@dataclass
class ToolContext:
    job: Path
    attachments: dict[str, Attachment] = field(default_factory=dict)
    cards: list[dict] = field(default_factory=list)  # things the UI should show in the chat (thumbnails, pickers)
    changed: bool = False  # True once the plan changed, so the page knows to refresh
    log: Callable[[str], None] = lambda message: None


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, dict]  # name -> JSON-schema-ish {"type", "description", ...}
    required: tuple[str, ...]
    fn: Callable[[ToolContext, dict], dict]
    mutating: bool = False
    clip_ids: Callable[[dict], list[int]] = lambda a: []  # slot clips the change may overwrite or delete
    label: Callable[[dict], str] = lambda a: ""  # shown while it runs / as the undo label


REGISTRY: dict[str, Tool] = {}


def _p(type_: str, description: str, enum: list | tuple | None = None) -> dict:
    return {"type": type_, "description": description, **({"enum": list(enum)} if enum else {})}


def tool(name: str, description: str, params: dict[str, dict] | None = None, required: tuple[str, ...] = (),
         mutating: bool = False, clip_ids: Callable[[dict], list[int]] | None = None,
         label: Callable[[dict], str] | None = None):
    def register(fn: Callable[[ToolContext, dict], dict]):
        REGISTRY[name] = Tool(name, description, params or {}, required, fn, mutating,
                              clip_ids or (lambda a: []), label or (lambda a: name.replace("_", " ")))
        return fn
    return register


# ----------------------------------------------------------------------------- argument helpers
def _int(args: dict, key: str) -> int:
    try:
        v = args[key]
        if isinstance(v, bool) or (isinstance(v, float) and v != int(v)):
            raise ValueError
        return int(v)
    except (KeyError, TypeError, ValueError):
        raise ToolError(f"'{key}' must be a whole number (got {args.get(key)!r}).") from None


def _num(args: dict, key: str, default: float | None = None) -> float:
    if args.get(key) is None:
        if default is None:
            raise ToolError(f"'{key}' is required.")
        return default
    try:
        return float(args[key])
    except (TypeError, ValueError):
        raise ToolError(f"'{key}' must be a number in seconds (got {args.get(key)!r}).") from None


def _opt_num(args: dict, key: str) -> float | None:
    return None if args.get(key) is None else _num(args, key)


def _str(args: dict, key: str, default: str | None = None) -> str:
    v = args.get(key)
    if v is None or not str(v).strip():
        if default is None:
            raise ToolError(f"'{key}' is required.")
        return default
    return str(v).strip()


def _slot_id(args: dict, key: str = "slot_id") -> int:
    return _int(args, key)


# ----------------------------------------------------------------------------- views the model reads
def slot_view(item: dict) -> dict:
    asset = item.get("asset") or {}
    view = {
        "id": item["id"], "start": item["timestamp_start"], "end": item["timestamp_end"],
        "layout": item.get("layout", "overlay"), "kind": item.get("asset_type", "video"),
        "removed": item.get("enabled") is False,
        "has_clip": asset.get("status") == "ok",
        "source": (asset.get("title") or item.get("search_query") or "")[:70],
        "says": (item.get("english_meaning") or item.get("spoken_text") or "")[:90],
    }
    if item.get("layout") == "pip":
        view["pip"] = item.get("pip")
    return view


def plan_view(job: Path) -> dict:
    plan = workflow.load_plan(job)
    meta = workflow.read_job_meta(job)
    items = sorted(plan["items"], key=lambda i: i["timestamp_start"])
    return {
        "video_topic": plan.get("video_topic", ""), "video_duration": plan.get("video_duration"),
        "style": meta.get("style", "overlay"),
        "slots": [slot_view(i) for i in items],
        "texts": [{k: t[k] for k in ("id", "text", "start", "end", "pos", "size", "color")}
                  for t in plan.get("texts", [])],
        "free_gaps": [list(g) for g in workflow.free_gaps(job)],
        "can_undo": snapshots.can_undo(job),
    }


def plan_summary(job: Path) -> str:
    """Compact text version for the system prompt (the model can call get_plan for a fresh one)."""
    try:
        return json.dumps(plan_view(job), ensure_ascii=False, separators=(",", ":"))
    except (OSError, ValueError, KeyError):
        return "(no plan yet)"


def _load_transcript(job: Path) -> dict:
    f = job / "transcript.json"
    if not f.is_file():
        raise ToolError("There is no transcript for this job yet.")
    return json.loads(f.read_text(encoding="utf-8"))


_NORM = re.compile(r"[^\wऀ-ॿ]+")


def _norm(word: str) -> str:
    return _NORM.sub("", word.lower())


def find_phrase(words: list[dict], phrase: str, threshold: float = 0.72, limit: int = 8) -> list[dict]:
    """Fuzzy search the word-level transcript (speech recognition spells names loosely, e.g. 'Koli'/'Kohli')."""
    tokens = [t for t in (_norm(w) for w in phrase.split()) if t]
    if not tokens:
        return []
    norm = [_norm(w.get("word", "")) for w in words]
    hits = []
    for i in range(len(words) - len(tokens) + 1):
        scores = [difflib.SequenceMatcher(None, tokens[j], norm[i + j]).ratio() for j in range(len(tokens))]
        if min(scores) >= threshold:
            hits.append((sum(scores) / len(scores), i))
    out, used = [], []
    for score, i in sorted(hits, reverse=True):
        if any(abs(i - u) < len(tokens) for u in used):
            continue
        used.append(i)
        lo, hi = max(0, i - 4), min(len(words), i + len(tokens) + 4)
        out.append({"start": words[i]["start"], "end": words[i + len(tokens) - 1]["end"],
                    "heard_as": " ".join(w["word"] for w in words[i:i + len(tokens)]),
                    "context": " ".join(w["word"] for w in words[lo:hi]), "match": round(score, 2)})
        if len(out) >= limit:
            break
    return sorted(out, key=lambda h: h["start"])


# ----------------------------------------------------------------------------- read tools
@tool("get_plan", "Current state of the project: every slot (id, start/end seconds, layout, kind, source, what is "
      "being said), text overlays, free gaps where a new slot fits, video length, style. Call it whenever you need "
      "ids or times; the user may also have edited things in the UI.")
def _get_plan(ctx: ToolContext, args: dict) -> dict:
    return plan_view(ctx.job)


@tool("get_transcript", "What the speaker says, with timestamps in seconds. Optionally limit to a time range.",
      {"start": _p("number", "range start in seconds (optional)"), "end": _p("number", "range end in seconds (optional)")})
def _get_transcript(ctx: ToolContext, args: dict) -> dict:
    lo, hi = _opt_num(args, "start"), _opt_num(args, "end")
    segs = [s for s in _load_transcript(ctx.job)["segments"]
            if (hi is None or s["start"] < hi) and (lo is None or s["end"] > lo)]
    out, size = [], 0
    for s in segs:
        row = {"start": s["start"], "end": s["end"], "text": s["text"]}
        size += len(row["text"]) + 30
        if size > MAX_TRANSCRIPT_CHARS:
            return {"segments": out, "truncated": True, "note": "Ask for a narrower start/end range for the rest."}
        out.append(row)
    return {"segments": out}


@tool("find_in_transcript", "Find WHEN the speaker says a word or phrase (fuzzy, so spelling variations of names "
      "still match). Returns start/end seconds of each match. Use this for 'where I say ...' requests.",
      {"phrase": _p("string", "the word(s) as spoken, e.g. 'Virat Kohli'")}, required=("phrase",))
def _find_in_transcript(ctx: ToolContext, args: dict) -> dict:
    phrase = _str(args, "phrase")
    hits = find_phrase(_load_transcript(ctx.job)["words"], phrase)
    return {"matches": hits} if hits else {"matches": [], "note": "Not found. Try a shorter or differently spelled "
                                           "phrase, or read get_transcript."}


# ----------------------------------------------------------------------------- edit tools (existing workflow)
def _ids(*keys: str) -> Callable[[dict], list[int]]:
    def f(args: dict) -> list[int]:
        return [int(args[k]) for k in keys if isinstance(args.get(k), (int, float)) and not isinstance(args.get(k), bool)]
    return f


@tool("set_query", "Change a slot's YouTube search query and fetch a fresh clip for it (takes a few seconds).",
      {"slot_id": _p("integer", "slot id"), "query": _p("string", "new English search query, 3-6 plain words")},
      required=("slot_id", "query"), mutating=True, clip_ids=_ids("slot_id"),
      label=lambda a: f"Searching a new clip for slot {a.get('slot_id')}")
def _set_query(ctx: ToolContext, args: dict) -> dict:
    return {"ok": True, "slot": slot_view(workflow.set_query(ctx.job, _slot_id(args), _str(args, "query")))}


@tool("regenerate", "Replace a slot's clip with a different video for the same query.",
      {"slot_id": _p("integer", "slot id")}, required=("slot_id",), mutating=True, clip_ids=_ids("slot_id"),
      label=lambda a: f"Finding a different clip for slot {a.get('slot_id')}")
def _regenerate(ctx: ToolContext, args: dict) -> dict:
    return {"ok": True, "slot": slot_view(workflow.regenerate(ctx.job, _slot_id(args)))}


@tool("remove_slot", "Take a slot off the video (the speaker stays on screen there). It keeps its place so it can "
      "be restored. Use delete_slot to free its time range completely.",
      {"slot_id": _p("integer", "slot id")}, required=("slot_id",), mutating=True,
      label=lambda a: f"Removing slot {a.get('slot_id')}")
def _remove_slot(ctx: ToolContext, args: dict) -> dict:
    workflow.set_enabled(ctx.job, _slot_id(args), False)
    return {"ok": True}


@tool("restore_slot", "Bring back a slot that was removed.", {"slot_id": _p("integer", "slot id")},
      required=("slot_id",), mutating=True, label=lambda a: f"Restoring slot {a.get('slot_id')}")
def _restore_slot(ctx: ToolContext, args: dict) -> dict:
    workflow.set_enabled(ctx.job, _slot_id(args), True)
    return {"ok": True}


@tool("delete_slot", "Permanently delete a slot and its clip, freeing its time range.",
      {"slot_id": _p("integer", "slot id")}, required=("slot_id",), mutating=True, clip_ids=_ids("slot_id"),
      label=lambda a: f"Deleting slot {a.get('slot_id')}")
def _delete_slot(ctx: ToolContext, args: dict) -> dict:
    workflow.delete_slot(ctx.job, _slot_id(args))
    return {"ok": True}


@tool("set_timing", "Move or resize a slot on the timeline. It cannot overlap neighbours, start before 1 s, last "
      "under 1.5 s, or be longer than its clip (unless refetch is true, which downloads a longer clip and may "
      "pick a different video).",
      {"slot_id": _p("integer", "slot id"), "start": _p("number", "new start, seconds"),
       "end": _p("number", "new end, seconds"), "refetch": _p("boolean", "fetch a longer clip if needed")},
      required=("slot_id", "start", "end"), mutating=True,
      clip_ids=lambda a: _ids("slot_id")(a) if a.get("refetch") else [],
      label=lambda a: f"Retiming slot {a.get('slot_id')}")
def _set_timing(ctx: ToolContext, args: dict) -> dict:
    item = workflow.set_timing(ctx.job, _slot_id(args), _num(args, "start"), _num(args, "end"),
                               refetch=bool(args.get("refetch")))
    return {"ok": True, "slot": slot_view(item)}


@tool("swap_slots", "Trade the clips between two slots. Timing, layout and removed-state stay with each slot.",
      {"slot_a": _p("integer", "first slot id"), "slot_b": _p("integer", "second slot id")},
      required=("slot_a", "slot_b"), mutating=True, clip_ids=_ids("slot_a", "slot_b"),
      label=lambda a: f"Swapping slots {a.get('slot_a')} and {a.get('slot_b')}")
def _swap_slots(ctx: ToolContext, args: dict) -> dict:
    warnings = workflow.swap_slots(ctx.job, _int(args, "slot_a"), _int(args, "slot_b"))
    return {"ok": True, "warnings": warnings}


@tool("set_layout", f"Set a slot's layout: {', '.join(LAYOUTS)}. overlay = clip fills the whole screen over the "
      "speaker; split = clip on the top half, speaker on the bottom half; pip = small picture-in-picture box over "
      "the speaker (use set_pip for its position and size).",
      {"slot_id": _p("integer", "slot id"), "layout": _p("string", "the layout", LAYOUTS)},
      required=("slot_id", "layout"), mutating=True, label=lambda a: f"Changing slot {a.get('slot_id')} layout")
def _set_layout(ctx: ToolContext, args: dict) -> dict:
    return {"ok": True, "slot": slot_view(workflow.set_layout(ctx.job, _slot_id(args), _str(args, "layout")))}


@tool("set_pip", "Place a slot as a small picture-in-picture box and set where it sits and how big it is.",
      {"slot_id": _p("integer", "slot id"), "position": _p("string", "where on screen", PIP_POSITIONS),
       "size": _p("number", "box width as a fraction of the screen width, 0.3 (small) to 0.95 (nearly full); "
                            "default 0.6")},
      required=("slot_id",), mutating=True, label=lambda a: f"Placing slot {a.get('slot_id')} as picture-in-picture")
def _set_pip(ctx: ToolContext, args: dict) -> dict:
    item = workflow.set_pip(ctx.job, _slot_id(args), pos=args.get("position"), scale=_opt_num(args, "size"))
    return {"ok": True, "slot": slot_view(item)}


@tool("set_framing", "Zoom/pan animation of a slot's clip. Use a preset, or set zoom (1 = fill, up to 3) and pan "
      "(0..1, 0.5 = centre) for the start and end of the slot; unspecified values keep their current setting.",
      {"slot_id": _p("integer", "slot id"),
       "preset": _p("string", "zoom_in, zoom_out, hold or hold_zoomed", list(FRAMING_PRESETS)),
       "zoom_start": _p("number", "zoom at the start"), "zoom_end": _p("number", "zoom at the end"),
       "pan_x_start": _p("number", "0 left .. 1 right, at the start"), "pan_x_end": _p("number", "same, at the end"),
       "pan_y_start": _p("number", "0 top .. 1 bottom, at the start"), "pan_y_end": _p("number", "same, at the end")},
      required=("slot_id",), mutating=True, label=lambda a: f"Framing slot {a.get('slot_id')}")
def _set_framing(ctx: ToolContext, args: dict) -> dict:
    sid = _slot_id(args)
    plan = workflow.load_plan(ctx.job)
    cur = framing_mod.normalize(workflow._item(plan, sid).get("framing"))
    if args.get("preset"):
        if args["preset"] not in FRAMING_PRESETS:
            raise ToolError(f"Unknown preset. Choose one of: {', '.join(FRAMING_PRESETS)}.")
        z0, z1 = FRAMING_PRESETS[args["preset"]]
        cur = {"start": {"z": z0, "x": 0.5, "y": 0.5}, "end": {"z": z1, "x": 0.5, "y": 0.5}}
    for end in ("start", "end"):
        for key, arg in (("z", "zoom"), ("x", "pan_x"), ("y", "pan_y")):
            v = _opt_num(args, f"{arg}_{end}")
            if v is not None:
                cur[end][key] = v
    return {"ok": True, "framing": workflow.set_framing(ctx.job, sid, cur)["framing"]}


@tool("apply_style", "Set the video style for ALL slots: overlay, split or hybrid (a mix). Resets every slot's "
      "layout except picture-in-picture ones. Only do this if the user asks for a whole-video style.",
      {"style": _p("string", "the style", STYLES)}, required=("style",), mutating=True,
      label=lambda a: f"Applying the {a.get('style')} style")
def _apply_style(ctx: ToolContext, args: dict) -> dict:
    workflow.apply_style(ctx.job, _str(args, "style"))
    return {"ok": True}


@tool("set_split_face_y", "Where the speaker's face is cropped from in split slots: 0 = top of the speaker video, "
      "1 = bottom.", {"value": _p("number", "0..1")}, required=("value",), mutating=True,
      label=lambda a: "Moving the facecam crop")
def _set_split_face_y(ctx: ToolContext, args: dict) -> dict:
    workflow.set_split_face_y(ctx.job, _num(args, "value"))
    return {"ok": True}


# ----------------------------------------------------------------------------- text overlays
_TEXT_PARAMS = {
    "position": _p("string", "where on screen", list(TEXT_POSITIONS)),
    "size": _p("string", "text size", list(TEXT_SIZES)),
    "color": _p("string", "a colour name or #RRGGBB: " + ", ".join(TEXT_COLORS)),
    "font": _p("string", "font style", list(TEXT_FONTS)),
}


def _style_args(args: dict) -> dict:
    return {k: args.get(src) for k, src in (("pos", "position"), ("size", "size"), ("color", "color"),
                                            ("font", "font")) if args.get(src) is not None}


@tool("add_text", "Add a timed text overlay (caption, meme text, quote, label) on the video.",
      {"text": _p("string", "the text, up to 120 characters"), "start": _p("number", "start, seconds"),
       "end": _p("number", "end, seconds"), **_TEXT_PARAMS},
      required=("text", "start", "end"), mutating=True, label=lambda a: "Adding text")
def _add_text(ctx: ToolContext, args: dict) -> dict:
    return {"ok": True, "text": workflow.add_text(ctx.job, _str(args, "text"), _num(args, "start"),
                                                  _num(args, "end"), **_style_args(args))}


@tool("update_text", "Change a text overlay (any of: text, start, end, position, size, color, font).",
      {"text_id": _p("integer", "text overlay id"), "text": _p("string", "new text"),
       "start": _p("number", "new start"), "end": _p("number", "new end"), **_TEXT_PARAMS},
      required=("text_id",), mutating=True, label=lambda a: "Editing text")
def _update_text(ctx: ToolContext, args: dict) -> dict:
    fields = {"text": args.get("text"), "start": _opt_num(args, "start"), "end": _opt_num(args, "end"),
              **_style_args(args)}
    return {"ok": True, "text": workflow.update_text(ctx.job, _int(args, "text_id"), **fields)}


@tool("remove_text", "Delete a text overlay.", {"text_id": _p("integer", "text overlay id")},
      required=("text_id",), mutating=True, label=lambda a: "Removing text")
def _remove_text(ctx: ToolContext, args: dict) -> dict:
    workflow.remove_text(ctx.job, _int(args, "text_id"))
    return {"ok": True}


# ----------------------------------------------------------------------------- undo
@tool("undo", "Undo the most recent change (the same as the Undo button). Can be repeated for older changes.")
def _undo(ctx: ToolContext, args: dict) -> dict:
    label = snapshots.undo(ctx.job)
    if label is None:
        return {"ok": False, "note": "Nothing to undo."}
    ctx.changed = True
    return {"ok": True, "undone": label}


# ----------------------------------------------------------------------------- dispatch + declarations
def declarations() -> list:
    """Gemini FunctionDeclarations for every registered tool."""
    from google.genai import types

    def schema(p: dict) -> dict:
        s = {"type": p["type"].upper(), "description": p.get("description", "")}
        if p.get("enum"):
            s["enum"] = p["enum"]
        return s

    out = []
    for t in REGISTRY.values():
        decl: dict[str, Any] = {"name": t.name, "description": t.description}
        if t.params:
            decl["parameters"] = {"type": "OBJECT", "properties": {k: schema(v) for k, v in t.params.items()},
                                  "required": list(t.required)}
        out.append(types.FunctionDeclaration(**decl))
    return out


def status_label(name: str, args: dict) -> str:
    t = REGISTRY.get(name)
    return (t.label(args) if t else name) + "..."


def run_tool(ctx: ToolContext, name: str, args: dict | None) -> dict:
    """Run one tool call and return a JSON-able result for the model. Never raises."""
    t = REGISTRY.get(name)
    args = dict(args or {})
    if t is None:
        return {"error": f"Unknown tool '{name}'. Available: {', '.join(REGISTRY)}."}
    missing = [r for r in t.required if args.get(r) is None]
    if missing:
        return {"error": f"Missing required argument(s): {', '.join(missing)}."}
    snapped = False
    try:
        if t.mutating:
            snapshots.take(ctx.job, t.label(args) or name, t.clip_ids(args))
            snapped = True
        result = t.fn(ctx, args)
        if t.mutating:
            ctx.changed = True
        return result
    except (ToolError, workflow.WorkflowError) as exc:
        message = str(exc)
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the chat
        message = f"{type(exc).__name__}: {exc}"
    if snapped:  # the call failed: put everything back exactly as it was
        snapshots.undo(ctx.job)
    return {"error": message}


# Registered last: these tools need the helpers defined above.
from src.assistant import media_tools  # noqa: E402,F401
