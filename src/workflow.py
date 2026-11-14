"""Orchestration used by the Streamlit app. The plan.json on disk is the single source of truth, so the UI
can reload (or reopen an old job) at any time and every action here is a small, testable function."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from src import config
from src.cleanup import purge_job_temp
from src.planning import pipeline as planning
from src.planning import recommend
from src.planning.layouts import LAYOUTS, STYLES, assign_layouts
from src.planning.prompt import INTRO_SAFE_S
from src.rendering import framing as framing_mod
from src.rendering import pipeline as rendering
from src.rendering.ass import MAX_TEXT_LEN, normalize_text_style
from src.rendering.compose import PIP_POSITIONS, normalize_pip
from src.sourcing import pipeline as sourcing
from src.sourcing.youtube import _probe as probe_clip
from src.transcription import pipeline as transcription

MIN_SLOT_S = 1.5  # shortest slot the editor allows (the planner itself aims for 3-5 s)
MIN_FREE_S = 0.3  # the chat is not bound by the editor's rules; this only stops a zero-length slot
EDIT_GAP_S = 0.1  # smallest gap the editor allows between two slots
# What travels with a clip when two slots swap places. Timing, layout and removed-state stay with the slot.
SWAP_FIELDS = ("asset", "search_query", "alternative_queries", "spoken_text", "english_meaning", "reasoning",
               "framing", "user_edited_query")

SPOKEN_LANGUAGES = {
    "hi": "Hindi / Hinglish", "en": "English", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
    "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "ar": "Arabic", "id": "Indonesian", "ja": "Japanese", "ko": "Korean",
}


class WorkflowError(RuntimeError):
    pass


def jobs_root(root: Path | None = None) -> Path:
    return (root or config.workdir_root()) / "jobs"


def list_jobs(root: Path | None = None) -> list[Path]:
    """Jobs the app created (finished or resumable), newest first."""
    base = jobs_root(root)
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if (p / "plan.json").is_file() or (p / "job.json").is_file()),
                  key=lambda p: p.name, reverse=True)


def read_job_meta(job_dir: Path) -> dict:
    f = job_dir / "job.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def write_job_meta(job_dir: Path, **fields) -> dict:
    meta = {**read_job_meta(job_dir), **fields}
    (job_dir / "job.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def source_video_path(job_dir: Path) -> Path | None:
    p = read_job_meta(job_dir).get("source_video")
    return Path(p) if p and Path(p).is_file() else None


def create_job(video_path: str | Path, language: str, copy_into_job: bool = False,
               root: Path | None = None, total: int | None = None, notes: str = "",
               style: str = "overlay") -> Path:
    """Create a job folder for a video. `copy_into_job=True` is for uploads (the copy is purged on cleanup).

    `total` (1-15) is how many B-rolls to plan; `notes` are optional creator hints for the AI;
    `style` is the video style: "overlay", "split" or "hybrid" (see planning/layouts.py)."""
    if style not in STYLES:
        raise WorkflowError(f"Unknown video style '{style}'. Choose one of: {', '.join(STYLES)}.")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise WorkflowError(f"Video not found: {video_path}")
    job = config.new_job_dir(video_path, root=jobs_root(root).parent)
    source = video_path
    if copy_into_job:
        source = job / f"source{video_path.suffix.lower() or '.mp4'}"
        shutil.copyfile(video_path, source)
    write_job_meta(job, source_video=str(source), language=language, source_name=video_path.name,
                   total_brolls=config.clamp_brolls(config.total_brolls() if total is None else total),
                   notes=" ".join((notes or "").split()), style=style)
    return job


def analyse(job_dir: Path, log: Callable[[str], None] = print) -> dict:
    """First step after a video is chosen: transcribe it (reused later by planning), then ask Gemini how many
    B-rolls suit it. The suggestion is stored in job.json so the slider can be preset (and survives a reload)."""
    meta = read_job_meta(job_dir)
    video = source_video_path(job_dir)
    if video is None:
        raise WorkflowError("The source video for this job is missing.")
    if not (job_dir / "transcript.json").is_file():
        log("Transcribing audio with Groq Whisper...")
        transcription.run(video, meta["language"], job_dir=job_dir)
    transcript = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
    log("Asking Gemini how many B-rolls suit this video...")
    suggestion = recommend.recommend_count(transcript)
    write_job_meta(job_dir, suggested_brolls=suggestion["count"], suggested_reason=suggestion["reason"],
                   suggested_source=suggestion["source"],
                   video_duration=float(transcript.get("video_duration") or transcript.get("duration") or 0.0))
    return suggestion


def set_language(job_dir: Path, language: str) -> None:
    """The spoken language changed: the transcript (and anything derived from it) no longer applies."""
    if read_job_meta(job_dir).get("language") == language:
        return
    write_job_meta(job_dir, language=language)
    for name in ("transcript.json", "plan.json", "final.mp4"):
        (job_dir / name).unlink(missing_ok=True)
    for cache in job_dir.glob("captions_*.json"):
        cache.unlink()
    shutil.rmtree(job_dir / "clips", ignore_errors=True)
    meta = read_job_meta(job_dir)
    for key in ("suggested_brolls", "suggested_reason", "suggested_source"):
        meta.pop(key, None)
    (job_dir / "job.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def discard_draft(job_dir: Path | None) -> None:
    """Delete a job that was only analysed (never planned), e.g. because another video replaced it."""
    if (job_dir and job_dir.parent.name == "jobs" and (job_dir / "job.json").is_file()
            and not (job_dir / "plan.json").is_file()):  # only ever a job folder the app made, never a plan
        shutil.rmtree(job_dir, ignore_errors=True)


def start_planning(job_dir: Path, total: int, notes: str, style: str) -> None:
    """The user pressed 'Build B-roll plan': remember the chosen count, notes and video style."""
    if style not in STYLES:
        raise WorkflowError(f"Unknown video style '{style}'.")
    write_job_meta(job_dir, total_brolls=config.clamp_brolls(total), notes=" ".join((notes or "").split()),
                   style=style)


def replan(job_dir: Path, total: int, notes: str) -> None:
    """Throw away this job's plan and downloaded clips (keeping the transcript) so it can be planned again
    with a new B-roll count and/or notes. The next run of `run_pipeline` rebuilds everything."""
    write_job_meta(job_dir, total_brolls=config.clamp_brolls(total), notes=" ".join((notes or "").split()))
    for name in ("plan.json", "final.mp4"):
        (job_dir / name).unlink(missing_ok=True)
    for cache in job_dir.glob("captions_*.json"):
        cache.unlink()
    shutil.rmtree(job_dir / "clips", ignore_errors=True)


def needs_pipeline(job_dir: Path) -> bool:
    """True if the job still has work to do: no plan yet, or video slots that were never sourced.
    (Slots whose search failed are left for the user to fix, and rendered jobs keep their asset records.)"""
    if not (job_dir / "plan.json").is_file():
        return True
    return any(i["asset_type"] == "video" and i.get("enabled") is not False and not i.get("asset")
               for i in load_plan(job_dir).get("items", []))


def run_pipeline(job_dir: Path, log: Callable[[str], None] = print, total: int | None = None) -> None:
    """Transcribe -> plan -> source clips. Each step is skipped if its output already exists (resumable)."""
    meta = read_job_meta(job_dir)
    video = source_video_path(job_dir)
    if video is None:
        raise WorkflowError("The source video for this job is missing. Point the job at the video file first.")
    if not (job_dir / "transcript.json").is_file():
        log("Transcribing audio with Groq Whisper...")
        transcription.run(video, meta["language"], job_dir=job_dir)
    if not (job_dir / "plan.json").is_file():
        log("Planning B-rolls with Gemini...")
        planning.run(job_dir, total=total or meta.get("total_brolls"), notes=meta.get("notes", ""),
                     style=meta.get("style", "overlay"))
    log("Downloading B-roll clips...")
    sourcing.run(job_dir, log=log)


def load_plan(job_dir: Path) -> dict:
    return json.loads((job_dir / "plan.json").read_text(encoding="utf-8"))


def _save(job_dir: Path, plan: dict) -> None:
    (job_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def _item(plan: dict, item_id: int) -> dict:
    for it in plan["items"]:
        if it["id"] == item_id:
            return it
    raise WorkflowError(f"No B-roll with id {item_id}.")


def _resource(job_dir: Path, item: dict, skip_ids: set[str]) -> None:
    try:
        item["asset"] = sourcing.source_item(job_dir, item, skip_ids=skip_ids)
    except sourcing.SourcingError as exc:
        item["asset"] = {"status": "failed", "error": str(exc), "tried_ids": sorted(skip_ids)}


def regenerate(job_dir: Path, item_id: int) -> dict:
    """Pick a different video for the same query (already-tried videos are skipped)."""
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    _resource(job_dir, item, set((item.get("asset") or {}).get("tried_ids", [])))
    _save(job_dir, plan)
    return item


def set_query(job_dir: Path, item_id: int, query: str) -> dict:
    """Replace a bad search query and fetch a fresh clip for it."""
    query = " ".join(query.split())
    if not query:
        raise WorkflowError("The search query cannot be empty.")
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    item["search_query"] = query
    item["user_edited_query"] = True
    _resource(job_dir, item, set())  # new query: earlier tried videos no longer apply
    _save(job_dir, plan)
    return item


def set_enabled(job_dir: Path, item_id: int, enabled: bool) -> None:
    plan = load_plan(job_dir)
    _item(plan, item_id)["enabled"] = enabled
    _save(job_dir, plan)


def _clip_length(job_dir: Path, item: dict) -> float | None:
    """How much footage the slot's downloaded clip has (None if unknown, e.g. already cleaned up)."""
    asset = item.get("asset") or {}
    if asset.get("status") != "ok":
        return None
    if asset.get("clip_duration"):
        return float(asset["clip_duration"])
    f = job_dir / asset.get("path", "")
    if f.is_file():
        got, ok = probe_clip(f)
        return got if ok else None
    return None


def timing_limits(job_dir: Path, item_id: int) -> dict:
    """The room a slot has on the timeline: earliest start, latest end and the longest allowed length.
    Used by the editor's sliders and by `set_timing`."""
    plan = load_plan(job_dir)
    items = sorted(plan["items"], key=lambda i: i["timestamp_start"])
    item = _item(plan, item_id)
    pos = items.index(item)
    lo = INTRO_SAFE_S if pos == 0 else max(INTRO_SAFE_S, items[pos - 1]["timestamp_end"] + EDIT_GAP_S)
    duration = float(plan.get("video_duration") or 0.0)
    hi = duration if pos == len(items) - 1 else min(duration, items[pos + 1]["timestamp_start"] - EDIT_GAP_S)
    return {"min_start": round(lo, 2), "max_end": round(hi, 2), "max_len": _clip_length(job_dir, item)}


def set_timing(job_dir: Path, item_id: int, start: float, end: float, refetch: bool = False,
               free: bool = False) -> dict:
    """Move/resize a slot on the timeline. It cannot cross or touch its neighbours, so slot order (and with it
    the ids and clip file names) never changes. A slot cannot be longer than its downloaded clip unless
    `refetch=True`, which downloads a fresh (possibly different) clip for the new length.

    `free=True` (the chat assistant) drops the editor's rules: any start from 0 s, short slots, and overlapping
    other slots are all fine (the later slot is drawn on top). Only the video's length and the clip's length bind."""
    if free:
        plan = load_plan(job_dir)
        start, end = _check_free(plan, start, end)
        lim = {"max_len": _clip_length(job_dir, _item(plan, item_id))}
    else:
        start, end = round(float(start), 2), round(float(end), 2)
        if end - start < MIN_SLOT_S - 1e-6:
            raise WorkflowError(f"A slot must last at least {MIN_SLOT_S:g} seconds.")
        lim = timing_limits(job_dir, item_id)
        if start < lim["min_start"] - 1e-6:
            raise WorkflowError(f"This slot cannot start before {lim['min_start']:g}s "
                                "(the speaker owns the opening, and slots can't overlap).")
        if end > lim["max_end"] + 1e-6:
            raise WorkflowError(f"This slot cannot end after {lim['max_end']:g}s "
                                "(the video ends there, or the next slot begins).")
    if not refetch and lim["max_len"] is not None and end - start > lim["max_len"] + 0.05:
        raise WorkflowError(f"The downloaded clip is only {lim['max_len']:.1f}s long. Shorten the slot, "
                            "or fetch a longer clip (this may pick a different video).")
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    item["timestamp_start"], item["timestamp_end"] = start, end
    if refetch:
        _resource(job_dir, item, set())
    if free:
        plan["items"].sort(key=lambda i: i["timestamp_start"])  # keep the list in timeline order
    _save(job_dir, plan)
    return item


def _swap_clip_files(job_dir: Path, a: dict, b: dict) -> None:
    """Keep the invariant 'item N owns clips/clip_NN.<ext>' when two items trade clips, so a later
    'Regenerate' on one slot can never overwrite the other slot's file. Renames the files and rewrites the
    path inside each asset record (the records themselves are exchanged by the caller)."""
    moves = []  # (current file or None, new relative path, the asset record to update)
    for owner, new_owner in ((a, b), (b, a)):
        asset = owner.get("asset") or {}
        if not asset.get("path"):
            continue
        old = job_dir / asset["path"]
        moves.append((old if old.is_file() else None, f"clips/clip_{new_owner['id']:02d}{old.suffix}", asset))
    parked = []
    for old, new_rel, asset in moves:  # park first so same-named targets cannot collide
        if old is not None:
            tmp = old.with_name(old.name + ".swap")
            old.rename(tmp)
            parked.append((tmp, job_dir / new_rel))
        asset["path"] = new_rel
    for tmp, target in parked:
        tmp.replace(target)


def swap_slots(job_dir: Path, id_a: int, id_b: int) -> list[str]:
    """Trade the clips (and what goes with them: query, spoken text, framing) between two slots.
    Timing, layout and the removed/kept state stay with the slot. Returns warnings, e.g. a clip that is
    now shorter than its new slot."""
    if id_a == id_b:
        raise WorkflowError("Pick two different slots to swap.")
    plan = load_plan(job_dir)
    a, b = _item(plan, id_a), _item(plan, id_b)
    clips = {id_a: _clip_length(job_dir, a), id_b: _clip_length(job_dir, b)}  # measured before anything moves
    _swap_clip_files(job_dir, a, b)
    for field in SWAP_FIELDS:
        va, vb = a.pop(field, None), b.pop(field, None)
        if vb is not None:
            a[field] = vb
        if va is not None:
            b[field] = va
    _save(job_dir, plan)
    warnings = []
    for slot, clip_from in ((a, id_b), (b, id_a)):  # each slot now plays the other one's clip
        length, have = slot["timestamp_end"] - slot["timestamp_start"], clips[clip_from]
        if have is not None and length > have + 0.05:
            warnings.append(f"Slot #{slot['id']} is {length:.1f}s but its new clip has only {have:.1f}s. "
                            "Shorten the slot or fetch a longer clip.")
    return warnings


def set_framing(job_dir: Path, item_id: int, framing: dict) -> dict:
    """Store zoom/pan keyframes (clamped) for a slot."""
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    item["framing"] = framing_mod.normalize(framing)
    _save(job_dir, plan)
    return item


def set_layout(job_dir: Path, item_id: int, layout: str) -> dict:
    if layout not in LAYOUTS:
        raise WorkflowError(f"Unknown layout '{layout}'. Choose one of: {', '.join(LAYOUTS)}.")
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    item["layout"] = layout
    if layout == "pip":
        item["pip"] = normalize_pip(item.get("pip"))
    _save(job_dir, plan)
    return item


def set_pip(job_dir: Path, item_id: int, pos: str | None = None, scale: float | None = None) -> dict:
    """Where (and how big) a picture-in-picture slot sits. Also switches the slot to the pip layout."""
    if pos is not None and pos not in PIP_POSITIONS:
        raise WorkflowError(f"Unknown position '{pos}'. Choose one of: {', '.join(PIP_POSITIONS)}.")
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    cur = normalize_pip(item.get("pip"))
    item["pip"] = normalize_pip({"pos": pos or cur["pos"], "scale": cur["scale"] if scale is None else scale})
    item["layout"] = "pip"
    _save(job_dir, plan)
    return item


def apply_style(job_dir: Path, style: str) -> None:
    """Switch the video style and reset every slot's layout to it. Hybrid restores the AI's own picks.
    Picture-in-picture slots are a deliberate choice, so they keep their layout."""
    if style not in STYLES:
        raise WorkflowError(f"Unknown video style '{style}'.")
    write_job_meta(job_dir, style=style)
    if not (job_dir / "plan.json").is_file():
        return
    plan = load_plan(job_dir)
    items = sorted(plan["items"], key=lambda i: i["timestamp_start"])
    for item, layout in zip(items, assign_layouts([i.get("ai_layout") for i in items], style)):
        if item.get("layout") != "pip":
            item["layout"] = layout
    plan["style"] = style
    _save(job_dir, plan)


# ----------------------------------------------------------------------------- adding / replacing slots (chat)
def _check_free(plan: dict, start: float, end: float) -> tuple[float, float]:
    """The chat's only rules: stay inside the video (a range running past the end is trimmed to it) and last at
    least a moment. Starting at 0 s and overlapping other slots are fine."""
    start, end = max(round(float(start), 2), 0.0), round(float(end), 2)
    duration = float(plan.get("video_duration") or 0.0)
    if duration:
        start, end = min(start, duration), min(end, duration)
    if end - start < MIN_FREE_S - 1e-6:
        where = f" (the video is {duration:g}s long)" if duration else ""
        raise WorkflowError(f"The slot would be shorter than {MIN_FREE_S:g}s{where}. Give it a longer time range.")
    return start, end


def overlap_notes(job_dir: Path, start: float, end: float, ignore_id: int | None = None) -> list[str]:
    """Plain-text notes about slots a range covers (only meaningful in free mode), for the assistant to mention."""
    notes = []
    for it in load_plan(job_dir)["items"]:
        if it["id"] != ignore_id and it.get("enabled") is not False \
                and start < it["timestamp_end"] and end > it["timestamp_start"]:
            notes.append(f"It overlaps slot #{it['id']} ({it['timestamp_start']:g}s to {it['timestamp_end']:g}s); "
                         "where they overlap, the later-starting one is on top.")
    return notes


def _check_room(plan: dict, start: float, end: float, ignore_id: int | None = None,
                free: bool = False) -> tuple[float, float]:
    """Validate a new slot's time range against the video and the other slots; returns the rounded range."""
    if free:
        return _check_free(plan, start, end)
    start, end = round(float(start), 2), round(float(end), 2)
    if end - start < MIN_SLOT_S - 1e-6:
        raise WorkflowError(f"A slot must last at least {MIN_SLOT_S:g} seconds.")
    if start < INTRO_SAFE_S - 1e-6:
        raise WorkflowError(f"A slot cannot start before {INTRO_SAFE_S:g}s (the speaker owns the opening).")
    duration = float(plan.get("video_duration") or 0.0)
    if duration and end > duration + 1e-6:
        raise WorkflowError(f"The video is only {duration:g}s long, so a slot cannot end at {end:g}s.")
    for it in plan["items"]:
        if it["id"] == ignore_id:
            continue
        if start < it["timestamp_end"] + EDIT_GAP_S - 1e-6 and end > it["timestamp_start"] - EDIT_GAP_S + 1e-6:
            state = " (removed, but it still holds its place; delete it to free the space)" \
                if it.get("enabled") is False else ""
            raise WorkflowError(f"That overlaps slot #{it['id']} ({it['timestamp_start']:g}s to "
                                f"{it['timestamp_end']:g}s){state}.")
    return start, end


def check_room(job_dir: Path, start: float, end: float, ignore_id: int | None = None,
               free: bool = False) -> tuple[float, float]:
    """Raise WorkflowError unless a slot fits at [start, end]; used to fail fast before slow downloads. Returns the
    (possibly trimmed) range. `free=True` applies only the chat's minimal rules."""
    return _check_room(load_plan(job_dir), start, end, ignore_id, free)


def read_transcript_words(job_dir: Path) -> list[dict]:
    """Word-level transcript ([{word, start, end}]), or [] when the job has none."""
    f = job_dir / "transcript.json"
    return json.loads(f.read_text(encoding="utf-8")).get("words", []) if f.is_file() else []


def free_gaps(job_dir: Path) -> list[tuple[float, float]]:
    """Stretches of the timeline where a new slot would fit (at least MIN_SLOT_S long)."""
    plan = load_plan(job_dir)
    duration = float(plan.get("video_duration") or 0.0)
    gaps, cursor = [], INTRO_SAFE_S
    for it in sorted(plan["items"], key=lambda i: i["timestamp_start"]):
        gap_end = it["timestamp_start"] - EDIT_GAP_S
        if gap_end - cursor >= MIN_SLOT_S:
            gaps.append((round(cursor, 2), round(gap_end, 2)))
        cursor = max(cursor, it["timestamp_end"] + EDIT_GAP_S)
    if duration - cursor >= MIN_SLOT_S:
        gaps.append((round(cursor, 2), round(duration, 2)))
    return gaps


def _file_clip(job_dir: Path, clip_file: Path, item_id: int) -> str:
    """Move a finished clip to clips/clip_NN.<ext> (the slot owns that name) and return the relative path."""
    clips = job_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    target = clips / f"clip_{item_id:02d}{clip_file.suffix.lower()}"
    for old in clips.glob(f"clip_{item_id:02d}.*"):
        if old.resolve() != clip_file.resolve():
            old.unlink(missing_ok=True)  # a stale clip with another extension, or the one being replaced
    if clip_file.resolve() != target.resolve():
        shutil.move(str(clip_file), str(target))
    return f"clips/{target.name}"


def _check_clip_fits(start: float, end: float, asset: dict) -> None:
    have = asset.get("clip_duration")
    if have is not None and end - start > float(have) + 0.05:
        raise WorkflowError(f"The clip has only {float(have):.1f}s but the slot is {end - start:.1f}s. "
                            "Use a shorter slot or a longer clip.")


def add_slot(job_dir: Path, start: float, end: float, clip_file: Path, asset: dict, asset_type: str = "image",
             layout: str = "overlay", pip: dict | None = None, spoken_text: str = "", meaning: str = "",
             reason: str = "Added from the chat assistant", free: bool = False) -> dict:
    """Add a brand-new slot (e.g. a picture or a pasted YouTube clip) at [start, end]. `clip_file` is a finished
    clip; it is moved into clips/ under the new slot's id. Nothing is changed if the range doesn't fit.
    `free=True` (the chat) allows starting at 0 s and overlapping other slots."""
    if asset_type not in ("video", "image"):
        raise WorkflowError(f"Unknown asset type '{asset_type}'.")
    if layout not in LAYOUTS:
        raise WorkflowError(f"Unknown layout '{layout}'. Choose one of: {', '.join(LAYOUTS)}.")
    plan = load_plan(job_dir)
    start, end = _check_room(plan, start, end, free=free)
    _check_clip_fits(start, end, asset)
    new_id = max((i["id"] for i in plan["items"]), default=0) + 1
    asset = {**asset, "status": "ok", "path": _file_clip(job_dir, Path(clip_file), new_id)}
    item = {
        "id": new_id, "asset_type": asset_type, "timestamp_start": start, "timestamp_end": end,
        "search_query": asset.get("query_used") or asset.get("title") or "", "alternative_queries": [],
        "spoken_text": spoken_text, "english_meaning": meaning, "reasoning": reason,
        "layout": layout, "ai_layout": layout if layout in ("overlay", "split") else "overlay",
        "enabled": True, "user_edited_query": True, "asset": asset,
    }
    if layout == "pip":
        item["pip"] = normalize_pip(pip)
    plan["items"].append(item)
    plan["items"].sort(key=lambda i: i["timestamp_start"])
    _save(job_dir, plan)
    return item


def set_asset(job_dir: Path, item_id: int, clip_file: Path, asset: dict, asset_type: str = "image") -> dict:
    """Put a different finished clip into an existing slot (timing, layout and position stay)."""
    if asset_type not in ("video", "image"):
        raise WorkflowError(f"Unknown asset type '{asset_type}'.")
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    _check_clip_fits(item["timestamp_start"], item["timestamp_end"], asset)
    item["asset"] = {**asset, "status": "ok", "path": _file_clip(job_dir, Path(clip_file), item_id)}
    item["asset_type"] = asset_type
    item["framing"] = framing_mod.default_framing()
    item["enabled"] = True
    if asset.get("query_used") or asset.get("title"):
        item["search_query"] = asset.get("query_used") or asset.get("title")
    item["user_edited_query"] = True
    _save(job_dir, plan)
    return item


def delete_slot(job_dir: Path, item_id: int) -> None:
    """Remove a slot completely (unlike `set_enabled`, which keeps its place) and delete its clip file."""
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    plan["items"] = [i for i in plan["items"] if i["id"] != item_id]
    _save(job_dir, plan)
    for f in (job_dir / "clips").glob(f"clip_{item['id']:02d}.*"):
        f.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- text overlays (chat)
MIN_TEXT_S = 0.5


def _check_text(plan: dict, text: str, start: float, end: float) -> tuple[str, float, float]:
    text = " ".join(str(text).split())
    if not text:
        raise WorkflowError("The text cannot be empty.")
    if len(text) > MAX_TEXT_LEN:
        raise WorkflowError(f"Keep the text under {MAX_TEXT_LEN} characters.")
    start, end = round(float(start), 2), round(float(end), 2)
    if start < 0 or end - start < MIN_TEXT_S - 1e-6:
        raise WorkflowError(f"Text must last at least {MIN_TEXT_S:g}s and start at 0s or later.")
    duration = float(plan.get("video_duration") or 0.0)
    if duration and end > duration + 1e-6:
        raise WorkflowError(f"The video is only {duration:g}s long, so the text cannot end at {end:g}s.")
    return text, start, end


def add_text(job_dir: Path, text: str, start: float, end: float, **style) -> dict:
    """A timed text overlay on the video. `style` may hold pos, size, color, font (see rendering/ass.py)."""
    plan = load_plan(job_dir)
    text, start, end = _check_text(plan, text, start, end)
    texts = plan.setdefault("texts", [])
    entry = {"id": max((t["id"] for t in texts), default=0) + 1, "text": text, "start": start, "end": end,
             **normalize_text_style(style)}
    texts.append(entry)
    _save(job_dir, plan)
    return entry


def _text(plan: dict, text_id: int) -> dict:
    for t in plan.get("texts", []):
        if t["id"] == text_id:
            return t
    raise WorkflowError(f"No text overlay with id {text_id}.")


def update_text(job_dir: Path, text_id: int, **fields) -> dict:
    """Change any of: text, start, end, pos, size, color, font."""
    plan = load_plan(job_dir)
    entry = _text(plan, text_id)
    merged = {**entry, **{k: v for k, v in fields.items() if v is not None}}
    entry["text"], entry["start"], entry["end"] = _check_text(plan, merged["text"], merged["start"], merged["end"])
    entry.update(normalize_text_style(merged))
    _save(job_dir, plan)
    return entry


def remove_text(job_dir: Path, text_id: int) -> None:
    plan = load_plan(job_dir)
    _text(plan, text_id)
    plan["texts"] = [t for t in plan["texts"] if t["id"] != text_id]
    _save(job_dir, plan)


def set_split_face_y(job_dir: Path, value: float) -> None:
    """Where the facecam half of a split slot is cropped from the speaker video (0 = top, 1 = bottom)."""
    write_job_meta(job_dir, split_face_y=round(min(max(float(value), 0.0), 1.0), 2))


def render_final(job_dir: Path, caption_mode: str | None, auto_clean: bool = True,
                 log: Callable[[str], None] = print, root: Path | None = None) -> tuple[Path, int]:
    """Render final.mp4, then (optionally) purge temp media. Returns (final path, bytes freed)."""
    video = source_video_path(job_dir)
    if video is None:
        raise WorkflowError("The source video for this job is missing.")
    plan = load_plan(job_dir)
    plan["status"] = "approved"
    _save(job_dir, plan)
    final = rendering.run(job_dir, video, caption_mode=caption_mode, log=log)
    freed = purge_job_temp(job_dir, root=root) if auto_clean else 0
    if auto_clean:
        log("Cleaned temporary clips and audio.")
    return final, freed
