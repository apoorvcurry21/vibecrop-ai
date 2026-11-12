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
from src.planning.layouts import LAYOUTS, STYLES, assign_layouts
from src.planning.prompt import INTRO_SAFE_S
from src.rendering import framing as framing_mod
from src.rendering import pipeline as rendering
from src.sourcing import pipeline as sourcing
from src.sourcing.youtube import _probe as probe_clip
from src.transcription import pipeline as transcription

MIN_SLOT_S = 1.5  # shortest slot the editor allows (the planner itself aims for 3-5 s)
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


def set_timing(job_dir: Path, item_id: int, start: float, end: float, refetch: bool = False) -> dict:
    """Move/resize a slot on the timeline. It cannot cross or touch its neighbours, so slot order (and with it
    the ids and clip file names) never changes. A slot cannot be longer than its downloaded clip unless
    `refetch=True`, which downloads a fresh (possibly different) clip for the new length."""
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
        raise WorkflowError(f"Unknown layout '{layout}'.")
    plan = load_plan(job_dir)
    item = _item(plan, item_id)
    item["layout"] = layout
    _save(job_dir, plan)
    return item


def apply_style(job_dir: Path, style: str) -> None:
    """Switch the video style and reset every slot's layout to it. Hybrid restores the AI's own picks."""
    if style not in STYLES:
        raise WorkflowError(f"Unknown video style '{style}'.")
    write_job_meta(job_dir, style=style)
    if not (job_dir / "plan.json").is_file():
        return
    plan = load_plan(job_dir)
    items = sorted(plan["items"], key=lambda i: i["timestamp_start"])
    for item, layout in zip(items, assign_layouts([i.get("ai_layout") for i in items], style)):
        item["layout"] = layout
    plan["style"] = style
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
