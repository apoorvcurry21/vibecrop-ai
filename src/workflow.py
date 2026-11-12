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
from src.rendering import pipeline as rendering
from src.sourcing import pipeline as sourcing
from src.transcription import pipeline as transcription

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
               root: Path | None = None, total: int | None = None, notes: str = "") -> Path:
    """Create a job folder for a video. `copy_into_job=True` is for uploads (the copy is purged on cleanup).

    `total` (1-15) is how many B-rolls to plan; `notes` are optional creator hints for the AI."""
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
                   notes=" ".join((notes or "").split()))
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
        planning.run(job_dir, total=total or meta.get("total_brolls"), notes=meta.get("notes", ""))
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
