"""Purge temporary media after a render, and report disk usage. Refuses to touch anything outside the jobs folder."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.config import workdir_root

KEEP_NAMES = {"final.mp4", "plan.json", "transcript.json", "job.json"}
KEEP_PREFIXES = ("captions_",)  # cached caption rewrites are tiny and avoid repeat Gemini calls
BUDGET_GB = 32.0


class CleanupError(RuntimeError):
    pass


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def purge_job_temp(job_dir: str | Path, root: Path | None = None) -> int:
    """Delete clips, audio, subtitle files and any copied source video; keep the final video and JSON.

    Returns bytes freed. Raises CleanupError unless `job_dir` is directly inside <workdir>/jobs.
    """
    job = Path(job_dir).resolve()
    jobs_root = ((root or workdir_root()) / "jobs").resolve()
    if job.parent != jobs_root or not job.is_dir():
        raise CleanupError(f"Refusing to clean '{job}': it is not a job folder inside {jobs_root}.")
    freed = 0
    for p in job.iterdir():
        if p.name in KEEP_NAMES or p.name.startswith(KEEP_PREFIXES):
            continue
        freed += _size(p)
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return freed


def workdir_usage_bytes(root: Path | None = None) -> int:
    base = root or workdir_root()
    return _size(base) if base.exists() else 0


def over_budget(root: Path | None = None, budget_gb: float = BUDGET_GB) -> bool:
    return workdir_usage_bytes(root) > budget_gb * 1024**3


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"
