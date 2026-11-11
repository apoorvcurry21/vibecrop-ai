"""Runtime configuration shared by every pipeline phase.

Secrets come from the environment (.env locally) or Colab Secrets. Nothing here
creates directories at import time.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"
DEFAULT_MAX_VIDEO_SECONDS = 180  # reels are 1-2 min; small guard for headroom
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"  # 2.5-* models 404 for new keys; 3.7/3.8 timed out in testing
DEFAULT_GEMINI_FALLBACKS = "gemini-3.1-flash-lite"
GEMINI_TIMEOUT_MS = 120_000
DEFAULT_TOTAL_BROLLS = 10  # the video/image split is chosen by Gemini, not fixed


class ConfigError(RuntimeError):
    """Missing key, missing binary or other setup problem the user must fix."""


def is_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def get_secret(name: str) -> str:
    """Read a secret from the environment, falling back to Colab Secrets."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if is_colab():
        try:
            from google.colab import userdata

            value = (userdata.get(name) or "").strip()
        except Exception:  # SecretNotFoundError / NotebookAccessError
            value = ""
        if value:
            return value
    where = "Colab Secrets (with Notebook access on)" if is_colab() else "your .env file"
    raise ConfigError(f"{name} is not set. Add it to {where}.")


def whisper_model() -> str:
    return os.environ.get("WHISPER_MODEL", "").strip() or DEFAULT_WHISPER_MODEL


def gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL


def gemini_fallback_models() -> list[str]:
    """Models tried in order if the primary keeps failing (overload, 404). Comma-separated in .env."""
    raw = os.environ.get("GEMINI_FALLBACK_MODELS")
    raw = DEFAULT_GEMINI_FALLBACKS if raw is None else raw
    return [m.strip() for m in raw.split(",") if m.strip()]


def enable_images() -> bool:
    """Images are off for now (video-only MVP). Set ENABLE_IMAGES=1 to let Gemini plan image B-rolls."""
    return os.environ.get("ENABLE_IMAGES", "").strip().lower() in {"1", "true", "yes"}


def total_brolls() -> int:
    raw = os.environ.get("TOTAL_BROLLS", "").strip()
    return int(raw) if raw else DEFAULT_TOTAL_BROLLS


def max_video_seconds() -> float:
    raw = os.environ.get("MAX_VIDEO_SECONDS", "").strip()
    return float(raw) if raw else float(DEFAULT_MAX_VIDEO_SECONDS)


def workdir_root() -> Path:
    """Root for temp media. Set VIBECROP_WORKDIR; falls back to Colab disk or ./workdir."""
    raw = os.environ.get("VIBECROP_WORKDIR", "").strip()
    if raw:
        return Path(raw)
    if is_colab():
        return Path("/content/vibecrop_work")
    return Path.cwd() / "workdir"


def new_job_dir(video_path: str | os.PathLike, root: Path | None = None) -> Path:
    """Create and return a fresh per-video job folder under the workdir root."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(video_path).stem).strip("_") or "video"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job = (root or workdir_root()) / "jobs" / f"{stamp}_{stem}"
    job.mkdir(parents=True, exist_ok=False)
    return job


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ConfigError(
            f"'{name}' was not found on PATH. Install FFmpeg and make sure "
            f"`{name} -version` works in a fresh terminal."
        )
    return path
