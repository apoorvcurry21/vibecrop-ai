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
