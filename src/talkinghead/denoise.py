"""Background-noise removal with ffmpeg's `arnndn` (RNNoise, a small neural denoiser that runs on the CPU).

`arnndn` needs a model file (~300 KB). It is downloaded once from the public GregorR/rnnoise-models repository into
the models folder. If the download is impossible, a model-free filter (`afftdn`) is used instead and the cleanup
report says so."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import requests

from src import config
from src.config import require_binary

MODEL_NAME = "bd.rnnn"  # "beguiling-drafter": the general-purpose speech/noise model
MODEL_URL = "https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafter-2018-08-30/bd.rnnn"
MIN_MODEL_BYTES = 100_000
SAMPLE_RATE = 48000  # RNNoise works at 48 kHz

ARNNDN = "arnndn"
AFFTDN = "afftdn"


class DenoiseError(RuntimeError):
    pass


def ensure_model(log: Callable[[str], None] = print) -> Path | None:
    """The RNNoise model file, downloading it on first use. None if it cannot be had."""
    target = config.models_dir() / MODEL_NAME
    if target.is_file() and target.stat().st_size >= MIN_MODEL_BYTES:
        return target
    try:
        log("Downloading the noise-removal model (one time, 300 KB)...")
        r = requests.get(MODEL_URL, timeout=30)
        r.raise_for_status()
        if len(r.content) < MIN_MODEL_BYTES:
            raise ValueError("the file is too small to be the model")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(target)
        return target
    except Exception as exc:  # noqa: BLE001 - offline, blocked, a changed URL: fall back, do not fail the job
        log(f"Could not get the noise-removal model ({str(exc)[:80]}); using the simpler filter instead.")
        return None


def _run(src: Path, dst: Path, af: str, cwd: Path | None = None) -> bool:
    r = subprocess.run([require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                        "-af", af, "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)],
                       capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def denoise(src: Path, dst: Path, log: Callable[[str], None] = print) -> str:
    """Write a denoised mono 48 kHz WAV of `src`'s audio to `dst`. Returns the method used ("arnndn" or "afftdn")."""
    model = ensure_model(log)
    if model is not None:
        # The model is referenced by file name with cwd = its folder: no drive-letter escaping in the filter.
        if _run(src, dst, f"highpass=f=60,arnndn=m={model.name}", cwd=model.parent):
            return ARNNDN
        log("The noise-removal filter failed; using the simpler one instead.")
    if _run(src, dst, "highpass=f=80,afftdn=nf=-30"):
        return AFFTDN
    raise DenoiseError("Could not process the audio of this video.")
