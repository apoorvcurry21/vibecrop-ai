"""Talking-head cleanup: raw video -> a video with clean audio (+ its transcript) that becomes the job's source.

    1. extract the audio, remove background noise (arnndn)
    2. transcribe the DENOISED audio with Groq Whisper (cleaner audio = better words and timestamps)
    3. ONE ffmpeg pass: polish the voice (EQ, compressor, loudness) and write source_clean.mp4. The picture is copied
       as it is (no re-encode), and nothing is cut: the user cuts pauses and retakes themselves before uploading.

Nothing is cut, so the transcript's timestamps are already right for the new video. The caller (workflow) points the
job at the returned video and transcript. Nothing here touches plan.json.
"""
from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from src import config, perf
from src.rendering.compose import video_encoder_args
from src.talkinghead import denoise, voice
from src.transcription.audio import AudioExtractionError, extract_audio, probe

REPORT_NAME = "cleanup_report.json"
CLEAN_NAME = "source_clean.mp4"
CRF = 17  # only used if the picture cannot be copied: the final export encodes again, so keep this one high quality


class CleanupError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- the audio pass
def build_polish_cmd(source: Path, audio: Path, out: Path, copy_video: bool) -> list[str]:
    """Picture from `source` (copied, or re-encoded if `copy_video` is False), sound from the denoised `audio`
    with the voice chain on top."""
    video_args = ["-c:v", "copy"] if copy_video else video_encoder_args("veryfast", CRF, hw=True)
    return [config.require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0", "-af", voice.voice_filter(), *video_args,
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]


def polish_video(source: Path, audio: Path, out: Path) -> None:
    """Write `out`: the picture of `source` with the denoised, polished sound. Copying the picture is instant and keeps
    its quality; if that is not possible for this file (an odd codec for mp4) the picture is re-encoded instead."""
    for copy_video in (True, False):
        out.unlink(missing_ok=True)
        r = subprocess.run(build_polish_cmd(source, audio, out, copy_video), capture_output=True, text=True)
        if r.returncode == 0 and out.is_file():
            return
    raise CleanupError(f"FFmpeg could not write the cleaned video: {r.stderr.strip()[-500:]}")


# ---------------------------------------------------------------------------------------------------- the pipeline
@contextmanager
def _timed(job_dir: Path, timings: dict, name: str):
    t0 = time.perf_counter()
    with perf.stage(job_dir, f"cleanup:{name}"):
        yield
    timings[name] = round(time.perf_counter() - t0, 2)


def clean(job_dir: Path, source: Path, language: str, log: Callable[[str], None] = print,
          transcribe_fn: Callable[[Path, str], dict] | None = None) -> dict:
    """Run the whole cleanup. Returns {"video": Path, "transcript": dict, "report": dict} and writes the report file.
    Writes only temporary files and `source_clean.mp4` into `job_dir`."""
    job_dir = Path(job_dir)
    info = probe(source)
    if not info["has_audio"]:
        raise AudioExtractionError("This video has no audio track to clean.")
    limit = config.max_video_seconds()
    if info["duration"] > limit:
        raise AudioExtractionError(
            f"Video is {info['duration']:.0f}s; the limit is {limit:.0f}s (set MAX_VIDEO_SECONDS to change it).")
    duration = info["duration"]
    warnings: list[str] = []
    timings: dict[str, float] = {}
    temps = [job_dir / n for n in ("raw.wav", "denoised.wav", "audio.flac")]

    try:
        with _timed(job_dir, timings, "denoise"):
            log("Removing background noise...")
            raw, den = temps[0], temps[1]
            if not _extract_wav(source, raw):
                raise AudioExtractionError("Could not read the audio of this video.")
            method = denoise.denoise(raw, den, log)
        if method != denoise.ARNNDN:
            warnings.append("Used the simpler noise filter (the noise-removal model was not available).")

        with _timed(job_dir, timings, "transcribe"):
            log("Transcribing the cleaned audio (Groq Whisper)...")
            flac = extract_audio(den, temps[2])
            if transcribe_fn is not None:
                transcript = transcribe_fn(flac, language)
            else:
                from src.transcription.groq_whisper import transcribe
                transcript = transcribe(flac, language)

        out = job_dir / CLEAN_NAME
        with _timed(job_dir, timings, "polish"):
            log("Polishing the voice...")
            polish_video(source, den, out)
        new_duration = probe(out)["duration"]
        transcript = {**transcript, "duration": round(new_duration, 3), "video_duration": round(new_duration, 3),
                      "video": out.name}

        report = {"original_duration": round(duration, 2), "clean_duration": round(new_duration, 2),
                  "denoise": method, "warnings": warnings, "seconds": timings}
        (job_dir / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"video": out, "transcript": transcript, "report": report}
    finally:
        for t in temps:
            t.unlink(missing_ok=True)


def _extract_wav(source: Path, dst: Path) -> bool:
    r = subprocess.run([config.require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-vn",
                        "-ac", "1", "-ar", str(denoise.SAMPLE_RATE), str(dst)], capture_output=True, text=True)
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0
