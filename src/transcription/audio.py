"""Audio extraction: any video -> 16 kHz mono FLAC (Whisper's native format)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.config import require_binary


class AudioExtractionError(RuntimeError):
    pass


def probe(video_path: Path) -> dict:
    """Return {'duration': float, 'has_audio': bool} using ffprobe."""
    ffprobe = require_binary("ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration:stream=codec_type",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioExtractionError(f"ffprobe could not read '{video_path}': {result.stderr.strip()}")
    info = json.loads(result.stdout or "{}")
    duration = float(info.get("format", {}).get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return {"duration": duration, "has_audio": has_audio}


def extract_audio(video_path: str | Path, out_path: str | Path) -> Path:
    """Extract the audio track of `video_path` to `out_path` as 16 kHz mono FLAC."""
    video_path, out_path = Path(video_path), Path(out_path)
    if not video_path.is_file():
        raise AudioExtractionError(f"Video not found: {video_path}")
    ffmpeg = require_binary("ffmpeg")
    cmd = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.is_file():
        raise AudioExtractionError(f"FFmpeg audio extraction failed: {result.stderr.strip()}")
    return out_path
