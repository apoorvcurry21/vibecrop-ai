"""Phase 1 entry point: video -> FLAC audio -> Groq Whisper -> transcript.json.

CLI:  python -m src.transcription.pipeline path\\to\\video.mp4 --language en
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import config
from src.transcription.audio import AudioExtractionError, extract_audio, probe
from src.transcription.groq_whisper import TranscriptionError, transcribe


def run(video_path: str | Path, language: str, job_dir: Path | None = None,
        model: str | None = None, client: Any = None) -> Path:
    """Run Phase 1 and return the path of the saved transcript.json."""
    video_path = Path(video_path)
    info = probe(video_path)
    if not info["has_audio"]:
        raise AudioExtractionError("This video has no audio track to transcribe.")
    limit = config.max_video_seconds()
    if info["duration"] > limit:
        raise AudioExtractionError(
            f"Video is {info['duration']:.0f}s; the limit is {limit:.0f}s (set MAX_VIDEO_SECONDS to change it)."
        )

    job_dir = job_dir or config.new_job_dir(video_path)
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = extract_audio(video_path, job_dir / "audio.flac")
    transcript = transcribe(audio_path, language, model=model, client=client)
    transcript["video"] = video_path.name
    transcript["video_duration"] = round(info["duration"], 3)

    out = job_dir / "transcript.json"
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeCrop AI - Phase 1 transcription")
    parser.add_argument("video", help="Path to the raw talking-head video")
    parser.add_argument("--language", required=True, help="Spoken language code, e.g. en, hi")
    parser.add_argument("--model", default=None, help="Override the Whisper model (default from .env / turbo)")
    args = parser.parse_args()
    try:
        out = run(args.video, args.language, model=args.model)
    except (config.ConfigError, AudioExtractionError, TranscriptionError) as exc:
        raise SystemExit(f"Error: {exc}")
    data = json.loads(out.read_text(encoding="utf-8"))
    print(f"Saved {out}")
    print(f"{len(data['segments'])} segments, {len(data['words'])} words, language={data['language']}")


if __name__ == "__main__":
    main()
