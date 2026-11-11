"""Groq-hosted Whisper transcription with word- and segment-level timestamps."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.config import get_secret, whisper_model

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # Groq free-tier upload limit
_LANG_RE = re.compile(r"^[a-z]{2,3}$")


class TranscriptionError(RuntimeError):
    pass


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict or an SDK response object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize(response: Any, language: str, model: str) -> dict:
    """Convert a Groq verbose_json response into VibeCrop's transcript schema."""
    words = [
        {
            "word": str(_get(w, "word", "")).strip(),
            "start": round(float(_get(w, "start", 0.0)), 3),
            "end": round(float(_get(w, "end", 0.0)), 3),
        }
        for w in (_get(response, "words") or [])
    ]
    segments = [
        {
            "id": int(_get(s, "id", i)),
            "start": round(float(_get(s, "start", 0.0)), 3),
            "end": round(float(_get(s, "end", 0.0)), 3),
            "text": str(_get(s, "text", "")).strip(),
        }
        for i, s in enumerate(_get(response, "segments") or [])
    ]
    return {
        "language": language,
        "model": model,
        "duration": round(float(_get(response, "duration", 0.0) or 0.0), 3),
        "text": str(_get(response, "text", "")).strip(),
        "segments": segments,
        "words": words,
    }


def transcribe(audio_path: str | Path, language: str, model: str | None = None, client: Any = None) -> dict:
    """Transcribe `audio_path`. `language` is an ISO-639-1 code chosen by the user (e.g. 'en', 'hi')."""
    audio_path = Path(audio_path)
    language = (language or "").strip().lower()
    if not _LANG_RE.match(language):
        raise TranscriptionError(f"Invalid language code '{language}'. Use an ISO code such as 'en' or 'hi'.")
    if not audio_path.is_file():
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    size = audio_path.stat().st_size
    if size > MAX_AUDIO_BYTES:
        raise TranscriptionError(
            f"Audio is {size / 1e6:.1f} MB, over Groq's 25 MB limit. Use a shorter video."
        )

    model = model or whisper_model()
    if client is None:
        from groq import Groq

        client = Groq(api_key=get_secret("GROQ_API_KEY"), max_retries=3)

    try:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, audio_path.read_bytes()),
            model=model,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            temperature=0.0,
        )
    except Exception as exc:  # surface API errors without leaking request details
        raise TranscriptionError(f"Groq transcription failed: {exc}") from exc

    transcript = normalize(response, language, model)
    if not transcript["words"]:
        raise TranscriptionError("Groq returned no words. Is there speech in the video?")
    return transcript
