"""Caption language modes. 'original' keeps Whisper's text; 'english' and 'hinglish' rewrite it with Gemini
chunk by chunk, keeping the original timings."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src.captions.chunks import retime
from src.gemini_client import GeminiError, content, generate_with_fallback, make_client, model_chain

MODES = {"original": "Original (as spoken)", "english": "English", "hinglish": "Roman Hindi (Hinglish)"}


class CaptionError(RuntimeError):
    pass


class _Out(BaseModel):
    id: int
    text: str


class _Rewrite(BaseModel):
    chunks: list[_Out]


_COMMON = (
    "You rewrite short on-screen caption chunks for a Reel. Input is a JSON list of {id, text}; the text was "
    "auto-transcribed, so English words may be spelled phonetically in Devanagari and names may be wrong "
    "(for example 'बैलन डियोर' is 'Ballon d'Or'). Use the video topic to fix names. Return exactly one output "
    "per input with the same id, in the same order. Keep each caption about as long as the input (2-5 words), "
    "never merge or split chunks, never add commentary."
)
SYSTEM = {
    "english": _COMMON + " Translate each chunk into natural, concise English.",
    "hinglish": _COMMON + (
        " Write each chunk in Roman Hindi (Hinglish), the way people type on WhatsApp: Hindi words in Latin "
        "letters, English words in correct English spelling. Do NOT translate Hindi into English."
    ),
}


def rewrite_chunks(chunks: list[dict], mode: str, language: str, context: str = "", client: Any = None,
                   sleep: Callable[[float], None] = time.sleep, max_attempts: int = 3) -> list[dict]:
    """Return chunks with rewritten word text for `mode`; 'original' (or English->English) is a no-op."""
    if mode not in MODES:
        raise CaptionError(f"Unknown caption mode '{mode}'.")
    if mode == "original" or (mode == "english" and language == "en") or not chunks:
        return chunks
    client = client or make_client()
    payload = [{"id": c["id"], "text": " ".join(w["text"] for w in c["words"])} for c in chunks]
    history = [content("user", f"Spoken language: {language}\nVideo topic: {context or 'unknown'}\n\n"
                               + json.dumps(payload, ensure_ascii=False))]
    want = [c["id"] for c in chunks]
    problem = ""
    for _ in range(max_attempts):
        try:
            text, _model = generate_with_fallback(client, model_chain(), history, sleep, SYSTEM[mode], _Rewrite)
        except GeminiError as exc:
            raise CaptionError(str(exc)) from exc
        try:
            out = _Rewrite.model_validate(json.loads(text)).chunks
            if [o.id for o in out] != want:
                problem = f"Return exactly {len(want)} items with ids {want[0]}..{want[-1]} in order."
            elif any(not o.text.strip() for o in out):
                problem = "Every item needs non-empty text."
            else:
                by_id = {o.id: o.text.strip() for o in out}
                return [
                    {**c, "words": retime(by_id[c["id"]], c["start"], c["end"], c["words"])} for c in chunks
                ]
        except (json.JSONDecodeError, ValidationError) as exc:
            problem = f"Reply was not valid JSON for the schema: {exc}"
        history += [content("model", text), content("user", f"Fix this and return the full JSON again: {problem}")]
    raise CaptionError(f"Gemini could not rewrite the captions: {problem}")
