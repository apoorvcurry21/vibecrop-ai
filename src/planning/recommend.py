"""Suggest how many B-rolls suit a reel. Gemini reads the transcript and answers with a count and a one-line
reason; the user sees it as the preset of the count slider and can still change it.

If Gemini is unavailable (no key, overloaded, bad reply) a length-based formula is used instead, so analysing a
video never fails because of this step."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from src import config
from src.gemini_client import GeminiError, content, generate_with_fallback, make_client, model_chain
from src.planning.prompt import VIDEO_MAX_S, VIDEO_MIN_S, feasible_max

SECONDS_PER_BROLL = 8.0  # the formula fallback, and the guidance given to Gemini


class CountSuggestion(BaseModel):
    count: int  # how many B-roll clips to lay over the video
    reason: str  # one short English sentence the user will read


SYSTEM_PROMPT = f"""You advise a short-form video editor (Instagram Reels / YouTube Shorts) on how MANY B-roll clips to lay over a talking-head video. You receive the timestamped transcript.

Decide from the CONTENT, not just the length:
- Count real topic shifts and concrete, filmable subjects (people, places, events, objects, actions, emotions). More distinct visual moments = more B-roll.
- Fast, punchy, story-driven speech with many scene changes suits more B-roll; slow, abstract, personal or advice-style talk where the speaker's face matters suits fewer.
- Each clip lasts {VIDEO_MIN_S:g}-{VIDEO_MAX_S:g} seconds and clips need breathing room between them, so B-roll should cover roughly a third to a half of the video at most. A typical reel needs about one clip per {SECONDS_PER_BROLL:g} seconds, more or fewer depending on the content.
- Never exceed the maximum stated in the request.

Reply with JSON: "count" (an integer) and "reason" (ONE short English sentence, at most 20 words, saying why this number suits the video; no preamble)."""


def clamp_count(duration: float, n: int) -> int:
    """Keep a suggestion inside what the app allows and what physically fits the video."""
    return max(1, min(config.MAX_BROLLS, feasible_max(duration), int(n)))


def formula_count(duration: float) -> int:
    return clamp_count(duration, round(duration / SECONDS_PER_BROLL))


def _user_prompt(transcript: dict, duration: float) -> str:
    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in transcript.get("segments", [])]
    return (
        f"Spoken language: {transcript.get('language', 'unknown')}\n"
        f"Video duration: {duration:.1f} seconds\n"
        f"Maximum allowed: {clamp_count(duration, config.MAX_BROLLS)}\n\n"
        "TRANSCRIPT\n" + "\n".join(lines)
    )


def recommend_count(transcript: dict, client: Any = None, models: list[str] | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> dict:
    """Returns {"count", "reason", "source"}: source is "gemini", or "formula" when Gemini could not answer
    (the reason then says so)."""
    duration = float(transcript.get("video_duration") or transcript.get("duration") or 0.0)
    try:
        client = client or make_client()
        text, _ = generate_with_fallback(client, models or model_chain(), [content("user", _user_prompt(transcript, duration))],
                                         sleep, SYSTEM_PROMPT, CountSuggestion)
        parsed = CountSuggestion.model_validate(json.loads(text))
        return {"count": clamp_count(duration, parsed.count), "reason": " ".join(parsed.reason.split()),
                "source": "gemini"}
    except (GeminiError, config.ConfigError, ValidationError, json.JSONDecodeError) as exc:
        n = formula_count(duration)
        why = "no Gemini key" if isinstance(exc, config.ConfigError) else "the AI did not answer"
        return {"count": n, "source": "formula",
                "reason": f"About one every {SECONDS_PER_BROLL:g} seconds of video ({why}, so this is a length-based guess)."}
