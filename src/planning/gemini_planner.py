"""Gemini as the video editor: transcript in, validated B-roll plan out."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from src.config import GEMINI_TIMEOUT_MS, gemini_fallback_models, gemini_model, get_secret, total_brolls
from src.planning.prompt import SYSTEM_PROMPT, build_user_prompt
from src.planning.schema import PlanResponse
from src.planning.validate import snap_and_sort, validate

TRANSIENT_CODES = {429, 500, 503, 504}


class PlanningError(RuntimeError):
    pass


def _content(role: str, text: str):
    from google.genai import types

    return types.Content(role=role, parts=[types.Part(text=text)])


def _generate(client: Any, model: str, history: list, sleep: Callable[[float], None]) -> str:
    """One Gemini call; retries transient API errors (rate limit / overload) with backoff."""
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=PlanResponse,
        temperature=0.4,
    )
    for attempt in range(1, 5):
        try:
            return client.models.generate_content(model=model, contents=history, config=config).text or ""
        except errors.APIError as exc:
            if getattr(exc, "code", None) in TRANSIENT_CODES and attempt < 4:
                sleep(5 * attempt)
                continue
            raise PlanningError(f"Gemini request failed: {exc}") from exc
    raise PlanningError("Gemini request failed after retries.")


def plan(transcript: dict, total: int | None = None, model: str | None = None, client: Any = None,
         max_attempts: int = 3, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Ask Gemini for `total` B-rolls, snap them to word boundaries, validate, retry with feedback."""
    total = total or total_brolls()
    models = [model or gemini_model()] + [m for m in gemini_fallback_models() if m != (model or gemini_model())]
    if client is None:
        from google import genai

        from google.genai import types

        client = genai.Client(
            api_key=get_secret("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),  # never hang silently
        )

    duration = float(transcript.get("video_duration") or transcript.get("duration") or 0.0)
    history = [_content("user", build_user_prompt(transcript, total))]
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        while True:
            try:
                text = _generate(client, models[0], history, sleep)
                break
            except PlanningError:
                if len(models) == 1:
                    raise
                models.pop(0)  # primary exhausted its retries: fall back to the next model
        try:
            parsed = PlanResponse.model_validate(json.loads(text))
            items = snap_and_sort(parsed.items, transcript.get("words", []), duration)
            errors = validate(items, total, duration)
        except (json.JSONDecodeError, ValidationError) as exc:
            parsed, items, errors = None, [], [f"Your reply was not valid JSON for the schema: {exc}"]
        if not errors:
            return {
                "language": transcript.get("language"),
                "video_duration": duration,
                "model": models[0],  # the model that actually produced the plan
                "video_topic": parsed.video_topic,
                "status": "draft",  # becomes "approved" after the user reviews it in the UI
                "items": [{"id": n, **it.model_dump()} for n, it in enumerate(items, 1)],
            }
        history += [
            _content("model", text),
            _content("user", "Fix these problems and return the full corrected JSON:\n- " + "\n- ".join(errors)),
        ]
    raise PlanningError(f"Gemini could not produce a valid plan after {max_attempts} attempts: " + "; ".join(errors))
