"""Shared Gemini plumbing: client, JSON-mode call with transient retry and model fallback."""
from __future__ import annotations

import time
from typing import Any, Callable

from src.config import GEMINI_TIMEOUT_MS, gemini_fallback_models, gemini_model, get_secret

TRANSIENT_CODES = {429, 500, 503, 504}


class GeminiError(RuntimeError):
    pass


def make_client(timeout_ms: int = GEMINI_TIMEOUT_MS) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=get_secret("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=timeout_ms),  # never hang silently
    )


def content(role: str, text: str):
    from google.genai import types

    return types.Content(role=role, parts=[types.Part(text=text)])


def generate(client: Any, model: str, history: list, sleep: Callable[[float], None],
             system_prompt: str, schema: Any) -> str:
    """One JSON-mode Gemini call; retries transient API errors (rate limit / overload) with backoff."""
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.4,
    )
    for attempt in range(1, 5):
        try:
            return client.models.generate_content(model=model, contents=history, config=config).text or ""
        except errors.APIError as exc:
            if getattr(exc, "code", None) in TRANSIENT_CODES and attempt < 4:
                sleep(5 * attempt)
                continue
            raise GeminiError(f"Gemini request failed: {exc}") from exc
    raise GeminiError("Gemini request failed after retries.")


def model_chain(primary: str | None = None) -> list[str]:
    first = primary or gemini_model()
    return [first] + [m for m in gemini_fallback_models() if m != first]


def generate_with_fallback(client: Any, models: list[str], history: list, sleep: Callable[[float], None],
                           system_prompt: str, schema: Any) -> tuple[str, str]:
    """Try each model in order; returns (text, model_used)."""
    last: Exception | None = None
    for m in models:
        try:
            return generate(client, m, history, sleep, system_prompt, schema), m
        except GeminiError as exc:
            last = exc
    raise GeminiError(str(last))


def default_sleep() -> Callable[[float], None]:
    return time.sleep
