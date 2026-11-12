"""Gemini as the video editor: transcript in, validated B-roll plan out."""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from src.config import enable_images, gemini_fallback_models, gemini_model, total_brolls
from src.gemini_client import GeminiError, content as _content, generate, make_client
from src.planning.layouts import assign_layouts
from src.planning.prompt import build_user_prompt, get_system_prompt
from src.planning.schema import PlanResponse
from src.planning.validate import resolve_layout, snap_and_sort, validate


class PlanningError(RuntimeError):
    pass


def _generate(client: Any, model: str, history: list, sleep: Callable[[float], None],
              system_prompt: str) -> str:
    try:
        return generate(client, model, history, sleep, system_prompt, PlanResponse)
    except GeminiError as exc:
        raise PlanningError(str(exc)) from exc


def plan(transcript: dict, total: int | None = None, model: str | None = None, client: Any = None,
         max_attempts: int = 3, sleep: Callable[[float], None] = time.sleep,
         images: bool | None = None, notes: str = "", style: str = "overlay") -> dict:
    """Ask Gemini for `total` B-rolls, snap them to word boundaries, validate, retry with feedback.

    `notes` are optional creator hints (correct names, topics, things to show or avoid).
    `style` ("overlay" | "split" | "hybrid") decides each slot's layout; only hybrid asks Gemini for a mix."""
    total = total or total_brolls()
    images = enable_images() if images is None else images
    system_prompt = get_system_prompt(images, hybrid=style == "hybrid")
    models = [model or gemini_model()] + [m for m in gemini_fallback_models() if m != (model or gemini_model())]
    if client is None:
        client = make_client()

    duration = float(transcript.get("video_duration") or transcript.get("duration") or 0.0)
    history = [_content("user", build_user_prompt(transcript, total, notes))]
    errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        while True:
            try:
                text = _generate(client, models[0], history, sleep, system_prompt)
                break
            except PlanningError:
                if len(models) == 1:
                    raise
                models.pop(0)  # primary exhausted its retries: fall back to the next model
        try:
            parsed = PlanResponse.model_validate(json.loads(text))
            words = transcript.get("words", [])
            items = resolve_layout(snap_and_sort(parsed.items, words, duration), words, duration)
            errors = validate(items, total, duration, allow_images=images)
        except (json.JSONDecodeError, ValidationError) as exc:
            parsed, items, errors = None, [], [f"Your reply was not valid JSON for the schema: {exc}"]
        if not errors:
            ai_picks = [it.layout for it in items]
            layouts = assign_layouts(ai_picks, style)
            rows = [{"id": n, **it.model_dump(), "layout": lay} for n, (it, lay) in enumerate(zip(items, layouts), 1)]
            if style == "hybrid":
                for row, pick in zip(rows, ai_picks):
                    row["ai_layout"] = pick  # remembered so "apply the hybrid style" can restore the AI's mix
            return {
                "language": transcript.get("language"),
                "video_duration": duration,
                "model": models[0],  # the model that actually produced the plan
                "video_topic": parsed.video_topic,
                "notes": " ".join((notes or "").split()),
                "status": "draft",  # becomes "approved" after the user reviews it in the UI
                "style": style,
                "items": rows,
            }
        history += [
            _content("model", text),
            _content("user", "Fix these problems and return the full corrected JSON:\n- " + "\n- ".join(errors)),
        ]
    raise PlanningError(f"Gemini could not produce a valid plan after {max_attempts} attempts: " + "; ".join(errors))
