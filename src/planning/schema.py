"""Typed schema Gemini must return. Count/timing rules are enforced in validate.py."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class BRollItem(BaseModel):
    asset_type: Literal["video", "image"]
    timestamp_start: float  # seconds
    timestamp_end: float  # seconds
    search_query: str  # ONE English query, 3-6 plain words (the user can edit it in the UI)
    alternative_queries: list[str]  # 2 different-angle English queries (used by "Regenerate")
    spoken_text: str  # words from the transcript this B-roll covers
    english_meaning: str  # short English meaning of what is being said
    reasoning: str  # why this visual fits the meaning (one sentence)
    layout: Literal["overlay", "split"] = "overlay"  # only asked for in the hybrid style (see planning/layouts.py)


class PlanResponse(BaseModel):
    video_topic: str  # one-line English summary of the whole video
    items: list[BRollItem]
