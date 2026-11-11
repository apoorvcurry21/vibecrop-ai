"""Snap Gemini's timestamps to spoken-word boundaries and check the plan's rules."""
from __future__ import annotations

from src.planning.prompt import (
    IMAGE_MAX_S, IMAGE_MIN_S, INTRO_SAFE_S, MAX_GAP_S, MIN_GAP_S, VIDEO_MAX_S, VIDEO_MIN_S,
)
from src.planning.schema import BRollItem

EPS = 0.01


def _bounds(asset_type: str) -> tuple[float, float]:
    return (VIDEO_MIN_S, VIDEO_MAX_S) if asset_type == "video" else (IMAGE_MIN_S, IMAGE_MAX_S)


def _nearest(values: list[float], target: float) -> float:
    return min(values, key=lambda v: abs(v - target))


def snap_item(item: BRollItem, words: list[dict], duration: float) -> BRollItem:
    """Move start/end onto word boundaries and fit the duration into the allowed range."""
    lo, hi = _bounds(item.asset_type)
    if not words:
        start, end = item.timestamp_start, item.timestamp_end
    else:
        starts = [w["start"] for w in words]
        ends = [w["end"] for w in words]
        start = _nearest(starts, item.timestamp_start)
        end = _nearest(ends, item.timestamp_end)
        if not (lo - EPS <= end - start <= hi + EPS):
            target = start + (lo + hi) / 2
            candidates = [e for e in ends if lo - EPS <= e - start <= hi + EPS]
            end = _nearest(candidates, target) if candidates else start + min(max(end - start, lo), hi)
    end = min(end, duration)
    return item.model_copy(update={"timestamp_start": round(start, 3), "timestamp_end": round(end, 3)})


def snap_and_sort(items: list[BRollItem], words: list[dict], duration: float) -> list[BRollItem]:
    return sorted((snap_item(i, words, duration) for i in items), key=lambda i: i.timestamp_start)


def validate(items: list[BRollItem], total: int, duration: float) -> list[str]:
    """Return a list of human-readable problems; empty means the plan is valid."""
    errors: list[str] = []
    if len(items) != total:
        errors.append(f"Return exactly {total} items; you returned {len(items)}.")
    prev_end = None
    for n, it in enumerate(items, 1):
        lo, hi = _bounds(it.asset_type)
        d = it.timestamp_end - it.timestamp_start
        tag = f"Item {n} ({it.asset_type} at {it.timestamp_start:.2f}s)"
        if not lo - EPS <= d <= hi + EPS:
            errors.append(f"{tag} lasts {d:.2f}s; {it.asset_type} items must last {lo:g}-{hi:g}s.")
        if it.timestamp_start < INTRO_SAFE_S - EPS:
            errors.append(f"{tag} starts before {INTRO_SAFE_S:g}s.")
        if it.timestamp_end > duration + EPS:
            errors.append(f"{tag} ends after the video ends ({duration:.2f}s).")
        n_words = len(it.search_query.split())
        if not 2 <= n_words <= 8:
            errors.append(f"{tag} search_query has {n_words} words; use one string of 3-6 plain words.")
        if len([q for q in it.alternative_queries if q.strip()]) < 2:
            errors.append(f"{tag} needs 2 alternative_queries.")
        if prev_end is not None and it.timestamp_start < prev_end + MIN_GAP_S - EPS:
            errors.append(f"{tag} overlaps or is closer than {MIN_GAP_S:g}s to the previous item.")
        if prev_end is not None and it.timestamp_start - prev_end > MAX_GAP_S + EPS:
            errors.append(
                f"No B-roll between {prev_end:.1f}s and {it.timestamp_start:.1f}s; "
                f"keep gaps under {MAX_GAP_S:g}s by adding or moving an item there."
            )
        prev_end = it.timestamp_end
    return errors
