"""Video styles -> per-slot layout ("overlay" or "split").

Styles:
  overlay - every B-roll covers the whole frame over the speaker (the original look);
  split   - every B-roll takes the top half, the speaker the bottom half, with a border between;
  hybrid  - a mix: Gemini suggests a layout per slot, code keeps the mix sensible.
"""
from __future__ import annotations

STYLES = ("overlay", "split", "hybrid")
PLANNER_LAYOUTS = ("overlay", "split")  # what a video style / the AI planner can assign
LAYOUTS = PLANNER_LAYOUTS + ("pip",)  # "pip" (picture-in-picture) is only ever chosen by the user or the chat assistant
MAX_RUN = 3  # hybrid: never more than this many slots in a row with the same layout

STYLE_LABELS = {
    "overlay": "Overlay: the b-roll covers the whole frame while your voice keeps running",
    "split": "Split: b-roll on top, your facecam below, a border between them",
    "hybrid": "Hybrid: a mix of both. The AI picks the layout per b-roll and you can change it",
}


def _alternate(n: int) -> list[str]:
    return ["overlay" if i % 2 == 0 else "split" for i in range(n)]


def assign_layouts(suggested: list[str | None], style: str) -> list[str]:
    """One layout per slot for the given style.

    For hybrid, `suggested` (the AI's picks, in time order) is used as-is except that a run longer than
    MAX_RUN gets its next slot flipped, and an all-same answer (or no answer) falls back to alternating."""
    n = len(suggested)
    if style == "overlay":
        return ["overlay"] * n
    if style == "split":
        return ["split"] * n
    picks = [s if s in PLANNER_LAYOUTS else None for s in suggested]
    if n >= 2 and (None in picks or len(set(picks)) < 2):
        return _alternate(n)
    out: list[str] = []
    for p in picks:
        if len(out) >= MAX_RUN and all(x == p for x in out[-MAX_RUN:]):
            p = "split" if p == "overlay" else "overlay"
        out.append(p or "overlay")
    return out
