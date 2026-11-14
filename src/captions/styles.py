"""Caption style presets and word emphasis.

A style is plain data (font, size, colours, how many words per line, whether the active word pops or gets a
highlight box). `rendering/ass.py` turns it into ASS styles and events. `plan["captions"]` stores the user's pick:

    {"style": "bold_pop", "emphasis": [{"id": 1, "start": 3.2, "end": 3.7, "text": "never", "color": "FF3B30", "scale": 1.4}]}

Emphasis is keyed by time (not by word index) so it still lands on the right word when the caption language
is rewritten to English / Hinglish (the word count differs there, the timing does not).
"""
from __future__ import annotations

import re

HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")
DEFAULT_STYLE = "classic"
MIN_SCALE, MAX_SCALE = 1.0, 1.8
DEFAULT_EMPHASIS_SCALE = 1.35
MAX_EMPHASIS = 60

# ASS colours are BGR: "&H00BBGGRR&".
STYLES: dict[str, dict] = {
    "classic": {
        "label": "Classic",
        "about": "White bold text, black outline, the word being spoken turns yellow.",
        "font": "Arial", "size": 74, "size_deva": 70, "bold": True,
        "outline": 7, "shadow": 2, "uppercase": False, "max_words": 4,
        "primary": "FFFFFF", "active": "FFFF00",  # RGB hex; the yellow the app has always used
        "pop": False, "box": False, "margin_v": 400,
    },
    "bold_pop": {
        "label": "Bold pop (Hormozi)",
        "about": "Very large ALL CAPS, 1-3 words at a time, the spoken word turns green and pops.",
        "font": "Impact", "size": 104, "size_deva": 84, "bold": False,
        "outline": 9, "shadow": 3, "uppercase": True, "max_words": 3,
        "primary": "FFFFFF", "active": "39FF14",
        "pop": True, "box": False, "margin_v": 400,
    },
    "karaoke_box": {
        "label": "Karaoke box",
        "about": "Clean bold text; the spoken word sits on a yellow highlight box.",
        "font": "Arial", "size": 78, "size_deva": 72, "bold": True,
        "outline": 6, "shadow": 0, "uppercase": False, "max_words": 3,
        "primary": "FFFFFF", "active": "000000",  # active text is black, on the box
        "box_color": "FFD60A",
        "pop": False, "box": True, "margin_v": 400,
    },
}
NAMED_COLORS = {
    "white": "FFFFFF", "yellow": "FFD60A", "red": "FF3B30", "green": "34C759", "blue": "0A84FF",
    "orange": "FF9500", "pink": "FF2D92", "black": "000000", "cyan": "40DCFF",
}


class CaptionStyleError(ValueError):
    pass


def style_ids() -> list[str]:
    return list(STYLES)


def get_style(style_id: str | None) -> dict:
    return STYLES.get(style_id or DEFAULT_STYLE, STYLES[DEFAULT_STYLE])


def normalize_color(color) -> str | None:
    """'red' / '#FF3B30' / 'ff3b30' -> 'FF3B30'. None or '' means "keep the style's own colour"."""
    if color is None or str(color).strip() == "":
        return None
    c = str(color).strip().lower()
    if c in NAMED_COLORS:
        return NAMED_COLORS[c]
    m = HEX_COLOR.match(c)
    if not m:
        raise CaptionStyleError(f"Unknown colour '{color}'. Use a name ({', '.join(NAMED_COLORS)}) or #RRGGBB.")
    return m.group(1).upper()


def normalize_scale(scale) -> float:
    try:
        s = float(DEFAULT_EMPHASIS_SCALE if scale is None else scale)
    except (TypeError, ValueError) as exc:
        raise CaptionStyleError("Scale must be a number between 1.0 and 1.8.") from exc
    return round(min(max(s, MIN_SCALE), MAX_SCALE), 2)


def caption_settings(plan: dict | None) -> dict:
    """The plan's caption settings with defaults filled in and unknown values dropped."""
    raw = (plan or {}).get("captions") or {}
    style = raw.get("style") if raw.get("style") in STYLES else DEFAULT_STYLE
    emphasis = [e for e in raw.get("emphasis") or [] if isinstance(e, dict) and "start" in e and "end" in e]
    return {"style": style, "emphasis": emphasis}


def fit_chunks(chunks: list[dict], max_words: int) -> list[dict]:
    """Split chunks longer than `max_words` into balanced pieces (4 words at max 3 -> 2 + 2, not 3 + 1).
    Works on already rewritten chunks too, so a style change never costs another Gemini call."""
    out: list[dict] = []
    for chunk in chunks:
        words = chunk.get("words") or []
        if len(words) <= max_words:
            out.append(chunk)
            continue
        parts = -(-len(words) // max_words)  # ceil
        size, extra = divmod(len(words), parts)
        i = 0
        for p in range(parts):
            n = size + (1 if p < extra else 0)
            piece = words[i:i + n]
            i += n
            out.append({"start": piece[0]["start"], "end": piece[-1]["end"], "words": piece})
    return [{**c, "id": n} for n, c in enumerate(out)]


def emphasis_for(word: dict, emphasis: list[dict]) -> dict | None:
    """The emphasis entry that covers this word (its midpoint falls inside the entry's time range)."""
    mid = (float(word["start"]) + float(word["end"])) / 2
    for e in emphasis:
        if float(e["start"]) - 0.02 <= mid <= float(e["end"]) + 0.02:
            return e
    return None
