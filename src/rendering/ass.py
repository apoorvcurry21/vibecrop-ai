"""Build an ASS subtitle file (rendered by FFmpeg/libass): creator credits plus highlighted captions."""
from __future__ import annotations

import os
import re
from pathlib import Path

OUT_W, OUT_H = 1080, 1920
WHITE, YELLOW = "&H00FFFFFF&", "&H0000FFFF&"  # ASS colours are BGR
DEVANAGARI = re.compile("[ऀ-ॿ]")
LATIN_FONT = "Arial"
DEVA_FONT = "Nirmala UI" if os.name == "nt" else "Noto Sans Devanagari"  # Arial has no Devanagari glyphs


def header() -> str:
    fmt = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
           "MarginL, MarginR, MarginV, Encoding")
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: {fmt}
Style: Credit,{LATIN_FONT},34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.5,0,2,40,40,150,1
Style: CaptionLatin,{LATIN_FONT},74,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,400,1
Style: CaptionDeva,{DEVA_FONT},70,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(seconds: float) -> str:
    cs = int(round(max(seconds, 0.0) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    """Strip characters that libass would treat as override tags or line breaks."""
    return " ".join(text.replace("{", "(").replace("}", ")").replace("\\", "/").split())


def credit_event(start: float, end: float, channel: str) -> str:
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Credit,,0,0,0,,{escape('Video: ' + channel)}"


def caption_events(chunks: list[dict]) -> list[str]:
    """One event per spoken word: the whole chunk is shown, the active word is yellow."""
    events: list[str] = []
    for ci, chunk in enumerate(chunks):
        words = chunk["words"]
        if not words:
            continue
        style = "CaptionDeva" if any(DEVANAGARI.search(w["text"]) for w in words) else "CaptionLatin"
        next_start = chunks[ci + 1]["words"][0]["start"] if ci + 1 < len(chunks) and chunks[ci + 1]["words"] else None
        for i, w in enumerate(words):
            start = w["start"]
            if i + 1 < len(words):
                end = words[i + 1]["start"]
            else:
                end = chunk["end"] + 0.15
                if next_start is not None:
                    end = min(end, next_start)  # never overlap the next caption
            end = max(end, start + 0.05)
            line = " ".join(
                f"{{\\c{YELLOW}}}{escape(x['text'])}{{\\c{WHITE}}}" if j == i else escape(x["text"])
                for j, x in enumerate(words)
            )
            events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{line}")
    return events


def write_ass(path: Path, events: list[str]) -> Path:
    path.write_text(header() + "\n".join(events) + "\n", encoding="utf-8")
    return path
