"""Build an ASS subtitle file (rendered by FFmpeg/libass). Phase 3: creator credits. Phase 4 adds captions."""
from __future__ import annotations

from pathlib import Path

OUT_W, OUT_H = 1080, 1920

HEADER = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Credit,Arial,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.5,0,2,40,40,150,1

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


def write_ass(path: Path, events: list[str]) -> Path:
    path.write_text(HEADER + "\n".join(events) + "\n", encoding="utf-8")
    return path
