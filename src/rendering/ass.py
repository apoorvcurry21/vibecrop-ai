"""Build an ASS subtitle file (rendered by FFmpeg/libass): creator credits plus highlighted captions."""
from __future__ import annotations

import os
import re
from pathlib import Path

from src.captions import styles

OUT_W, OUT_H = 1080, 1920
# Split slots: captions are centred on the border line (alignment 5 = middle of the screen) and the credit
# sits just above the border, at the bottom of the b-roll half.
SPLIT_CREDIT_MARGIN_V = OUT_H // 2 + 120
WHITE, YELLOW = "&H00FFFFFF&", "&H0000FFFF&"  # ASS colours are BGR
DEVANAGARI = re.compile("[ऀ-ॿ]")
LATIN_FONT = "Arial"
DEVA_FONT = "Nirmala UI" if os.name == "nt" else "Noto Sans Devanagari"  # Arial has no Devanagari glyphs


def bgr(rgb: str) -> str:
    """'FF3B30' (RGB hex) -> the ASS colour '&H00303BFF&' (BGR)."""
    rgb = rgb.upper()
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&"


BOX_PAD = 14  # how far the highlight box of the karaoke style reaches around the active word


def _style_line(name: str, font: str, size: int, bold: bool, primary: str, outline_colour: str, outline: float,
                shadow: float, alignment: int, margin_v: int) -> str:
    c = lambda rgb: bgr(rgb).rstrip("&")  # noqa: E731 - Style lines write colours without the closing "&"
    return (f"Style: {name},{font},{size},{c(primary)},{c(primary)},{c(outline_colour)},&H64000000,"
            f"{-1 if bold else 0},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},60,60,{margin_v},1")


def caption_style_lines(style_id: str = styles.DEFAULT_STYLE) -> list[str]:
    """ASS style lines for one caption preset: the normal and split-slot variants, in Latin and Devanagari, plus
    (karaoke) the layer that draws the highlight box behind the spoken word."""
    st = styles.get_style(style_id)
    lines = []
    for name, font, size, bold, align, margin in (
        ("CaptionLatin", st["font"], st["size"], st["bold"], 2, st["margin_v"]),
        ("CaptionDeva", DEVA_FONT, st["size_deva"], True, 2, st["margin_v"]),
        ("CaptionLatinSplit", st["font"], st["size"], st["bold"], 5, 0),
        ("CaptionDevaSplit", DEVA_FONT, st["size_deva"], True, 5, 0),
    ):
        lines.append(_style_line(name, font, size, bold, st["primary"], "000000", st["outline"], st["shadow"],
                                 align, margin))
        if st["box"]:  # same metrics, so the box lines up exactly with the word underneath
            lines.append(_style_line("Active" + name.removeprefix("Caption"), font, size, bold, st["active"],
                                     st["box_color"], BOX_PAD, 0, align, margin))
    return lines


def header(style_id: str = styles.DEFAULT_STYLE) -> str:
    fmt = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
           "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
           "MarginL, MarginR, MarginV, Encoding")
    cap = caption_style_lines(style_id)
    by_name = {line.split(":", 1)[1].split(",", 1)[0].strip(): line for line in cap}
    caption_block = "\n".join([by_name["CaptionLatin"], by_name["CaptionDeva"]])
    split_block = "\n".join([by_name["CaptionLatinSplit"], by_name["CaptionDevaSplit"]])
    active_block = "".join("\n" + line for name, line in by_name.items() if name.startswith("Active"))
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: {fmt}
Style: Credit,{LATIN_FONT},34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.5,0,2,40,40,150,1
{caption_block}
Style: CreditSplit,{LATIN_FONT},34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2.5,0,2,40,40,{SPLIT_CREDIT_MARGIN_V},1
{split_block}
Style: Text,{LATIN_FONT},104,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,2,8,60,60,0,1{active_block}

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


def credit_event(start: float, end: float, channel: str, split: bool = False, text: str | None = None) -> str:
    """The small credit line. `text` overrides the default "Video: <channel>" (e.g. "Image: example.com")."""
    style = "CreditSplit" if split else "Credit"
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{escape(text or 'Video: ' + channel)}"


# ---------------------------------------------------------------- text overlays (set from the chat assistant)
# position -> (ASS alignment, x, y). The anchor point is where the text is pinned, so "bottom" text grows upwards.
TEXT_POSITIONS = {
    "top-left": (7, 60, 200), "top": (8, 540, 200), "top-right": (9, 1020, 200),
    "center-left": (4, 60, 960), "center": (5, 540, 960), "center-right": (6, 1020, 960),
    "bottom-left": (1, 60, 1260), "bottom": (2, 540, 1260), "bottom-right": (3, 1020, 1260),
}
TEXT_SIZES = {"small": 54, "medium": 76, "large": 104, "xl": 140}
TEXT_COLORS = {
    "white": "FFFFFF", "yellow": "FFD60A", "red": "FF3B30", "green": "34C759", "blue": "0A84FF",
    "orange": "FF9500", "pink": "FF2D92", "black": "000000", "cyan": "40DCFF",
}
TEXT_FONTS = {"bold": "Arial", "impact": "Impact", "serif": "Georgia"}  # all drawn bold with a dark outline
TEXT_DEFAULT = {"pos": "top", "size": "large", "color": "white", "font": "bold"}
MAX_TEXT_LEN = 120
HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")


def normalize_text_style(spec: dict | None) -> dict:
    """Validate pos/size/color/font. Unknown values fall back to the defaults; size may also be a number of
    pixels (30-220) and color a '#RRGGBB' hex."""
    spec = spec or {}
    pos = spec.get("pos") if spec.get("pos") in TEXT_POSITIONS else TEXT_DEFAULT["pos"]
    size = spec.get("size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        size = int(min(max(size, 30), 220))
    elif size not in TEXT_SIZES:
        size = TEXT_DEFAULT["size"]
    color = str(spec.get("color") or "").strip().lower()
    if color not in TEXT_COLORS:
        m = HEX_COLOR.match(color)
        color = m.group(1).upper() if m else TEXT_DEFAULT["color"]
    font = spec.get("font") if spec.get("font") in TEXT_FONTS else TEXT_DEFAULT["font"]
    return {"pos": pos, "size": size, "color": color, "font": font}


def _ass_color(color: str) -> str:
    rgb = TEXT_COLORS.get(color, color)
    return f"&H00{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&"  # ASS colours are BGR


def text_events(texts: list[dict]) -> list[str]:
    """One timed text overlay per entry of plan["texts"] (each {text, start, end, pos, size, color, font})."""
    events = []
    for t in texts or []:
        body = escape(str(t.get("text", "")))[:MAX_TEXT_LEN]
        if not body or float(t.get("end", 0)) <= float(t.get("start", 0)):
            continue
        s = normalize_text_style(t)
        an, x, y = TEXT_POSITIONS[s["pos"]]
        size = TEXT_SIZES.get(s["size"], s["size"])
        font = DEVA_FONT if DEVANAGARI.search(body) else TEXT_FONTS[s["font"]]
        tags = f"{{\\an{an}\\pos({x},{y})\\fn{font}\\fs{size}\\c{_ass_color(s['color'])}\\fad(150,150)}}"
        events.append(f"Dialogue: 2,{ass_time(float(t['start']))},{ass_time(float(t['end']))},Text,,0,0,0,,{tags}{body}")
    return events


POP_FROM = 0.82  # the spoken word starts at this fraction of its size and grows to full size (bold pop style)
POP_MS = 90


def _num(x: float) -> str:
    return f"{x:.0f}"


def _word_markup(text: str, active: bool, emph: dict | None, st: dict, layer: str = "base") -> str:
    """ASS text for one word. `layer` is "base" (the whole line) or "box" (the karaoke highlight layer, where only
    the spoken word is visible). Both layers carry the same size tags on every word, so the two lines have the
    same width and height and the box lines up exactly with its word."""
    body = escape(text)
    if st["uppercase"]:
        body = body.upper()
    e_color = emph.get("color") if emph else None
    scale = float(emph.get("scale", 1.0)) if emph else 1.0
    pop = st["pop"] and active and layer == "base"
    size = ""
    if pop:
        base = scale * 100
        size = (f"\\fscx{_num(base * POP_FROM)}\\fscy{_num(base * POP_FROM)}"
                f"\\t(0,{POP_MS},\\fscx{_num(base)}\\fscy{_num(base)})")
    elif scale != 1.0:
        size = f"\\fscx{_num(scale * 100)}\\fscy{_num(scale * 100)}"

    if layer == "box":
        if not active:  # invisible, but it still takes up its space
            return f"{{\\alpha&HFF&{size}}}{body}{{\\r}}"
        tags = (f"\\c{bgr(e_color)}" if e_color else "") + size
        return f"{{{tags}}}{body}{{\\r}}" if tags else body

    color = e_color or (st["active"] if active and not st["box"] else None)
    tags = (f"\\c{bgr(color)}" if color else "") + size
    if not tags:
        return body
    if color and not size:
        return f"{{{tags}}}{body}{{\\c{bgr(st['primary'])}}}"  # the long-standing classic form
    return f"{{{tags}}}{body}{{\\r}}"


def caption_events(chunks: list[dict], split_spans: list[tuple[float, float]] | None = None,
                   style_id: str = styles.DEFAULT_STYLE, emphasis: list[dict] | None = None) -> list[str]:
    """One event per spoken word: the whole chunk is shown and the spoken word is highlighted the way the style
    says (yellow, green + pop, or a highlight box). Words covered by an `emphasis` entry get their own colour/size.

    Time inside a split slot (`split_spans`) uses the centred-on-the-border caption style. A word that straddles
    a slot boundary is cut there, so the caption moves exactly when the layout changes."""
    st = styles.get_style(style_id)
    spans = split_spans or []
    emph = emphasis or []
    events: list[str] = []
    for ci, chunk in enumerate(chunks):
        words = chunk["words"]
        if not words:
            continue
        base_style = "CaptionDeva" if any(DEVANAGARI.search(w["text"]) for w in words) else "CaptionLatin"
        next_start = chunks[ci + 1]["words"][0]["start"] if ci + 1 < len(chunks) and chunks[ci + 1]["words"] else None
        marks = [styles.emphasis_for(w, emph) for w in words]
        for i, w in enumerate(words):
            start = w["start"]
            if i + 1 < len(words):
                end = words[i + 1]["start"]
            else:
                end = chunk["end"] + 0.15
                if next_start is not None:
                    end = min(end, next_start)  # never overlap the next caption
            end = max(end, start + 0.05)
            line = " ".join(_word_markup(x["text"], j == i, marks[j], st) for j, x in enumerate(words))
            box_line = None
            if st["box"]:
                box_line = " ".join(_word_markup(x["text"], j == i, marks[j], st, layer="box")
                                    for j, x in enumerate(words))
            cuts = sorted({c for span in spans for c in span if start < c < end})
            points = [start, *cuts, end]
            for seg_start, seg_end in zip(points, points[1:]):
                suffix = "Split" if any(a <= seg_start < b for a, b in spans) else ""
                t0, t1 = ass_time(seg_start), ass_time(seg_end)
                events.append(f"Dialogue: 1,{t0},{t1},{base_style}{suffix},,0,0,0,,{line}")
                if box_line is not None:
                    # Its own layer: libass pushes overlapping events of the SAME layer apart (so the box would
                    # float above the line), but events on different layers are drawn in place, on top.
                    active_style = "Active" + base_style.removeprefix("Caption") + suffix
                    events.append(f"Dialogue: 2,{t0},{t1},{active_style},,0,0,0,,{box_line}")
    return events


def write_ass(path: Path, events: list[str], style_id: str = styles.DEFAULT_STYLE) -> Path:
    path.write_text(header(style_id) + "\n".join(events) + "\n", encoding="utf-8")
    return path
