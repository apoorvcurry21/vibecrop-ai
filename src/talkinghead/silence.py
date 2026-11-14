"""Long pauses: find silences with ffmpeg `silencedetect`, then turn them into ranges to cut.

Only pauses of at least MIN_SILENCE_S are touched, and KEEP_PAD_S of each pause is left on both sides, so speech keeps
a natural breath (a 1.5 s pause becomes about 0.4 s). A cut never reaches into a word the transcript placed there."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.config import require_binary

SILENCE_DB = -35  # quieter than this counts as silence (the audio is already denoised)
DETECT_MIN_S = 0.4  # silencedetect's own minimum; the real threshold is MIN_SILENCE_S
MIN_SILENCE_S = 0.6  # pauses shorter than this are natural rhythm: leave them
KEEP_PAD_S = 0.18  # how much of a cut pause stays on each side
MIN_CUT_S = 0.15
WORD_GUARD_S = 0.03  # a cut stays this far away from any word


def detect(audio: Path, duration: float) -> list[tuple[float, float]]:
    """(start, end) of every silence in `audio`."""
    r = subprocess.run([require_binary("ffmpeg"), "-nostdin", "-hide_banner", "-i", str(audio), "-af",
                        f"silencedetect=noise={SILENCE_DB}dB:d={DETECT_MIN_S}", "-f", "null", "-"],
                       capture_output=True, text=True)
    out, start = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            start = max(float(m.group(1)), 0.0)
            continue
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and start is not None:
            out.append((start, float(m.group(1))))
            start = None
    if start is not None:  # the file ends while still silent
        out.append((start, duration))
    return out


def _subtract(span: tuple[float, float], blocks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`span` minus every block (blocks sorted by start)."""
    pieces, cur = [], span[0]
    for b0, b1 in blocks:
        if b1 <= cur or b0 >= span[1]:
            continue
        if b0 > cur:
            pieces.append((cur, b0))
        cur = max(cur, b1)
    if cur < span[1]:
        pieces.append((cur, span[1]))
    return pieces


def cuts_from_silence(silences: list[tuple[float, float]], words: list[dict], duration: float,
                      min_s: float = MIN_SILENCE_S, pad: float = KEEP_PAD_S) -> list[tuple[float, float]]:
    """The ranges to remove: the inside of every long silence, minus any word."""
    blocks = sorted((w["start"] - WORD_GUARD_S, w["end"] + WORD_GUARD_S) for w in words)
    cuts: list[tuple[float, float]] = []
    for a, b in silences:
        if b - a < min_s:
            continue
        s = 0.0 if a < 0.05 else a + pad  # silence at the very start / end goes (leaving only the pad next to speech)
        e = duration if b > duration - 0.05 else b - pad
        if e - s < MIN_CUT_S:
            continue
        cuts += [(x, y) for x, y in _subtract((s, e), blocks) if y - x >= MIN_CUT_S]
    return cuts
