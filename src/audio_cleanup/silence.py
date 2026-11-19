"""Long pauses: find silences with ffmpeg `silencedetect`, then turn them into ranges to cut.

Pure audio: nothing here looks at the transcript. The retired version subtracted the transcript's word-timestamps
from every candidate pause, and on Hindi/Hinglish those timestamps are not precise enough - it kept almost every
long pause instead of cutting it. Only pauses of at least MIN_SILENCE_S are touched, and KEEP_PAD_S of each pause
is left on both sides so speech keeps a natural breath (a 1.5 s pause becomes about 1.1 s)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.config import require_binary

SILENCE_DB = -35  # quieter than this counts as silence (the detection audio is already denoised)
DETECT_MIN_S = 0.25  # silencedetect's own minimum; the real threshold is MIN_SILENCE_S
MIN_SILENCE_S = 0.3  # pauses shorter than this are natural rhythm: leave them (aggressive on purpose)
KEEP_PAD_S = 0.12  # how much of a cut pause stays on each side
MIN_CUT_S = 0.1


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


def cuts_from_silence(silences: list[tuple[float, float]], duration: float, min_s: float = MIN_SILENCE_S,
                      pad: float = KEEP_PAD_S) -> list[dict]:
    """Candidate cuts: the inside of every long-enough silence, with a small pad left on each side."""
    cuts = []
    for a, b in silences:
        if b - a < min_s:
            continue
        s = 0.0 if a < 0.05 else a + pad  # silence at the very start / end goes entirely but for the pad next to speech
        e = duration if b > duration - 0.05 else b - pad
        if e - s < MIN_CUT_S:
            continue
        cuts.append({"start": round(s, 3), "end": round(e, 3), "kind": "silence", "text": ""})
    return cuts


def detect_cuts(audio: Path, duration: float) -> list[dict]:
    return cuts_from_silence(detect(audio, duration), duration)
