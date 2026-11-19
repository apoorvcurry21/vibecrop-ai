"""Stutters and retakes found straight from the transcript: pattern-matching on the words, no AI call.

A stutter is a short word/phrase said twice (or more) right next to itself ("the the", "I I I"): every copy but the
last is cut. A retake is a sentence restarted right after a failed attempt (the same first few words come back
within a few words, at a sentence boundary): the earlier attempt is cut, the last one kept.

This project used to send candidates through Groq then Gemini for a second opinion. Dropped on purpose: no API
cost or latency, nothing to go down, and the patterns below (exact repeats, sentence-boundary restarts within a
short span) are narrow enough that false positives should be rare."""
from __future__ import annotations

import re

MIN_WORDS = 6
MAX_REMOVED_FRACTION = 0.40  # cutting more than this of the video as retakes is not plausible: cut none instead
STUTTER_GAP_S = 0.8  # repeats further apart than this are not a stutter
RESTART_WORDS = 3  # a restart = the same first 3 words again
RESTART_MAX_SPAN = 25  # the failed take is at most this many words
RESTART_BOUNDARY_GAP_S = 0.25
SENTENCE_END = (".", "?", "!", "।", "…")  # includes the Devanagari danda
CUT_PAD_S = 0.05


def _norm_word(word: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]+", "", word.lower())


def _same(words: list[dict], i: int, j: int, length: int) -> bool:
    if i + length > len(words) or j + length > len(words):
        return False
    a, b = [_norm_word(w["word"]) for w in words[i:i + length]], [_norm_word(w["word"]) for w in words[j:j + length]]
    return a == b and all(a)


def stutter_at(words: list[dict], i: int) -> tuple[int, int] | None:
    """(from, to) of a stutter starting at word i: a 1-3 word unit said 2+ times in a row; everything but the LAST
    repetition is cut, so the speaker's words stay. None if there is no stutter at i."""
    for length in (1, 2, 3):
        reps = 1
        while (_same(words, i, i + reps * length, length)
               and words[i + reps * length]["start"] - words[i + reps * length - 1]["end"] < STUTTER_GAP_S):
            reps += 1
        if reps >= 2:
            return i, i + (reps - 1) * length - 1
    return None


def stutter_candidates(words: list[dict]) -> list[dict]:
    out, i = [], 0
    while i < len(words) - 1:
        hit = stutter_at(words, i)
        if hit:
            out.append({"from": hit[0], "to": hit[1], "reason": "stutter"})
            i = hit[1] + 1
        else:
            i += 1
    return out


def _starts_sentence(words: list[dict], i: int) -> bool:
    return (i == 0 or words[i - 1]["word"].rstrip().endswith(SENTENCE_END)
            or words[i]["start"] - words[i - 1]["end"] >= RESTART_BOUNDARY_GAP_S)


def restart_candidates(words: list[dict]) -> list[dict]:
    """A line that begins again right after a failed first attempt: the same first words come back within a few
    words, at a sentence boundary. The earlier attempt [i, j-1] is the candidate cut."""
    out, i, n = [], 0, len(words)
    while i < n - RESTART_WORDS:
        found = None
        if _starts_sentence(words, i):
            for j in range(i + RESTART_WORDS, min(n, i + RESTART_MAX_SPAN + 1)):
                if _starts_sentence(words, j) and _same(words, i, j, RESTART_WORDS):
                    found = j
                    break
        if found:
            out.append({"from": i, "to": found - 1, "reason": "retake"})
            i = found
        else:
            i += 1
    return out


def repair(words: list[dict], c: dict) -> dict | None:
    """A stutter cut must leave the last repetition (the raw scan can overreach); a stutter that turns out not to
    be a real repetition is dropped. Retakes pass through unchanged."""
    if c["reason"] != "stutter":
        return c
    hit = stutter_at(words, c["from"])
    return {"from": hit[0], "to": hit[1], "reason": "stutter"} if hit else None


def sanitize(raw: list[dict], n_words: int) -> list[dict]:
    """In-range proposals, sorted, overlaps merged (keeping the wider span)."""
    out = [c for c in raw if 0 <= c["from"] <= c["to"] < n_words]
    out.sort(key=lambda c: c["from"])
    merged: list[dict] = []
    for c in out:
        if merged and c["from"] <= merged[-1]["to"]:
            merged[-1]["to"] = max(merged[-1]["to"], c["to"])
        else:
            merged.append(c)
    return merged


def to_time_cuts(words: list[dict], cuts: list[dict], duration: float) -> list[dict]:
    """Word-number cuts -> time ranges, snapped just outside the removed words and clear of the kept neighbours."""
    out = []
    for c in cuts:
        a, b = c["from"], c["to"]
        lo = words[a - 1]["end"] if a > 0 else 0.0
        hi = words[b + 1]["start"] if b + 1 < len(words) else duration
        start = min(max(lo + 0.01, words[a]["start"] - CUT_PAD_S), words[a]["start"])
        end = max(min(hi - 0.01, words[b]["end"] + CUT_PAD_S), words[b]["end"])
        out.append({"start": round(start, 3), "end": round(min(end, duration), 3), "kind": c["reason"],
                    "text": " ".join(w["word"] for w in words[a:b + 1])})
    return out


def detect_cuts(words: list[dict], duration: float) -> list[dict]:
    """{"start", "end", "kind": "stutter" | "retake", "text"} for every stutter/retake found in the transcript."""
    if len(words) < MIN_WORDS:
        return []
    pool = stutter_candidates(words) + restart_candidates(words)
    repaired = [x for x in (repair(words, c) for c in pool) if x]
    # the same exact range can match both patterns (e.g. a repeated 3-word phrase); keep the retake label for it,
    # since restart_candidates is added after stutter_candidates, a later entry for the same (from, to) wins here
    unique = list({(c["from"], c["to"]): c for c in repaired}.values())
    merged = sanitize(unique, len(words))
    cuts = to_time_cuts(words, merged, duration)
    if duration > 0 and sum(c["end"] - c["start"] for c in cuts) > MAX_REMOVED_FRACTION * duration:
        return []  # implausible: distrust the scan rather than cut almost the whole video
    return cuts
