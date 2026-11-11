"""Group Whisper word timestamps into short caption chunks (2-4 words) and re-time rewritten text."""
from __future__ import annotations

SENTENCE_END = (".", "?", "!", "।", "…")  # includes the Devanagari danda


def chunk_words(words: list[dict], max_words: int = 4, max_chars: int = 26, max_gap: float = 0.6) -> list[dict]:
    """Return [{'id','start','end','words':[{'text','start','end'}]}], breaking at pauses and sentence ends."""
    chunks: list[list[dict]] = []
    cur: list[dict] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            chunks.append(cur)
            cur = []

    for w in words:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["text"]) + 1 for x in cur) + len(text)
            if len(cur) >= max_words or gap > max_gap or chars > max_chars:
                flush()
        cur.append({"text": text, "start": float(w["start"]), "end": float(w["end"])})
        if text.endswith(SENTENCE_END):
            flush()
    flush()
    return [{"id": i, "start": c[0]["start"], "end": c[-1]["end"], "words": c} for i, c in enumerate(chunks)]


def retime(text: str, start: float, end: float, original_words: list[dict]) -> list[dict]:
    """Give rewritten caption text word timings: reuse the originals if the word count matches,
    otherwise spread the words across the chunk in proportion to their length."""
    tokens = text.split()
    if not tokens:
        return []
    if len(tokens) == len(original_words):
        return [{"text": t, "start": o["start"], "end": o["end"]} for t, o in zip(tokens, original_words)]
    total = sum(len(t) + 1 for t in tokens)
    out, cursor = [], start
    for t in tokens:
        span = (end - start) * (len(t) + 1) / total
        out.append({"text": t, "start": round(cursor, 3), "end": round(cursor + span, 3)})
        cursor += span
    return out
