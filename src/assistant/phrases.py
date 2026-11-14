"""'Where I say X' for the chat tools: find a phrase in the video (spoken transcript first, then any cached English /
Hinglish caption text) and pick which of the matches to act on."""
from __future__ import annotations

import json
from pathlib import Path

from src import workflow
from src.assistant.tools import ToolError, _int, find_phrase

MAX_HITS = 30
OCCURRENCES = ("all", "first", "last", "nth")


def _word_pools(job: Path) -> list[list[dict]]:
    pools: list[list[dict]] = []
    try:
        pools.append(workflow.read_transcript_words(job))
    except (OSError, ValueError):
        pass
    for cache in sorted(job.glob("captions_*.json")):
        try:
            chunks = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pools.append([{"word": w["text"], "start": w["start"], "end": w["end"]}
                      for c in chunks for w in c.get("words", [])])
    return pools


def search(job: Path, phrase: str) -> list[dict]:
    """Matches ({start, end, heard_as, context, match}) in time order. Raises ToolError when there are none."""
    for pool in _word_pools(job):
        hits = find_phrase(pool, phrase, limit=MAX_HITS)
        if hits:
            return hits
    raise ToolError("Could not find those words. Try find_in_transcript or get_transcript for the exact spelling.")


def select(hits: list[dict], args: dict) -> list[dict]:
    """Apply the tool's `occurrence` (all, first, last, nth + `which`) to the matches."""
    which = args.get("occurrence") or "all"
    if which not in OCCURRENCES:
        raise ToolError(f"occurrence must be one of: {', '.join(OCCURRENCES)}.")
    if which == "first":
        return hits[:1]
    if which == "last":
        return hits[-1:]
    if which == "nth":
        n = _int(args, "which")
        if not 1 <= n <= len(hits):
            raise ToolError(f"There are only {len(hits)} match(es); 'which' must be between 1 and {len(hits)}.")
        return [hits[n - 1]]
    return hits
