"""The local sound-effect library: `<sfx_dir>/index.json` plus the audio files it points to.

    {"sounds": [{"id": "fah", "name": "Fah", "aliases": ["fahhh"], "file": "fah.mp3", "duration": 1.4,
                 "source": "catalog" | "youtube" | "web", "url": "...", "tags": []}]}

Files are `<sfx_dir>/<id>.mp3` (built from the catalog) or `<sfx_dir>/_added/<id>.mp3` (fetched on request). Plan
entries refer to a sound by its id, so they survive the per-job temp cleanup.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import threading
from pathlib import Path

from src import config

INDEX_NAME = "index.json"
MATCH_MIN = 0.72  # below this a name is not considered "in the library"
_lock = threading.Lock()
_NORM = re.compile(r"[^\wऀ-ॿ]+")


class SfxError(RuntimeError):
    pass


def _norm(text: str) -> str:
    return _NORM.sub(" ", str(text).lower()).strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "sound"


def index_path() -> Path:
    return config.sfx_dir() / INDEX_NAME


def load() -> list[dict]:
    """Every sound in the library (empty list if it has not been built yet)."""
    try:
        data = json.loads(index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s for s in data.get("sounds", []) if isinstance(s, dict) and s.get("id") and s.get("file")]


def _write(sounds: list[dict]) -> None:
    """Atomic: a crash while writing never leaves a half-written index."""
    d = config.sfx_dir()
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"sounds": sounds}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, index_path())
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def get(lib_id: str) -> dict | None:
    return next((s for s in load() if s["id"] == lib_id), None)


def path_of(entry: dict) -> Path:
    return config.sfx_dir() / entry["file"]


def available(entry: dict) -> bool:
    return path_of(entry).is_file()


def register(entry: dict) -> dict:
    """Add or replace one sound in the index. The audio file must already be inside the sfx folder."""
    if not available(entry):
        raise SfxError(f"The sound file {entry['file']} is missing.")
    with _lock:
        sounds = [s for s in load() if s["id"] != entry["id"]]
        sounds.append(entry)
        sounds.sort(key=lambda s: s["id"])
        _write(sounds)
    return entry


def unique_id(base: str) -> str:
    taken = {s["id"] for s in load()}
    cand, n = base, 2
    while cand in taken:
        cand = f"{base}_{n}"
        n += 1
    return cand


def score(query: str, entry: dict) -> float:
    """1.0 for an exact name/alias/id; otherwise how well the words line up (fuzzy, so 'fahh' still finds 'fah')."""
    q = _norm(query)
    if not q:
        return 0.0
    names = [entry["name"], entry["id"].replace("_", " "), *entry.get("aliases", [])]
    best = 0.0
    for name in names:
        n = _norm(name)
        if not n:
            continue
        if q == n:
            return 1.0
        s = difflib.SequenceMatcher(None, q, n).ratio()
        q_tokens, n_tokens = set(q.split()), set(n.split())
        if q_tokens and q_tokens <= n_tokens:  # every asked word is in the name ("sus" in "among us sus")
            s = max(s, 0.85 if len(q) >= 3 else 0.0)
        elif n_tokens and n_tokens <= q_tokens:  # the name is inside a longer request ("fah sound")
            s = max(s, 0.8)
        best = max(best, s)
    return best


def find(query: str, limit: int = 5, minimum: float = MATCH_MIN) -> list[dict]:
    """Library sounds that match `query`, best first, each as a copy with a `score` field."""
    scored = [({**s}, score(query, s)) for s in load() if available(s)]
    hits = [dict(s, score=round(sc, 2)) for s, sc in sorted(scored, key=lambda p: -p[1]) if sc >= minimum]
    return hits[:limit]


def names() -> list[str]:
    return [s["name"] for s in load() if available(s)]
