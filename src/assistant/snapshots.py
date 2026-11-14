"""Undo for chat edits: a snapshot is taken before every change and restored by `undo`.

A snapshot holds plan.json + job.json and a copy of only the clip files the change can overwrite (a few MB at
most), plus the list of clip file names that existed, so clips created by the change are removed on undo.
Snapshots live in <job>/history/ (deleted with the other temp files by the post-render cleanup); only the newest
MAX_SNAPSHOTS are kept."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

MAX_SNAPSHOTS = 10
HISTORY = "history"


def _dir(job: Path) -> Path:
    return job / HISTORY


def _numbered(job: Path) -> list[Path]:
    base = _dir(job)
    if not base.is_dir():
        return []
    return sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))


def take(job: Path, label: str, clip_ids: tuple[int, ...] | list[int] = ()) -> Path:
    """Save the current state. `clip_ids` are the slot ids whose clip file the coming change may overwrite or
    delete (regenerate, swap, replace, delete...)."""
    existing = _numbered(job)
    snap = _dir(job) / f"{(int(existing[-1].name) + 1 if existing else 1):06d}"
    (snap / "clips").mkdir(parents=True)
    for name in ("plan.json", "job.json"):
        if (job / name).is_file():
            shutil.copy2(job / name, snap / name)
    clips = job / "clips"
    saved = []
    for cid in dict.fromkeys(clip_ids):
        for f in clips.glob(f"clip_{cid:02d}.*"):
            shutil.copy2(f, snap / "clips" / f.name)
            saved.append(f.name)
    names = sorted(p.name for p in clips.iterdir() if p.is_file()) if clips.is_dir() else []
    (snap / "meta.json").write_text(json.dumps({"label": label, "time": time.time(), "saved": saved,
                                                "clip_names": names}), encoding="utf-8")
    for old in _numbered(job)[:-MAX_SNAPSHOTS]:
        shutil.rmtree(old, ignore_errors=True)
    return snap


def can_undo(job: Path) -> bool:
    return bool(_numbered(job))


def last_label(job: Path) -> str | None:
    snaps = _numbered(job)
    if not snaps:
        return None
    return json.loads((snaps[-1] / "meta.json").read_text(encoding="utf-8")).get("label")


def discard_latest(job: Path) -> None:
    """Drop the newest snapshot without restoring it (the change it guarded failed before doing anything)."""
    snaps = _numbered(job)
    if snaps:
        shutil.rmtree(snaps[-1], ignore_errors=True)


def undo(job: Path) -> str | None:
    """Restore the newest snapshot and delete it. Returns its label, or None if there is nothing to undo."""
    snaps = _numbered(job)
    if not snaps:
        return None
    snap = snaps[-1]
    meta = json.loads((snap / "meta.json").read_text(encoding="utf-8"))
    clips = job / "clips"
    clips.mkdir(exist_ok=True)
    keep = set(meta["clip_names"])
    for f in clips.iterdir():  # clips the undone change created
        if f.is_file() and f.name not in keep:
            f.unlink(missing_ok=True)
    for name in meta["saved"]:  # clips it overwrote or deleted
        shutil.copy2(snap / "clips" / name, clips / name)
    for name in ("plan.json", "job.json"):
        if (snap / name).is_file():
            shutil.copy2(snap / name, job / name)
    shutil.rmtree(snap, ignore_errors=True)
    return meta.get("label")
