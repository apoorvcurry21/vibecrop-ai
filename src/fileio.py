"""Crash-safe file writes. The new text goes into a temp file next to the target and then replaces it in one step, so
the target is always either the old complete file or the new complete file, never a half-written one (plan.json is the
single source of truth: a torn write would make the whole job unreadable)."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

REPLACE_TRIES = 8  # on Windows os.replace fails for a moment while another thread is reading the target


def write_text_atomic(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == REPLACE_TRIES - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        Path(tmp).unlink(missing_ok=True)  # only still there if the replace failed


def write_json_atomic(path: str | Path, data: Any, indent: int | None = 2) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=indent))
