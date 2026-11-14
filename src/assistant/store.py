"""Per-job chat history in <job>/chat.json: text, attachment names and small UI cards only (no media), so it stays a
few KB. Written once per finished turn, never per token."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

MAX_MESSAGES = 200
FILE = "chat.json"


def load(job: Path) -> list[dict]:
    f = job / FILE
    if not f.is_file():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [m for m in data if isinstance(m, dict) and m.get("role") in ("user", "assistant")] \
        if isinstance(data, list) else []


def save(job: Path, messages: list[dict]) -> None:
    tmp = job / (FILE + ".tmp")
    tmp.write_text(json.dumps(messages[-MAX_MESSAGES:], ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, job / FILE)  # atomic: a crash never leaves a half-written history


def append(job: Path, role: str, text: str, files: list[str] | None = None, cards: list[dict] | None = None) -> list[dict]:
    messages = load(job)
    msg = {"role": role, "text": text, "time": round(time.time())}
    if files:
        msg["files"] = files
    if cards:
        msg["cards"] = cards
    messages.append(msg)
    save(job, messages)
    return messages
