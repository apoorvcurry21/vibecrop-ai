"""Tiny stage timer: `with perf.stage(job_dir, "render"): ...` appends one line to <job>/perf.log.

It never raises (a full disk must not break a render) and costs a few microseconds, so it is always on.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def record(job_dir: Path | str | None, name: str, seconds: float) -> None:
    if not job_dir:
        return
    try:
        with (Path(job_dir) / "perf.log").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')}  {name:<24} {seconds:7.2f}s\n")
    except OSError:
        pass


@contextmanager
def stage(job_dir: Path | str | None, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(job_dir, name, time.perf_counter() - t0)
