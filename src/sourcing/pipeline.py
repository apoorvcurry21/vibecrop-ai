"""Phase 3a entry point: plan.json -> clips/clip_XX.mp4 (+ asset metadata written back into plan.json).

CLI:  python -m src.sourcing.pipeline D:\\vibecrop_tmp\\jobs\\<job>  [--only 1 4]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from src import config
from src.fileio import write_json_atomic
from src.sourcing import relevance
from src.sourcing.youtube import SourcingError, source_video_clip


def source_item(job_dir: Path, item: dict, skip_ids: set[str] | frozenset[str] = frozenset(), checker=None) -> dict:
    """Source one plan item. Used by the batch run and by the UI's 'Regenerate Asset' button.
    Each downloaded clip is checked against what is being said (relevance.py) unless RELEVANCE_CHECK=0."""
    slot = item["timestamp_end"] - item["timestamp_start"]
    queries = [item["search_query"], *item.get("alternative_queries", [])]
    stem = job_dir / "clips" / f"clip_{item['id']:02d}"
    checker = checker or relevance.default_checker()
    meaning = item.get("english_meaning") or item.get("spoken_text") or ""
    verify = (lambda clip, length, query: checker.check(clip, length, meaning, query)) if checker else None
    meta = source_video_clip(queries, slot, stem, skip_ids=skip_ids, verify=verify)
    meta["path"] = f"clips/{meta['path']}"  # relative to the job folder
    return meta


def _source_safely(job_dir: Path, item: dict, skip_ids: set[str]) -> dict:
    """Worker body: never raises, so one bad slot cannot take the others down."""
    try:
        return source_item(job_dir, item, skip_ids=skip_ids)
    except SourcingError as exc:
        return {"status": "failed", "error": str(exc)}
    except Exception as exc:  # anything unexpected (network, ffmpeg, ...) fails this slot only; "Regenerate" can retry
        return {"status": "failed", "error": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"}


def run(job_dir: str | Path, only: list[int] | None = None, force: bool = False,
        log: Callable[[str], None] = print, workers: int | None = None) -> Path:
    """Source every video slot. Slots download in parallel (each is a separate search + short section download, so
    they are network-bound); only this thread logs and writes plan.json, so neither needs a lock."""
    job_dir = Path(job_dir)
    plan_path = job_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    todo = []
    for item in plan["items"]:
        if item["asset_type"] != "video" or (only and item["id"] not in only):
            continue
        asset = item.get("asset") or {}
        if not force and asset.get("status") == "ok" and (job_dir / asset.get("path", "")).is_file():
            log(f"[{item['id']:>2}] already sourced, skipping")
            continue
        todo.append((item, set(asset.get("tried_ids", []))))
    if not todo:
        return plan_path

    n_workers = max(1, min(workers or config.sourcing_workers(), len(todo)))
    log(f"Downloading {len(todo)} clip(s), {n_workers} at a time...")
    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="source") as pool:
        futures = {pool.submit(_source_safely, job_dir, item, skip): item for item, skip in todo}
        for fut in as_completed(futures):
            item = futures[fut]
            item["asset"] = fut.result()
            a = item["asset"]
            if a.get("status") == "ok":
                note = f"  (may not fit: {a.get('relevance_reason', '')})" if a.get("relevance") == "mismatch" else ""
                log(f"[{item['id']:>2}] ok  {a['channel']} - {a['title'][:60]} @ {a['window_start']}s{note}")
            else:
                log(f"[{item['id']:>2}] FAILED: {a.get('error')}")
            write_json_atomic(plan_path, plan)  # save as we go
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeCrop AI - Phase 3 media sourcing (video)")
    parser.add_argument("job", help="Job folder containing plan.json")
    parser.add_argument("--only", type=int, nargs="*", help="Only (re)source these item ids")
    parser.add_argument("--force", action="store_true", help="Re-download even if a clip exists")
    args = parser.parse_args()
    try:
        plan_path = run(args.job, only=args.only, force=args.force)
    except (config.ConfigError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    ok = sum((i.get("asset") or {}).get("status") == "ok" for i in plan["items"])
    print(f"{ok}/{len(plan['items'])} clips sourced")


if __name__ == "__main__":
    main()
