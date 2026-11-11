"""Phase 3a entry point: plan.json -> clips/clip_XX.mp4 (+ asset metadata written back into plan.json).

CLI:  python -m src.sourcing.pipeline D:\\vibecrop_tmp\\jobs\\<job>  [--only 1 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src import config
from src.sourcing.youtube import SourcingError, source_video_clip


def source_item(job_dir: Path, item: dict, skip_ids: set[str] | frozenset[str] = frozenset()) -> dict:
    """Source one plan item. Used by the batch run and by the UI's 'Regenerate Asset' button."""
    slot = item["timestamp_end"] - item["timestamp_start"]
    queries = [item["search_query"], *item.get("alternative_queries", [])]
    stem = job_dir / "clips" / f"clip_{item['id']:02d}"
    meta = source_video_clip(queries, slot, stem, skip_ids=skip_ids)
    meta["path"] = f"clips/{meta['path']}"  # relative to the job folder
    return meta


def run(job_dir: str | Path, only: list[int] | None = None, force: bool = False,
        log: Callable[[str], None] = print) -> Path:
    job_dir = Path(job_dir)
    plan_path = job_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    for item in plan["items"]:
        if item["asset_type"] != "video" or (only and item["id"] not in only):
            continue
        asset = item.get("asset") or {}
        if not force and asset.get("status") == "ok" and (job_dir / asset.get("path", "")).is_file():
            log(f"[{item['id']:>2}] already sourced, skipping")
            continue
        log(f"[{item['id']:>2}] searching: {item['search_query']}")
        try:
            item["asset"] = source_item(job_dir, item, skip_ids=set(asset.get("tried_ids", [])))
            a = item["asset"]
            log(f"     ok  {a['channel']} - {a['title'][:60]} @ {a['window_start']}s")
        except SourcingError as exc:
            item["asset"] = {"status": "failed", "error": str(exc)}
            log(f"     FAILED: {exc}")
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")  # save as we go
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
