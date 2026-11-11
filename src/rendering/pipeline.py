"""Phase 3b entry point: plan.json + clips + original video -> final.mp4.

CLI:  python -m src.rendering.pipeline D:\\vibecrop_tmp\\jobs\\<job> "D:\\vids\\video.mp4"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src import config
from src.rendering.ass import credit_event, write_ass
from src.rendering.compose import Overlay, RenderError, render


def run(job_dir: str | Path, main_video: str | Path, credits: bool = True,
        log: Callable[[str], None] = print) -> Path:
    job_dir, main_video = Path(job_dir), Path(main_video)
    if not main_video.is_file():
        raise RenderError(f"Main video not found: {main_video}")
    plan = json.loads((job_dir / "plan.json").read_text(encoding="utf-8"))

    overlays: list[Overlay] = []
    events: list[str] = []
    for item in sorted(plan["items"], key=lambda i: i["timestamp_start"]):
        asset = item.get("asset") or {}
        clip = job_dir / asset.get("path", "")
        if item["asset_type"] != "video" or asset.get("status") != "ok" or not clip.is_file():
            log(f"[{item['id']:>2}] no clip, speaker stays on screen for this slot")
            continue
        start, end = item["timestamp_start"], item["timestamp_end"]
        overlays.append(Overlay(clip, start, end))
        if credits and asset.get("channel"):
            events.append(credit_event(start, end, asset["channel"]))

    ass_name = None
    if events:
        write_ass(job_dir / "credits.ass", events)
        ass_name = "credits.ass"  # relative: FFmpeg runs with cwd=job_dir

    log(f"Rendering {len(overlays)} B-roll overlays...")
    return render(main_video, overlays, job_dir / "final.mp4", subtitles_file=ass_name, cwd=job_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeCrop AI - Phase 3 render")
    parser.add_argument("job", help="Job folder containing plan.json and clips/")
    parser.add_argument("video", help="The original raw video")
    parser.add_argument("--no-credits", action="store_true", help="Do not burn creator credit lines")
    args = parser.parse_args()
    try:
        out = run(args.job, args.video, credits=not args.no_credits)
    except (config.ConfigError, RenderError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
