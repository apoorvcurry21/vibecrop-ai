"""Phase 2 entry point: transcript.json -> plan.json.

CLI:  python -m src.planning.pipeline D:\\vibecrop_tmp\\jobs\\<job>   (or the transcript.json path)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src import config
from src.planning.gemini_planner import PlanningError, plan


def run(transcript_path: str | Path, total: int | None = None, model: str | None = None,
        client: Any = None) -> Path:
    transcript_path = Path(transcript_path)
    if transcript_path.is_dir():
        transcript_path = transcript_path / "transcript.json"
    if not transcript_path.is_file():
        raise PlanningError(f"Transcript not found: {transcript_path}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    result = plan(transcript, total=total, model=model, client=client)
    result["source_transcript"] = transcript_path.name
    out = transcript_path.parent / "plan.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeCrop AI - Phase 2 Gemini planning")
    parser.add_argument("job", help="Job folder or transcript.json from Phase 1")
    parser.add_argument("--total", type=int, default=None, help="Number of B-rolls (default 10)")
    parser.add_argument("--model", default=None, help="Override the Gemini model")
    args = parser.parse_args()
    try:
        out = run(args.job, total=args.total, model=args.model)
    except (config.ConfigError, PlanningError) as exc:
        raise SystemExit(f"Error: {exc}")
    data = json.loads(out.read_text(encoding="utf-8"))
    videos = sum(i["asset_type"] == "video" for i in data["items"])
    print(f"Saved {out}")
    print(f"{len(data['items'])} B-rolls: {videos} video, {len(data['items']) - videos} image | topic: {data['video_topic']}")


if __name__ == "__main__":
    main()
