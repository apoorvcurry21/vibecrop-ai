"""Phase 3b/4 render: plan.json + clips + original video (+ captions) -> final.mp4.

CLI:  python -m src.rendering.pipeline D:\\vibecrop_tmp\\jobs\\<job> "D:\\vids\\video.mp4" [--captions hinglish]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from src import config
from src.captions.chunks import chunk_words
from src.captions.rewrite import MODES, CaptionError, rewrite_chunks
from src.rendering.ass import caption_events, credit_event, write_ass
from src.rendering.compose import DEFAULT_FACE_Y, LAYOUTS, Overlay, RenderError, render


def load_caption_chunks(job_dir: Path, mode: str, transcript: dict, topic: str = "",
                        client=None, log: Callable[[str], None] = print) -> list[dict]:
    """Chunk the transcript and rewrite it for `mode`. The rewrite is cached per job so Gemini runs once."""
    cache = job_dir / f"captions_{mode}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    chunks = rewrite_chunks(chunk_words(transcript["words"]), mode, transcript.get("language", ""),
                            context=topic, client=client)
    cache.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return chunks


def run(job_dir: str | Path, main_video: str | Path, credits: bool = True, caption_mode: str | None = None,
        log: Callable[[str], None] = print, client=None) -> Path:
    job_dir, main_video = Path(job_dir), Path(main_video)
    if not main_video.is_file():
        raise RenderError(f"Main video not found: {main_video}")
    plan = json.loads((job_dir / "plan.json").read_text(encoding="utf-8"))

    job_meta_file = job_dir / "job.json"
    job_meta = json.loads(job_meta_file.read_text(encoding="utf-8")) if job_meta_file.is_file() else {}
    face_y = float(job_meta.get("split_face_y", DEFAULT_FACE_Y))

    overlays: list[Overlay] = []
    events: list[str] = []
    split_spans: list[tuple[float, float]] = []
    for item in sorted(plan["items"], key=lambda i: i["timestamp_start"]):
        asset = item.get("asset") or {}
        clip = job_dir / asset.get("path", "")
        if item.get("enabled") is False:
            log(f"[{item['id']:>2}] removed by you, speaker stays on screen")
            continue
        if item["asset_type"] != "video" or asset.get("status") != "ok" or not clip.is_file():
            log(f"[{item['id']:>2}] no clip, speaker stays on screen for this slot")
            continue
        start, end = item["timestamp_start"], item["timestamp_end"]
        layout = item.get("layout") if item.get("layout") in LAYOUTS else "overlay"
        overlays.append(Overlay(clip, start, end, layout=layout, framing=item.get("framing")))
        if layout == "split":
            split_spans.append((start, end))
        if credits and asset.get("channel"):
            events.append(credit_event(start, end, asset["channel"], split=layout == "split"))

    if caption_mode:
        if caption_mode not in MODES:
            raise RenderError(f"Unknown caption mode '{caption_mode}'.")
        transcript = json.loads((job_dir / "transcript.json").read_text(encoding="utf-8"))
        try:
            context = plan.get("video_topic", "")
            if plan.get("notes"):  # the creator's own spellings of names fix the caption text too
                context += f". Creator notes (correct names/spellings): {plan['notes']}"
            chunks = load_caption_chunks(job_dir, caption_mode, transcript, context, client, log)
        except CaptionError as exc:
            log(f"Caption rewrite failed ({exc}); using the original captions instead.")
            chunks = load_caption_chunks(job_dir, "original", transcript)
        events += caption_events(chunks, split_spans)

    ass_name = None
    if events:
        write_ass(job_dir / "overlays.ass", events)
        ass_name = "overlays.ass"  # relative: FFmpeg runs with cwd=job_dir

    log(f"Rendering {len(overlays)} B-roll overlays" + (f" + {caption_mode} captions..." if caption_mode else "..."))
    return render(main_video, overlays, job_dir / "final.mp4", subtitles_file=ass_name, cwd=job_dir, face_y=face_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="VibeCrop AI - render")
    parser.add_argument("job", help="Job folder containing plan.json and clips/")
    parser.add_argument("video", help="The original raw video")
    parser.add_argument("--no-credits", action="store_true", help="Do not burn creator credit lines")
    parser.add_argument("--captions", choices=sorted(MODES), default=None, help="Burn in captions")
    args = parser.parse_args()
    try:
        out = run(args.job, args.video, credits=not args.no_credits, caption_mode=args.captions)
    except (config.ConfigError, RenderError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
