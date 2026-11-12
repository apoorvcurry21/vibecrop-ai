"""Find and download a short, muted video-only slice of a YouTube video (no full downloads)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

MIN_SOURCE_S = 30.0  # skip tiny clips/shorts (often logos, memes or vertical already)
MAX_SOURCE_S = 1200.0  # skip very long videos (streams, full matches): slow and mostly filler
SEARCH_N = 10
MAX_DOWNLOAD_TRIES = 3
WINDOW_FRACTION = 0.25  # start ~25% in: past intros/logos, usually in real footage
PAD_S = 4.0  # extra footage beyond the slot: the editor can lengthen a slot a little without re-downloading
# Video-only (so the clip is silent by construction), up to 1080p, H.264 preferred.
VIDEO_FORMAT = (
    "bestvideo[height<=1080][vcodec^=avc1]/bestvideo[height<=1080][ext=mp4]/"
    "bestvideo[height<=1080]/best[height<=1080]"
)


class SourcingError(RuntimeError):
    pass


def pick_window(source_duration: float, slot_seconds: float) -> tuple[float, float]:
    """Blind 'smart' window: ~25% into the video, clamped so it fits with margin."""
    length = slot_seconds + PAD_S
    start = max(3.0, source_duration * WINDOW_FRACTION)
    start = min(start, max(0.0, source_duration - length - 2.0))
    return round(start, 2), round(start + length, 2)


def search_candidates(query: str, skip_ids: set[str] | frozenset[str] = frozenset(),
                      limit: int = SEARCH_N) -> list[dict]:
    """YouTube search via yt-dlp (metadata only). Returns usable candidates in rank order."""
    from yt_dlp import YoutubeDL

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as exc:
        raise SourcingError(f"YouTube search failed for '{query}': {exc}") from exc

    candidates = []
    for e in (info or {}).get("entries") or []:
        vid, dur = e.get("id"), e.get("duration")
        if not vid or vid in skip_ids or dur is None:
            continue
        if not MIN_SOURCE_S <= float(dur) <= MAX_SOURCE_S:
            continue
        if e.get("live_status") in ("is_live", "is_upcoming"):
            continue
        candidates.append({
            "id": vid,
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "duration": float(dur),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
        })
    return candidates


def _probe(path: Path) -> tuple[float, bool]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip()), out.returncode == 0
    except ValueError:
        return 0.0, False


def download_section(candidate: dict, start: float, end: float, out_stem: Path) -> Path:
    """Download only [start, end] seconds of the stream (yt-dlp --download-sections, via FFmpeg)."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import download_range_func

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {
        "quiet": True, "no_warnings": True, "noprogress": True,
        "format": VIDEO_FORMAT,
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,  # frame-accurate cut instead of snapping to a keyframe
        "outtmpl": str(out_stem) + ".%(ext)s",
        "overwrites": True,
        "retries": 3,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(candidate["url"], download=True)
    downloads = (info or {}).get("requested_downloads") or []
    path = Path(downloads[0]["filepath"]) if downloads else None
    if path is None or not path.is_file():
        matches = sorted(out_stem.parent.glob(out_stem.name + ".*"))
        path = matches[0] if matches else None
    if path is None:
        raise SourcingError("yt-dlp finished but no file was written.")
    return path


def source_video_clip(queries: list[str], slot_seconds: float, out_stem: Path,
                      skip_ids: set[str] | frozenset[str] = frozenset(),
                      max_tries: int = MAX_DOWNLOAD_TRIES) -> dict:
    """Try each query in turn; download the first candidate that yields a valid clip.

    Returns metadata (video id, url, title, channel, window, file path). `skip_ids` lets the
    UI's "Regenerate" ask for a different video than the ones already tried.
    """
    tries, last_error = 0, "no usable search results"
    seen = set(skip_ids)
    for query in [q for q in queries if q and q.strip()]:
        try:
            candidates = search_candidates(query, skip_ids=seen)
        except SourcingError as exc:
            last_error = str(exc)
            continue
        for cand in candidates:
            if tries >= max_tries:
                raise SourcingError(f"Gave up after {tries} downloads. Last error: {last_error}")
            seen.add(cand["id"])
            tries += 1
            start, end = pick_window(cand["duration"], slot_seconds)
            try:
                path = download_section(cand, start, end, out_stem)
            except Exception as exc:  # unavailable, geo/age blocked, bot check, ...
                last_error = f"{cand['id']}: {str(exc).splitlines()[0][:200]}"
                continue
            got, ok = _probe(path)
            if not ok or got < slot_seconds - 0.3:
                last_error = f"{cand['id']}: downloaded clip too short ({got:.1f}s)"
                path.unlink(missing_ok=True)
                continue
            return {
                "status": "ok", "source": "youtube", "query_used": query,
                "video_id": cand["id"], "url": cand["url"], "title": cand["title"],
                "channel": cand["channel"], "source_duration": cand["duration"],
                "window_start": start, "window_end": end, "path": path.name, "clip_duration": round(got, 2),
                "tried_ids": sorted(seen),  # so "Regenerate" can ask for a different video
            }
    raise SourcingError(f"No clip found. Last error: {last_error}")
