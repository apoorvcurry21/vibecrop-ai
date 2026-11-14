"""A short clip out of a YouTube link the user pasted.

Where in the video: the user's own timestamp if they gave one; otherwise Gemini watches the video (Gemini reads
YouTube links natively) and picks the best moment for what the user described. Only the needed seconds are
downloaded (reusing sourcing.youtube.download_section)."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel

from src import config
from src.gemini_client import GeminiError, generate_with_fallback, make_client
from src.sourcing import youtube

MAX_AUTO_SEARCH_S = 40 * 60  # longer videos: ask the user for a rough time instead of making Gemini watch an hour
YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}


class YtError(RuntimeError):
    pass


class Window(BaseModel):
    start: float  # seconds into the video where the moment begins
    reason: str  # one short sentence: what is on screen then


def video_id(url: str) -> str | None:
    """The 11-character id of a youtube.com / youtu.be link (watch, shorts, embed, live), else None."""
    p = urlparse((url or "").strip())
    if (p.hostname or "").lower() not in YT_HOSTS:
        return None
    if p.hostname.endswith("youtu.be"):
        cand = p.path.strip("/").split("/")[0]
    elif p.path == "/watch":
        cand = (parse_qs(p.query).get("v") or [""])[0]
    else:
        m = re.match(r"^/(?:shorts|embed|live|v)/([^/?]+)", p.path)
        cand = m.group(1) if m else ""
    return cand if re.fullmatch(r"[A-Za-z0-9_-]{11}", cand or "") else None


def parse_time(value: Any) -> float | None:
    """'2:15' -> 135.0, '1:02:03' -> 3723.0, '90' or 90 -> 90.0, '1m30s' -> 90.0. None if empty/unreadable."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if value >= 0 else None
    s = str(value).strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if ":" in s:
        try:
            nums = [float(x) for x in s.split(":")]
        except ValueError:
            return None
        if len(nums) > 3 or any(n < 0 for n in nums):
            return None
        return sum(n * 60 ** i for i, n in enumerate(reversed(nums)))
    m = re.fullmatch(r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?", s)
    if m and any(m.groups()):
        h, mi, sec = (float(g or 0) for g in m.groups())
        return h * 3600 + mi * 60 + sec
    return None


def video_info(url: str) -> dict:
    """Title, channel and length of the video (metadata only, nothing is downloaded)."""
    from yt_dlp import YoutubeDL

    vid = video_id(url)
    if not vid:
        raise YtError("That doesn't look like a YouTube link (youtube.com/watch?v=..., youtu.be/..., /shorts/...).")
    clean = f"https://www.youtube.com/watch?v={vid}"
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(clean, download=False)
    except Exception as exc:  # noqa: BLE001 - private / removed / age-gated / blocked
        raise YtError(f"I couldn't open that video: {str(exc).splitlines()[0][:160]}") from exc
    if info.get("live_status") in ("is_live", "is_upcoming"):
        raise YtError("That is a live stream, so I can't cut a clip from it.")
    return {"id": vid, "url": clean, "title": info.get("title") or "", "channel": info.get("channel") or
            info.get("uploader") or "", "duration": float(info.get("duration") or 0.0)}


def find_window(info: dict, about: str, length: float, client: Any = None, models: list[str] | None = None,
                sleep=time.sleep) -> tuple[float, str]:
    """Ask Gemini to watch the video and return (start second, why) of the best `length`-second moment."""
    from google.genai import types

    if info["duration"] > MAX_AUTO_SEARCH_S:
        raise YtError("That video is very long. Tell me roughly when the moment happens (for example 'around "
                      "12:30') and I'll cut from there.")
    prompt = (f"Watch this video. Find the single best moment, {length:g} seconds long, that shows: {about}\n"
              f"The video is {info['duration']:.0f} seconds long. Reply with `start` (seconds from the beginning "
              f"where the moment begins; the whole {length:g} seconds must stay inside the video and show it, "
              "not an intro, logo, title card or still image) and `reason` (one short sentence about what is on "
              "screen).")
    history = [types.Content(role="user", parts=[types.Part(file_data=types.FileData(file_uri=info["url"])),
                                                 types.Part(text=prompt)])]
    try:
        text, _ = generate_with_fallback(client or make_client(), models or config.chat_model_chain(), history,
                                         sleep, "You find precise moments in videos for a video editor.", Window)
        win = Window.model_validate_json(text)
    except (GeminiError, ValueError) as exc:
        raise YtError(f"I couldn't pick a moment automatically ({exc}). Tell me a time like 2:15 instead.") from exc
    start = min(max(win.start, 0.0), max(info["duration"] - length - 0.5, 0.0))
    return round(start, 2), win.reason.strip()


def cut_clip(info: dict, source_start: float, length: float, out_stem: Path) -> tuple[Path, float]:
    """Download [source_start, source_start+length+PAD] of the video. Returns (file, its real length)."""
    if source_start >= info["duration"] - 0.5:
        raise YtError(f"The video is only {info['duration']:.0f}s long, so it has nothing at {source_start:.0f}s.")
    end = min(source_start + length + youtube.PAD_S, info["duration"])
    try:
        path = youtube.download_section({"url": info["url"]}, source_start, end, out_stem)
    except Exception as exc:  # noqa: BLE001 - unavailable / blocked / bot check
        raise YtError(f"I couldn't download that part: {str(exc).splitlines()[0][:160]}") from exc
    got, ok = youtube._probe(path)
    if not ok or got < min(length, 1.5) - 0.3:
        path.unlink(missing_ok=True)
        raise YtError("The downloaded piece was too short. Try a start time earlier in the video.")
    return path, round(got, 2)
