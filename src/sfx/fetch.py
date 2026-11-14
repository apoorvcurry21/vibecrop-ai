"""Bring a new sound into the library: from a YouTube link (with or without a time), or just by name.

Only the needed seconds of audio are downloaded (yt-dlp section download). Every sound is trimmed of silence, capped
in length, peak-normalised and stored as a small mp3, so all library sounds play at a similar level."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from src import config
from src.assistant import ytclip
from src.config import require_binary
from src.sfx import library
from src.sfx.library import SfxError
from src.sourcing import youtube

SFX_MAX_S = 8.0  # the longest a stored sound can be
PEAK_DB = -3.0  # every stored sound is scaled so its loudest point sits here
MIN_SOUND_S = 0.15
DEFAULT_LINK_LEN = 3.0
MAX_LINK_LEN = 15.0
NAME_PREFER_S = 20.0  # a search result this short is assumed to BE the sound: use it from the start
NAME_SEARCH_MAX_S = 120.0
NAME_TRIES = 3
READ_LIMIT_S = 30  # never decode more than this from a downloaded file


class FetchError(SfxError):
    pass


def _ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([require_binary("ffmpeg"), "-nostdin", "-y", "-hide_banner", *args],
                          capture_output=True, text=True)


def process_audio(src: Path, dst: Path, max_s: float = SFX_MAX_S) -> float:
    """Trim leading/trailing silence, cap at `max_s`, peak-normalise to PEAK_DB, write an mp3. Returns its length."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    trim = ("silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.03,areverse,"
            "silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.03,areverse")
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "trimmed.wav"
        r = _ffmpeg(["-loglevel", "error", "-t", str(READ_LIMIT_S), "-i", str(src), "-vn", "-af", trim,
                     "-t", f"{max_s:.3f}", "-ar", "44100", "-ac", "2", str(wav)])
        if r.returncode != 0 or not wav.is_file():
            raise FetchError(f"Could not read that audio: {r.stderr.strip()[-200:]}")
        length = youtube.probe_audio_duration(wav)
        if length < MIN_SOUND_S:
            raise FetchError("That audio is silent or too short to use.")
        peak = _max_volume_db(wav)
        if peak is None or peak < -60:
            raise FetchError("That audio is silent.")
        gain = max(-20.0, min(30.0, PEAK_DB - peak))
        fade = max(length - 0.02, 0.0)
        r = _ffmpeg(["-loglevel", "error", "-i", str(wav), "-af", f"volume={gain:.2f}dB,afade=t=out:st={fade:.3f}:d=0.02",
                     "-c:a", "libmp3lame", "-b:a", "128k", str(dst)])
        if r.returncode != 0 or not dst.is_file():
            raise FetchError(f"Could not save the sound: {r.stderr.strip()[-200:]}")
    return round(youtube.probe_audio_duration(dst), 2)


def _max_volume_db(path: Path) -> float | None:
    r = _ffmpeg(["-i", str(path), "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", r.stderr)
    return float(m.group(1)) if m else None


def add_to_library(audio: Path, lib_id: str, name: str, source: str, url: str = "", aliases: list[str] | None = None,
                   subdir: str = "_added", **extra: Any) -> dict:
    """Process `audio` into <sfx_dir>/<subdir>/<lib_id>.mp3 and register it."""
    rel = f"{subdir}/{lib_id}.mp3" if subdir else f"{lib_id}.mp3"
    duration = process_audio(audio, config.sfx_dir() / rel, max_s=extra.pop("max_s", SFX_MAX_S))
    return library.register({"id": lib_id, "name": name, "aliases": aliases or [], "file": rel, "duration": duration,
                             "source": source, "url": url, **extra})


def _download(cand: dict, start: float, end: float, work: Path) -> Path:
    try:
        return youtube.download_audio_section(cand, start, end, work / "dl")
    except Exception as exc:  # noqa: BLE001 - unavailable / blocked / bot check
        raise FetchError(f"I couldn't download that audio: {str(exc).splitlines()[0][:160]}") from exc


def from_youtube(url: str, about: str = "", from_time: Any = None, length: float | None = None,
                 name: str | None = None, client: Any = None, log: Callable[[str], None] = print) -> dict:
    """A sound from a YouTube link. `from_time` ('0:42', 42) says where it starts; without it Gemini listens to the
    video and picks the moment that matches `about`."""
    try:
        info = ytclip.video_info(url)
    except ytclip.YtError as exc:
        raise FetchError(str(exc)) from exc
    length = min(max(float(length or DEFAULT_LINK_LEN), 0.3), MAX_LINK_LEN)
    start = ytclip.parse_time(from_time)
    why = ""
    if start is None:
        if info["duration"] <= length + 1.0:
            start = 0.0
        elif not about.strip():
            raise FetchError("Tell me which sound to look for in that video, or when it happens (for example 0:42).")
        else:
            log("Listening to the video to find the sound...")
            try:
                start, why = ytclip.find_window(info, about.strip(), length, client=client, sound=True)
            except ytclip.YtError as exc:
                raise FetchError(str(exc)) from exc
    if start >= info["duration"] - 0.3:
        raise FetchError(f"That video is only {info['duration']:.0f}s long, so it has nothing at {start:.0f}s.")
    end = min(start + length + 0.5, info["duration"])
    work = Path(tempfile.mkdtemp(prefix="sfx_"))
    try:
        audio = _download(info, start, end, work)
        lib_id = library.unique_id(f"yt_{info['id']}_{int(start)}")
        label = (name or about or info["title"] or lib_id).strip()[:60]
        entry = add_to_library(audio, lib_id, label, "youtube", info["url"], max_s=length + 0.3,
                               window_start=round(start, 2), video_title=info["title"][:100])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    entry["why"] = why
    return entry


def download_by_search(query: str, about: str, work: Path, client: Any = None,
                       log: Callable[[str], None] = print) -> tuple[Path, dict]:
    """Search YouTube for `query` and download audio of the best match. Returns (audio file, the video's metadata).
    Short results are taken from their start; a longer one is listened to by Gemini to find the sound."""
    try:
        cands = youtube.search_candidates(query, limit=8, min_s=0.5, max_s=NAME_SEARCH_MAX_S)
    except youtube.SourcingError as exc:
        raise FetchError(str(exc)) from exc
    if not cands:
        raise FetchError(f"I couldn't find '{about}' on YouTube.")
    ordered = [c for c in cands if c["duration"] <= NAME_PREFER_S] + [c for c in cands if c["duration"] > NAME_PREFER_S]
    last = "no usable result"
    for cand in ordered[:NAME_TRIES]:
        try:
            if cand["duration"] <= NAME_PREFER_S:
                start, end = 0.0, cand["duration"]
            else:
                log("Listening to a longer video to find the sound...")
                start, _ = ytclip.find_window(cand, about, DEFAULT_LINK_LEN, client=client, sound=True)
                end = min(start + DEFAULT_LINK_LEN + 0.5, cand["duration"])
            return _download(cand, start, end, work), cand
        except (FetchError, ytclip.YtError) as exc:
            last = str(exc)
    raise FetchError(f"I couldn't get '{about}': {last}")


def from_name(name: str, client: Any = None, log: Callable[[str], None] = print) -> dict:
    """A sound found on YouTube by name ('fah', 'sad violin'), saved into the library under that name."""
    name = " ".join(name.split())[:60]
    if not name:
        raise FetchError("Tell me the name of the sound.")
    work = Path(tempfile.mkdtemp(prefix="sfx_"))
    try:
        audio, cand = download_by_search(f"{name} sound effect", name, work, client, log)
        lib_id = library.unique_id(f"web_{library.slugify(name)}")
        return add_to_library(audio, lib_id, name, "web", cand["url"], video_title=cand["title"][:100])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_catalog_entry(row: dict, client: Any = None, log: Callable[[str], None] = print) -> dict:
    """One catalog sound into <sfx_dir>/<id>.mp3 (used by scripts/build_sfx_library.py)."""
    work = Path(tempfile.mkdtemp(prefix="sfx_"))
    try:
        audio, cand = download_by_search(row["query"], row["name"], work, client, log)
        return add_to_library(audio, row["id"], row["name"], "catalog", cand["url"], aliases=row["aliases"],
                              subdir="", video_title=cand["title"][:100])
    finally:
        shutil.rmtree(work, ignore_errors=True)
