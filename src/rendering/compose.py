"""FFmpeg composition: speaker video + framed, muted B-roll -> 1080x1920 reel.

Two layouts per B-roll slot:
  overlay - the clip fills the whole frame on top of the speaker (the voice keeps running underneath);
  split   - the clip fills the top half, the speaker (facecam) the bottom half, a thin border between.
"""
from __future__ import annotations

import functools
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src import config
from src.config import require_binary
from src.rendering.framing import clip_filter, fill_crop, is_default

OUT_W, OUT_H, FPS = 1080, 1920, 30
SPLIT_Y = OUT_H // 2  # the border sits on this line: b-roll above, facecam below
SPLIT_H = OUT_H - SPLIT_Y
BORDER_H = 8
BORDER_COLOR = "white"
DEFAULT_FACE_Y = 0.4  # where in the vertical speaker video the facecam half is cropped (0 = top, 1 = bottom)

# Fill the 9:16 frame by scaling up (aspect preserved, no stretching), then crop the centre.
FILL_CROP = fill_crop(OUT_W, OUT_H)

LAYOUTS = ("overlay", "split", "pip")

# Picture-in-picture: a small framed clip over the speaker. `pip = {"pos": ..., "scale": ...}` where scale is the
# box width as a fraction of the frame width. Edges keep clear of the app UI at the top and of the captions at the
# bottom (captions sit around y = 1400-1520).
PIP_ASPECT = 4 / 3  # box width / height
PIP_BORDER = 6
PIP_MARGIN_X, PIP_MARGIN_TOP, PIP_MARGIN_BOTTOM = 50, 190, 440
PIP_POSITIONS = ("top-left", "top", "top-right", "center", "bottom-left", "bottom", "bottom-right")
PIP_DEFAULT = {"pos": "top-right", "scale": 0.6}
PIP_MIN_SCALE, PIP_MAX_SCALE = 0.3, 0.95


class RenderError(RuntimeError):
    pass


@dataclass
class Overlay:
    path: Path
    start: float  # position on the main timeline
    end: float
    layout: str = "overlay"
    framing: dict | None = None  # zoom/pan keyframes (see framing.py); None = plain centre fill
    src_w: int = 0  # clip size; probed by render() when zoom/pan needs it
    src_h: int = 0
    pip: dict | None = None  # picture-in-picture position/size (layout "pip" only)


@dataclass
class Sound:
    """A sound effect mixed into the audio: `path` plays from `start` seconds on the video timeline."""
    path: Path
    start: float
    volume: float = 1.0


AUDIO_RATE = 44100


def has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return "audio" in out.stdout


def build_audio_graph(sounds: list[Sound], first_input: int, main_has_audio: bool, main_duration: float) -> str:
    """Mix the speaker's audio with every sound effect: each effect is delayed to its start and scaled by its
    volume, then all are summed (no automatic level change), with a limiter so a loud overlap cannot clip.
    The result is as long as the speaker's audio. `first_input` is the ffmpeg input index of the first sound."""
    fmt = f"aresample={AUDIO_RATE},aformat=channel_layouts=stereo"
    if main_has_audio:
        parts = [f"[0:a:0]{fmt}[a0]"]
    else:  # a silent speaker video still needs a base track to mix the effects onto
        parts = [f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=0:{max(main_duration, 0.1):.3f}[a0]"]
    labels = ["[a0]"]
    for n, snd in enumerate(sounds):
        ms = max(0, round(snd.start * 1000))
        parts.append(f"[{first_input + n}:a:0]{fmt},volume={snd.volume:.3f},adelay={ms}|{ms}[s{n}]")
        labels.append(f"[s{n}]")
    parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=first:normalize=0:dropout_transition=0,"
                 "alimiter=limit=0.97[aout]")
    return ";".join(parts)


def normalize_pip(pip: dict | None) -> dict:
    """Clamp and complete a pip dict; anything unknown falls back to the default."""
    pip = pip or {}
    pos = pip.get("pos") if pip.get("pos") in PIP_POSITIONS else PIP_DEFAULT["pos"]
    try:
        scale = float(pip.get("scale", PIP_DEFAULT["scale"]))
    except (TypeError, ValueError):
        scale = PIP_DEFAULT["scale"]
    return {"pos": pos, "scale": round(min(max(scale, PIP_MIN_SCALE), PIP_MAX_SCALE), 2)}


def pip_geometry(pip: dict | None) -> tuple[int, int, int, int]:
    """(width, height, x, y) of the picture-in-picture box in the 1080x1920 frame."""
    p = normalize_pip(pip)
    w = round(OUT_W * p["scale"] / 2) * 2
    h = round(w / PIP_ASPECT / 2) * 2
    vert, _, horiz = p["pos"].partition("-")  # "top-left" -> ("top", "left"); "top" -> ("top", "")
    x ={"left": PIP_MARGIN_X, "right": OUT_W - w - PIP_MARGIN_X}.get(horiz, (OUT_W - w) // 2)
    y = {"top": PIP_MARGIN_TOP, "bottom": OUT_H - h - PIP_MARGIN_BOTTOM}.get(vert, (OUT_H - h) // 2)
    return w, h, max(0, min(x, OUT_W - w)), max(0, min(y, OUT_H - h))


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        w, h = out.stdout.strip().split(",")[:2]
        return int(w), int(h)
    except ValueError as exc:
        raise RenderError(f"Could not read the size of clip {path.name}.") from exc


def box_for(layout: str, pip: dict | None = None) -> tuple[int, int]:
    """Size of the area a clip is framed into."""
    if layout == "pip":
        return pip_geometry(pip)[:2]
    return (OUT_W, SPLIT_Y) if layout == "split" else (OUT_W, OUT_H)


def build_filtergraph(overlays: list[Overlay], subtitles_file: str | None = None,
                      face_y: float = DEFAULT_FACE_Y) -> str:
    """Chain: normalise main -> per slot add the clip (full frame, or top half + facecam + border)
    -> (optional) burn subtitles."""
    splits = [ov for ov in overlays if ov.layout == "split"]
    base = f"[0:v]{FILL_CROP},fps={FPS},format=yuv420p"
    if splits:  # the speaker video is needed again for each split slot's facecam half
        copies = "".join(f"[m{k}]" for k in range(1, len(splits) + 1))
        parts = [f"{base},split={len(splits) + 1}[b0]{copies}"]
    else:
        parts = [f"{base}[b0]"]

    face_top = round((OUT_H - SPLIT_H) * min(max(face_y, 0.0), 1.0))
    k = 0
    for n, ov in enumerate(overlays, 1):
        w, h = box_for(ov.layout, ov.pip)
        dur = ov.end - ov.start
        # Clip timestamps are shifted to start at the slot's start, so frames line up with the speaker.
        chain = clip_filter(ov.framing, dur, w, h, ov.src_w, ov.src_h, FPS)
        gate = f"enable='between(t,{ov.start:.3f},{ov.end:.3f})'"
        if ov.layout == "pip":
            _, _, px, py = pip_geometry(ov.pip)
            parts.append(f"[{n}:v]{chain},format=yuv420p,tpad=stop_mode=clone:stop_duration={dur:.3f},"
                         f"drawbox=x=0:y=0:w=iw:h=ih:color={BORDER_COLOR}:t={PIP_BORDER},"
                         f"setpts=PTS+{ov.start:.3f}/TB[c{n}]")
            parts.append(f"[b{n - 1}][c{n}]overlay={px}:{py}:{gate}:eof_action=pass[b{n}]")
            continue
        if ov.layout != "split":
            parts.append(f"[{n}:v]{chain},format=yuv420p,setpts=PTS+{ov.start:.3f}/TB[c{n}]")
            parts.append(f"[b{n - 1}][c{n}]overlay={gate}:eof_action=pass[b{n}]")
            continue
        k += 1
        # Freeze the last frame if the clip is a little shorter than the slot, so the top half never flashes
        # the speaker.
        parts.append(f"[{n}:v]{chain},format=yuv420p,tpad=stop_mode=clone:stop_duration={dur:.3f},"
                     f"setpts=PTS+{ov.start:.3f}/TB[c{n}]")
        parts.append(f"[m{k}]crop={OUT_W}:{SPLIT_H}:0:{face_top},setsar=1[f{n}]")
        parts.append(f"[b{n - 1}][c{n}]overlay=0:0:{gate}:eof_action=pass[t{n}]")
        parts.append(f"[t{n}][f{n}]overlay=0:{SPLIT_Y}:{gate}[u{n}]")
        parts.append(f"[u{n}]drawbox=x=0:y={SPLIT_Y - BORDER_H // 2}:w={OUT_W}:h={BORDER_H}:"
                     f"color={BORDER_COLOR}:t=fill:{gate}[b{n}]")
    last = f"b{len(overlays)}"
    parts.append(f"[{last}]subtitles={subtitles_file}[v]" if subtitles_file else f"[{last}]null[v]")
    return ";".join(parts)


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True if this ffmpeg lists h264_nvenc AND a 1-frame test encode works (the listing alone is not enough:
    it says yes on machines without an NVIDIA GPU or driver)."""
    try:
        ffmpeg = require_binary("ffmpeg")
        listed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        if "h264_nvenc" not in listed.stdout:
            return False
        test = subprocess.run(
            [ffmpeg, "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, text=True, timeout=30)
        return test.returncode == 0
    except Exception:  # noqa: BLE001 - any probe failure just means "use the CPU"
        return False


def video_encoder_args(preset: str, crf: int, hw: bool = False) -> list[str]:
    """libx264 by default. With `hw=True` and VIBECROP_ENCODER=auto|nvenc, the NVIDIA encoder if it really works."""
    pref = config.video_encoder_pref()
    if hw and pref != "cpu" and nvenc_available():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf + 1), "-b:v", "0",
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
                          str(path)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def parse_progress_line(line: str, total_s: float) -> float | None:
    """One line of `ffmpeg -progress pipe:1` -> fraction done (0-1), or None for lines that carry no time."""
    key, _, value = line.strip().partition("=")
    if key not in ("out_time_us", "out_time_ms") or total_s <= 0:  # (out_time_ms is in microseconds, too)
        return None
    try:
        return min(max(int(value) / 1_000_000 / total_s, 0.0), 1.0)
    except ValueError:  # "N/A" before the first frame
        return None


def _run_ffmpeg(cmd: list[str], cwd: Path | None, total_s: float, progress: Callable[[float], None] | None):
    """Run ffmpeg. With a `progress` callback it streams `-progress` output and reports the fraction done."""
    if progress is None:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:  # a file, so stderr cannot fill
        proc = subprocess.Popen([*cmd[:1], "-progress", "pipe:1", "-nostats", *cmd[1:]], stdout=subprocess.PIPE,
                                stderr=err, text=True, cwd=str(cwd) if cwd else None)
        last = -1.0
        for line in proc.stdout:  # type: ignore[union-attr]
            frac = parse_progress_line(line, total_s)
            if frac is not None and frac - last >= 0.01:
                last = frac
                try:
                    progress(frac)
                except Exception:  # noqa: BLE001 - a broken UI callback must never kill the render
                    progress = lambda _f: None  # noqa: E731
        proc.wait()
        err.seek(0)
        return subprocess.CompletedProcess(cmd, proc.returncode, "", err.read())


def render(main_video: Path, overlays: list[Overlay], out_path: Path,
           subtitles_file: str | None = None, cwd: Path | None = None,
           face_y: float = DEFAULT_FACE_Y, preset: str = "veryfast", crf: int = 20,
           progress: Callable[[float], None] | None = None, hw: bool = False,
           sounds: list[Sound] | None = None) -> Path:
    """Render the final video. Only the main video's audio is used: B-roll audio is never mapped.

    `subtitles_file` must be relative to `cwd` (avoids Windows drive-letter escaping in filters).
    `progress(fraction)` is called while ffmpeg runs; `hw=True` allows the GPU encoder (final export only).
    `sounds` are sound effects mixed into the speaker's audio (see build_audio_graph).
    """
    ffmpeg = require_binary("ffmpeg")
    for ov in overlays:  # zoom/pan needs the clip's real size
        if not is_default(ov.framing) and not (ov.src_w and ov.src_h):
            ov.src_w, ov.src_h = probe_size(ov.path)
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(main_video)]
    for ov in overlays:
        # -t trims the clip to its slot length; -an drops any audio the source might carry.
        cmd += ["-an", "-t", f"{ov.end - ov.start:.3f}", "-i", str(ov.path)]
    graph = build_filtergraph(overlays, subtitles_file, face_y)
    audio_map = ["-map", "0:a:0?"]
    sounds = [snd for snd in (sounds or []) if snd.path.is_file()]
    if sounds:
        for snd in sounds:
            cmd += ["-i", str(snd.path)]
        graph += ";" + build_audio_graph(sounds, len(overlays) + 1, has_audio(main_video), probe_duration(main_video))
        audio_map = ["-map", "[aout]"]
    cmd += [
        "-filter_complex", graph,
        "-map", "[v]", *audio_map,
        *video_encoder_args(preset, crf, hw),
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out_path),
    ]
    total_s = probe_duration(main_video) if progress else 0.0
    result = _run_ffmpeg(cmd, cwd, total_s, progress)
    if result.returncode != 0 or not out_path.is_file():
        raise RenderError(f"FFmpeg render failed: {result.stderr.strip()[-800:]}")
    if progress:
        try:
            progress(1.0)
        except Exception:  # noqa: BLE001 - the video is already written; a broken progress bar must not fail it
            pass
    return out_path
