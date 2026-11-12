"""FFmpeg composition: speaker video + framed, muted B-roll -> 1080x1920 reel.

Two layouts per B-roll slot:
  overlay - the clip fills the whole frame on top of the speaker (the voice keeps running underneath);
  split   - the clip fills the top half, the speaker (facecam) the bottom half, a thin border between.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

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

LAYOUTS = ("overlay", "split")


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


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        w, h = out.stdout.strip().split(",")[:2]
        return int(w), int(h)
    except ValueError as exc:
        raise RenderError(f"Could not read the size of clip {path.name}.") from exc


def box_for(layout: str) -> tuple[int, int]:
    """Size of the area a clip is framed into."""
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
        w, h = box_for(ov.layout)
        dur = ov.end - ov.start
        # Clip timestamps are shifted to start at the slot's start, so frames line up with the speaker.
        chain = clip_filter(ov.framing, dur, w, h, ov.src_w, ov.src_h, FPS)
        gate = f"enable='between(t,{ov.start:.3f},{ov.end:.3f})'"
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


def render(main_video: Path, overlays: list[Overlay], out_path: Path,
           subtitles_file: str | None = None, cwd: Path | None = None,
           face_y: float = DEFAULT_FACE_Y) -> Path:
    """Render the final video. Only the main video's audio is used: B-roll audio is never mapped.

    `subtitles_file` must be relative to `cwd` (avoids Windows drive-letter escaping in filters).
    """
    ffmpeg = require_binary("ffmpeg")
    for ov in overlays:  # zoom/pan needs the clip's real size
        if not is_default(ov.framing) and not (ov.src_w and ov.src_h):
            ov.src_w, ov.src_h = probe_size(ov.path)
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(main_video)]
    for ov in overlays:
        # -t trims the clip to its slot length; -an drops any audio the source might carry.
        cmd += ["-an", "-t", f"{ov.end - ov.start:.3f}", "-i", str(ov.path)]
    cmd += [
        "-filter_complex", build_filtergraph(overlays, subtitles_file, face_y),
        "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    if result.returncode != 0 or not out_path.is_file():
        raise RenderError(f"FFmpeg render failed: {result.stderr.strip()[-800:]}")
    return out_path
