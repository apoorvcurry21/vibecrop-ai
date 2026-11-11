"""FFmpeg composition: speaker video + center-cropped, muted B-roll overlays -> 1080x1920 reel."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import require_binary

OUT_W, OUT_H, FPS = 1080, 1920, 30

# Fill the 9:16 frame by scaling up (aspect preserved, no stretching), then crop the centre.
FILL_CROP = f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},setsar=1"


class RenderError(RuntimeError):
    pass


@dataclass
class Overlay:
    path: Path
    start: float  # position on the main timeline
    end: float


def build_filtergraph(overlays: list[Overlay], subtitles_file: str | None = None) -> str:
    """Chain: normalise main -> overlay each clip during its slot -> (optional) burn subtitles."""
    parts = [f"[0:v]{FILL_CROP},fps={FPS},format=yuv420p[b0]"]
    for n, ov in enumerate(overlays, 1):
        # Clip timestamps are shifted to start at the slot's start, so frames line up with the speaker.
        parts.append(
            f"[{n}:v]setpts=PTS-STARTPTS,fps={FPS},{FILL_CROP},format=yuv420p,"
            f"setpts=PTS+{ov.start:.3f}/TB[c{n}]"
        )
        parts.append(
            f"[b{n - 1}][c{n}]overlay=enable='between(t,{ov.start:.3f},{ov.end:.3f})':eof_action=pass[b{n}]"
        )
    last = f"b{len(overlays)}"
    parts.append(f"[{last}]subtitles={subtitles_file}[v]" if subtitles_file else f"[{last}]null[v]")
    return ";".join(parts)


def render(main_video: Path, overlays: list[Overlay], out_path: Path,
           subtitles_file: str | None = None, cwd: Path | None = None) -> Path:
    """Render the final video. Only the main video's audio is used: B-roll audio is never mapped.

    `subtitles_file` must be relative to `cwd` (avoids Windows drive-letter escaping in filters).
    """
    ffmpeg = require_binary("ffmpeg")
    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(main_video)]
    for ov in overlays:
        # -t trims the clip to its slot length; -an drops any audio the source might carry.
        cmd += ["-an", "-t", f"{ov.end - ov.start:.3f}", "-i", str(ov.path)]
    cmd += [
        "-filter_complex", build_filtergraph(overlays, subtitles_file),
        "-map", "[v]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None)
    if result.returncode != 0 or not out_path.is_file():
        raise RenderError(f"FFmpeg render failed: {result.stderr.strip()[-800:]}")
    return out_path
