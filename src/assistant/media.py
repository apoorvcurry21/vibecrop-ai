"""Turning what the user attached ("att_1") into a local file, and cutting attached videos.

The model only ever sees short ids, never paths. (Web image search and AI image generation were removed: the user adds
their own pictures and clips, in the chat or with the "Add picture / clip" panel.)"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from src.assistant import assets
from src.assistant.tools import ToolContext, ToolError
from src.config import require_binary

MAX_VIDEO_TRIM_S = 60.0
ID_RE = re.compile(r"^att_\d+$")


class MediaError(RuntimeError):
    pass


def resolve_image(job: Path, ctx: ToolContext, ref: str) -> tuple[Path, dict]:
    """Turn 'att_1' into (local image file, asset fields)."""
    ref = (ref or "").strip()
    if not ID_RE.match(ref):
        raise ToolError("Give me the id of a picture the user attached (att_1, att_2...). I can't fetch pictures from "
                        "the web: ask the user to attach the picture.")
    att = ctx.attachments.get(ref) or assets.load_attachment(job, ref)
    if att is None or att.kind != "image":
        raise ToolError(f"'{ref}' is not an attached picture.")
    return att.path, {"origin": "attachment", "title": att.name}


# ----------------------------------------------------------------------------- attached videos
def trim_video(src: Path, dest: Path, from_s: float, length: float) -> float:
    """Cut [from_s, from_s+length] out of a video file as a silent H.264 clip, capped at 1080p. Returns its length."""
    if length <= 0 or length > MAX_VIDEO_TRIM_S:
        raise ToolError(f"A clip must be between 0 and {MAX_VIDEO_TRIM_S:g} seconds.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-ss", f"{max(from_s, 0):.3f}",
           "-t", f"{length:.3f}", "-i", str(src), "-an", "-vf", "scale='min(1080,iw)':-2:flags=bicubic,setsar=1",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dest.is_file():
        raise MediaError(f"Could not cut that video: {r.stderr.strip()[-250:]}")
    from src.sourcing.youtube import _probe

    got, ok = _probe(dest)
    if not ok or got < 0.5:
        dest.unlink(missing_ok=True)
        raise MediaError("There is no video at that point in the file. Pick an earlier start.")
    return got
