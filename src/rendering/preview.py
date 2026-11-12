"""Editor previews: what a B-roll will look like in the 9:16 reel, before rendering.

Everything here goes through the same code as the final render (`framing.clip_filter`, `framing.crop_rect`,
`compose.render`), so the preview cannot disagree with the export.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

from src.config import require_binary
from src.rendering import framing as fr
from src.rendering.compose import FPS, OUT_H, OUT_W, Overlay, RenderError, box_for, probe_size, render

PREVIEW_W = 360  # width of the still previews, in pixels
START_COLOR, END_COLOR = (255, 214, 10), (64, 220, 255)


class PreviewError(RuntimeError):
    pass


def _ffmpeg_png(args: list[str]) -> bytes:
    cmd = [require_binary("ffmpeg"), "-nostdin", "-loglevel", "error", *args, "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0 or not out.stdout:
        raise PreviewError(f"Could not make the preview: {out.stderr.decode(errors='replace').strip()[-300:]}")
    return out.stdout


def frame_png(clip: Path, framing: dict | None, layout: str, duration: float, at: str,
              clip_len: float | None = None) -> bytes:
    """One still of the framed B-roll box (the whole 9:16 frame for overlay, the top half for split):
    `at` = "start" or "end" keyframe, i.e. the first or the last frame of the slot."""
    w, h = box_for(layout)
    sw, sh = probe_size(clip)
    chain = fr.clip_filter(framing, duration, w, h, sw, sh, FPS)
    last = max(0, int(min(duration, clip_len or duration) * FPS) - 1)
    n = 0 if at == "start" else last
    return _ffmpeg_png(["-an", "-t", f"{duration:.3f}", "-i", str(clip), "-vf",
                        f"{chain},select=eq(n\\,{n}),scale={PREVIEW_W}:-2"])


def _dashed_rect(draw, box, color, width=3, dash=10):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = int(length // dash)
        for i in range(0, steps, 2):
            t0, t1 = i * dash / length, min((i + 1) * dash / length, 1)
            draw.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                      fill=color, width=width)


def source_view_png(clip: Path, framing: dict | None, layout: str, width: int = 640) -> bytes:
    """The WHOLE source clip (e.g. 16:9) with the part that ends up in the reel outlined: a solid box for the
    start keyframe and a dashed one for the end keyframe. Everything outside the start box is dimmed."""
    from PIL import Image, ImageDraw

    sw, sh = probe_size(clip)
    img = Image.open(io.BytesIO(_ffmpeg_png(["-i", str(clip), "-vf", f"scale={width}:-2"]))).convert("RGB")
    k = img.width / sw
    w, h = box_for(layout)
    f = fr.normalize(framing)

    def rect(kf):
        left, top, cw, ch = fr.crop_rect(sw, sh, w, h, kf["z"], kf["x"], kf["y"])
        return (left * k, top * k, (left + cw) * k - 1, (top + ch) * k - 1)

    start_box, end_box = rect(f["start"]), rect(f["end"])
    dim = Image.new("RGB", img.size, (0, 0, 0))
    mask = Image.new("L", img.size, 130)
    ImageDraw.Draw(mask).rectangle(start_box, fill=0)
    img = Image.composite(dim, img, mask)
    draw = ImageDraw.Draw(img)
    draw.rectangle(start_box, outline=START_COLOR, width=3)
    if not fr.is_static(f):
        _dashed_rect(draw, end_box, END_COLOR)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def render_clip_preview(clip: Path, main_video: Path | None, start: float, end: float, layout: str,
                        framing: dict | None, face_y: float, out_path: Path) -> Path:
    """A short real render of just this slot: the speaker video (or grey if it is gone) with the B-roll laid
    over it exactly as in the final export. Fast settings, same filter graph."""
    ffmpeg = require_binary("ffmpeg")
    duration = max(end - start, 0.1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg = out_path.with_name(out_path.stem + "_speaker.mp4")
    if main_video is not None and main_video.is_file():
        src = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(main_video)]
    else:
        src = ["-f", "lavfi", "-i", f"color=c=gray:s={OUT_W}x{OUT_H}:r={FPS}:d={duration:.3f}"]
    r = subprocess.run([ffmpeg, "-nostdin", "-y", "-loglevel", "error", *src, "-an", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p", str(seg)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not seg.is_file():
        raise PreviewError(f"Could not cut the speaker video: {r.stderr.strip()[-300:]}")
    try:
        return render(seg, [Overlay(clip, 0.0, duration, layout=layout, framing=framing)], out_path,
                      face_y=face_y, preset="ultrafast", crf=28)
    except RenderError as exc:
        raise PreviewError(str(exc)) from exc
    finally:
        seg.unlink(missing_ok=True)
