"""Turn an image into a short silent still clip, so pictures flow through the same pipeline as video B-roll
(framing / Ken-Burns zoom, overlay / split / pip layouts, previews, swap, render, cleanup)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.config import require_binary

STILL_SECONDS = 10.0  # long enough for any slot; a static clip this long is a few hundred KB
MAX_SIDE = 2160  # keeps zoom (up to 3x) sharp without huge files
FPS = 30


class StillError(RuntimeError):
    pass


def normalize_image(src: Path, dest: Path) -> tuple[int, int]:
    """Open any common image (jpg/png/webp/...), fix EXIF rotation, flatten transparency onto black, cap the
    size and save as an even-sized PNG. Raises StillError if the file is not a readable image."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(src) as im:
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (0, 0, 0))
                bg.paste(im, mask=im.getchannel("A"))
                im = bg
            else:
                im = im.convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            w, h = im.size
            w, h = max(2, w - w % 2), max(2, h - h % 2)  # libx264 + yuv420p need even sizes
            if (w, h) != im.size:
                im = im.crop((0, 0, w, h))
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, format="PNG")
            return w, h
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise StillError(f"That file is not a readable image ({exc}).") from exc


def image_to_clip(image: Path, out_path: Path, seconds: float = STILL_SECONDS) -> dict:
    """Write `out_path` (.mp4): the image held for `seconds`. Returns {"clip_duration", "width", "height"}."""
    norm = out_path.with_name(out_path.stem + "_src.png")
    try:
        w, h = normalize_image(image, norm)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-loop", "1", "-framerate",
               str(FPS), "-t", f"{seconds:.3f}", "-i", str(norm), "-an", "-c:v", "libx264", "-preset", "veryfast",
               "-tune", "stillimage", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out_path.is_file():
            raise StillError(f"Could not turn the image into a clip: {r.stderr.strip()[-300:]}")
        return {"clip_duration": float(seconds), "width": w, "height": h}
    finally:
        norm.unlink(missing_ok=True)
