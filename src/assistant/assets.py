"""User-supplied and fetched media for the chat assistant: saving attachments safely into <job>/assets/.

Everything the model can refer to is an id (att_1, ...), never a path. Files are renamed on save, so a hostile file
name can never escape the folder."""
from __future__ import annotations

import io
import re
from pathlib import Path

from src.assistant.tools import Attachment

ASSETS = "assets"
STATIC_DIR = Path(__file__).parent / "static"  # the layout mockup images
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
MAX_ATTACHMENT_BYTES = 200 * 1024 * 1024  # a short clip; far above any picture
VISION_SIDE = 1024  # what the model sees of a picture (a smaller copy keeps the request fast and cheap)


class AssetError(RuntimeError):
    pass


def assets_dir(job: Path) -> Path:
    d = job / ASSETS
    d.mkdir(parents=True, exist_ok=True)
    return d


def _next_number(job: Path) -> int:
    nums = [int(m.group(1)) for p in assets_dir(job).glob("att_*") if (m := re.match(r"att_(\d+)", p.name))]
    return max(nums, default=0) + 1


def save_attachment(job: Path, name: str, data: bytes) -> Attachment:
    """Store an uploaded file as assets/att_N.<ext>. Only images and short videos are accepted."""
    ext = Path(name or "").suffix.lower()
    if ext in IMAGE_EXT:
        kind = "image"
    elif ext in VIDEO_EXT:
        kind = "video"
    else:
        raise AssetError(f"'{name}' is not a picture or video I can use (png, jpg, webp, mp4, mov...).")
    if not data or len(data) > MAX_ATTACHMENT_BYTES:
        raise AssetError(f"'{name}' is empty or too big (limit {MAX_ATTACHMENT_BYTES // 1024 // 1024} MB).")
    n = _next_number(job)
    path = assets_dir(job) / f"att_{n}{ext}"
    path.write_bytes(data)
    return Attachment(id=f"att_{n}", path=path, kind=kind, name=Path(name).name[:80])


def load_attachment(job: Path, att_id: str, name: str = "") -> Attachment | None:
    """Re-find a saved attachment by id (e.g. after the page reloaded)."""
    if not re.fullmatch(r"att_\d+", att_id or ""):
        return None
    for p in assets_dir(job).glob(att_id + ".*"):
        kind = "image" if p.suffix.lower() in IMAGE_EXT else "video"
        return Attachment(id=att_id, path=p, kind=kind, name=name or p.name)
    return None


def vision_jpeg(path: Path) -> bytes | None:
    """A small JPEG copy of a picture for the model to look at (None if it cannot be read)."""
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((VISION_SIDE, VISION_SIDE))
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=85)
            return out.getvalue()
    except (OSError, ValueError):
        return None
