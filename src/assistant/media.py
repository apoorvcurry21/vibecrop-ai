"""Getting pictures and clips for the chat assistant: web image search, AI image generation, and turning whatever the
user refers to ("att_1", "res_3", "gen_2", a link) into a local image file.

The model only ever sees short ids. Search results are remembered per job in assets/candidates.json, so the model
never has to repeat (or invent) a URL for them."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from src.assistant import assets, fetch
from src.assistant.tools import ToolContext, ToolError
from src.config import require_binary

CANDIDATES = "candidates.json"
THUMB_SIDE = 320
SEARCH_COUNT = 6
MAX_VIDEO_TRIM_S = 60.0
DEFAULT_IMAGE_MODELS = ("gemini-3.1-flash-lite-image", "gemini-3.1-flash-image", "gemini-2.5-flash-image")
ID_RE = re.compile(r"^(att|res|gen|url)_\d+$")


class MediaError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- bookkeeping
def _registry(job: Path) -> dict:
    f = assets.assets_dir(job) / CANDIDATES
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    except (OSError, ValueError):
        return {}


def _save_registry(job: Path, data: dict) -> None:
    (assets.assets_dir(job) / CANDIDATES).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _next_id(job: Path, prefix: str) -> str:
    nums = [int(m.group(1)) for p in assets.assets_dir(job).glob(f"{prefix}_*")
            if (m := re.match(rf"{prefix}_(\d+)", p.name))]
    nums += [int(k.split("_")[1]) for k in _registry(job) if k.startswith(prefix + "_")]
    return f"{prefix}_{max(nums, default=0) + 1}"


def _domain(url: str) -> str:
    host = urlparse(url or "").hostname or ""
    return host[4:] if host.startswith("www.") else host


def _thumb(data: bytes, dest: Path) -> None:
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((THUMB_SIDE, THUMB_SIDE))
        im.save(dest, format="JPEG", quality=80)


# ----------------------------------------------------------------------------- web image search
def search_images(job: Path, query: str, count: int = SEARCH_COUNT) -> list[dict]:
    """Search the web for pictures. Returns [{id, title, domain, width, height, thumb}] (thumb is relative to
    the job). Thumbnails download in parallel; the full picture is only fetched for the one that gets used."""
    from ddgs import DDGS

    count = max(1, min(int(count), 10))
    try:
        found = DDGS(timeout=10).images(query, max_results=count * 2, safesearch="moderate")
    except Exception as exc:  # noqa: BLE001 - ddgs raises its own family of errors (rate limit, no results...)
        raise MediaError(f"Image search failed ({type(exc).__name__}). Try again in a moment.") from exc
    rows = [r for r in found or [] if r.get("image") and (r.get("thumbnail") or r.get("image"))]
    if not rows:
        raise MediaError(f"No pictures found for '{query}'. Try different words.")
    reg = _registry(job)
    base = int(_next_id(job, "res").split("_")[1])
    picked = [(f"res_{base + i}", r) for i, r in enumerate(rows[:count + 2])]  # +2 spare for dead links

    def grab(item):
        rid, r = item
        try:
            data = fetch.fetch_image(r.get("thumbnail") or r["image"], max_bytes=3 * 1024 * 1024)
            dest = assets.assets_dir(job) / f"{rid}_thumb.jpg"
            _thumb(data, dest)
            return rid, r, dest
        except Exception:  # noqa: BLE001 - one dead thumbnail just drops that result
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        loaded = [g for g in pool.map(grab, picked) if g]
    got = loaded[:count]
    for _, _, spare in loaded[count:]:  # thumbnails of the spare results are not needed
        spare.unlink(missing_ok=True)
    if not got:
        raise MediaError("The pictures found could not be loaded. Try again or search for something else.")
    out = []
    for rid, r, dest in got:
        reg[rid] = {"url": r["image"], "page": r.get("url", ""), "title": (r.get("title") or "")[:120],
                    "domain": _domain(r.get("url") or r["image"])}
        out.append({"id": rid, "title": reg[rid]["title"], "domain": reg[rid]["domain"],
                    "width": r.get("width"), "height": r.get("height"), "thumb": f"{assets.ASSETS}/{dest.name}"})
    _save_registry(job, reg)
    return out


# ----------------------------------------------------------------------------- AI image generation
def image_model_chain() -> list[str]:
    first = os.environ.get("GEMINI_IMAGE_MODEL", "").strip()
    return ([first] if first else []) + [m for m in DEFAULT_IMAGE_MODELS if m != first]


def generate_image(job: Path, prompt: str, client=None, models: list[str] | None = None) -> dict:
    """Make a picture with a Gemini image model and save it as assets/gen_N.png. Returns {id, path}."""
    from google.genai import errors, types

    from src.gemini_client import make_client

    prompt = " ".join((prompt or "").split())
    if not prompt:
        raise MediaError("Tell me what the picture should show.")
    client = client or make_client()
    cfg = types.GenerateContentConfig(response_modalities=["IMAGE"],
                                      image_config=types.ImageConfig(aspect_ratio="9:16"))
    last = "no image came back"
    for model in models or image_model_chain():
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 429:
                last = "quota"
            elif code in (400, 403, 404):
                last = f"{model} is not available for this key"
            else:
                last = f"{type(exc).__name__}"
            continue
        cand = (getattr(resp, "candidates", None) or [None])[0]
        parts = list(getattr(getattr(cand, "content", None), "parts", None) or [])
        blobs = [p.inline_data.data for p in parts if getattr(p, "inline_data", None) and p.inline_data.data]
        if blobs:
            gid = _next_id(job, "gen")
            dest = assets.assets_dir(job) / f"{gid}.png"
            from PIL import Image

            with Image.open(io.BytesIO(blobs[0])) as im:
                im.convert("RGB").save(dest, format="PNG")
            return {"id": gid, "path": f"{assets.ASSETS}/{dest.name}"}
        last = "refused"
    if last == "quota":
        raise MediaError("Image generation isn't available on this Gemini key (its image models need a "
                         "billing-enabled key). I can search the web for a picture instead.")
    if last == "refused":
        raise MediaError("Gemini wouldn't make that picture (it often refuses real people and brands). I can "
                         "search the web for a real photo instead.")
    raise MediaError(f"Could not generate the picture ({last}). I can search the web for one instead.")


# ----------------------------------------------------------------------------- resolve a reference to a file
def resolve_image(job: Path, ctx: ToolContext, ref: str) -> tuple[Path, dict]:
    """Turn 'att_1' / 'gen_2' / 'res_3' / an image link into (local image file, asset fields).
    The asset fields carry the small credit line shown in the video (web pictures only)."""
    ref = (ref or "").strip()
    if ID_RE.match(ref):
        kind = ref.split("_")[0]
        if kind == "att":
            att = ctx.attachments.get(ref) or assets.load_attachment(job, ref)
            if att is None or att.kind != "image":
                raise ToolError(f"'{ref}' is not an attached picture.")
            return att.path, {"origin": "attachment", "title": att.name}
        if kind == "gen":
            f = assets.assets_dir(job) / f"{ref}.png"
            if not f.is_file():
                raise ToolError(f"There is no generated picture '{ref}'.")
            return f, {"origin": "generated", "title": "AI-generated picture"}
        if kind == "res":
            info = _registry(job).get(ref)
            if not info:
                raise ToolError(f"There is no search result '{ref}'. Search again first.")
            f = assets.assets_dir(job) / f"{ref}.img"
            if not f.is_file():
                f.write_bytes(_fetch(info["url"]))
            return f, {"origin": "web", "title": info["title"], "credit": _credit(info["domain"]),
                       "url": info["page"] or info["url"]}
        raise ToolError(f"Unknown picture id '{ref}'.")
    if ref.lower().startswith(("http://", "https://")):
        rid = _next_id(job, "url")
        f = assets.assets_dir(job) / f"{rid}.img"
        f.write_bytes(_fetch(ref))
        return f, {"origin": "web", "title": _domain(ref), "credit": _credit(_domain(ref)), "url": ref}
    raise ToolError("Give me a picture id (att_1, res_2, gen_3) or a link that starts with http.")


def _fetch(url: str) -> bytes:
    try:
        return fetch.fetch_image(url)
    except fetch.FetchError as exc:
        raise ToolError(str(exc)) from exc


def _credit(domain: str) -> str | None:
    return f"Image: {domain}" if domain else None


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
