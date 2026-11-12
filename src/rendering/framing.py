"""Per-B-roll framing: how a source clip is zoomed and panned inside its target box (the 9:16 frame, or the
top half of a split layout).

This module is the single source of truth for the crop. The renderer (`compose.py`) and the editor preview
(`preview.py`) both call it, so what the user sees while adjusting zoom/pan is what ends up in the video.

Model: two keyframes, `start` and `end`, each `{z, x, y}`:
  z = zoom, 1.0 (fill the box, like the old centre crop) to MAX_ZOOM;
  x, y = pan, 0..1 (0.5 = centre), the position of the crop window inside the zoomed clip.
The values are interpolated linearly over the slot, so `start` is the first frame and `end` the last.
"""
from __future__ import annotations

from copy import deepcopy

MIN_ZOOM, MAX_ZOOM = 1.0, 3.0
DEFAULT_KEYFRAME = {"z": 1.0, "x": 0.5, "y": 0.5}
STATIC_EPS = 1e-6


def default_framing() -> dict:
    return {"start": dict(DEFAULT_KEYFRAME), "end": dict(DEFAULT_KEYFRAME)}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _keyframe(raw: dict | None) -> dict:
    raw = raw or {}
    return {
        "z": round(_clamp(raw.get("z", 1.0), MIN_ZOOM, MAX_ZOOM), 4),
        "x": round(_clamp(raw.get("x", 0.5), 0.0, 1.0), 4),
        "y": round(_clamp(raw.get("y", 0.5), 0.0, 1.0), 4),
    }


def normalize(framing: dict | None) -> dict:
    """Clamp and complete a framing dict. Missing/None means the default centre fill (old behaviour)."""
    framing = framing or {}
    start = _keyframe(framing.get("start"))
    end = _keyframe(framing.get("end", framing.get("start")))
    return {"start": start, "end": end}


def is_static(framing: dict | None) -> bool:
    f = normalize(framing)
    return all(abs(f["start"][k] - f["end"][k]) < STATIC_EPS for k in ("z", "x", "y"))


def keyframe_at(framing: dict | None, progress: float) -> dict:
    """Interpolated {z, x, y} at progress 0..1 through the slot."""
    f = normalize(framing)
    p = _clamp(progress, 0.0, 1.0)
    return {k: f["start"][k] + (f["end"][k] - f["start"][k]) * p for k in ("z", "x", "y")}


def cover_scale(src_w: float, src_h: float, target_w: float, target_h: float) -> float:
    """Scale at which the source just covers the target box (zoom 1.0)."""
    return max(target_w / src_w, target_h / src_h)


def crop_rect(src_w: float, src_h: float, target_w: float, target_h: float,
              z: float, x: float, y: float) -> tuple[float, float, float, float]:
    """The part of the SOURCE (left, top, width, height in source pixels) that ends up in the target box.

    The preview's crop rectangle on the full source frame comes from here. At z=1, x=y=0.5 it is exactly the
    old centre crop."""
    z = _clamp(z, MIN_ZOOM, MAX_ZOOM)
    f = cover_scale(src_w, src_h, target_w, target_h) * z
    scaled_w, scaled_h = src_w * f, src_h * f
    left = (scaled_w - target_w) * _clamp(x, 0, 1) / f
    top = (scaled_h - target_h) * _clamp(y, 0, 1) / f
    return left, top, target_w / f, target_h / f


def _lerp(a: float, b: float, duration: float) -> str:
    """ffmpeg expression for a value moving linearly from a to b over `duration` seconds (t = clip time)."""
    if abs(a - b) < STATIC_EPS:
        return f"{a:.5f}"
    return f"({a:.5f}+({b - a:.5f})*min(t/{max(duration, 0.05):.4f},1))"


def framing_filter(framing: dict | None, duration: float, target_w: int, target_h: int,
                   src_w: int, src_h: int, fps: int = 30) -> str:
    """ffmpeg filter chain: clip -> exactly target_w x target_h, zoomed/panned per `framing`.

    Frames are scaled per frame (`scale eval=frame`) then cropped; the pan offsets are computed from the known
    source size because `crop` only reads its input size once. Rounding the scaled size up to even numbers can
    shift the crop by at most one pixel."""
    f = normalize(framing)
    s, e = f["start"], f["end"]
    cover = f"max({target_w},{target_h}*{src_w}/{src_h})"
    zoom = _lerp(s["z"], e["z"], duration)
    px, py = _lerp(s["x"], e["x"], duration), _lerp(s["y"], e["y"], duration)
    if is_static(f):
        scale = (f"scale=w='ceil({cover}*{zoom}/2)*2':h='ceil(ow*ih/iw/2)*2':flags=bicubic")
    else:
        scale = (f"scale=w='ceil({cover}*{zoom}/2)*2':h='ceil(ow*ih/iw/2)*2':eval=frame:flags=bicubic")
    # Scaled size of the current frame, from the known source size (see docstring).
    sw = f"({cover}*{zoom})"
    sh = f"({cover}*{zoom}*{src_h}/{src_w})"
    crop = (f"crop={target_w}:{target_h}:x='max(0,({sw}-{target_w})*{px})':"
            f"y='max(0,({sh}-{target_h})*{py})'")
    return f"setpts=PTS-STARTPTS,fps={fps},{scale},{crop},setsar=1"


def fill_crop(w: int, h: int) -> str:
    """Scale up (aspect preserved, no stretching) to fill w x h, then crop the centre."""
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1"


def is_default(framing: dict | None) -> bool:
    """True for the untouched framing (z=1, centred): it keeps the original fill-crop filter exactly."""
    return normalize(framing) == default_framing()


def clip_filter(framing: dict | None, duration: float, w: int, h: int,
                src_w: int = 0, src_h: int = 0, fps: int = 30) -> str:
    """Filter chain turning a clip into exactly w x h (timestamps reset to 0). This is what both the final
    render and the editor preview use. Source size is only needed once framing has been changed."""
    if is_default(framing):
        return f"setpts=PTS-STARTPTS,fps={fps},{fill_crop(w, h)}"
    if not (src_w and src_h):
        raise ValueError("The clip's width and height are needed to apply zoom/pan.")
    return framing_filter(framing, duration, w, h, src_w, src_h, fps)


def with_framing(item: dict) -> dict:
    """The item's framing (normalised), never mutating the item."""
    return deepcopy(normalize(item.get("framing")))
