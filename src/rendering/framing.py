"""Per-B-roll framing: how a source clip is zoomed and panned inside its target box (the 9:16 frame, or the
top half of a split layout).

This module is the single source of truth for the crop. The renderer (`compose.py`) and the editor preview
(`preview.py`) both call it, so what the user sees while adjusting zoom/pan is what ends up in the video.

Model: a `base` and two keyframes, `start` and `end`, each `{z, x, y}`:
  base "fit" (the default): z = 1.0 shows the WHOLE clip inside the box, nothing is cut; the empty space is filled
    with a blurred copy of the clip. Zooming in (z > 1) goes from there.
  base "fill": z = 1.0 fills the box and crops the clip's edges (the old behaviour; "Fill screen" in the editor).
  z = zoom, 1.0 to MAX_ZOOM;
  x, y = pan, 0..1 (0.5 = centre), the position of the window inside a clip that is bigger than the box.
The values are interpolated linearly over the slot, so `start` is the first frame and `end` the last.
Picture-in-picture boxes always use the fill behaviour.
"""
from __future__ import annotations

from copy import deepcopy

MIN_ZOOM, MAX_ZOOM = 1.0, 4.0
BASES = ("fit", "fill")
BLUR_DIV = 8  # the blurred backdrop is made at 1/8 size and scaled up: cheap, and it is blurry anyway
SAME_SHAPE = 0.02  # a clip within 2% of the box's shape has no bars to fill: fit and fill are the same
DEFAULT_KEYFRAME = {"z": 1.0, "x": 0.5, "y": 0.5}
STATIC_EPS = 1e-6


def default_framing() -> dict:
    return {"base": "fit", "start": dict(DEFAULT_KEYFRAME), "end": dict(DEFAULT_KEYFRAME)}


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
    """Clamp and complete a framing dict. Missing/None means the default: the whole clip, centred.

    A saved framing without a `base` is from before the fit mode: untouched ones (z=1, centred) become "fit" like every
    new slot, ones the user zoomed/panned keep the fill behaviour they were set up with."""
    framing = framing or {}
    start = _keyframe(framing.get("start"))
    end = _keyframe(framing.get("end", framing.get("start")))
    base = framing.get("base")
    if base not in BASES:
        untouched = start == end == _keyframe(DEFAULT_KEYFRAME)
        base = "fit" if untouched else "fill"
    return {"base": base, "start": start, "end": end}


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
    """True when the keyframes are untouched (z=1, centred), whatever the base. For the fill behaviour (and always
    for pip) that keeps the original fill-crop filter exactly."""
    f = normalize(framing)
    return f["start"] == f["end"] == _keyframe(DEFAULT_KEYFRAME)


def clip_filter(framing: dict | None, duration: float, w: int, h: int,
                src_w: int = 0, src_h: int = 0, fps: int = 30) -> str:
    """Filter chain turning a clip into exactly w x h (timestamps reset to 0). This is what both the final
    render and the editor preview use. Source size is only needed once framing has been changed."""
    if is_default(framing):
        return f"setpts=PTS-STARTPTS,fps={fps},{fill_crop(w, h)}"
    if not (src_w and src_h):
        raise ValueError("The clip's width and height are needed to apply zoom/pan.")
    return framing_filter(framing, duration, w, h, src_w, src_h, fps)


def uses_fit(framing: dict | None, src_w: int = 0, src_h: int = 0, w: int = 0, h: int = 0) -> bool:
    """True when the clip is shown whole with a blurred backdrop. A clip that already has the box's shape (a vertical
    clip in the 9:16 frame) has no bars to fill, so it takes the plain fill path (identical picture, cheaper)."""
    if normalize(framing)["base"] != "fit":
        return False
    return not (src_w and src_h and w and h and abs((src_w / src_h) / (w / h) - 1) < SAME_SHAPE)


def clip_graph(framing: dict | None, duration: float, w: int, h: int, src_w: int, src_h: int, fps: int,
               src: str, out: str, tag: str | int) -> str:
    """Filter-graph piece `[src]...[out]` that turns a clip into exactly w x h. This is what both the final render and
    the editor preview use. `tag` keeps the inner labels unique when several slots are in one graph.

    Fit: the whole clip is scaled to fit inside the box (times the zoom), centred (or panned, when zoomed past the box)
    on a blurred copy of itself. It needs no source size: the filters read the real one."""
    f = normalize(framing)
    if not uses_fit(f, src_w, src_h, w, h):
        return f"[{src}]{clip_filter(f, duration, w, h, src_w, src_h, fps)}[{out}]"
    s, e = f["start"], f["end"]
    zoom = _lerp(s["z"], e["z"], duration)
    px, py = _lerp(s["x"], e["x"], duration), _lerp(s["y"], e["y"], duration)
    fit = f"min({w}/iw,{h}/ih)"
    frame_eval = "" if is_static(f) else ":eval=frame"
    bw, bh = max(2, w // BLUR_DIV // 2 * 2), max(2, h // BLUR_DIV // 2 * 2)
    return (
        f"[{src}]setpts=PTS-STARTPTS,fps={fps},split[a{tag}][b{tag}];"
        f"[a{tag}]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=6:2,"
        f"scale={w}:{h}:flags=bilinear,setsar=1[g{tag}];"
        f"[b{tag}]scale=w='trunc({fit}*iw*{zoom}/2)*2':h='trunc({fit}*ih*{zoom}/2)*2'{frame_eval}:flags=bicubic[f{tag}];"
        f"[g{tag}][f{tag}]overlay=x='if(gt(w,W),(W-w)*{px},(W-w)/2)':y='if(gt(h,H),(H-h)*{py},(H-h)/2)',setsar=1[{out}]"
    )


def with_framing(item: dict) -> dict:
    """The item's framing (normalised), never mutating the item."""
    return deepcopy(normalize(item.get("framing")))
