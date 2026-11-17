"""Framing (zoom / pan keyframes): pure maths plus real FFmpeg on a synthetic striped clip."""
import shutil
import subprocess
from pathlib import Path

import pytest

from src.rendering.framing import (
    MAX_ZOOM, clip_graph, crop_rect, default_framing, framing_filter, is_default, is_static, keyframe_at, normalize,
    uses_fit,
)

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


# ----------------------------------------------------------------------------- pure maths
def test_default_framing_is_the_old_centre_crop():
    left, top, w, h = crop_rect(1920, 1080, 1080, 1920, 1.0, 0.5, 0.5)
    assert w == pytest.approx(607.5) and h == pytest.approx(1080)  # full height, 9:16 wide
    assert left == pytest.approx((1920 - 607.5) / 2) and top == pytest.approx(0)
    assert is_static(default_framing()) and is_static(None)


def test_zoom_shrinks_the_window_and_pan_moves_it():
    _, _, w1, h1 = crop_rect(1920, 1080, 1080, 1920, 1.0, 0.5, 0.5)
    left, top, w2, h2 = crop_rect(1920, 1080, 1080, 1920, 2.0, 1.0, 0.0)
    assert w2 == pytest.approx(w1 / 2) and h2 == pytest.approx(h1 / 2)
    assert left + w2 == pytest.approx(1920) and top == pytest.approx(0)  # pan 1,0 = right/top edge


def test_split_box_has_a_different_aspect():
    _, _, w, h = crop_rect(1920, 1080, 1080, 960, 1.0, 0.5, 0.5)
    assert w / h == pytest.approx(1080 / 960)


def test_normalize_clamps_and_completes():
    f = normalize({"start": {"z": 9, "x": -1}, "end": {"z": 0.2, "y": 7}})
    assert f["start"] == {"z": MAX_ZOOM, "x": 0.0, "y": 0.5}
    assert f["end"] == {"z": 1.0, "x": 0.5, "y": 1.0}
    assert normalize({"start": {"z": 2}})["end"]["z"] == 2  # missing end = hold the start


def test_keyframe_interpolation():
    f = {"start": {"z": 1, "x": 0, "y": 0.5}, "end": {"z": 3, "x": 1, "y": 0.5}}
    mid = keyframe_at(f, 0.5)
    assert (mid["z"], mid["x"], mid["y"]) == pytest.approx((2.0, 0.5, 0.5))
    assert keyframe_at(f, -4)["z"] == 1 and keyframe_at(f, 9)["z"] == 3


# ----------------------------------------------------------------------------- real ffmpeg
def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def pixel(video: Path, n: int, x: int, y: int) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-vf",
         f"select=eq(n\\,{n}),format=rgb24,crop=1:1:{x}:{y}", "-frames:v", "1", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return out[0], out[1], out[2]


def near(rgb, target, tol=60):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


RED, GREEN, BLUE = (255, 0, 0), (0, 128, 0), (0, 0, 255)  # ffmpeg "green" is #008000
DUR, FPS = 3.0, 30


@pytest.fixture(scope="module")
def stripes(tmp_path_factory):
    """1920x1080 clip: red | green | blue vertical thirds, so the pan position is visible in the colour."""
    clip = tmp_path_factory.mktemp("framing") / "stripes.mp4"
    ff("-f", "lavfi", "-i", f"color=c=green:s=1920x1080:r=30:d={DUR}", "-vf",
       "drawbox=x=0:y=0:w=640:h=1080:color=red:t=fill,drawbox=x=1280:y=0:w=640:h=1080:color=blue:t=fill",
       "-pix_fmt", "yuv420p", str(clip))
    return clip


def render_framed(clip: Path, framing, out: Path, tw=1080, th=1920) -> Path:
    ff("-i", str(clip), "-vf", framing_filter(framing, DUR, tw, th, 1920, 1080) + ",format=yuv420p",
       "-c:v", "libx264", "-crf", "12", str(out))
    return out


@needs_ffmpeg
def test_static_pan_picks_the_stripe(stripes, tmp_path):
    for x, colour in ((0.0, RED), (0.5, GREEN), (1.0, BLUE)):
        fr = {"start": {"z": 1, "x": x, "y": 0.5}, "end": {"z": 1, "x": x, "y": 0.5}}
        out = render_framed(stripes, fr, tmp_path / f"s{x}.mp4")
        assert near(pixel(out, 10, 540, 960), colour), f"pan x={x}"


@needs_ffmpeg
def test_animated_pan_moves_from_red_to_blue(stripes, tmp_path):
    fr = {"start": {"z": 1, "x": 0, "y": 0.5}, "end": {"z": 1, "x": 1, "y": 0.5}}
    out = render_framed(stripes, fr, tmp_path / "anim.mp4")
    assert near(pixel(out, 0, 540, 960), RED)  # first frame = start keyframe
    assert near(pixel(out, int(DUR * FPS / 2), 540, 960), GREEN)  # halfway: passing the middle
    assert near(pixel(out, int(DUR * FPS) - 1, 540, 960), BLUE)  # last frame = end keyframe


@needs_ffmpeg
def test_zoom_in_narrows_the_view(stripes, tmp_path):
    # Pan fully left at z=1: the 9:16 window is 607 px wide, entirely red in the left stripe. Zooming in while
    # staying panned left must keep red; zooming out from far right shows blue at the start, then wider.
    fr = {"start": {"z": 1, "x": 0, "y": 0.5}, "end": {"z": 3, "x": 0, "y": 0.5}}
    out = render_framed(stripes, fr, tmp_path / "zin.mp4")
    assert near(pixel(out, 0, 540, 960), RED) and near(pixel(out, int(DUR * FPS) - 1, 540, 960), RED)
    # a pixel at the far right of the frame sees the window edge; at z=1 it is still inside the red stripe
    assert near(pixel(out, 0, 1070, 960), RED)


@needs_ffmpeg
def test_output_size_matches_the_target_box(stripes, tmp_path):
    out = render_framed(stripes, {"start": {"z": 1.5, "x": 0.3, "y": 0.7}, "end": {"z": 2, "x": 0.6, "y": 0.2}},
                        tmp_path / "split.mp4", tw=1080, th=960)
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert size.strip() == "1080,960"


# ----------------------------------------------------------------------------- fit mode (the whole clip, blurred backdrop)
def test_base_is_fit_by_default_and_old_zoomed_framings_keep_filling():
    assert normalize(None)["base"] == "fit" and default_framing()["base"] == "fit"
    assert normalize({"start": {"z": 1}, "end": {"z": 1}})["base"] == "fit"  # an old untouched framing
    assert normalize({"start": {"z": 1.6}, "end": {"z": 1.6}})["base"] == "fill"  # the user zoomed it back then
    assert normalize({"base": "fill", "start": {"z": 1}})["base"] == "fill"
    assert normalize({"base": "nonsense"})["base"] == "fit"
    assert is_default({"base": "fill"}) and is_default(None) and not is_default({"start": {"z": 2}})


def test_a_clip_with_the_boxs_own_shape_needs_no_backdrop():
    assert uses_fit(None) and uses_fit(None, 1920, 1080, 1080, 1920)
    assert not uses_fit(None, 1080, 1920, 1080, 1920)  # vertical clip in the 9:16 frame: fit == fill
    assert not uses_fit({"base": "fill"}, 1920, 1080, 1080, 1920)


def render_graph(clip: Path, framing, out: Path, tw=1080, th=1920) -> Path:
    graph = clip_graph(framing, DUR, tw, th, 1920, 1080, FPS, "0:v", "s", 1)
    ff("-i", str(clip), "-filter_complex", graph + ";[s]format=yuv420p[v]", "-map", "[v]", "-c:v", "libx264",
       "-crf", "12", str(out))
    return out


@needs_ffmpeg
def test_fit_shows_the_whole_clip_with_a_blurred_backdrop(stripes, tmp_path):
    out = render_graph(stripes, None, tmp_path / "fit.mp4")
    # the clip is 1080 wide x 607 tall in the middle: both outer stripes are visible, nothing is cut
    assert near(pixel(out, 10, 10, 960), RED) and near(pixel(out, 10, 540, 960), GREEN) and near(pixel(out, 10, 1070, 960), BLUE)
    assert not near(pixel(out, 10, 10, 960 - 400), RED, 40)  # above the clip: the soft backdrop, not the sharp stripe
    old = render_framed(stripes, default_framing(), tmp_path / "fill.mp4")  # the fill path crops the stripes away
    assert near(pixel(old, 10, 10, 960), GREEN)
    size = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                           "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert size.strip() == "1080,1920"


@needs_ffmpeg
def test_fit_zoomed_past_the_box_pans_and_an_animated_zoom_renders(stripes, tmp_path):
    for x, colour in ((0.0, RED), (1.0, BLUE)):  # z=3: the clip is 3240 wide in a 1080 box
        fr = {"base": "fit", "start": {"z": 3, "x": x, "y": 0.5}, "end": {"z": 3, "x": x, "y": 0.5}}
        assert near(pixel(render_graph(stripes, fr, tmp_path / f"p{x}.mp4"), 10, 540, 960), colour), f"pan {x}"
    anim = render_graph(stripes, {"base": "fit", "start": {"z": 1, "x": .5, "y": .5}, "end": {"z": 1.5, "x": .5, "y": .5}},
                        tmp_path / "anim.mp4", tw=1080, th=960)  # the split box
    assert near(pixel(anim, 0, 10, 480), RED)  # first frame: the whole clip, red stripe at the edge
    assert near(pixel(anim, int(DUR * FPS) - 1, 540, 480), GREEN)
