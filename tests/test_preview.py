"""Editor previews use the same crop as the render: compare a still against the real final frame."""
import io
import shutil
import subprocess
from pathlib import Path

import pytest

from src.rendering.compose import Overlay, RenderError, render
from src.rendering.preview import frame_png, render_clip_preview, source_view_png

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
RED, GREEN, BLUE = (255, 0, 0), (0, 128, 0), (0, 0, 255)


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("preview")
    clip = d / "stripes.mp4"  # 1920x1080: red | green | blue thirds
    ff("-f", "lavfi", "-i", "color=c=green:s=1920x1080:r=30:d=4", "-vf",
       "drawbox=x=0:y=0:w=640:h=1080:color=red:t=fill,drawbox=x=1280:y=0:w=640:h=1080:color=blue:t=fill",
       "-pix_fmt", "yuv420p", str(clip))
    main = d / "main.mp4"
    ff("-f", "lavfi", "-i", "color=c=yellow:s=1080x1920:r=30:d=6", "-pix_fmt", "yuv420p", str(main))
    return d, clip, main


def px(png: bytes, x: int, y: int):
    from PIL import Image
    return Image.open(io.BytesIO(png)).convert("RGB").getpixel((x, y))


def near(rgb, target, tol=60):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


@needs_ffmpeg
def test_still_frames_follow_the_start_and_end_keyframes(media):
    _, clip, _ = media
    fr = {"start": {"z": 1, "x": 0, "y": 0.5}, "end": {"z": 1, "x": 1, "y": 0.5}}
    start = frame_png(clip, fr, "overlay", 3.0, "start")
    end = frame_png(clip, fr, "overlay", 3.0, "end")
    assert near(px(start, 180, 320), RED) and near(px(end, 180, 320), BLUE)
    # 360 wide, 9:16 -> 640 high for the full frame; the split box is 9:8 -> 320 high
    assert Image_size(start) == (360, 640) and Image_size(frame_png(clip, None, "split", 3.0, "start")) == (360, 320)


def Image_size(png: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(png)).size


@needs_ffmpeg
def test_preview_frame_matches_the_real_render(media, tmp_path):
    _, clip, main = media
    fr = {"start": {"z": 2, "x": 1, "y": 0.5}, "end": {"z": 2, "x": 1, "y": 0.5}}  # zoomed, panned right = blue
    out = render(main, [Overlay(clip, 1.0, 4.0, framing=fr)], tmp_path / "final.mp4")
    got = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "2.5", "-i", str(out), "-frames:v", "1",
                          "-vf", "format=rgb24,crop=1:1:540:960", "-f", "rawvideo", "-"], capture_output=True).stdout
    assert near(tuple(got[:3]), BLUE) and near(px(frame_png(clip, fr, "overlay", 3.0, "start"), 180, 320), BLUE)


@needs_ffmpeg
def test_source_view_outlines_the_crop_and_dims_the_rest(media):
    _, clip, _ = media
    png = source_view_png(clip, {"start": {"z": 1, "x": 0, "y": 0.5}, "end": {"z": 1, "x": 1, "y": 0.5}}, "overlay")
    assert Image_size(png) == (640, 360)
    # start box = leftmost 9:16 window (0..202 px of 640): red inside, and dimmed green outside it
    assert near(px(png, 100, 180), RED, tol=70) and px(png, 400, 180)[1] < 80  # centre green is darkened


@needs_ffmpeg
def test_playable_preview_is_a_real_render_of_just_this_slot(media, tmp_path):
    _, clip, main = media
    out = render_clip_preview(clip, main, 1.0, 3.5, "split", None, 0.4, tmp_path / "p" / "prev.mp4")
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout
    assert abs(float(dur) - 2.5) < 0.3
    assert not list((tmp_path / "p").glob("*_speaker.mp4"))  # the temporary speaker cut is removed
    # no speaker video (already cleaned up): still renders, with grey instead
    assert render_clip_preview(clip, None, 1.0, 3.0, "overlay", None, 0.4, tmp_path / "p" / "g.mp4").is_file()


def test_missing_clip_is_a_clear_error(tmp_path):
    with pytest.raises(RenderError, match="Could not read the size"):
        frame_png(tmp_path / "nope.mp4", None, "overlay", 3.0, "start")
