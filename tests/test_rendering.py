"""Phase 3b tests: real FFmpeg on synthetic media (no network)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.rendering import pipeline
from src.rendering.ass import credit_event, escape
from src.rendering.compose import Overlay, build_filtergraph, render

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def pixel(video: Path, t: float, x: int, y: int) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-vf", f"format=rgb24,crop=1:1:{x}:{y}", "-f", "rawvideo", "-"],  # RGB first: yuv420p can't crop 1x1
        capture_output=True, check=True,
    ).stdout
    return out[0], out[1], out[2]


def near(rgb, target, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


@pytest.fixture
def media(tmp_path):
    main = tmp_path / "main.mp4"  # vertical red speaker video with a 440 Hz tone, 8 s
    ff("-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=30:d=8", "-f", "lavfi", "-i", "sine=frequency=440:d=8",
       "-shortest", "-pix_fmt", "yuv420p", str(main))
    clip = tmp_path / "clip.mp4"  # 16:9 green clip with a blue centre stripe AND an audio track
    ff("-f", "lavfi", "-i", "color=c=green:s=1920x1080:r=25:d=5", "-f", "lavfi", "-i", "sine=frequency=880:d=5",
       "-vf", "drawbox=x=640:y=0:w=640:h=1080:color=blue:t=fill", "-shortest", "-pix_fmt", "yuv420p", str(clip))
    return main, clip


def streams(path: Path, kind: str) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=width,height,codec_name",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


@needs_ffmpeg
def test_overlay_is_centre_cropped_timed_and_muted(media, tmp_path):
    main, clip = media
    out = render(main, [Overlay(clip, 2.0, 5.0)], tmp_path / "out.mp4")
    v = streams(out, "v")[0]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert len(streams(out, "a")) == 1  # only the speaker's audio; the clip's tone is never mapped
    assert near(pixel(out, 1.0, 540, 960), (255, 0, 0))  # before the slot: speaker
    assert near(pixel(out, 3.5, 540, 960), (0, 0, 255))  # during: blue centre stripe of the clip
    assert near(pixel(out, 3.5, 5, 5), (0, 0, 255))  # corners are blue too = centre crop, no stretch
    assert near(pixel(out, 6.5, 540, 960), (255, 0, 0))  # after the slot: speaker again


@needs_ffmpeg
def test_credit_line_is_burned_in_during_slot_only(media, tmp_path):
    main, clip = media
    job = tmp_path / "job"
    (job / "clips").mkdir(parents=True)
    shutil.copy(clip, job / "clips" / "clip_01.mp4")
    (job / "plan.json").write_text(json.dumps({"items": [{
        "id": 1, "asset_type": "video", "timestamp_start": 2.0, "timestamp_end": 5.0,
        "asset": {"status": "ok", "path": "clips/clip_01.mp4", "channel": "Test Channel"},
    }]}), encoding="utf-8")
    out = pipeline.run(job, main, log=lambda *_: None)

    def white_pixels(t):
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
             "-vf", "crop=1080:140:0:1650,format=gray", "-f", "rawvideo", "-"],
            capture_output=True, check=True).stdout
        return sum(b > 235 for b in raw)

    assert white_pixels(3.5) > 100  # white credit text over the blue clip
    assert white_pixels(1.0) == 0 and white_pixels(6.5) == 0


@needs_ffmpeg
def test_missing_clip_leaves_speaker_visible(media, tmp_path):
    main, _ = media
    job = tmp_path / "job"
    job.mkdir()
    (job / "plan.json").write_text(json.dumps({"items": [{
        "id": 1, "asset_type": "video", "timestamp_start": 2.0, "timestamp_end": 5.0,
        "asset": {"status": "failed", "error": "x"},
    }]}), encoding="utf-8")
    out = pipeline.run(job, main, log=lambda *_: None)
    assert near(pixel(out, 3.5, 540, 960), (255, 0, 0))


def test_filtergraph_overlays_in_order_and_without_audio():
    g = build_filtergraph([Overlay(Path("a.mp4"), 1.0, 4.0), Overlay(Path("b.mp4"), 6.0, 9.0)], "credits.ass")
    assert "[b0][c1]overlay=enable='between(t,1.000,4.000)'" in g
    assert "[b1][c2]overlay=enable='between(t,6.000,9.000)'" in g
    assert g.endswith("[b2]subtitles=credits.ass[v]")
    assert "crop=1080:1920" in g and "force_original_aspect_ratio=increase" in g


def test_ass_credit_escaping_and_timing():
    assert escape("A{b}\\c\nd") == "A(b)/c d"
    line = credit_event(61.5, 63.25, "FC {Barcelona}")
    assert "0:01:01.50,0:01:03.25,Credit" in line and "Video: FC (Barcelona)" in line
