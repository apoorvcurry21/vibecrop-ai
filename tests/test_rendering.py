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


@needs_ffmpeg
def test_removed_slot_keeps_speaker(media, tmp_path):
    main, clip = media
    job = tmp_path / "job"
    (job / "clips").mkdir(parents=True)
    shutil.copy(clip, job / "clips" / "clip_01.mp4")
    (job / "plan.json").write_text(json.dumps({"items": [{
        "id": 1, "asset_type": "video", "timestamp_start": 2.0, "timestamp_end": 5.0, "enabled": False,
        "asset": {"status": "ok", "path": "clips/clip_01.mp4", "channel": "C"},
    }]}), encoding="utf-8")
    out = pipeline.run(job, main, log=lambda *_: None)
    assert near(pixel(out, 3.5, 540, 960), (255, 0, 0))


@needs_ffmpeg
@pytest.mark.parametrize("script_word", ["hello", "नमस्ते"])
def test_captions_are_burned_in_with_yellow_highlight(media, tmp_path, script_word):
    main, _ = media
    job = tmp_path / "job"
    job.mkdir()
    (job / "plan.json").write_text(json.dumps({"video_topic": "t", "items": []}), encoding="utf-8")
    (job / "transcript.json").write_text(json.dumps({"language": "hi", "words": [
        {"word": script_word, "start": 1.0, "end": 2.5}, {"word": "world", "start": 2.5, "end": 4.0}]}),
        encoding="utf-8")
    out = pipeline.run(job, main, caption_mode="original", log=lambda *_: None)

    def band_counts(t):
        raw = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
             "-vf", "format=rgb24,crop=1080:260:0:1330", "-f", "rawvideo", "-"],
            capture_output=True, check=True).stdout
        px = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        yellow = sum(p[0] > 200 and p[1] > 200 and p[2] < 90 for p in px)
        white = sum(min(p) > 235 for p in px)
        return yellow, white

    yellow, white = band_counts(1.5)  # first word active: yellow word + white second word
    assert yellow > 300 and white > 300
    assert band_counts(0.5) == (0, 0)  # no caption before the first word


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


# ----------------------------------------------------------------------------- layouts (overlay / split / hybrid)
def split_main(tmp_path) -> Path:
    """Speaker video: yellow above y=1000, red below, so the facecam crop position is visible."""
    main = tmp_path / "main_split.mp4"
    ff("-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=30:d=8", "-f", "lavfi", "-i", "sine=frequency=440:d=8",
       "-vf", "drawbox=x=0:y=0:w=1080:h=1000:color=yellow:t=fill", "-shortest", "-pix_fmt", "yuv420p", str(main))
    return main


@needs_ffmpeg
def test_split_layout_has_broll_on_top_facecam_below_and_a_border(media, tmp_path):
    _, clip = media
    main = split_main(tmp_path)
    out = render(main, [Overlay(clip, 2.0, 5.0, layout="split")], tmp_path / "split.mp4", face_y=0.4)
    assert near(pixel(out, 1.0, 540, 400), (255, 255, 0))  # before the slot: plain speaker (yellow top)
    assert near(pixel(out, 3.5, 540, 400), (0, 0, 255))  # top half: the clip's blue centre stripe
    assert near(pixel(out, 3.5, 5, 400), (0, 128, 0))  # ...centre-cropped, so the far left is its green
    assert near(pixel(out, 3.5, 540, 960), (255, 255, 255), tol=30)  # the border line
    # facecam = speaker rows 384..1344: yellow until 1000 (output y 960 + 616 = 1576), red after
    assert near(pixel(out, 3.5, 540, 1200), (255, 255, 0))
    assert near(pixel(out, 3.5, 540, 1800), (255, 0, 0))
    assert near(pixel(out, 6.5, 540, 1800), (255, 0, 0)) and near(pixel(out, 6.5, 540, 400), (255, 255, 0))
    assert len(streams(out, "a")) == 1


@needs_ffmpeg
def test_hybrid_mixes_overlay_and_split_slots_in_one_render(media, tmp_path):
    main, clip = media
    out = render(main, [Overlay(clip, 1.0, 3.0), Overlay(clip, 4.0, 6.0, layout="split")], tmp_path / "hyb.mp4")
    assert near(pixel(out, 2.0, 540, 1700), (0, 0, 255))  # overlay slot: the clip covers the whole frame
    assert near(pixel(out, 5.0, 540, 400), (0, 0, 255)) and near(pixel(out, 5.0, 540, 1700), (255, 0, 0))  # split
    assert near(pixel(out, 3.5, 540, 1700), (255, 0, 0))  # between slots: speaker only


@needs_ffmpeg
def test_zoomed_overlay_uses_the_framing(media, tmp_path):
    _, clip = media
    pan_left = {"start": {"z": 1, "x": 0, "y": 0.5}, "end": {"z": 1, "x": 0, "y": 0.5}}
    out = render(media[0], [Overlay(clip, 2.0, 5.0, framing=pan_left)], tmp_path / "pan.mp4")
    assert near(pixel(out, 3.5, 540, 960), (0, 128, 0))  # panned to the left edge: green, not the blue centre
    assert near(pixel(out, 3.5, 5, 5), (0, 128, 0))


def test_filtergraph_split_adds_facecam_border_and_gates_everything_to_the_slot():
    g = build_filtergraph([Overlay(Path("c.mp4"), 2.0, 5.0, layout="split")], face_y=0.5)
    assert "split=2[b0][m1]" in g and "crop=1080:960:0:480" in g  # facecam half, 50% down the speaker
    assert "overlay=0:0:enable='between(t,2.000,5.000)'" in g and "overlay=0:960:enable=" in g
    assert "drawbox=x=0:y=956:w=1080:h=8" in g and "between(t,2.000,5.000)" in g
    plain = build_filtergraph([Overlay(Path("c.mp4"), 2.0, 5.0)])
    assert "split=" not in plain and "drawbox" not in plain  # overlay layout is unchanged


def test_captions_and_credits_move_for_split_slots():
    from src.rendering.ass import caption_events, credit_event, header
    chunks = [{"start": 3.0, "end": 4.0, "words": [{"text": "hello", "start": 3.0, "end": 3.5}]},
              {"start": 8.0, "end": 9.0, "words": [{"text": "bye", "start": 8.0, "end": 8.5}]}]
    ev = caption_events(chunks, split_spans=[(2.0, 5.0)])
    assert "CaptionLatinSplit" in ev[0] and "CaptionLatin," in ev[1]  # only the word inside the split slot moves
    assert ",CreditSplit," in credit_event(2.0, 5.0, "Chan", split=True) and ",Credit," in credit_event(2.0, 5.0, "Chan")
    assert "Style: CaptionLatinSplit" in header() and ",5,60,60,0,1" in header()  # alignment 5 = centred


@needs_ffmpeg
def test_pipeline_passes_layout_and_framing_and_facecam_focus(media, tmp_path):
    main = split_main(tmp_path)
    job = tmp_path / "job"
    (job / "clips").mkdir(parents=True)
    (job / "clips" / "clip_01.mp4").write_bytes(media[1].read_bytes())
    (job / "job.json").write_text(json.dumps({"split_face_y": 0.0}))
    (job / "plan.json").write_text(json.dumps({"items": [{
        "id": 1, "asset_type": "video", "timestamp_start": 2.0, "timestamp_end": 5.0, "layout": "split",
        "asset": {"status": "ok", "path": "clips/clip_01.mp4", "channel": "Chan"}}]}))
    out = pipeline.run(job, main, credits=False, caption_mode=None, log=lambda m: None)
    assert near(pixel(out, 3.5, 540, 400), (0, 0, 255))  # top: b-roll
    assert near(pixel(out, 3.5, 540, 1500), (255, 255, 0))  # face_y=0 -> facecam starts at speaker row 0: yellow


def test_a_caption_straddling_a_slot_boundary_switches_position_at_the_boundary():
    from src.rendering.ass import caption_events
    chunks = [{"start": 4.0, "end": 5.5, "words": [{"text": "of", "start": 4.0, "end": 4.3},
                                                    {"text": "video", "start": 4.35, "end": 4.9},
                                                    {"text": "now", "start": 5.0, "end": 5.4}]}]
    ev = caption_events(chunks, split_spans=[(4.5, 8.0)])
    # the "video" word runs 4.35 -> 5.0 and crosses 4.5: normal style first, then the split style
    styles = [e.split(",")[3] for e in ev]
    assert styles[:3] == ["CaptionLatin", "CaptionLatin", "CaptionLatinSplit"] and ev[1].split(",")[2] == "0:00:04.50"
    assert ev[2].split(",")[1] == "0:00:04.50"
