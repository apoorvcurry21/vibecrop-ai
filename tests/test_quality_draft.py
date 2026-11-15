"""Export keeps the speaker video's frame rate and quality; the full-reel draft preview. Real ffmpeg, no network."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import workflow
from src.cleanup import purge_job_temp
from src.rendering import compose, pipeline
from src.rendering.compose import Overlay, SourceProfile, probe_profile, render, snap_fps

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def info(path: Path, kind: str = "v:0") -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries",
                          "stream=width,height,avg_frame_rate,bit_rate,sample_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, _, den = s.get("avg_frame_rate", "0/1").partition("/")
    s["fps"] = float(num) / float(den) if float(den or 0) else 0.0  # an audio stream says 0/0
    return s


def speaker(path: Path, fps: int, seconds: float = 3, noisy: bool = False, size: str = "1080x1920") -> Path:
    src = f"testsrc2=s={size}:r={fps}:d={seconds}"
    vf = ["-vf", "noise=alls=40:allf=t"] if noisy else []
    ff("-f", "lavfi", "-i", src, "-f", "lavfi", "-i", f"sine=frequency=440:r=48000:d={seconds}", *vf, "-shortest",
       "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", str(path))
    return path


def clip(path: Path) -> Path:
    ff("-f", "lavfi", "-i", "color=c=green:s=1920x1080:r=25:d=3", "-pix_fmt", "yuv420p", str(path))
    return path


def make_job(tmp_path: Path, main: Path) -> Path:
    job = tmp_path / "jobs" / "20260101_000000_x"
    (job / "clips").mkdir(parents=True)
    clip(job / "clips" / "clip_01.mp4")
    (job / "plan.json").write_text(json.dumps({"video_topic": "t", "video_duration": 3.0, "items": [{
        "id": 1, "asset_type": "video", "timestamp_start": 0.5, "timestamp_end": 2.0,
        "asset": {"status": "ok", "path": "clips/clip_01.mp4", "channel": "C"}}]}), encoding="utf-8")
    (job / "job.json").write_text(json.dumps({"language": "en", "source_name": "a.mp4", "style": "overlay",
                                              "source_video": str(main)}), encoding="utf-8")
    return job


# ---- reading the source ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, want", [(59.76, 60), (59.94, 60), (60.0, 60), (29.97, 30), (23.976, 24), (25.0, 25),
                                         (120.0, 60), (10.0, 24), (0.0, 30), (None, 30)])
def test_snap_fps_rounds_to_a_whole_rate_between_24_and_60(value, want):
    assert snap_fps(value) == want


def test_bitrate_cap_follows_the_source_with_a_floor_and_audio_stays_good():
    assert SourceProfile(60, 28000, 256).maxrate_kbps == 28000  # never more than the source had
    assert SourceProfile(30, 3000, 128).maxrate_kbps == 8000  # a poor source is not squeezed further
    assert SourceProfile(30, 0, 0).maxrate_kbps == 16000  # unknown rate: a generous cap
    assert [SourceProfile(30, 0, k).out_audio_kbps for k in (0, 128, 256, 640)] == [192, 192, 256, 320]


@needs_ffmpeg
def test_probe_profile_reads_frame_rate_and_bitrates(tmp_path):
    p = probe_profile(speaker(tmp_path / "s.mp4", 50))
    assert p.fps == 50 and p.video_kbps > 0 and p.audio_kbps > 0
    assert probe_profile(tmp_path / "missing.mp4") == SourceProfile()  # unreadable: the old defaults


# ---- the export -----------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_export_keeps_60_fps_and_the_voice_sample_rate(tmp_path):
    main = speaker(tmp_path / "main60.mp4", 60)
    job = make_job(tmp_path, main)
    out = pipeline.run(job, main, log=lambda m: None)
    v = info(out)
    assert (v["width"], v["height"]) == (1080, 1920) and v["fps"] == pytest.approx(60, abs=0.5)
    assert info(out, "a:0")["sample_rate"] == "48000"
    assert not (job / "draft.mp4").exists()


@needs_ffmpeg
def test_export_never_goes_over_the_bitrate_cap(tmp_path):
    main = speaker(tmp_path / "noisy.mp4", 60, seconds=2, noisy=True)
    free = int(info(render(main, [], tmp_path / "free.mp4", fps=60, crf=17))["bit_rate"]) / 1000
    capped_out = render(main, [], tmp_path / "capped.mp4", fps=60, crf=17, maxrate_kbps=3000)
    capped = int(info(capped_out)["bit_rate"]) / 1000
    assert free > 6000  # the noisy picture at CRF 17 really wants far more than the cap...
    assert capped < free * 0.6  # ...and the cap holds it down
    # the rate buffer (2 x cap) can be spent once at the start, so a 2 s clip may exceed the cap by buffer / duration
    assert capped <= 3000 + 2 * 3000 / 2 + 300
    assert info(capped_out)["fps"] == pytest.approx(60, abs=0.5)


@needs_ffmpeg
def test_a_30_fps_source_still_exports_at_30(tmp_path):
    main = speaker(tmp_path / "main30.mp4", 30)
    out = pipeline.run(make_job(tmp_path, main), main, log=lambda m: None)
    assert info(out)["fps"] == pytest.approx(30, abs=0.5)


def test_graph_uses_the_given_frame_rate_and_only_scales_when_asked():
    g = compose.build_filtergraph([Overlay(Path("c.mp4"), 1.0, 2.0)], fps=60)
    assert "fps=60" in g and "fps=30" not in g and "scale=540" not in g and g.endswith("[b1]null[v]")
    d = compose.build_filtergraph([Overlay(Path("c.mp4"), 1.0, 2.0)], "o.ass", scale_to=(540, 960))
    assert d.endswith("[b1]subtitles=o.ass,scale=540:960[v]")
    assert compose.build_filtergraph([], scale_to=(540, 960)).endswith("[b0]scale=540:960[v]")


# ---- the draft preview ----------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_draft_is_small_and_fast_settings_and_leaves_no_final(tmp_path):
    main = speaker(tmp_path / "main60.mp4", 60)
    job = make_job(tmp_path, main)
    out = pipeline.run(job, main, log=lambda m: None, draft=True)
    v = info(out)
    assert out.name == "draft.mp4" and (v["width"], v["height"]) == (540, 960)
    assert v["fps"] == pytest.approx(30, abs=0.5)  # a draft does not need the source's 60 fps
    assert not (job / "final.mp4").exists()


@needs_ffmpeg
def test_render_draft_changes_nothing_and_knows_when_it_is_out_of_date(tmp_path):
    main = speaker(tmp_path / "main.mp4", 30)
    job = make_job(tmp_path, main)
    plan_before = (job / "plan.json").read_bytes()
    assert workflow.draft_is_current(job, None) is None  # no draft yet
    out = workflow.render_draft(job, None, log=lambda m: None)
    assert out == job / "draft.mp4" and out.is_file()
    assert (job / "plan.json").read_bytes() == plan_before  # not approved, nothing rewritten
    assert (job / "clips" / "clip_01.mp4").is_file() and main.is_file()  # nothing purged
    assert workflow.draft_is_current(job, None) is True
    assert workflow.draft_is_current(job, "english") is False  # another caption language: a different reel
    workflow.set_enabled(job, 1, False)  # the user edits the plan
    assert workflow.draft_is_current(job, None) is False


def test_the_post_render_purge_removes_the_draft(tmp_path):
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    for n in ("final.mp4", "draft.mp4", "plan.json"):
        (job / n).write_bytes(b"x")
    purge_job_temp(job, root=tmp_path)
    assert sorted(p.name for p in job.iterdir()) == ["final.mp4", "plan.json"]


def test_the_review_page_offers_the_draft_and_warns_when_it_is_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    main = tmp_path / "main.mp4"
    main.write_bytes(b"v")
    job = tmp_path / "jobs" / "20260101_000000_x"
    job.mkdir(parents=True)
    (job / "plan.json").write_text(json.dumps({"video_topic": "t", "video_duration": 15.0, "items": []}))
    (job / "job.json").write_text(json.dumps({"language": "en", "source_name": "a.mp4", "style": "overlay",
                                              "source_video": str(main)}))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(b.label.startswith("Preview full reel") for b in at.button)
    assert not any("out of date" in w.value for w in at.warning)  # no draft yet

    (job / "draft.mp4").write_bytes(b"d")
    workflow.write_job_meta(job, draft_sig="from-an-older-plan")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("out of date" in w.value for w in at.warning)
