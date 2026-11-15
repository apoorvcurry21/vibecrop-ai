"""Speed work: parallel B-roll downloads, single-copy uploads, render progress and encoder choice, stage timing."""
import json
import subprocess
import threading
import time

import pytest

from src import config, perf, workflow
from src.rendering import compose
from src.rendering.compose import Overlay, parse_progress_line, render, video_encoder_args
from src.sourcing import pipeline as sourcing


# ---- parallel sourcing -------------------------------------------------------------------------------------------

def _plan(tmp_path, n):
    items = [{"id": i, "asset_type": "video", "timestamp_start": 2.0 * i, "timestamp_end": 2.0 * i + 1.5,
              "search_query": f"q{i}", "alternative_queries": []} for i in range(1, n + 1)]
    (tmp_path / "plan.json").write_text(json.dumps({"items": items}), encoding="utf-8")


def test_sourcing_runs_slots_in_parallel_and_keeps_every_result(monkeypatch, tmp_path):
    _plan(tmp_path, 6)
    seen_threads = set()

    def slow(job_dir, item, skip_ids=frozenset()):
        seen_threads.add(threading.current_thread().name)
        time.sleep(0.3)
        if item["id"] == 3:
            raise sourcing.SourcingError("blocked")
        return {"status": "ok", "path": f"clips/clip_{item['id']:02d}.mp4", "channel": "C", "title": "T",
                "window_start": 1.0}

    monkeypatch.setattr(sourcing, "source_item", slow)
    logs = []
    t0 = time.perf_counter()
    sourcing.run(tmp_path, log=logs.append, workers=4)
    elapsed = time.perf_counter() - t0

    items = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))["items"]
    assert [i["id"] for i in items] == [1, 2, 3, 4, 5, 6]  # order untouched
    assert [i["asset"]["status"] for i in items] == ["ok", "ok", "failed", "ok", "ok", "ok"]
    assert "blocked" in items[2]["asset"]["error"]
    assert len(seen_threads) > 1
    assert elapsed < 6 * 0.3 * 0.7  # sequential would take 1.8 s; 4 workers take about 0.6 s
    assert any("FAILED" in m for m in logs)


def test_sourcing_with_one_worker_is_sequential(monkeypatch, tmp_path):
    _plan(tmp_path, 3)
    active, peak = [0], [0]
    lock = threading.Lock()

    def fake(job_dir, item, skip_ids=frozenset()):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return {"status": "ok", "path": "clips/x.mp4", "channel": "C", "title": "T", "window_start": 0}

    monkeypatch.setattr(sourcing, "source_item", fake)
    sourcing.run(tmp_path, log=lambda *_: None, workers=1)
    assert peak[0] == 1


def test_sourcing_workers_env_is_clamped(monkeypatch):
    monkeypatch.delenv("SOURCING_WORKERS", raising=False)
    assert config.sourcing_workers() == config.DEFAULT_SOURCING_WORKERS
    for raw, want in (("2", 2), ("0", 1), ("99", config.MAX_SOURCING_WORKERS), ("abc", config.DEFAULT_SOURCING_WORKERS)):
        monkeypatch.setenv("SOURCING_WORKERS", raw)
        assert config.sourcing_workers() == want


# ---- upload ------------------------------------------------------------------------------------------------------

def test_upload_is_written_once_straight_into_the_job(tmp_path):
    job = workflow.create_job_from_upload("My Clip.MOV", memoryview(b"video-bytes"), "hi", root=tmp_path, notes=" a  b ")
    meta = workflow.read_job_meta(job)
    assert (job / "source.mov").read_bytes() == b"video-bytes"
    assert meta["source_video"] == str(job / "source.mov") and meta["source_name"] == "My Clip.MOV"
    assert meta["notes"] == "a b" and meta["language"] == "hi"
    assert not (tmp_path / "uploads").exists()  # no staging copy any more
    with pytest.raises(workflow.WorkflowError):
        workflow.create_job_from_upload("a.mp4", b"x", "hi", root=tmp_path, style="nope")


# ---- render progress and encoder ---------------------------------------------------------------------------------

def test_parse_progress_line():
    assert parse_progress_line("out_time_us=2000000", 8.0) == pytest.approx(0.25)
    assert parse_progress_line("out_time_ms=4000000", 8.0) == pytest.approx(0.5)  # ffmpeg's ms key is microseconds
    assert parse_progress_line("out_time_us=99000000", 8.0) == 1.0
    assert parse_progress_line("out_time_us=N/A", 8.0) is None
    assert parse_progress_line("frame=12", 8.0) is None
    assert parse_progress_line("out_time_us=1", 0.0) is None


def test_encoder_defaults_to_the_cpu_and_uses_nvenc_only_when_allowed(monkeypatch):
    monkeypatch.setattr(compose, "nvenc_available", lambda: True)
    monkeypatch.setenv("VIBECROP_ENCODER", "auto")
    assert "libx264" in video_encoder_args("veryfast", 20)  # previews and tests never use the GPU
    assert "h264_nvenc" in video_encoder_args("veryfast", 20, hw=True)
    monkeypatch.setenv("VIBECROP_ENCODER", "cpu")
    assert "libx264" in video_encoder_args("veryfast", 20, hw=True)
    monkeypatch.setattr(compose, "nvenc_available", lambda: False)
    monkeypatch.setenv("VIBECROP_ENCODER", "nvenc")
    assert "libx264" in video_encoder_args("veryfast", 20, hw=True)  # asked for the GPU, but there is none


def test_the_bitrate_cap_goes_to_both_encoders_and_is_absent_without_one(monkeypatch):
    monkeypatch.setattr(compose, "nvenc_available", lambda: True)
    monkeypatch.setenv("VIBECROP_ENCODER", "auto")
    for hw in (False, True):
        a = video_encoder_args("veryfast", 17, hw=hw, maxrate_kbps=20000)
        assert a[a.index("-maxrate") + 1] == "20000k" and a[a.index("-bufsize") + 1] == "40000k"
    assert "-maxrate" not in video_encoder_args("veryfast", 20) and "-maxrate" not in video_encoder_args("veryfast", 20, hw=True)


def _ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def test_render_reports_progress_and_matches_the_plain_render(tmp_path):
    main, clip = tmp_path / "main.mp4", tmp_path / "clip.mp4"
    _ff("-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=30:d=4", "-f", "lavfi", "-i", "sine=frequency=440:d=4",
        "-shortest", "-pix_fmt", "yuv420p", str(main))
    _ff("-f", "lavfi", "-i", "color=c=green:s=1920x1080:r=25:d=3", "-pix_fmt", "yuv420p", str(clip))
    seen = []
    out = render(main, [Overlay(clip, 1.0, 3.0)], tmp_path / "a.mp4", progress=seen.append)
    assert out.is_file() and seen and seen[-1] == 1.0
    assert all(0.0 <= f <= 1.0 for f in seen) and seen == sorted(seen)

    def boom(_f):
        raise RuntimeError("UI went away")

    assert render(main, [Overlay(clip, 1.0, 3.0)], tmp_path / "b.mp4", progress=boom).is_file()  # callback errors are ignored
    plain = render(main, [Overlay(clip, 1.0, 3.0)], tmp_path / "c.mp4")
    assert plain.stat().st_size == pytest.approx(out.stat().st_size, rel=0.02)


# ---- timing log --------------------------------------------------------------------------------------------------

def test_perf_stage_appends_a_line_and_never_raises(tmp_path):
    with perf.stage(tmp_path, "render"):
        time.sleep(0.01)
    assert "render" in (tmp_path / "perf.log").read_text(encoding="utf-8")
    with perf.stage(tmp_path / "missing-dir", "x"):  # unwritable: silently ignored
        pass
    with perf.stage(None, "x"):
        pass
