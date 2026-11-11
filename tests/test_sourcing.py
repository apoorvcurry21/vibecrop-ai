"""Phase 3a tests: yt-dlp is mocked, so no network is needed."""
import json
from pathlib import Path

import pytest

import src.sourcing.youtube as yt
from src.sourcing import pipeline
from src.sourcing.youtube import SourcingError, pick_window, source_video_clip


def test_pick_window_starts_about_25_percent_in():
    start, end = pick_window(200.0, 4.0)
    assert start == 50.0 and end == pytest.approx(55.0)  # 4 s slot + 1 s padding


def test_pick_window_clamps_inside_short_videos():
    start, end = pick_window(35.0, 5.0)
    assert end <= 35.0 and start >= 0.0


def test_search_filters_short_long_live_and_skipped(monkeypatch):
    entries = [
        {"id": "short", "duration": 10},
        {"id": "long", "duration": 9000},
        {"id": "live", "duration": 100, "live_status": "is_live"},
        {"id": "seen", "duration": 100},
        {"id": "good", "duration": 100, "title": "T", "channel": "C"},
        {"id": "nodur", "duration": None},
    ]

    class FakeYDL:
        def __init__(self, opts): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, q, download=False):
            assert q.startswith("ytsearch")
            return {"entries": entries}

    import yt_dlp
    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    out = yt.search_candidates("q", skip_ids={"seen"})
    assert [c["id"] for c in out] == ["good"] and out[0]["channel"] == "C"


def _fake_env(monkeypatch, tmp_path, fail_ids=(), duration=4.5):
    cands = {"q1": [{"id": "a", "title": "A", "channel": "ChA", "duration": 120.0, "url": "u/a"},
                    {"id": "b", "title": "B", "channel": "ChB", "duration": 120.0, "url": "u/b"}],
             "q2": [{"id": "c", "title": "C", "channel": "ChC", "duration": 120.0, "url": "u/c"}]}
    monkeypatch.setattr(yt, "search_candidates",
                        lambda q, skip_ids=frozenset(), limit=10: [c for c in cands.get(q, []) if c["id"] not in skip_ids])

    def fake_download(cand, start, end, stem):
        if cand["id"] in fail_ids:
            raise RuntimeError("Sign in to confirm you're not a bot\nmore")
        p = Path(str(stem) + ".mp4")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(yt, "download_section", fake_download)
    monkeypatch.setattr(yt, "_probe", lambda p: (duration, True))


def test_falls_through_failed_downloads_to_next_candidate(monkeypatch, tmp_path):
    _fake_env(monkeypatch, tmp_path, fail_ids={"a"})
    meta = source_video_clip(["q1"], 4.0, tmp_path / "clip_01")
    assert meta["video_id"] == "b" and meta["channel"] == "ChB" and meta["path"] == "clip_01.mp4"
    assert meta["tried_ids"] == ["a", "b"]


def test_uses_alternative_query_when_first_has_no_results(monkeypatch, tmp_path):
    _fake_env(monkeypatch, tmp_path)
    meta = source_video_clip(["none", "q2"], 4.0, tmp_path / "clip_02")
    assert meta["video_id"] == "c" and meta["query_used"] == "q2"


def test_skip_ids_forces_a_different_video(monkeypatch, tmp_path):
    _fake_env(monkeypatch, tmp_path)
    meta = source_video_clip(["q1"], 4.0, tmp_path / "clip_03", skip_ids={"a"})
    assert meta["video_id"] == "b"


def test_gives_up_after_max_tries_and_rejects_short_clips(monkeypatch, tmp_path):
    _fake_env(monkeypatch, tmp_path, fail_ids={"a", "b", "c"})
    with pytest.raises(SourcingError):
        source_video_clip(["q1", "q2"], 4.0, tmp_path / "clip_04", max_tries=2)
    _fake_env(monkeypatch, tmp_path, duration=1.0)  # downloaded file far too short
    with pytest.raises(SourcingError, match="too short|No clip"):
        source_video_clip(["q1"], 4.0, tmp_path / "clip_05")


def test_pipeline_records_assets_and_failures(monkeypatch, tmp_path):
    _fake_env(monkeypatch, tmp_path, fail_ids={"c"})
    items = [
        {"id": 1, "asset_type": "video", "timestamp_start": 2.0, "timestamp_end": 6.0,
         "search_query": "q1", "alternative_queries": []},
        {"id": 2, "asset_type": "video", "timestamp_start": 20.0, "timestamp_end": 24.0,
         "search_query": "q2", "alternative_queries": []},
        {"id": 3, "asset_type": "image", "timestamp_start": 40.0, "timestamp_end": 43.0,
         "search_query": "q1", "alternative_queries": []},
    ]
    (tmp_path / "plan.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    pipeline.run(tmp_path, log=lambda *_: None)
    out = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))["items"]
    assert out[0]["asset"]["status"] == "ok" and out[0]["asset"]["path"] == "clips/clip_01.mp4"
    assert out[1]["asset"]["status"] == "failed" and "bot" in out[1]["asset"]["error"]
    assert "asset" not in out[2]  # images are not sourced in the video-only MVP
