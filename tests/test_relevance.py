"""The clip relevance check: a light Gemini model looks at a few frames of each downloaded clip. Gemini and yt-dlp are
mocked; ffmpeg runs for real on a synthetic clip."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.sourcing.youtube as yt
from src import config
from src.gemini_client import GeminiError
from src.sourcing import pipeline, relevance
from src.sourcing.youtube import source_video_clip

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_clip(path: Path, seconds=5) -> Path:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=duration={seconds}:size=320x240:rate=10", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


class FakeModels:
    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return SimpleNamespace(text=json.dumps(step))


def client(*script):
    return SimpleNamespace(models=FakeModels(*script))


@needs_ffmpeg
def test_small_frames_are_cut_from_the_clip(tmp_path):
    frames = relevance.extract_frames(make_clip(tmp_path / "c.mp4"), 5.0)
    assert len(frames) == len(relevance.FRAME_POINTS) and all(f[:2] == b"\xff\xd8" for f in frames)  # JPEG magic bytes
    assert relevance.extract_frames(tmp_path / "missing.mp4", 5.0) == []


@needs_ffmpeg
def test_a_fitting_clip_is_kept_and_an_unrelated_one_is_flagged_with_the_reason(tmp_path):
    clip = make_clip(tmp_path / "c.mp4")
    ok = relevance.Checker(client({"match": True, "reason": "stadium"}), models=["m"])
    assert ok.check(clip, 5.0, "a cricket match", "cricket stadium") == (True, "")
    bad = relevance.Checker(client({"match": False, "reason": "  just a   logo "}), models=["m"])
    assert bad.check(clip, 5.0, "a cricket match", "cricket stadium") == (False, "just a logo")


@needs_ffmpeg
def test_gemini_gets_the_meaning_the_query_and_the_pictures(tmp_path):
    c = client({"match": True, "reason": "ok"})
    relevance.Checker(c, models=["m"]).check(make_clip(tmp_path / "c.mp4"), 5.0, "he scores a goal", "football goal")
    model, contents = c.models.calls[0]
    parts = contents[0].parts
    assert "he scores a goal" in parts[0].text and "football goal" in parts[0].text
    assert len(parts) == 1 + len(relevance.FRAME_POINTS) and all(
        p.inline_data.mime_type == "image/jpeg" for p in parts[1:])


@needs_ffmpeg
def test_trouble_means_unknown_never_a_rejection_and_repeated_trouble_pauses_the_check(tmp_path):
    clip = make_clip(tmp_path / "c.mp4")
    now = [1000.0]
    c = client(GeminiError("quota"))
    chk = relevance.Checker(c, models=["m"], clock=lambda: now[0])
    assert chk.check(clip, 5.0, "x", "y") == (None, "")
    assert not chk.paused()
    assert chk.check(clip, 5.0, "x", "y") == (None, "")
    assert chk.paused()  # two failures in a row: stop asking for a minute
    calls = len(c.models.calls)
    assert chk.check(clip, 5.0, "x", "y") == (None, "") and len(c.models.calls) == calls  # no call while paused
    now[0] += relevance.PAUSE_S + 1
    assert not chk.paused()


@needs_ffmpeg
def test_a_network_timeout_is_just_unknown_too(tmp_path):
    """Regression: 'The read operation timed out' (httpx.ReadTimeout) used to escape and fail the whole job."""
    import httpx

    clip = make_clip(tmp_path / "c.mp4")
    chk = relevance.Checker(client(httpx.ReadTimeout("The read operation timed out")), models=["m"])
    assert chk.check(clip, 5.0, "x", "y") == (None, "")


def test_one_crashing_slot_does_not_take_the_job_down(monkeypatch, tmp_path):
    def boom(job_dir, item, skip_ids=frozenset(), checker=None):
        raise RuntimeError("something unexpected")

    monkeypatch.setattr(pipeline, "source_item", boom)
    out = pipeline._source_safely(tmp_path, {"id": 1}, set())
    assert out["status"] == "failed" and "something unexpected" in out["error"]


def test_a_missing_key_or_an_unreadable_clip_is_just_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(relevance, "make_client", lambda *a, **k: (_ for _ in ()).throw(config.ConfigError("no key")))
    chk = relevance.Checker(models=["m"])
    broken = tmp_path / "c.mp4"
    broken.write_bytes(b"not a video")
    assert chk.check(broken, 5.0, "x", "y") == (None, "")  # ffmpeg cannot read it
    if shutil.which("ffmpeg"):
        good = make_clip(tmp_path / "g.mp4")
        assert chk.check(good, 5.0, "x", "y") == (None, "")  # frames fine, but no API key


def test_the_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RELEVANCE_CHECK", "0")
    assert relevance.default_checker() is None
    monkeypatch.setenv("RELEVANCE_CHECK", "1")
    assert relevance.default_checker() is not None and config.relevance_enabled()


# ----------------------------------------------------------------------------- inside the download loop
def fake_env(monkeypatch, duration=4.5):
    cands = [{"id": i, "title": i.upper(), "channel": f"Ch{i}", "duration": 120.0, "url": f"u/{i}"} for i in "abc"]
    monkeypatch.setattr(yt, "search_candidates",
                        lambda q, skip_ids=frozenset(), limit=10: [c for c in cands if c["id"] not in skip_ids])

    def fake_download(cand, start, end, stem):
        p = Path(str(stem) + ".mp4")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(cand["id"].encode())  # the file says which video it came from
        return p

    monkeypatch.setattr(yt, "download_section", fake_download)
    monkeypatch.setattr(yt, "_probe", lambda p: (duration, True))


def test_an_off_topic_clip_is_replaced_by_the_next_candidate(monkeypatch, tmp_path):
    fake_env(monkeypatch)
    seen = []

    def verify(path, length, query):
        seen.append(path.read_bytes())
        return (False, "a logo") if path.read_bytes() == b"a" else (True, "")

    meta = source_video_clip(["q"], 4.0, tmp_path / "clip_01", verify=verify)
    assert seen == [b"a", b"b"] and meta["video_id"] == "b" and meta["relevance"] == "match"
    assert meta["tried_ids"] == ["a", "b"]  # "Regenerate" will not offer the rejected video again
    assert (tmp_path / "clip_01.mp4").read_bytes() == b"b"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip_01.mp4"]  # no leftovers of the rejected clip


def test_when_every_clip_is_off_topic_the_first_is_still_used_and_marked(monkeypatch, tmp_path):
    fake_env(monkeypatch)
    meta = source_video_clip(["q"], 4.0, tmp_path / "clip_02", verify=lambda p, n, q: (False, "unrelated"))
    assert meta["video_id"] == "a" and meta["relevance"] == "mismatch" and meta["relevance_reason"] == "unrelated"
    assert meta["path"] == "clip_02.mp4" and (tmp_path / "clip_02.mp4").read_bytes() == b"a"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip_02.mp4"]
    assert meta["tried_ids"] == ["a", "b", "c"]


def test_unknown_keeps_the_first_clip_and_no_check_changes_nothing(monkeypatch, tmp_path):
    fake_env(monkeypatch)
    meta = source_video_clip(["q"], 4.0, tmp_path / "clip_03", verify=lambda p, n, q: (None, ""))
    assert meta["video_id"] == "a" and meta["relevance"] == "unchecked"
    meta = source_video_clip(["q"], 4.0, tmp_path / "clip_04")
    assert meta["video_id"] == "a" and meta["relevance"] == "unchecked"


def test_source_item_checks_each_clip_against_what_is_being_said(monkeypatch, tmp_path):
    fake_env(monkeypatch)
    asked = []

    class FakeChecker:
        def check(self, clip, length, meaning, query):
            asked.append((meaning, query))
            return True, ""

    item = {"id": 3, "search_query": "kohli cover drive", "alternative_queries": [], "timestamp_start": 2.0,
            "timestamp_end": 6.0, "english_meaning": "He plays a beautiful cover drive", "spoken_text": "x"}
    meta = pipeline.source_item(tmp_path, item, checker=FakeChecker())
    assert asked == [("He plays a beautiful cover drive", "kohli cover drive")]
    assert meta["relevance"] == "match" and meta["path"] == "clips/clip_03.mp4"


def test_source_item_skips_the_check_when_it_is_off(monkeypatch, tmp_path):
    fake_env(monkeypatch)  # RELEVANCE_CHECK=0 comes from conftest
    item = {"id": 1, "search_query": "q", "alternative_queries": [], "timestamp_start": 2.0, "timestamp_end": 6.0}
    assert pipeline.source_item(tmp_path, item)["relevance"] == "unchecked"
