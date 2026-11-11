"""Streamlit smoke tests using AppTest (no browser, no network)."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def make_job(root):
    job = root / "jobs" / "20260101_000000_demo"
    job.mkdir(parents=True)
    items = [{"id": i, "asset_type": "video", "timestamp_start": 2.0 * i, "timestamp_end": 2.0 * i + 4,
              "search_query": f"query {i}", "alternative_queries": ["a", "b"], "english_meaning": f"says {i}",
              "asset": {"status": "failed", "error": "x"} if i == 2 else {"status": "ok", "path": "clips/x.mp4",
                                                                           "channel": "Chan", "title": "T"}}
             for i in (1, 2, 3)]
    (job / "plan.json").write_text(json.dumps({"video_topic": "Demo topic", "items": items}), encoding="utf-8")
    (job / "job.json").write_text(json.dumps({"language": "hi", "source_name": "demo.mp4"}), encoding="utf-8")
    return job


def test_start_screen_renders_without_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Choose your video" in s.value for s in at.subheader)
    assert at.button[0].disabled  # "Build B-roll plan" needs a video first


def test_review_screen_shows_one_editable_row_per_item(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception
    assert [t.value for t in at.text_input if t.label == "Search query"] == ["query 1", "query 2", "query 3"]
    labels = [b.label for b in at.button]
    assert labels.count("Regenerate asset") == 3 and labels.count("Remove slot") == 3
    assert any("No clip found" in w.value for w in at.warning)  # the failed item is flagged
    assert "Demo topic" in " ".join(m.value for m in at.markdown)
