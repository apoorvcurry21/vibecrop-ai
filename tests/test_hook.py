"""The opening hook: the planner suggests 3 lines, one goes on the video as a text overlay, the review page can swap it."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src import workflow
from src.planning import pipeline as planning_pipeline
from src.planning.gemini_planner import plan
from src.planning.prompt import HOOK_REQUEST, build_user_prompt, clean_hooks

APP = str(Path(__file__).resolve().parent.parent / "app.py")
WORDS = [{"word": f"w{i}", "start": round(i * 0.5, 3), "end": round(i * 0.5 + 0.4, 3)} for i in range(200)]
TRANSCRIPT = {"language": "hi", "video_duration": 100.0, "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "namaste"}],
              "words": WORDS}


def reply(hooks=None, n=3):
    items = [dict(asset_type="video", timestamp_start=5 + i * 9, timestamp_end=9 + i * 9, search_query="stressed anxious person desk",
                  alternative_queries=["a", "b"], spoken_text="x", english_meaning="y", reasoning="z") for i in range(n)]
    body = {"video_topic": "topic", "items": items}
    if hooks is not None:
        body["hooks"] = hooks
    return json.dumps(body)


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.models = SimpleNamespace(generate_content=lambda model, contents, config: SimpleNamespace(text=self.text))


def make_job(root: Path, hooks=None, duration=30.0, texts=None) -> Path:
    job = root / "jobs" / "20260101_000000_hook"
    job.mkdir(parents=True)
    item = {"id": 1, "asset_type": "video", "timestamp_start": 5.0, "timestamp_end": 9.0, "search_query": "q",
            "alternative_queries": [], "english_meaning": "says", "layout": "overlay",
            "asset": {"status": "ok", "path": "clips/clip_01.mp4", "channel": "C", "title": "T", "clip_duration": 8.0,
                      "relevance": "mismatch", "relevance_reason": "just a logo"}}
    body = {"video_topic": "t", "video_duration": duration, "items": [item], "texts": texts or []}
    if hooks is not None:
        body["hooks"] = hooks
    (job / "plan.json").write_text(json.dumps(body))
    (job / "job.json").write_text(json.dumps({"language": "hi", "source_name": "demo.mp4", "style": "overlay"}))
    return job


def texts(job):
    return workflow.load_plan(job).get("texts", [])


# ----------------------------------------------------------------------------- planner
def test_the_planner_asks_for_hooks_and_returns_them_cleaned():
    assert HOOK_REQUEST in build_user_prompt(TRANSCRIPT, 3)
    out = plan(TRANSCRIPT, total=3, client=FakeClient(reply(['"Ye galti mat karna"', "Ye galti mat karna", "  Sach  kya hai? ", "x" * 80])))
    assert out["hooks"] == ["Ye galti mat karna", "Sach kya hai?"]  # quotes and spaces cleaned, duplicate and over-long dropped


def test_a_reply_without_hooks_is_still_a_valid_plan():
    out = plan(TRANSCRIPT, total=3, client=FakeClient(reply(None)))
    assert out["hooks"] == [] and len(out["items"]) == 3


def test_clean_hooks_keeps_at_most_three():
    assert clean_hooks(["a", "b", "c", "d"]) == ["a", "b", "c"] and clean_hooks(None) == []


def test_pipeline_saves_the_hooks_in_plan_json(tmp_path):
    (tmp_path / "transcript.json").write_text(json.dumps(TRANSCRIPT))
    out = planning_pipeline.run(tmp_path, total=3, client=FakeClient(reply(["One", "Two", "Three"])))
    assert json.loads(out.read_text(encoding="utf-8"))["hooks"] == ["One", "Two", "Three"]


# ----------------------------------------------------------------------------- workflow
def test_set_hook_puts_one_flagged_text_over_the_first_three_seconds(tmp_path):
    job = make_job(tmp_path, hooks=["First", "Second"])
    entry = workflow.set_hook(job, "  Ye  galti mat karna ")
    assert (entry["text"], entry["start"], entry["end"], entry["hook"]) == ("Ye galti mat karna", 0.3, 3.0, True)
    assert entry["pos"] == "top"
    workflow.set_hook(job, "Second")  # replaces, never stacks
    assert [t["text"] for t in texts(job)] == ["Second"]


def test_the_hook_lives_next_to_other_texts_and_does_not_touch_them(tmp_path):
    other = {"id": 1, "text": "GOAT", "start": 7.0, "end": 9.0, "pos": "top", "size": "large", "color": "white", "font": "bold"}
    job = make_job(tmp_path, texts=[other])
    workflow.set_hook(job, "Hook")
    workflow.set_hook(job, "Better hook")
    assert [(t["id"], t["text"]) for t in texts(job)] == [(1, "GOAT"), (3, "Better hook")]
    workflow.remove_hook(job)
    assert [t["text"] for t in texts(job)] == ["GOAT"]
    workflow.remove_hook(job)  # nothing to remove: fine


def test_a_short_video_shortens_the_hook_and_a_tiny_one_refuses_it(tmp_path):
    job = make_job(tmp_path, duration=2.0)
    assert workflow.set_hook(job, "Short") ["end"] == 2.0
    with pytest.raises(workflow.WorkflowError):
        workflow.set_hook(make_job(tmp_path / "b", duration=0.5), "Too short")
    with pytest.raises(workflow.WorkflowError):
        workflow.set_hook(job, "   ")


def test_default_hook_uses_the_first_suggestion_once_and_never_overrides_the_users_choice(tmp_path):
    job = make_job(tmp_path, hooks=["First", "Second"])
    assert workflow.apply_default_hook(job)["text"] == "First"
    workflow.set_hook(job, "Mine")
    assert workflow.apply_default_hook(job) is None and [t["text"] for t in texts(job)] == ["Mine"]
    assert workflow.apply_default_hook(make_job(tmp_path / "old", hooks=None)) is None  # an older plan: no suggestions


def test_run_pipeline_adds_the_default_hook_after_planning(tmp_path, monkeypatch):
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    (job / "job.json").write_text(json.dumps({"source_video": str(video), "language": "hi"}))
    (job / "transcript.json").write_text(json.dumps(TRANSCRIPT))

    def fake_plan(job_dir, **kw):
        (Path(job_dir) / "plan.json").write_text(json.dumps({"video_duration": 30.0, "items": [], "hooks": ["Open here"]}))

    monkeypatch.setattr(workflow.planning, "run", fake_plan)
    monkeypatch.setattr(workflow.sourcing, "run", lambda *a, **k: None)
    workflow.run_pipeline(job, log=lambda m: None)
    assert [(t["text"], t["hook"]) for t in texts(job)] == [("Open here", True)]


# ----------------------------------------------------------------------------- review page
def open_page(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path, **kw)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return job, at


def hook_radio(at):
    return next(r for r in at.radio if r.key and r.key.startswith("hook_pick_"))


def test_the_page_lists_the_suggestions_and_choosing_one_saves_it(tmp_path, monkeypatch):
    job, at = open_page(tmp_path, monkeypatch, hooks=["One", "Two", "Three"])
    assert hook_radio(at).options == ["One", "Two", "Three", "No hook"] and hook_radio(at).value == "No hook"
    hook_radio(at).set_value("Two").run()
    assert [(t["text"], t["hook"]) for t in texts(job)] == [("Two", True)]
    assert hook_radio(at).value == "Two"
    hook_radio(at).set_value("No hook").run()
    assert texts(job) == []


def test_your_own_hook_can_be_written(tmp_path, monkeypatch):
    job, at = open_page(tmp_path, monkeypatch, hooks=["One"])
    own = next(t for t in at.text_input if t.key and t.key.startswith("hook_own_"))
    own.set_value("Meri apni line").run()
    next(b for b in at.button if b.label == "Use my own hook").click().run()
    assert [t["text"] for t in texts(job)] == ["Meri apni line"]
    assert "Meri apni line" in hook_radio(at).options  # a custom hook shows up as the chosen option


def test_an_older_plan_without_suggestions_still_works_and_says_so(tmp_path, monkeypatch):
    job, at = open_page(tmp_path, monkeypatch, hooks=None)
    assert any("no hook suggestions" in i.value for i in at.info)
    assert hook_radio(at).options == ["No hook"]


def test_a_clip_judged_off_topic_is_flagged_on_the_page(tmp_path, monkeypatch):
    job, at = open_page(tmp_path, monkeypatch, hooks=["One"])
    assert any("may not fit what you say (just a logo)" in c.value for c in at.caption)
