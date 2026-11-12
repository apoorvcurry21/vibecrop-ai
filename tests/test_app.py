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
    slider = at.slider[0]
    assert (slider.label, slider.min, slider.max, slider.value) == ("How many B-rolls?", 1, 15, 10)
    notes = at.text_area[0]
    assert notes.label == "Notes for the AI (optional)" and notes.value == ""  # optional: empty is fine


def test_reload_restores_the_job_from_the_url_and_ignores_path_tricks(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["job"] = job.name  # what a page reload looks like: no session, job id in the URL
    at.run()
    assert not at.exception and len([t for t in at.text_input if t.label == "Search query"]) == 3

    evil = AppTest.from_file(APP, default_timeout=30)
    evil.query_params["job"] = "..\\..\\Windows"
    evil.run()
    assert not evil.exception and any("Choose your video" in s.value for s in evil.subheader)


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
    assert at.expander[0].label.startswith("Change the number of B-rolls")
    assert at.slider(key="re_count").value == 3  # no saved count -> falls back to the plan's size
    assert "Demo topic" in " ".join(m.value for m in at.markdown)


# ----------------------------------------------------------------------------- analyse-on-upload + slot editor
import shutil  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

from src import workflow  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def fake_analysis(count, reason, duration):
    """A stand-in for workflow.analyse: stores the suggestion in job.json the way the real one does."""
    def analyse(job_dir, log=print):
        workflow.write_job_meta(job_dir, suggested_brolls=count, suggested_reason=reason,
                                suggested_source="gemini", video_duration=duration)
        return {"count": count, "reason": reason, "source": "gemini"}
    return analyse


def test_adding_a_video_analyses_it_presets_the_slider_and_build_uses_the_chosen_style(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    video = tmp_path / "my.mp4"
    video.write_bytes(b"not really a video")
    calls = []
    inner = fake_analysis(7, "Fast story with many scenes.", 60.0)
    monkeypatch.setattr(workflow, "analyse", lambda job_dir, log=print: calls.append(1) or inner(job_dir))
    monkeypatch.setattr(workflow, "run_pipeline", lambda job, log=print, total=None: (job / "plan.json").write_text(
        json.dumps({"video_topic": "t", "video_duration": 60.0, "items": []})))

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value(str(video)).run()
    assert not at.exception and len(calls) == 1
    assert at.slider(key="count").value == 7  # preset from the AI's suggestion
    assert any("AI suggests **7**" in c.value for c in at.caption)
    at.run()
    assert len(calls) == 1  # a plain rerun does not analyse again
    assert not at.button[0].disabled

    at.slider(key="count").set_value(12).run()  # the user changes it
    assert at.slider(key="count").value == 12
    at.radio(key="style").set_value("hybrid").run()
    at.button[0].click().run()
    assert not at.exception
    meta = workflow.read_job_meta(Path(at.session_state["job_dir"]))
    assert meta["total_brolls"] == 12 and meta["style"] == "hybrid"


def test_changing_the_video_replaces_the_draft_job(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    monkeypatch.setattr(workflow, "analyse", fake_analysis(4, "x", 40.0))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value(str(a)).run()
    first = Path(at.session_state["draft_job"])
    at.text_input[0].set_value(str(b)).run()
    second = Path(at.session_state["draft_job"])
    assert first != second and not first.exists() and second.is_dir()  # the abandoned draft is cleaned up


def test_too_many_brolls_for_a_short_video_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(workflow, "analyse", fake_analysis(3, "Short.", 20.0))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value(str(video)).run()
    at.slider(key="count").set_value(15).run()
    assert any("physically fit" in w.value for w in at.warning)


def make_editable_job(root):
    """Two slots with real clips (16:9, red left third) so the editor and its previews have something to work on."""
    job = root / "jobs" / "20260101_000000_edit"
    (job / "clips").mkdir(parents=True)
    items = []
    for i, start in ((1, 2.0), (2, 9.0)):
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=green:s=1280x720:r=30:d=8", "-vf",
                        "drawbox=x=0:y=0:w=430:h=720:color=red:t=fill", "-pix_fmt", "yuv420p",
                        str(job / "clips" / f"clip_{i:02d}.mp4")], check=True)
        items.append({"id": i, "asset_type": "video", "timestamp_start": start, "timestamp_end": start + 4,
                      "search_query": f"query {i}", "alternative_queries": ["a", "b"], "english_meaning": f"says {i}",
                      "layout": "overlay",
                      "asset": {"status": "ok", "path": f"clips/clip_{i:02d}.mp4", "channel": "Chan", "title": "T",
                                "clip_duration": 8.0}})
    (job / "plan.json").write_text(json.dumps({"video_topic": "Edit", "video_duration": 30.0, "items": items}))
    (job / "job.json").write_text(json.dumps({"language": "hi", "source_name": "demo.mp4", "style": "overlay"}))
    return job


def open_editor(tmp_path, monkeypatch, item_id=1):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_editable_job(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["job_dir"] = str(job)
    at.run()
    at.toggle(key=f"edit{item_id}").set_value(True).run()
    return job, at


def button(at, label):
    return next(b for b in at.button if b.label == label)


@needs_ffmpeg
def test_slot_editor_opens_with_sliders_and_previews(tmp_path, monkeypatch):
    job, at = open_editor(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert [s.label for s in at.slider if s.label == "Zoom"] == ["Zoom", "Zoom"]  # start + end keyframe
    assert len(at.get("image")) == 2  # first-frame + last-frame stills
    assert button(at, "Swap with previous").disabled and not button(at, "Swap with next").disabled
    assert button(at, "Save framing").disabled  # nothing changed yet


@needs_ffmpeg
def test_zoom_preset_and_save_framing_are_stored_in_the_plan(tmp_path, monkeypatch):
    job, at = open_editor(tmp_path, monkeypatch)
    button(at, "Zoom in").click().run()
    assert [s.value for s in at.slider if s.label == "Zoom"] == [1.0, 1.6]
    assert not button(at, "Save framing").disabled
    assert any("Unsaved framing" in c.value for c in at.caption)
    button(at, "Save framing").click().run()
    saved = workflow.load_plan(job)["items"][0]["framing"]
    assert saved["start"]["z"] == 1.0 and saved["end"]["z"] == 1.6


@needs_ffmpeg
def test_switching_the_preview_to_16_9_shows_the_whole_clip(tmp_path, monkeypatch):
    job, at = open_editor(tmp_path, monkeypatch)
    at.radio(key="view1").set_value("16:9 (whole clip, crop outlined)").run()
    assert not at.exception and len(at.get("image")) == 1


@needs_ffmpeg
def test_layout_change_and_swap_from_the_editor_are_saved(tmp_path, monkeypatch):
    job, at = open_editor(tmp_path, monkeypatch)
    layout = next(r for r in at.radio if r.label == "Layout")
    layout.set_value("split").run()
    assert workflow.load_plan(job)["items"][0]["layout"] == "split"
    button(at, "Swap with next").click().run()
    plan = workflow.load_plan(job)
    assert [i["search_query"] for i in plan["items"]] == ["query 2", "query 1"]
    assert plan["items"][0]["layout"] == "split" and plan["items"][0]["timestamp_start"] == 2.0  # slot keeps layout+time


@needs_ffmpeg
def test_timing_is_saved_and_a_too_long_slot_is_refused(tmp_path, monkeypatch):
    job, at = open_editor(tmp_path, monkeypatch)
    next(s for s in at.slider if s.label.startswith("When it plays")).set_range(2.5, 6.0).run()
    button(at, "Save timing").click().run()
    item = workflow.load_plan(job)["items"][0]
    assert (item["timestamp_start"], item["timestamp_end"]) == (2.5, 6.0)
    with pytest.raises(workflow.WorkflowError):
        workflow.set_timing(job, 1, 2.5, 12.0)  # runs into slot 2 and past the 8 s clip
