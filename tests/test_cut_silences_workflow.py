"""How the workflow drives silence/retake cutting: detect on analyse, review (select/dismiss), apply, and the
toggle's on/off bookkeeping. Audio/video pipelines are mocked here; the real ffmpeg path is covered by
test_audio_cuts.py."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src import workflow

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def W(text, start, end):
    return {"word": text, "start": start, "end": end}


def fake_transcribe(video, language, job_dir=None, **k):
    (job_dir / "transcript.json").write_text(json.dumps(
        {"words": [W("hi", 0, 0.3)], "segments": [], "duration": 20.0, "video_duration": 20.0}))


def fake_candidates(audio, words, duration):
    return [{"id": 0, "selected": True, "start": 1.0, "end": 2.0, "kind": "silence", "text": ""},
            {"id": 1, "selected": True, "start": 5.0, "end": 6.0, "kind": "retake", "text": "say this"}]


def patch(monkeypatch, apply_result=None):
    calls = SimpleNamespace(detect=0, apply=0, denoise_audio=0)

    def prepare(video, job_dir, already_denoised, log):
        calls.denoise_audio += 1
        p = Path(job_dir) / "_det.wav"
        p.write_bytes(b"x")
        return p

    def detect(audio, words, duration):
        calls.detect += 1
        return fake_candidates(audio, words, duration)

    def apply(video, duration, cuts, transcript, out, work):
        calls.apply += 1
        Path(out).write_bytes(b"cut-video")
        return apply_result or {"video": out, "removed_s": 2.0,
                                "transcript": {**transcript, "video_duration": duration - 2.0, "duration": duration - 2.0,
                                              "words": [W("hi", 0, 0.3)]}}

    monkeypatch.setattr(workflow.transcription, "run", fake_transcribe)
    monkeypatch.setattr(workflow.audio_cuts, "prepare_detection_audio", prepare)
    monkeypatch.setattr(workflow.audio_cuts, "detect_cuts", detect)
    monkeypatch.setattr(workflow.audio_cuts, "apply_cuts", apply)
    monkeypatch.setattr(workflow.recommend, "recommend_count", lambda t: {"count": 3, "reason": "r", "source": "fallback"})
    return calls


def new_job(tmp_path, **kw) -> Path:
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"raw")
    return workflow.create_job(src, "en", root=tmp_path, **kw)


def test_analyse_detects_candidates_without_touching_the_video(tmp_path, monkeypatch):
    calls = patch(monkeypatch)
    job = new_job(tmp_path, cut_silences=True)
    workflow.analyse(job, log=lambda m: None)
    meta = workflow.read_job_meta(job)
    assert meta["cuts_detected"] and meta["cuts_applied"] is False and calls.detect == 1 and calls.denoise_audio == 1
    assert meta["pre_cut_video"]
    assert meta["source_video"] == str(tmp_path / "raw.mp4")  # not cut yet: still the original
    cuts = workflow.read_cuts_preview(job)
    assert [(c["id"], c["kind"]) for c in cuts] == [(0, "silence"), (1, "retake")]
    workflow.analyse(job, log=lambda m: None)
    assert calls.detect == 1  # a second analyse does not re-detect


def test_without_the_toggle_nothing_is_detected(tmp_path, monkeypatch):
    calls = patch(monkeypatch)
    job = new_job(tmp_path)
    workflow.analyse(job, log=lambda m: None)
    assert calls.detect == 0 and workflow.read_cuts_preview(job) is None


def test_set_cut_selection_flips_one_row(tmp_path, monkeypatch):
    patch(monkeypatch)
    job = new_job(tmp_path, cut_silences=True)
    workflow.analyse(job, log=lambda m: None)
    workflow.set_cut_selection(job, 1, False)
    cuts = workflow.read_cuts_preview(job)
    assert [(c["id"], c["selected"]) for c in cuts] == [(0, True), (1, False)]
    with pytest.raises(workflow.WorkflowError):
        workflow.set_cut_selection(job, 99, False)


def test_apply_cuts_updates_video_transcript_and_recommendation(tmp_path, monkeypatch):
    calls = patch(monkeypatch)
    job = new_job(tmp_path, cut_silences=True)
    workflow.analyse(job, log=lambda m: None)
    result = workflow.apply_cuts(job, log=lambda m: None)
    assert calls.apply == 1 and result["removed_s"] == 2.0
    meta = workflow.read_job_meta(job)
    assert meta["cuts_applied"] and meta["source_video"] == str(job / "source_cut.mp4")
    assert meta["video_duration"] == 18.0 and meta["suggested_brolls"] == 3
    assert workflow.read_cuts_preview(job) is None
    report = workflow.read_cuts_report(job)
    assert report["removed_s"] == 2.0 and len(report["cuts"]) == 2
    assert json.loads((job / "transcript.json").read_text(encoding="utf-8"))["video_duration"] == 18.0


def test_dismiss_keeps_everything_unchanged(tmp_path, monkeypatch):
    patch(monkeypatch)
    job = new_job(tmp_path, cut_silences=True)
    workflow.analyse(job, log=lambda m: None)
    original_video = workflow.read_job_meta(job)["source_video"]
    workflow.dismiss_cut_preview(job)
    meta = workflow.read_job_meta(job)
    assert meta["cuts_applied"] and meta["source_video"] == original_video
    assert workflow.read_cuts_preview(job) is None


def test_toggling_cut_silences_off_after_applying_reverts_the_video(tmp_path, monkeypatch):
    patch(monkeypatch)
    job = new_job(tmp_path, cut_silences=True)
    workflow.analyse(job, log=lambda m: None)
    pre_cut = workflow.read_job_meta(job)["pre_cut_video"]
    workflow.apply_cuts(job, log=lambda m: None)
    assert workflow.read_job_meta(job)["source_video"] == str(job / "source_cut.mp4")
    workflow.set_cut_silences(job, False)
    meta = workflow.read_job_meta(job)
    assert not meta["cut_silences"] and not meta.get("cuts_applied") and meta["source_video"] == pre_cut
    assert not (job / "cuts_report.json").exists() and not (job / "source_cut.mp4").exists()
    assert not (job / "transcript.json").exists()  # _reset_derived: the next analyse redoes it
    workflow.set_cut_silences(job, False)  # already off: nothing happens
    assert workflow.read_job_meta(job)["source_video"] == pre_cut


def test_apply_cuts_without_a_preview_raises(tmp_path, monkeypatch):
    patch(monkeypatch)
    job = new_job(tmp_path, cut_silences=True)
    with pytest.raises(workflow.WorkflowError):
        workflow.apply_cuts(job, log=lambda m: None)


# ---- the upload screen -----------------------------------------------------------------------------------------------

def fake_analysis(job_dir, log=print):
    workflow.write_job_meta(job_dir, suggested_brolls=4, suggested_reason="x", suggested_source="gemini", video_duration=30.0)
    return {"count": 4, "reason": "x", "source": "gemini"}


def test_the_checkbox_sets_the_flag_and_the_preview_blocks_step_2_until_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    video = tmp_path / "my.mp4"
    video.write_bytes(b"x")
    seen = []

    def fake_analyse(job_dir, log=print):
        seen.append(bool(workflow.read_job_meta(job_dir).get("cut_silences")))
        workflow.write_json_atomic(job_dir / "cuts_preview.json", {"cuts": [
            {"id": 0, "selected": True, "start": 1.0, "end": 2.0, "kind": "silence", "text": ""}]})
        workflow.write_job_meta(job_dir, cuts_detected=True, cuts_applied=False,
                               pre_cut_video=workflow.read_job_meta(job_dir)["source_video"])
        return fake_analysis(job_dir)

    monkeypatch.setattr(workflow, "analyse", fake_analyse)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and at.checkbox(key="cut_silences_th").value is False
    at.checkbox(key="cut_silences_th").set_value(True).run()
    at.text_input[0].set_value(str(video)).run()
    assert not at.exception and seen == [True]
    draft = Path(at.session_state["draft_job"])
    assert workflow.read_job_meta(draft)["cut_silences"] is True

    # the preview panel is showing and blocks step 2 (style) from rendering
    assert any("Review the cuts" in h.value for h in at.subheader)
    assert not any("Pick a video style" in h.value for h in at.subheader)

    skip = next(b for b in at.button if b.label == "Skip (keep everything)")
    skip.click().run()
    assert not at.exception
    assert workflow.read_job_meta(draft)["cuts_applied"] is True and workflow.read_cuts_preview(draft) is None
    assert any("Pick a video style" in h.value for h in at.subheader)  # resolved: step 2 now renders


def test_the_cuts_report_survives_the_post_render_purge(tmp_path):
    from src.cleanup import purge_job_temp

    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    for n in ("final.mp4", "cuts_report.json", "source_cut.mp4", "cuts_preview.json", "source.mp4"):
        (job / n).write_bytes(b"x")
    purge_job_temp(job, root=tmp_path)
    assert sorted(p.name for p in job.iterdir()) == ["cuts_report.json", "final.mp4"]
