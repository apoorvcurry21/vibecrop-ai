"""AI B-roll count suggestion and the analyse-on-upload workflow. Gemini and Groq are mocked."""
import json
from types import SimpleNamespace

import pytest

from src import config, workflow
from src.gemini_client import GeminiError
from src.planning import recommend
from src.planning.recommend import clamp_count, formula_count, recommend_count

TRANSCRIPT = {
    "language": "hi", "video_duration": 60.0, "duration": 60.0,
    "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "namaste doston"},
                 {"id": 1, "start": 4.0, "end": 9.0, "text": "aaj hum Messi ke baare mein baat karenge"}],
    "words": [],
}


class FakeClient:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(contents)
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(text=text)


def answer(count, reason="Fast story with many scenes."):
    return json.dumps({"count": count, "reason": reason})


def test_gemini_count_and_reason_are_returned():
    out = recommend_count(TRANSCRIPT, client=FakeClient(answer(7)), models=["m"])
    assert out == {"count": 7, "reason": "Fast story with many scenes.", "source": "gemini"}


def test_prompt_carries_transcript_duration_and_the_allowed_maximum():
    client = FakeClient(answer(5))
    recommend_count(TRANSCRIPT, client=client, models=["m"])
    prompt = client.calls[0][0].parts[0].text
    assert "Video duration: 60.0 seconds" in prompt and "Maximum allowed: 15" in prompt
    assert "[4.0-9.0] aaj hum Messi" in prompt


@pytest.mark.parametrize("duration,asked,expected", [
    (60.0, 40, 15),   # the app-wide cap
    (60.0, 0, 1),     # at least one
    (20.0, 15, 5),    # only 5 clips physically fit in 20 s
    (60.0, 6, 6),
])
def test_suggestion_is_clamped_to_what_fits(duration, asked, expected):
    assert clamp_count(duration, asked) == expected
    t = {**TRANSCRIPT, "video_duration": duration, "duration": duration}
    assert recommend_count(t, client=FakeClient(answer(asked)), models=["m"])["count"] == expected


def test_formula_is_about_one_per_eight_seconds():
    assert formula_count(64.0) == 8 and formula_count(120.0) == 15 and formula_count(5.0) == 1


@pytest.mark.parametrize("failure", [
    GeminiError("down"), "not json at all", json.dumps({"count": "many", "reason": "x"}),
])
def test_falls_back_to_the_formula_instead_of_failing(failure, monkeypatch):
    if isinstance(failure, GeminiError):
        monkeypatch.setattr(recommend, "generate_with_fallback", lambda *a, **k: (_ for _ in ()).throw(failure))
        client = object()
    else:
        client = FakeClient(failure)
    out = recommend_count(TRANSCRIPT, client=client, models=["m"])
    assert out["source"] == "formula" and out["count"] == 8 and "length-based" in out["reason"]


def test_missing_gemini_key_falls_back_too(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config, "is_colab", lambda: False)
    out = recommend_count(TRANSCRIPT)
    assert out["source"] == "formula" and "no Gemini key" in out["reason"]


# ----------------------------------------------------------------------------- workflow
@pytest.fixture
def draft(tmp_path, monkeypatch):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"v")
    job = workflow.create_job(src, "hi", root=tmp_path)
    calls = []

    def fake_transcribe(video, lang, job_dir=None):
        calls.append(lang)
        (job_dir / "transcript.json").write_text(json.dumps(TRANSCRIPT))

    monkeypatch.setattr(workflow.transcription, "run", fake_transcribe)
    monkeypatch.setattr(workflow.recommend, "recommend_count",
                        lambda t: {"count": 6, "reason": "Calm talk.", "source": "gemini"})
    return job, calls


def test_analyse_transcribes_once_and_stores_the_suggestion(draft):
    job, calls = draft
    out = workflow.analyse(job, log=lambda m: None)
    meta = workflow.read_job_meta(job)
    assert out["count"] == 6 and meta["suggested_brolls"] == 6 and meta["suggested_reason"] == "Calm talk."
    assert meta["suggested_source"] == "gemini" and meta["video_duration"] == 60.0
    workflow.analyse(job, log=lambda m: None)  # analysing again reuses the transcript
    assert calls == ["hi"]


def test_changing_the_language_invalidates_the_transcript_and_suggestion(draft):
    job, calls = draft
    workflow.analyse(job, log=lambda m: None)
    workflow.set_language(job, "hi")  # same language: nothing happens
    assert (job / "transcript.json").is_file()
    workflow.set_language(job, "en")
    meta = workflow.read_job_meta(job)
    assert meta["language"] == "en" and "suggested_brolls" not in meta and not (job / "transcript.json").exists()
    workflow.analyse(job, log=lambda m: None)
    assert calls == ["hi", "en"]


def test_planning_reuses_the_analysis_transcript_and_the_chosen_settings(draft, monkeypatch):
    job, calls = draft
    workflow.analyse(job, log=lambda m: None)
    workflow.start_planning(job, total=9, notes=" Messi  ", style="hybrid")
    seen = {}
    monkeypatch.setattr(workflow.planning, "run", lambda j, total=None, notes="", style="overlay": seen.update(
        total=total, notes=notes, style=style) or (j / "plan.json").write_text(json.dumps({"items": []})))
    monkeypatch.setattr(workflow.sourcing, "run", lambda j, log=print: None)
    workflow.run_pipeline(job, log=lambda m: None)
    assert calls == ["hi"]  # no second transcription
    assert seen == {"total": 9, "notes": "Messi", "style": "hybrid"}


def test_discard_draft_only_removes_unplanned_job_folders(draft, tmp_path):
    job, _ = draft
    planned = workflow.create_job(tmp_path / "v.mp4", "hi", root=tmp_path)
    (planned / "plan.json").write_text("{}")
    workflow.discard_draft(planned)
    assert planned.is_dir()  # has a plan: kept
    workflow.discard_draft(tmp_path)  # not a job folder: untouched
    assert tmp_path.is_dir()
    workflow.discard_draft(job)
    assert not job.exists()
    workflow.discard_draft(None)  # no draft yet: fine
