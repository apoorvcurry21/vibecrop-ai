"""Phase 2 tests. Gemini is mocked, so no API key or network is needed."""
import json
from types import SimpleNamespace

import pytest

from src.planning import pipeline
from src.planning.gemini_planner import PlanningError, plan
from src.planning.prompt import SYSTEM_PROMPT, build_user_prompt
from src.planning.schema import BRollItem
from src.planning.validate import snap_item, validate

DURATION = 100.0
WORDS = [{"word": f"w{i}", "start": round(i * 0.5, 3), "end": round(i * 0.5 + 0.4, 3)} for i in range(200)]
TRANSCRIPT = {
    "language": "hi", "video_duration": DURATION, "duration": DURATION,
    "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "namaste"}],
    "words": WORDS,
}


def item(kind="video", start=10.0, end=14.0, **kw):
    base = dict(asset_type=kind, timestamp_start=start, timestamp_end=end,
                search_query="stressed anxious person desk", alternative_queries=["a", "b"],
                spoken_text="x", english_meaning="y", reasoning="z")
    base.update(kw)
    return base


def good_items(n=10):
    return [item("video" if i % 3 else "image", 5 + i * 9, 5 + i * 9 + (4 if i % 3 else 3)) for i in range(n)]


def reply(items):
    return json.dumps({"video_topic": "topic", "items": items})


class FakeClient:
    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append(contents)
        return SimpleNamespace(text=self.texts.pop(0))


def test_valid_plan_any_mix_is_accepted():
    for kinds in (["video"] * 10, ["image"] * 10, ["video"] * 7 + ["image"] * 3):
        items = [item(k, 5 + i * 9, 5 + i * 9 + (4 if k == "video" else 3)) for i, k in enumerate(kinds)]
        out = plan(TRANSCRIPT, total=10, client=FakeClient(reply(items)))
        assert len(out["items"]) == 10 and out["status"] == "draft"
        assert sorted(i["asset_type"] for i in out["items"]) == sorted(kinds)


def test_snaps_to_word_boundaries():
    snapped = snap_item(BRollItem(**item(start=10.13, end=14.2)), WORDS, DURATION)
    assert snapped.timestamp_start in {w["start"] for w in WORDS}
    assert snapped.timestamp_end in {w["end"] for w in WORDS}
    assert 3.0 - 0.01 <= snapped.timestamp_end - snapped.timestamp_start <= 5.0 + 0.01


def test_retries_with_feedback_on_wrong_count():
    client = FakeClient(reply(good_items(9)), reply(good_items(10)))
    out = plan(TRANSCRIPT, total=10, client=client)
    assert len(out["items"]) == 10 and len(client.calls) == 2
    feedback = client.calls[1][-1].parts[0].text
    assert "exactly 10" in feedback


def test_retries_on_invalid_json_then_succeeds():
    client = FakeClient("not json", reply(good_items()))
    assert len(plan(TRANSCRIPT, total=10, client=client)["items"]) == 10


def test_gives_up_after_max_attempts():
    client = FakeClient(*[reply(good_items(8))] * 3)
    with pytest.raises(PlanningError):
        plan(TRANSCRIPT, total=10, client=client, max_attempts=3)


def test_validate_flags_overlap_and_intro_and_overrun():
    items = [BRollItem(**item(start=0.2, end=3.5)),
             BRollItem(**item(start=3.6, end=7.0)),
             BRollItem(**item(start=98.0, end=102.0))]
    errs = " ".join(validate(items, 3, DURATION))
    assert "before 1s" in errs and "overlaps" in errs and "after the video ends" in errs


def test_validate_flags_bad_query():
    items = [BRollItem(**item(start=5.0, end=8.0, search_query="success", alternative_queries=["a"]))]
    errs = " ".join(validate(items, 1, DURATION))
    assert "search_query has 1 words" in errs and "2 alternative_queries" in errs


def test_validate_flags_long_uncovered_stretch():
    items = [BRollItem(**item(start=5.0, end=8.0)), BRollItem(**item(start=60.0, end=63.0))]
    assert any("No B-roll between" in e for e in validate(items, 2, DURATION))


def test_falls_back_to_next_model_when_primary_fails(monkeypatch):
    from src.planning import gemini_planner

    used = []

    def fake_generate(client, model, history, sleep):
        used.append(model)
        if model == "primary":
            raise PlanningError("503")
        return reply(good_items())

    monkeypatch.setattr(gemini_planner, "_generate", fake_generate)
    monkeypatch.setattr(gemini_planner, "gemini_fallback_models", lambda: ["backup"])
    out = plan(TRANSCRIPT, total=10, model="primary", client=object())
    assert used == ["primary", "backup"] and out["model"] == "backup"


def test_prompt_contains_key_rules_and_transcript():
    assert "hit a wall" in SYSTEM_PROMPT and "NEVER a brick wall" in SYSTEM_PROMPT
    assert "English, Latin script" in SYSTEM_PROMPT
    text = build_user_prompt(TRANSCRIPT, 10)
    assert "exactly 10" in text and "[0.00-4.00] namaste" in text


def test_pipeline_writes_plan_json(tmp_path):
    (tmp_path / "transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    out = pipeline.run(tmp_path, total=10, client=FakeClient(reply(good_items())))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert out.name == "plan.json" and data["source_transcript"] == "transcript.json"
    assert [i["id"] for i in data["items"]] == list(range(1, 11))
