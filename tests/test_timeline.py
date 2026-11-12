"""Editor operations on a plan: timing, swapping clips between slots, framing, layout and video style."""
import json
from pathlib import Path

import pytest

from src import workflow
from src.workflow import WorkflowError


def make_job(tmp_path: Path, clip_len=9.0) -> Path:
    """Three 4 s slots (2-6, 8-12, 14-18) in a 30 s video; each owns clips/clip_NN.mp4 holding its own marker."""
    job = tmp_path / "jobs" / "j"
    (job / "clips").mkdir(parents=True)
    items = []
    for i, start in enumerate((2.0, 8.0, 14.0), 1):
        (job / "clips" / f"clip_{i:02d}.mp4").write_bytes(f"clip-{i}".encode())
        items.append({
            "id": i, "asset_type": "video", "timestamp_start": start, "timestamp_end": start + 4,
            "search_query": f"query {i}", "alternative_queries": ["a", "b"], "spoken_text": f"words {i}",
            "english_meaning": f"meaning {i}", "reasoning": f"why {i}", "layout": "overlay",
            "ai_layout": "split" if i == 2 else "overlay",
            "asset": {"status": "ok", "path": f"clips/clip_{i:02d}.mp4", "clip_duration": clip_len,
                      "channel": f"chan{i}", "tried_ids": []},
        })
    (job / "plan.json").write_text(json.dumps({"video_duration": 30.0, "items": items}), encoding="utf-8")
    (job / "job.json").write_text("{}", encoding="utf-8")
    return job


def item(job: Path, i: int) -> dict:
    return next(x for x in workflow.load_plan(job)["items"] if x["id"] == i)


# ----------------------------------------------------------------------------- timing
def test_set_timing_moves_and_resizes_a_slot(tmp_path):
    job = make_job(tmp_path)
    workflow.set_timing(job, 2, 7.0, 13.5)
    assert (item(job, 2)["timestamp_start"], item(job, 2)["timestamp_end"]) == (7.0, 13.5)
    assert item(job, 1)["timestamp_end"] == 6.0  # neighbours untouched


def test_timing_limits_follow_the_neighbours_and_the_video(tmp_path):
    job = make_job(tmp_path)
    assert workflow.timing_limits(job, 1) == {"min_start": 1.0, "max_end": 7.9, "max_len": 9.0}  # intro + next slot
    assert workflow.timing_limits(job, 2)["min_start"] == 6.1 and workflow.timing_limits(job, 2)["max_end"] == 13.9
    assert workflow.timing_limits(job, 3)["max_end"] == 30.0  # last slot: the end of the video


@pytest.mark.parametrize("start,end,message", [
    (2.0, 3.0, "at least"),          # too short
    (0.5, 5.0, "cannot start before"),  # inside the speaker's intro
    (5.0, 8.5, "cannot start before"),  # overlaps slot 1 (starts before slot 1 ends + gap): see below
    (8.0, 14.5, "cannot end after"),    # runs into slot 3
])
def test_set_timing_rejects_bad_edits(tmp_path, start, end, message):
    job = make_job(tmp_path)
    with pytest.raises(WorkflowError, match=message):
        workflow.set_timing(job, 2, start, end)
    assert item(job, 2)["timestamp_start"] == 8.0  # nothing was saved


def test_set_timing_cannot_exceed_the_downloaded_clip_unless_refetching(tmp_path, monkeypatch):
    job = make_job(tmp_path, clip_len=4.5)
    with pytest.raises(WorkflowError, match="only 4.5s"):
        workflow.set_timing(job, 2, 8.0, 13.0)  # 5 s from a 4.5 s clip
    fetched = {}
    monkeypatch.setattr(workflow, "_resource", lambda j, it, skip: fetched.update(id=it["id"], skip=set(skip)))
    workflow.set_timing(job, 2, 8.0, 13.0, refetch=True)
    assert fetched == {"id": 2, "skip": set()} and item(job, 2)["timestamp_end"] == 13.0


# ----------------------------------------------------------------------------- swap
def test_swap_moves_clips_but_keeps_timing_layout_and_renamed_files(tmp_path):
    job = make_job(tmp_path)
    workflow.set_framing(job, 1, {"start": {"z": 2}, "end": {"z": 2}})
    workflow.set_layout(job, 2, "split")
    assert workflow.swap_slots(job, 1, 2) == []
    one, two = item(job, 1), item(job, 2)
    assert (one["timestamp_start"], two["timestamp_start"]) == (2.0, 8.0)  # timing stays in the slot
    assert (one["layout"], two["layout"]) == ("overlay", "split")  # so does the layout
    assert (one["search_query"], two["search_query"]) == ("query 2", "query 1")
    assert (one["asset"]["channel"], two["asset"]["channel"]) == ("chan2", "chan1")
    assert two["framing"]["start"]["z"] == 2 and "framing" not in one  # framing travels with its clip
    # files follow the invariant "item N owns clip_NN"; contents moved with the clip
    assert one["asset"]["path"] == "clips/clip_01.mp4" and two["asset"]["path"] == "clips/clip_02.mp4"
    assert (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-2"
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"clip-1"
    assert not list((job / "clips").glob("*.swap"))


def test_swap_keeps_a_removed_slot_removed_and_handles_missing_files(tmp_path):
    job = make_job(tmp_path)
    workflow.set_enabled(job, 1, False)
    (job / "clips" / "clip_02.mp4").unlink()  # e.g. cleaned up already
    workflow.swap_slots(job, 1, 2)
    assert item(job, 1)["enabled"] is False  # removal belongs to the slot
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"clip-1"
    assert item(job, 1)["asset"]["path"] == "clips/clip_01.mp4"


def test_swap_warns_when_a_clip_is_too_short_for_its_new_slot(tmp_path):
    job = make_job(tmp_path)
    plan = workflow.load_plan(job)
    plan["items"][0]["asset"]["clip_duration"] = 4.0  # slot 1's clip is short
    plan["items"][1]["timestamp_end"] = 14.0 - 0.1  # slot 2 is now 5.9 s
    (job / "plan.json").write_text(json.dumps(plan))
    warnings = workflow.swap_slots(job, 1, 2)
    assert len(warnings) == 1 and "Slot #2" in warnings[0] and "4.0s" in warnings[0]


def test_swap_with_itself_is_an_error(tmp_path):
    with pytest.raises(WorkflowError):
        workflow.swap_slots(make_job(tmp_path), 2, 2)


# ----------------------------------------------------------------------------- framing / layout / style
def test_set_framing_clamps_values(tmp_path):
    job = make_job(tmp_path)
    workflow.set_framing(job, 3, {"start": {"z": 9, "x": -2}, "end": {"z": 1.5, "x": 0.9, "y": 0.1}})
    f = item(job, 3)["framing"]
    assert f["start"] == {"z": 3.0, "x": 0.0, "y": 0.5} and f["end"] == {"z": 1.5, "x": 0.9, "y": 0.1}


def test_set_layout_validates(tmp_path):
    job = make_job(tmp_path)
    with pytest.raises(WorkflowError):
        workflow.set_layout(job, 1, "diagonal")
    assert workflow.set_layout(job, 1, "split")["layout"] == "split"


def test_apply_style_resets_every_slot_and_hybrid_restores_the_ai_mix(tmp_path):
    job = make_job(tmp_path)
    workflow.apply_style(job, "split")
    assert {i["layout"] for i in workflow.load_plan(job)["items"]} == {"split"}
    assert workflow.read_job_meta(job)["style"] == "split"
    workflow.apply_style(job, "hybrid")  # AI picks were overlay, split, overlay
    assert [i["layout"] for i in workflow.load_plan(job)["items"]] == ["overlay", "split", "overlay"]
    workflow.apply_style(job, "overlay")
    assert {i["layout"] for i in workflow.load_plan(job)["items"]} == {"overlay"}
    with pytest.raises(WorkflowError):
        workflow.apply_style(job, "wild")


def test_split_face_y_is_stored_and_clamped(tmp_path):
    job = make_job(tmp_path)
    workflow.set_split_face_y(job, 3)
    assert workflow.read_job_meta(job)["split_face_y"] == 1.0


def test_create_job_stores_the_style_and_rejects_unknown_ones(tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"v")
    job = workflow.create_job(src, "hi", root=tmp_path, style="hybrid")
    assert workflow.read_job_meta(job)["style"] == "hybrid"
    with pytest.raises(WorkflowError):
        workflow.create_job(src, "hi", root=tmp_path, style="nope")
