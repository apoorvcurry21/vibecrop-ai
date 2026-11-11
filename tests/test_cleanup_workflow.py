import json
from pathlib import Path

import pytest

from src import workflow
from src.cleanup import CleanupError, fmt_bytes, over_budget, purge_job_temp, workdir_usage_bytes


def make_job(root: Path, name="20260101_000000_x") -> Path:
    job = root / "jobs" / name
    (job / "clips").mkdir(parents=True)
    (job / "clips" / "clip_01.mp4").write_bytes(b"x" * 1000)
    for n, size in {"audio.flac": 500, "overlays.ass": 10, "source.mp4": 2000, "final.mp4": 300,
                    "transcript.json": 5, "captions_hinglish.json": 5}.items():
        (job / n).write_bytes(b"y" * size)
    (job / "job.json").write_text("{}", encoding="utf-8")
    (job / "plan.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    return job


def test_purge_removes_temp_media_and_keeps_results(tmp_path):
    job = make_job(tmp_path)
    freed = purge_job_temp(job, root=tmp_path)
    assert freed == 1000 + 500 + 10 + 2000
    assert sorted(p.name for p in job.iterdir()) == [
        "captions_hinglish.json", "final.mp4", "job.json", "plan.json", "transcript.json"]


def test_purge_refuses_anything_outside_the_jobs_folder(tmp_path):
    make_job(tmp_path)
    other = tmp_path / "precious"
    other.mkdir()
    (other / "keep.txt").write_text("hi")
    for bad in (other, tmp_path, tmp_path / "jobs", tmp_path / "jobs" / "missing"):
        with pytest.raises(CleanupError):
            purge_job_temp(bad, root=tmp_path)
    assert (other / "keep.txt").exists()


def test_usage_and_budget(tmp_path):
    make_job(tmp_path)
    assert workdir_usage_bytes(tmp_path) > 3000 and not over_budget(tmp_path)
    assert over_budget(tmp_path, budget_gb=0.000001)
    assert fmt_bytes(512) == "512 B" and fmt_bytes(2048) == "2.0 KB"


# ---- workflow actions (sourcing/rendering mocked) ----
def write_plan(job: Path):
    items = [{"id": i, "asset_type": "video", "timestamp_start": 2.0 * i, "timestamp_end": 2.0 * i + 4,
              "search_query": f"query {i}", "alternative_queries": ["alt"],
              "asset": {"status": "ok", "path": f"clips/clip_0{i}.mp4", "tried_ids": ["aaa"]}} for i in (1, 2)]
    (job / "plan.json").write_text(json.dumps({"video_topic": "t", "items": items}), encoding="utf-8")


def test_regenerate_skips_tried_ids_and_records_failures(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    write_plan(job)
    seen = {}

    def fake_source(job_dir, item, skip_ids=frozenset()):
        seen["skip"] = set(skip_ids)
        return {"status": "ok", "path": "clips/new.mp4", "tried_ids": sorted(set(skip_ids) | {"bbb"})}

    monkeypatch.setattr(workflow.sourcing, "source_item", fake_source)
    workflow.regenerate(job, 1)
    assert seen["skip"] == {"aaa"}
    assert workflow.load_plan(job)["items"][0]["asset"]["tried_ids"] == ["aaa", "bbb"]

    def boom(job_dir, item, skip_ids=frozenset()):
        raise workflow.sourcing.SourcingError("blocked")

    monkeypatch.setattr(workflow.sourcing, "source_item", boom)
    workflow.regenerate(job, 2)
    assert workflow.load_plan(job)["items"][1]["asset"]["status"] == "failed"


def test_set_query_resets_skip_list_and_validates(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    write_plan(job)
    seen = {}
    monkeypatch.setattr(workflow.sourcing, "source_item",
                        lambda j, it, skip_ids=frozenset(): seen.update(q=it["search_query"], skip=set(skip_ids))
                        or {"status": "ok", "path": "clips/n.mp4"})
    workflow.set_query(job, 2, "  rafael   nadal serve ")
    assert seen == {"q": "rafael nadal serve", "skip": set()}
    item = workflow.load_plan(job)["items"][1]
    assert item["search_query"] == "rafael nadal serve" and item["user_edited_query"] is True
    with pytest.raises(workflow.WorkflowError):
        workflow.set_query(job, 2, "   ")
    with pytest.raises(workflow.WorkflowError):
        workflow.set_query(job, 99, "x")


def test_set_enabled_and_render_final_cleans(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    write_plan(job)
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"v")
    workflow.write_job_meta(job, source_video=str(video), language="hi")
    workflow.set_enabled(job, 1, False)
    assert workflow.load_plan(job)["items"][0]["enabled"] is False

    calls = {}

    def fake_render(job_dir, main_video, caption_mode=None, log=print):
        calls.update(mode=caption_mode, video=main_video)
        (job_dir / "final.mp4").write_bytes(b"f")
        return job_dir / "final.mp4"

    monkeypatch.setattr(workflow.rendering, "run", fake_render)
    final, freed = workflow.render_final(job, "hinglish", auto_clean=True, log=lambda m: None, root=tmp_path)
    assert calls["mode"] == "hinglish" and final.name == "final.mp4" and freed > 0
    assert not (job / "clips").exists() and video.exists()  # the user's own video is never deleted
    assert workflow.load_plan(job)["status"] == "approved"


def test_create_job_copy_and_listing(tmp_path):
    src = tmp_path / "in" / "My Clip.mp4"
    src.parent.mkdir()
    src.write_bytes(b"data")
    job = workflow.create_job(src, "hi", copy_into_job=True, root=tmp_path)
    assert job.parent == tmp_path / "jobs" and (job / "source.mp4").read_bytes() == b"data"
    assert workflow.read_job_meta(job)["language"] == "hi"
    assert workflow.list_jobs(tmp_path) == [job]  # unfinished jobs are listed so they can be resumed
    (tmp_path / "jobs" / "stray_folder").mkdir()  # folders the app did not create are ignored
    (job / "plan.json").write_text("{}")
    assert workflow.list_jobs(tmp_path) == [job]
    with pytest.raises(workflow.WorkflowError):
        workflow.create_job(tmp_path / "missing.mp4", "hi", root=tmp_path)
