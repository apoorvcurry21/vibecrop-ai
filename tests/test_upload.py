"""The 'Add picture / video' panel: the same placement rules, undo and cleanup as the chat, with no AI call."""
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from src import workflow
from src.assistant import snapshots, upload

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_job(root: Path) -> Path:
    job = root / "jobs" / "20260101_000000_up"
    (job / "clips").mkdir(parents=True)
    (job / "clips" / "clip_01.mp4").write_bytes(b"clip-1")
    item = {"id": 1, "asset_type": "video", "timestamp_start": 2.0, "timestamp_end": 6.0, "search_query": "q",
            "alternative_queries": [], "spoken_text": "", "english_meaning": "", "reasoning": "", "layout": "overlay",
            "ai_layout": "overlay", "asset": {"status": "ok", "path": "clips/clip_01.mp4", "clip_duration": 9.0}}
    (job / "plan.json").write_text(json.dumps({"video_duration": 30.0, "items": [item]}))
    (job / "job.json").write_text("{}")
    return job


def png(color="blue") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (120, 80), color).save(out, format="PNG")
    return out.getvalue()


def make_video(path: Path, seconds=6) -> bytes:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=duration={seconds}:size=160x120:rate=10", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path.read_bytes()


@needs_ffmpeg
def test_a_picture_becomes_a_slot_at_the_chosen_time_and_layout(tmp_path):
    job = make_job(tmp_path)
    slot, notes = upload.add_upload(job, "me.png", png(), start=10.0, length=4.0, layout="pip", position="bottom-left",
                                    size=0.5)
    assert (slot["id"], slot["start"], slot["end"], slot["layout"], slot["kind"]) == (2, 10.0, 14.0, "pip", "image")
    assert notes == []
    item = next(i for i in workflow.load_plan(job)["items"] if i["id"] == 2)
    assert item["pip"] == {"pos": "bottom-left", "scale": 0.5} and (job / item["asset"]["path"]).is_file()
    assert "credit" not in item["asset"]


@needs_ffmpeg
def test_a_video_is_cut_from_the_second_the_user_chose(tmp_path):
    job = make_job(tmp_path)
    data = make_video(tmp_path / "mine.mp4", seconds=8)
    slot, _ = upload.add_upload(job, "mine.mp4", data, start=12.0, length=3.0, from_second=2.0)
    assert (slot["start"], slot["end"], slot["kind"]) == (12.0, 15.0, "video")
    item = next(i for i in workflow.load_plan(job)["items"] if i["id"] == slot["id"])
    assert item["asset"]["source"] == "upload" and item["asset"]["clip_duration"] >= 3.0


@needs_ffmpeg
def test_overlapping_an_existing_slot_is_allowed_but_reported(tmp_path):
    job = make_job(tmp_path)
    _, notes = upload.add_upload(job, "me.png", png(), start=3.0, length=2.0)
    assert notes and "slot #1" in notes[0]


@needs_ffmpeg
def test_the_chat_undo_button_also_undoes_an_upload(tmp_path):
    job = make_job(tmp_path)
    upload.add_upload(job, "me.png", png(), start=10.0)
    assert snapshots.can_undo(job)
    snapshots.undo(job)
    assert [i["id"] for i in workflow.load_plan(job)["items"]] == [1] and not (job / "clips" / "clip_02.mp4").exists()


def test_bad_uploads_give_a_plain_error_and_change_nothing(tmp_path):
    job = make_job(tmp_path)
    before = (job / "plan.json").read_text()
    for name, data in (("notes.pdf", b"x"), ("empty.png", b""), ("broken.png", b"not an image"),
                       ("broken.mp4", b"not a video")):
        with pytest.raises(upload.UploadError):
            upload.add_upload(job, name, data, start=10.0)
    assert (job / "plan.json").read_text() == before and not snapshots.can_undo(job)


@needs_ffmpeg
def test_a_video_shorter_than_the_slot_is_refused_and_rolled_back(tmp_path):
    job = make_job(tmp_path)
    before = (job / "plan.json").read_text()
    data = make_video(tmp_path / "tiny.mp4", seconds=2)
    with pytest.raises(upload.UploadError, match="only has"):
        upload.add_upload(job, "tiny.mp4", data, start=10.0, length=5.0)
    assert (job / "plan.json").read_text() == before and not (job / "clips" / "clip_02.mp4").exists()
