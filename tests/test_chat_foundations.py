"""Foundations of the chat assistant: pip layout, text overlays, image stills, add/replace/delete slots, undo."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src import workflow
from src.assistant import snapshots
from src.cleanup import purge_job_temp
from src.rendering.ass import header, normalize_text_style, text_events
from src.rendering.compose import (OUT_H, OUT_W, PIP_POSITIONS, Overlay, build_filtergraph, normalize_pip,
                                   pip_geometry, render)
from src.sourcing.stills import StillError, image_to_clip, normalize_image
from src.workflow import WorkflowError

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def pixel(video: Path, t: float, x: int, y: int):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1",
         "-vf", f"format=rgb24,crop=1:1:{x}:{y}", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return out[0], out[1], out[2]


def near(rgb, target, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


def make_job(tmp_path: Path) -> Path:
    """Two 4 s slots (2-6, 14-18) in a 30 s video, so there are free gaps between and after them."""
    job = tmp_path / "jobs" / "j"
    (job / "clips").mkdir(parents=True)
    items = []
    for i, start in ((1, 2.0), (2, 14.0)):
        (job / "clips" / f"clip_{i:02d}.mp4").write_bytes(f"clip-{i}".encode())
        items.append({"id": i, "asset_type": "video", "timestamp_start": start, "timestamp_end": start + 4,
                      "search_query": f"q{i}", "alternative_queries": [], "spoken_text": "", "english_meaning": "",
                      "reasoning": "", "layout": "overlay", "ai_layout": "overlay",
                      "asset": {"status": "ok", "path": f"clips/clip_{i:02d}.mp4", "clip_duration": 9.0}})
    (job / "plan.json").write_text(json.dumps({"video_duration": 30.0, "items": items}), encoding="utf-8")
    (job / "job.json").write_text("{}", encoding="utf-8")
    return job


def new_clip(tmp_path: Path, name="new.mp4", content=b"new") -> Path:
    f = tmp_path / name
    f.write_bytes(content)
    return f


def ids(job):
    return [i["id"] for i in workflow.load_plan(job)["items"]]


# ----------------------------------------------------------------------------- pip geometry / filtergraph
def test_pip_geometry_stays_inside_the_frame_for_every_position_and_size():
    for pos in PIP_POSITIONS:
        for scale in (0.3, 0.6, 0.95):
            w, h, x, y = pip_geometry({"pos": pos, "scale": scale})
            assert w % 2 == 0 and h % 2 == 0
            assert 0 <= x and x + w <= OUT_W and 0 <= y and y + h <= OUT_H


def test_pip_geometry_corners_and_defaults():
    w, h, x, y = pip_geometry({"pos": "top-right", "scale": 0.5})
    assert (w, x) == (540, OUT_W - 540 - 50) and y == 190
    assert pip_geometry({"pos": "center"})[2] == (OUT_W - pip_geometry({"pos": "center"})[0]) // 2
    assert normalize_pip({"pos": "nowhere", "scale": 99}) == {"pos": "top-right", "scale": 0.95}
    assert normalize_pip(None) == {"pos": "top-right", "scale": 0.6}
    assert normalize_pip({"scale": "x"})["scale"] == 0.6


def test_pip_filtergraph_overlays_a_small_bordered_box_at_the_chosen_spot():
    g = build_filtergraph([Overlay(Path("c.mp4"), 2.0, 5.0, layout="pip", pip={"pos": "bottom-left", "scale": 0.5})])
    w, h, x, y = pip_geometry({"pos": "bottom-left", "scale": 0.5})
    assert f"overlay={x}:{y}:enable='between(t,2.000,5.000)'" in g and "drawbox" in g
    assert f"{w}:{h}" in g  # the clip is framed into the small box, not the whole frame


@needs_ffmpeg
def test_pip_renders_a_small_box_over_the_speaker_for_its_slot_only(tmp_path):
    main = tmp_path / "main.mp4"
    ff("-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=30:d=6", "-pix_fmt", "yuv420p", str(main))
    clip = tmp_path / "clip.mp4"
    ff("-f", "lavfi", "-i", "color=c=blue:s=1920x1080:r=25:d=5", "-pix_fmt", "yuv420p", str(clip))
    pip = {"pos": "top-left", "scale": 0.5}
    out = render(main, [Overlay(clip, 2.0, 4.0, layout="pip", pip=pip)], tmp_path / "out.mp4")
    w, h, x, y = pip_geometry(pip)
    assert near(pixel(out, 3.0, x + w // 2, y + h // 2), (0, 0, 255))  # inside the box: the clip
    assert near(pixel(out, 3.0, 1000, 1800), (255, 0, 0))  # everywhere else: still the speaker
    assert near(pixel(out, 1.0, x + w // 2, y + h // 2), (255, 0, 0))  # before the slot: speaker
    assert near(pixel(out, 5.0, x + w // 2, y + h // 2), (255, 0, 0))  # after it: speaker again


# ----------------------------------------------------------------------------- text overlays
def test_text_style_validation_falls_back_to_defaults():
    assert normalize_text_style({}) == {"pos": "top", "size": "large", "color": "white", "font": "bold"}
    s = normalize_text_style({"pos": "bottom-left", "size": 9999, "color": "#ff8800", "font": "impact"})
    assert s == {"pos": "bottom-left", "size": 220, "color": "FF8800", "font": "impact"}
    assert normalize_text_style({"pos": "x", "size": "huge", "color": "nope", "font": "comic"})["color"] == "white"


def test_text_events_have_timing_position_size_and_bgr_colour():
    ev = text_events([{"text": "GOAT {x}", "start": 3.0, "end": 5.5, "pos": "bottom", "size": "xl", "color": "red"}])
    assert len(ev) == 1
    line = ev[0]
    assert "0:00:03.00,0:00:05.50,Text" in line and "\\an2\\pos(540,1260)" in line and "\\fs140" in line
    assert "\\c&H00303BFF&" in line  # FF3B30 (red) written as BGR
    assert "GOAT (x)" in line  # braces cannot inject ASS tags
    assert text_events([{"text": "  ", "start": 1, "end": 2}, {"text": "late", "start": 5, "end": 4}]) == []
    assert "Style: Text," in header()


def test_add_update_remove_text(tmp_path):
    job = make_job(tmp_path)
    t = workflow.add_text(job, "  Kohli   is GOAT ", 3, 6, pos="top", color="yellow")
    assert t["id"] == 1 and t["text"] == "Kohli is GOAT" and t["color"] == "yellow"
    workflow.update_text(job, 1, text="New", end=8, size="small")
    saved = workflow.load_plan(job)["texts"][0]
    assert (saved["text"], saved["end"], saved["size"], saved["color"]) == ("New", 8.0, "small", "yellow")
    assert workflow.add_text(job, "two", 1, 2)["id"] == 2
    workflow.remove_text(job, 1)
    assert [x["id"] for x in workflow.load_plan(job)["texts"]] == [2]
    for bad in (("", 1, 2), ("x", 5, 5.2), ("x", 1, 31), ("y" * 500, 1, 2)):
        with pytest.raises(WorkflowError):
            workflow.add_text(job, *bad)
    with pytest.raises(WorkflowError, match="No text overlay"):
        workflow.remove_text(job, 99)


# ----------------------------------------------------------------------------- image stills
@needs_ffmpeg
def test_image_becomes_an_even_sized_still_clip(tmp_path):
    from PIL import Image

    img = tmp_path / "pic.png"
    Image.new("RGBA", (301, 201), (0, 128, 255, 255)).save(img)
    out = tmp_path / "clips" / "clip_09.mp4"
    info = image_to_clip(img, out, seconds=2.0)
    assert out.is_file() and info["width"] % 2 == 0 and info["height"] % 2 == 0
    assert not list(out.parent.glob("*_src.png"))  # the temp PNG is cleaned up
    got = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                         capture_output=True, text=True).stdout.strip()
    assert abs(float(got) - 2.0) < 0.2


def test_non_images_are_rejected(tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(StillError):
        normalize_image(bad, tmp_path / "o.png")


def test_big_images_are_capped(tmp_path):
    from PIL import Image

    big = tmp_path / "big.jpg"
    Image.new("RGB", (5000, 3001), "white").save(big)
    w, h = normalize_image(big, tmp_path / "o.png")
    assert max(w, h) <= 2160 and w % 2 == 0 and h % 2 == 0


# ----------------------------------------------------------------------------- add / replace / delete slots
def test_add_slot_inserts_in_time_order_with_a_fresh_id_and_moves_the_clip(tmp_path):
    job = make_job(tmp_path)
    item = workflow.add_slot(job, 8.0, 11.0, new_clip(tmp_path), {"clip_duration": 10.0, "credit": "Image: x.com"},
                             layout="pip", pip={"pos": "bottom", "scale": 0.7}, spoken_text="kohli")
    assert item["id"] == 3 and item["asset"]["path"] == "clips/clip_03.mp4" and item["asset_type"] == "image"
    assert (job / "clips" / "clip_03.mp4").read_bytes() == b"new"
    assert ids(job) == [1, 3, 2]  # list is in time order, ids are never reused
    assert item["pip"] == {"pos": "bottom", "scale": 0.7} and item["layout"] == "pip"


@pytest.mark.parametrize("start,end,message", [
    (6.0, 9.0, "overlaps slot #1"),   # touches slot 1 (needs a gap after it)
    (0.2, 3.0, "cannot start before"),  # speaker's intro
    (8.0, 9.0, "at least"),
    (28.0, 31.0, "only 30"),
    (12.0, 15.0, "overlaps slot #2"),
])
def test_add_slot_rejects_bad_ranges_and_changes_nothing(tmp_path, start, end, message):
    job = make_job(tmp_path)
    clip = new_clip(tmp_path)
    with pytest.raises(WorkflowError, match=message):
        workflow.add_slot(job, start, end, clip, {"clip_duration": 10.0})
    assert ids(job) == [1, 2] and clip.is_file() and not (job / "clips" / "clip_03.mp4").exists()


def test_add_slot_rejects_a_clip_shorter_than_the_slot(tmp_path):
    job = make_job(tmp_path)
    with pytest.raises(WorkflowError, match="only 3.0s"):
        workflow.add_slot(job, 8.0, 12.0, new_clip(tmp_path), {"clip_duration": 3.0})
    assert ids(job) == [1, 2]


def test_removed_slots_still_block_until_deleted(tmp_path):
    job = make_job(tmp_path)
    workflow.set_enabled(job, 1, False)
    with pytest.raises(WorkflowError, match="removed, but it still holds"):
        workflow.add_slot(job, 3.0, 5.0, new_clip(tmp_path), {"clip_duration": 10.0})
    workflow.delete_slot(job, 1)
    assert not (job / "clips" / "clip_01.mp4").exists() and ids(job) == [2]
    assert workflow.add_slot(job, 3.0, 5.0, new_clip(tmp_path), {"clip_duration": 10.0})["id"] == 3  # id 1 not reused


def test_free_gaps(tmp_path):
    job = make_job(tmp_path)
    assert workflow.free_gaps(job) == [(6.1, 13.9), (18.1, 30.0)]  # nothing fits before slot 1 (intro 1s + gap)
    workflow.add_slot(job, 7.0, 12.0, new_clip(tmp_path), {"clip_duration": 10.0})
    assert workflow.free_gaps(job) == [(12.1, 13.9), (18.1, 30.0)]  # the 0.8 s sliver before the new slot is too small


def test_set_asset_replaces_the_clip_and_resets_framing_but_keeps_timing_and_layout(tmp_path):
    job = make_job(tmp_path)
    workflow.set_layout(job, 2, "split")
    workflow.set_framing(job, 2, {"start": {"z": 2}, "end": {"z": 2}})
    item = workflow.set_asset(job, 2, new_clip(tmp_path, "pic.mp4", b"pic"),
                              {"clip_duration": 10.0, "title": "my pic"}, asset_type="image")
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"pic"
    assert item["asset_type"] == "image" and item["layout"] == "split" and item["timestamp_start"] == 14.0
    assert item["framing"]["start"]["z"] == 1.0 and item["search_query"] == "my pic"
    with pytest.raises(WorkflowError, match="only 2.0s"):
        workflow.set_asset(job, 2, new_clip(tmp_path, "s.mp4"), {"clip_duration": 2.0})
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"pic"  # failed replacement changed nothing


def test_set_layout_and_set_pip(tmp_path):
    job = make_job(tmp_path)
    assert workflow.set_layout(job, 1, "pip")["pip"] == {"pos": "top-right", "scale": 0.6}
    item = workflow.set_pip(job, 2, pos="bottom-right", scale=0.4)
    assert item["layout"] == "pip" and item["pip"] == {"pos": "bottom-right", "scale": 0.4}
    assert workflow.set_pip(job, 2, scale=0.8)["pip"] == {"pos": "bottom-right", "scale": 0.8}
    with pytest.raises(WorkflowError):
        workflow.set_pip(job, 2, pos="moon")
    with pytest.raises(WorkflowError):
        workflow.set_layout(job, 2, "diagonal")


def test_apply_style_keeps_pip_slots(tmp_path):
    job = make_job(tmp_path)
    workflow.set_pip(job, 1, pos="top")
    workflow.apply_style(job, "split")
    layouts = {i["id"]: i["layout"] for i in workflow.load_plan(job)["items"]}
    assert layouts == {1: "pip", 2: "split"}


# ----------------------------------------------------------------------------- undo
def test_undo_restores_plan_and_overwritten_clip(tmp_path):
    job = make_job(tmp_path)
    snapshots.take(job, "replace slot 2", clip_ids=[2])
    workflow.set_asset(job, 2, new_clip(tmp_path, "p.mp4", b"pic"), {"clip_duration": 10.0}, asset_type="image")
    assert snapshots.last_label(job) == "replace slot 2" and snapshots.can_undo(job)
    assert snapshots.undo(job) == "replace slot 2"
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"clip-2"
    assert workflow.load_plan(job)["items"][1]["asset_type"] == "video"
    assert not snapshots.can_undo(job) and snapshots.undo(job) is None


def test_undo_removes_clips_created_by_the_change(tmp_path):
    job = make_job(tmp_path)
    snapshots.take(job, "add slot")
    workflow.add_slot(job, 8.0, 11.0, new_clip(tmp_path), {"clip_duration": 10.0})
    assert (job / "clips" / "clip_03.mp4").exists()
    snapshots.undo(job)
    assert not (job / "clips" / "clip_03.mp4").exists() and ids(job) == [1, 2]


def test_undo_restores_a_deleted_slot_and_its_clip_and_texts(tmp_path):
    job = make_job(tmp_path)
    workflow.add_text(job, "hello", 3, 5)
    snapshots.take(job, "delete slot 1", clip_ids=[1])
    workflow.delete_slot(job, 1)
    workflow.remove_text(job, 1)
    snapshots.undo(job)
    assert ids(job) == [1, 2] and (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-1"
    assert workflow.load_plan(job)["texts"][0]["text"] == "hello"


def test_undo_reverts_a_swap(tmp_path):
    job = make_job(tmp_path)
    snapshots.take(job, "swap", clip_ids=[1, 2])
    workflow.swap_slots(job, 1, 2)
    assert (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-2"
    snapshots.undo(job)
    assert (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-1"
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"clip-2"
    assert workflow.load_plan(job)["items"][0]["asset"]["path"] == "clips/clip_01.mp4"


def test_snapshots_keep_only_the_newest_ten_and_undo_goes_in_order(tmp_path):
    job = make_job(tmp_path)
    for n in range(13):
        snapshots.take(job, f"edit {n}")
    assert len(snapshots._numbered(job)) == snapshots.MAX_SNAPSHOTS
    assert snapshots.undo(job) == "edit 12" and snapshots.undo(job) == "edit 11"


def test_discard_latest_drops_a_snapshot_without_restoring(tmp_path):
    job = make_job(tmp_path)
    snapshots.take(job, "a")
    workflow.set_layout(job, 1, "split")
    snapshots.discard_latest(job)
    assert not snapshots.can_undo(job)
    assert workflow.load_plan(job)["items"][0]["layout"] == "split"


# ----------------------------------------------------------------------------- cleanup
def test_cleanup_keeps_chat_json_but_removes_assets_and_history(tmp_path):
    job = make_job(tmp_path)
    (job / "chat.json").write_text("[]", encoding="utf-8")
    (job / "assets").mkdir()
    (job / "assets" / "att_1.png").write_bytes(b"x" * 100)
    snapshots.take(job, "x", clip_ids=[1])
    purge_job_temp(job, root=tmp_path)
    assert sorted(p.name for p in job.iterdir()) == ["chat.json", "job.json", "plan.json"]
