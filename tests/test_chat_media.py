"""Chat media: YouTube clips, attached files and the place_* tools.
Network, Gemini and yt-dlp are mocked; FFmpeg runs for real on synthetic media."""
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors, types
from PIL import Image

from src import workflow
from src.assistant import assets, media, snapshots, tools, ytclip
from src.assistant.tools import Attachment, ToolContext, run_tool

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_job(root: Path) -> Path:
    """Two 4 s slots (2-6 and 14-18) in a 30 s video; words 'virat koli' at 7 s."""
    job = root / "jobs" / "20260101_000000_media"
    (job / "clips").mkdir(parents=True)
    items = []
    for i, start in ((1, 2.0), (2, 14.0)):
        (job / "clips" / f"clip_{i:02d}.mp4").write_bytes(f"clip-{i}".encode())
        items.append({"id": i, "asset_type": "video", "timestamp_start": start, "timestamp_end": start + 4,
                      "search_query": f"q{i}", "alternative_queries": [], "spoken_text": "", "english_meaning": "",
                      "reasoning": "", "layout": "overlay", "ai_layout": "overlay",
                      "asset": {"status": "ok", "path": f"clips/clip_{i:02d}.mp4", "clip_duration": 9.0}})
    (job / "plan.json").write_text(json.dumps({"video_topic": "Cricket", "video_duration": 30.0, "items": items}))
    (job / "job.json").write_text("{}")
    (job / "transcript.json").write_text(json.dumps({"segments": [], "words": [
        {"word": w, "start": s, "end": s + 0.3} for w, s in (("virat", 7.0), ("koli", 7.4), ("is", 7.8), ("great", 8.0))]}))
    return job


def png_bytes(color="blue", size=(120, 80)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def slot(job, i):
    return next(x for x in workflow.load_plan(job)["items"] if x["id"] == i)


def ctx_for(job, *atts):
    return ToolContext(job=job, attachments={a.id: a for a in atts})


# ----------------------------------------------------------------------------- resolving references
def test_resolve_image_finds_the_attached_picture(tmp_path):
    job = make_job(tmp_path)
    att = assets.save_attachment(job, "mine.png", png_bytes())
    path, extra = media.resolve_image(job, ctx_for(job, att), "att_1")
    assert path == att.path and extra == {"origin": "attachment", "title": "mine.png"}
    # a fresh chat turn (new ToolContext) still finds it on disk
    assert media.resolve_image(job, ctx_for(job), "att_1")[0] == att.path


@pytest.mark.parametrize("ref", ["att_9", "res_7", "gen_4", "url_1", "../etc/passwd", "file:///c:/x.png",
                                 "https://example.com/a.png", "att_1/../x", ""])
def test_resolve_image_rejects_everything_but_attachments(tmp_path, ref):
    """Web links and res_/gen_ ids went away with web search and AI generation: only attachments are accepted."""
    job = make_job(tmp_path)
    with pytest.raises(tools.ToolError):
        media.resolve_image(job, ctx_for(job), ref)


def test_a_video_attachment_is_not_a_picture(tmp_path):
    job = make_job(tmp_path)
    vid = assets.save_attachment(job, "v.mp4", b"x")
    with pytest.raises(tools.ToolError, match="not an attached picture"):
        media.resolve_image(job, ctx_for(job, vid), "att_1")


# ----------------------------------------------------------------------------- YouTube helpers
@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
    ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=x", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://evil.com/watch?v=dQw4w9WgXcQ", None), ("https://youtube.com.evil.com/watch?v=dQw4w9WgXcQ", None),
    ("https://www.youtube.com/watch?v=short", None), ("not a url", None), ("", None),
])
def test_video_id_only_accepts_real_youtube_links(url, vid):
    assert ytclip.video_id(url) == vid


@pytest.mark.parametrize("raw,secs", [("2:15", 135), ("1:02:03", 3723), ("90", 90), (90, 90), ("1m30s", 90), ("45s", 45),
                                      ("1h", 3600), ("12.5", 12.5), (None, None), ("", None), ("abc", None),
                                      ("1:2:3:4", None), (-5, None), (True, None)])
def test_parse_time(raw, secs):
    assert ytclip.parse_time(raw) == secs


def fake_info(duration=300.0):
    return {"id": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "title": "Kohli football",
            "channel": "Sports Ch", "duration": duration}


def test_find_window_asks_gemini_to_watch_the_video_and_clamps_the_answer():
    seen = {}

    class Models:
        def generate_content(self, model, contents, config):
            seen["parts"] = contents[0].parts
            return SimpleNamespace(text=json.dumps({"start": 299.0, "reason": "Kohli juggles the ball"}))

    start, why = ytclip.find_window(fake_info(), "Kohli playing football", 5.0, client=SimpleNamespace(models=Models()),
                                    models=["m"])
    assert start == 294.5 and why == "Kohli juggles the ball"  # clamped so the whole clip stays inside the video
    assert seen["parts"][0].file_data.file_uri == fake_info()["url"] and "Kohli playing football" in seen["parts"][1].text


def test_find_window_refuses_very_long_videos_and_reports_model_trouble():
    with pytest.raises(ytclip.YtError, match="very long"):
        ytclip.find_window(fake_info(5 * 3600), "x", 5.0, client=object(), models=["m"])

    class Bad:
        def generate_content(self, **kw):
            raise errors.ClientError(403, {"error": {"message": "nope"}})

    with pytest.raises(ytclip.YtError, match="Tell me a time"):
        ytclip.find_window(fake_info(), "x", 5.0, client=SimpleNamespace(models=Bad()), models=["m"], sleep=lambda s: None)


def test_video_info_rejects_non_youtube_links_without_touching_the_network():
    with pytest.raises(ytclip.YtError, match="doesn't look like a YouTube link"):
        ytclip.video_info("https://evil.com/x")


# ----------------------------------------------------------------------------- place_* tools
@needs_ffmpeg
def test_place_attached_picture_as_a_new_slot_where_the_user_said(tmp_path):
    job = make_job(tmp_path)
    att = assets.save_attachment(job, "kohli.png", png_bytes("green", (300, 200)))
    ctx = ctx_for(job, att)
    out = run_tool(ctx, "place_image", {"image": "att_1", "start": 6.8, "layout": "pip", "position": "bottom-left",
                                        "size": 0.5})
    assert out["ok"] and out["slot"]["id"] == 3 and out["slot"]["layout"] == "pip" and ctx.changed
    item = slot(job, 3)
    assert (item["timestamp_start"], item["timestamp_end"], item["asset_type"]) == (6.8, 9.8, "image")
    assert item["pip"] == {"pos": "bottom-left", "scale": 0.5} and item["spoken_text"] == "virat koli is great"
    assert item["asset"]["status"] == "ok" and (job / item["asset"]["path"]).is_file()
    assert "credit" not in item["asset"]  # the user's own picture needs no credit line
    assert not list((job / "assets" / "_work").glob("*.mp4"))  # the temp clip was moved into clips/
    run_tool(ctx, "undo", {})
    assert [i["id"] for i in workflow.load_plan(job)["items"]] == [1, 2] and not (job / "clips" / "clip_03.mp4").exists()


@needs_ffmpeg
def test_place_image_into_an_existing_slot_keeps_its_time_and_layout(tmp_path):
    job = make_job(tmp_path)
    att = assets.save_attachment(job, "p.jpg", png_bytes())
    workflow.set_layout(job, 2, "split")
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "slot_id": 2})
    assert out["ok"] and (out["slot"]["start"], out["slot"]["layout"], out["slot"]["kind"]) == (14.0, "split", "image")
    run_tool(ctx_for(job), "undo", {})
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"clip-2" and slot(job, 2)["asset_type"] == "video"


@needs_ffmpeg
def test_the_chat_can_place_media_anywhere_even_at_the_very_start_or_over_other_slots(tmp_path):
    job = make_job(tmp_path)
    att = assets.save_attachment(job, "p.png", png_bytes())
    # the case that was refused before: the first 2.4 seconds, where the speaker "owns the opening"
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 0.0, "end": 2.4, "layout": "pip"})
    assert out["ok"] and (out["slot"]["start"], out["slot"]["end"]) == (0.0, 2.4)
    assert out["notes"][0].startswith("It overlaps slot #1")  # 0-2.4 s runs into slot 1 (2-6 s): allowed, but said
    # fully on top of an existing slot, and back to back with no gap
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 3.0, "end": 5.0})
    assert out["ok"] and "slot #1" in out["notes"][0]
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 6.0, "end": 8.0})
    assert out["ok"] and "notes" not in out  # merely touching slot 1 (2-6 s) at 6.0 is not an overlap, and no gap is needed
    assert sorted(i["id"] for i in workflow.load_plan(job)["items"]) == [1, 2, 3, 4, 5]
    assert [i["timestamp_start"] for i in workflow.load_plan(job)["items"]] == sorted(
        i["timestamp_start"] for i in workflow.load_plan(job)["items"])  # still kept in timeline order


@needs_ffmpeg
def test_free_placement_trims_a_range_that_runs_past_the_end_and_rejects_only_the_pointless(tmp_path):
    job = make_job(tmp_path)  # a 30 s video
    att = assets.save_attachment(job, "p.png", png_bytes())
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 28.0, "end": 35.0})
    assert out["ok"] and (out["slot"]["start"], out["slot"]["end"]) == (28.0, 30.0)  # trimmed to the video's end
    before = (job / "plan.json").read_text()
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 29.9, "end": 40.0})
    assert "shorter than 0.3s" in out["error"] and (job / "plan.json").read_text() == before
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 5.0, "end": 5.0})
    assert "shorter than 0.3s" in out["error"]
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": -3.0, "end": 1.0})
    assert out["ok"] and out["slot"]["start"] == 0.0  # a negative start just means 0
    assert "No B-roll with id 99" in run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "slot_id": 99})["error"]
    assert "Unknown layout" in run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 20, "layout": "x"})["error"]


def test_the_chat_can_move_a_slot_anywhere_but_the_editors_own_rules_are_unchanged(tmp_path):
    job = make_job(tmp_path)
    out = run_tool(ctx_for(job), "set_timing", {"slot_id": 1, "start": 0.0, "end": 9.0})  # over the 1 s intro + into slot 2's gap
    assert out["ok"] and out["slot"]["start"] == 0.0
    out = run_tool(ctx_for(job), "set_timing", {"slot_id": 2, "start": 8.0, "end": 12.0})  # right on top of slot 1
    assert out["ok"] and "slot #1" in out["notes"][0]
    out = run_tool(ctx_for(job), "set_timing", {"slot_id": 2, "start": 25.0, "end": 99.0})  # runs past the video: trimmed
    assert out["ok"] and out["slot"]["end"] == 30.0
    # the slider/editor path (workflow.set_timing without free=True) keeps its guard rails
    with pytest.raises(workflow.WorkflowError, match="cannot start before"):
        workflow.set_timing(job, 1, 0.2, 5.0)
    with pytest.raises(workflow.WorkflowError, match="at least"):
        workflow.set_timing(job, 1, 2.0, 2.5)
    with pytest.raises(workflow.WorkflowError, match="overlaps"):
        workflow.add_slot(job, 3.0, 5.0, Path("x"), {"clip_duration": 10.0})


def test_a_clip_still_cannot_be_longer_than_it_has_footage(tmp_path):
    job = make_job(tmp_path)  # clips have 9 s of footage
    out = run_tool(ctx_for(job), "set_timing", {"slot_id": 1, "start": 0.0, "end": 12.0})
    assert "only 9.0s" in out["error"] and slot(job, 1)["timestamp_start"] == 2.0  # unchanged


def test_a_failing_image_leaves_nothing_behind(tmp_path):
    job = make_job(tmp_path)
    bad = assets.save_attachment(job, "broken.png", b"not really a png")
    before = (job / "plan.json").read_text()
    out = run_tool(ctx_for(job, bad), "place_image", {"image": "att_1", "start": 20.0})
    assert "not a readable image" in out["error"] and (job / "plan.json").read_text() == before


def make_video(path: Path, seconds=8):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=s=640x360:r=25:d={seconds}", "-f", "lavfi", "-i", f"sine=d={seconds}", "-shortest",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


@needs_ffmpeg
def test_attached_video_is_cut_silent_and_placed(tmp_path):
    job = make_job(tmp_path)
    src = tmp_path / "in.mp4"
    make_video(src, 8)
    att = assets.save_attachment(job, "match.mp4", src.read_bytes())
    ctx = ctx_for(job, att)
    out = run_tool(ctx, "place_video_file", {"video": "att_1", "from_second": 2, "start": 20.0, "end": 24.0})
    assert out["ok"] and out["slot"]["kind"] == "video"
    clip = job / slot(job, 3)["asset"]["path"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(clip)],
                           capture_output=True, text=True).stdout.split()
    assert probe == ["video"]  # no audio stream
    too_far = run_tool(ctx, "place_video_file", {"video": "att_1", "from_second": 7, "start": 25.0, "end": 29.0})
    assert "only" in too_far["error"] and [i["id"] for i in workflow.load_plan(job)["items"]] == [1, 2, 3]
    assert "not an attached video" in run_tool(ctx_for(job), "place_video_file", {"video": "att_5", "start": 25})["error"]


def patch_youtube(monkeypatch, tmp_path, picked=None):
    calls = {}
    monkeypatch.setattr(ytclip, "video_info", lambda url: calls.setdefault("info", fake_info()) and fake_info())

    def find(info, about, length, **kw):
        calls["about"] = about
        return 61.0, "Kohli on the ball"

    def cut(info, start, length, stem):
        calls["cut"] = (start, length)
        f = Path(str(stem) + ".mp4")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"yt-clip")
        return f, length + 4.0

    monkeypatch.setattr(ytclip, "find_window", find)
    monkeypatch.setattr(ytclip, "cut_clip", cut)
    return calls


def test_youtube_clip_uses_the_users_timestamp_without_asking_gemini(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    calls = patch_youtube(monkeypatch, tmp_path)
    out = run_tool(ctx_for(job), "place_youtube_clip", {"url": "https://youtu.be/dQw4w9WgXcQ", "from_time": "2:15",
                                                         "start": 20.0, "end": 25.0})
    assert out["ok"] and calls["cut"] == (135.0, 5.0) and "about" not in calls
    asset = slot(job, 3)["asset"]
    assert (asset["source"], asset["channel"], asset["window_start"], asset["clip_duration"]) == ("youtube", "Sports Ch", 135.0, 9.0)
    assert out["from_video"]["why"] == "the time you gave" and (job / asset["path"]).read_bytes() == b"yt-clip"


def test_youtube_clip_without_a_time_lets_gemini_pick_the_moment(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    calls = patch_youtube(monkeypatch, tmp_path)
    out = run_tool(ctx_for(job), "place_youtube_clip", {"url": "https://youtu.be/dQw4w9WgXcQ", "about": "Kohli playing football",
                                                         "slot_id": 1})
    assert out["ok"] and calls["about"] == "Kohli playing football" and calls["cut"] == (61.0, 4.0)  # slot 1 is 4 s
    assert out["from_video"] == {"title": "Kohli football", "channel": "Sports Ch", "starts_at": 61.0, "why": "Kohli on the ball"}
    assert slot(job, 1)["asset"]["window_start"] == 61.0 and slot(job, 1)["timestamp_start"] == 2.0
    run_tool(ctx_for(job), "undo", {})
    assert (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-1"


def test_youtube_clip_errors_are_plain_and_change_nothing(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    before = (job / "plan.json").read_text()
    out = run_tool(ctx_for(job), "place_youtube_clip", {"url": "https://evil.com/x", "start": 20.0})
    assert "doesn't look like a YouTube link" in out["error"]
    patch_youtube(monkeypatch, tmp_path)
    out = run_tool(ctx_for(job), "place_youtube_clip", {"url": "https://youtu.be/dQw4w9WgXcQ", "start": 20.0})
    assert "what to look for" in out["error"]
    out = run_tool(ctx_for(job), "place_youtube_clip", {"url": "https://youtu.be/dQw4w9WgXcQ", "from_time": "1:00",
                                                         "start": 5.0, "end": 25.0})
    assert "under 12 seconds" in out["error"]  # a 20 s B-roll clip is too long; timing is not a limit
    assert (job / "plan.json").read_text() == before


def test_every_media_tool_is_registered_and_declared():
    for name in ("place_image", "place_video_file", "place_youtube_clip"):
        assert name in tools.REGISTRY
    assert "search_web_images" not in tools.REGISTRY and "generate_image" not in tools.REGISTRY  # removed on purpose
    assert {d.name for d in tools.declarations()} == set(tools.REGISTRY)
    for name in ("place_image", "place_video_file", "place_youtube_clip"):
        assert tools.REGISTRY[name].mutating


def test_cleanup_removes_assets_but_the_chat_survives(tmp_path):
    from src.cleanup import purge_job_temp

    job = make_job(tmp_path)
    assets.save_attachment(job, "p.png", png_bytes())
    (job / "chat.json").write_text("[]")
    purge_job_temp(job, root=tmp_path)
    assert not (job / "assets").exists() and (job / "chat.json").is_file()
