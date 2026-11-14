"""Chat media: safe fetching, web image search, image generation, YouTube clips, attached files and the place_* tools.
Network, DDGS, Gemini and yt-dlp are mocked; FFmpeg runs for real on synthetic media."""
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
from src.assistant import assets, fetch, media, snapshots, tools, ytclip
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


# ----------------------------------------------------------------------------- safe fetching
@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254", "::1", "0.0.0.0"])
def test_private_and_local_addresses_are_refused(monkeypatch, ip):
    fam = 10 if ":" in ip else 2
    monkeypatch.setattr(fetch.socket, "getaddrinfo", lambda *a, **k: [(fam, 1, 6, "", (ip, 80))])
    with pytest.raises(fetch.FetchError, match="private or local"):
        fetch.check_url("http://evil.example/pic.png")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.com/a.png", "javascript:alert(1)", "", "http://"])
def test_only_http_links_are_accepted(url):
    with pytest.raises(fetch.FetchError):
        fetch.check_url(url)


def test_a_public_address_passes(monkeypatch):
    monkeypatch.setattr(fetch.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    assert fetch.check_url("https://example.com/a.png") == "https://example.com/a.png"


class FakeResp:
    def __init__(self, status=200, headers=None, body=b"", redirect=None):
        self.status_code, self.body = status, body
        self.headers = {"Content-Type": "image/png", **(headers or {})}
        if redirect:
            self.headers["Location"] = redirect
        self.is_redirect = bool(redirect)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]


def patch_requests(monkeypatch, responses):
    import requests

    seen = []
    monkeypatch.setattr(fetch.socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", (
        "10.0.0.1" if host == "internal.local" else "93.184.216.34", 80))])

    def fake_get(url, **kw):
        seen.append(url)
        assert kw["allow_redirects"] is False  # redirects are followed by hand, each one re-checked
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)
    return seen


def test_fetch_returns_the_bytes_and_follows_redirects_safely(monkeypatch):
    seen = patch_requests(monkeypatch, [FakeResp(302, redirect="/real.png"), FakeResp(body=b"PNGDATA")])
    assert fetch.fetch_image("http://example.com/a") == b"PNGDATA"
    assert seen == ["http://example.com/a", "http://example.com/real.png"]


def test_a_redirect_to_a_private_host_is_blocked(monkeypatch):
    patch_requests(monkeypatch, [FakeResp(302, redirect="http://internal.local/secret.png")])
    with pytest.raises(fetch.FetchError, match="private or local"):
        fetch.fetch_image("http://example.com/a")


def test_fetch_rejects_pages_big_files_errors_and_redirect_loops(monkeypatch):
    patch_requests(monkeypatch, [FakeResp(headers={"Content-Type": "text/html"}, body=b"<html>")])
    with pytest.raises(fetch.FetchError, match="not a picture"):
        fetch.fetch_image("http://example.com/page")
    patch_requests(monkeypatch, [FakeResp(headers={"Content-Length": str(99 * 1024 * 1024)})])
    with pytest.raises(fetch.FetchError, match="too big"):
        fetch.fetch_image("http://example.com/huge.png")
    patch_requests(monkeypatch, [FakeResp(body=b"x" * 5000)])
    with pytest.raises(fetch.FetchError, match="too big"):
        fetch.fetch_image("http://example.com/lying.png", max_bytes=1000)  # no Content-Length: counted while reading
    patch_requests(monkeypatch, [FakeResp(404)])
    with pytest.raises(fetch.FetchError, match="404"):
        fetch.fetch_image("http://example.com/gone.png")
    patch_requests(monkeypatch, [FakeResp(302, redirect="/a") for _ in range(10)])
    with pytest.raises(fetch.FetchError, match="Too many redirects"):
        fetch.fetch_image("http://example.com/loop")


# ----------------------------------------------------------------------------- web image search
def patch_search(monkeypatch, rows, fetch_ok=True):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", lambda **kw: SimpleNamespace(images=lambda q, **k: rows))
    monkeypatch.setattr(fetch, "fetch_image", lambda url, max_bytes=0: png_bytes() if fetch_ok and "bad" not in url
                        else (_ for _ in ()).throw(fetch.FetchError("dead")))


ROWS = [{"title": f"Kohli {i}", "image": f"https://img{i}.example/k{i}.jpg", "thumbnail": f"https://t{i}.example/t.jpg",
         "url": f"https://www.site{i}.com/page", "width": 800, "height": 600} for i in range(1, 5)]


def test_search_downloads_thumbnails_and_remembers_the_full_urls(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    patch_search(monkeypatch, ROWS)
    out = media.search_images(job, "kohli football", count=3)
    assert [r["id"] for r in out] == ["res_1", "res_2", "res_3"] and out[0]["domain"] == "site1.com"
    assert (job / out[0]["thumb"]).is_file() and Image.open(job / out[0]["thumb"]).size[0] <= media.THUMB_SIDE
    reg = json.loads((job / "assets" / "candidates.json").read_text())
    assert reg["res_2"]["url"] == "https://img2.example/k2.jpg"
    again = media.search_images(job, "more kohli", count=2)  # ids keep counting up, never reuse
    assert [r["id"] for r in again] == ["res_4", "res_5"]


def test_search_skips_dead_thumbnails_and_reports_no_results(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    rows = [{**ROWS[0], "thumbnail": "https://bad.example/t.jpg"}, ROWS[1]]
    patch_search(monkeypatch, rows)
    assert [r["title"] for r in media.search_images(job, "x")] == ["Kohli 2"]
    patch_search(monkeypatch, [])
    with pytest.raises(media.MediaError, match="No pictures found"):
        media.search_images(job, "nothing")
    patch_search(monkeypatch, [{**ROWS[0], "thumbnail": "https://bad.example/t.jpg"}])
    with pytest.raises(media.MediaError, match="could not be loaded"):
        media.search_images(job, "x")


def test_search_errors_become_friendly_messages(tmp_path, monkeypatch):
    import ddgs

    def boom(**kw):
        raise RuntimeError("ratelimit")

    monkeypatch.setattr(ddgs, "DDGS", boom)
    with pytest.raises(media.MediaError, match="Image search failed"):
        media.search_images(make_job(tmp_path), "x")


# ----------------------------------------------------------------------------- image generation
class FakeImageModels:
    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def image_response(data=None):
    parts = [types.Part(inline_data=types.Blob(data=data, mime_type="image/png"))] if data else [types.Part(text="no")]
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


def test_generated_image_is_saved_and_numbered(tmp_path):
    job = make_job(tmp_path)
    client = SimpleNamespace(models=FakeImageModels(image_response(png_bytes("red")), image_response(png_bytes())))
    out = media.generate_image(job, "  a red   cricket ball ", client=client, models=["m"])
    assert out == {"id": "gen_1", "path": "assets/gen_1.png"} and (job / out["path"]).is_file()
    assert client.models.calls[0][1] == "a red cricket ball"
    assert client.models.calls[0][2].image_config.aspect_ratio == "9:16"
    assert media.generate_image(job, "again", client=client, models=["m"])["id"] == "gen_2"


def test_generation_falls_back_between_models_and_explains_failures(tmp_path):
    job = make_job(tmp_path)
    quota = errors.ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
    client = SimpleNamespace(models=FakeImageModels(quota, image_response(png_bytes())))
    assert media.generate_image(job, "x", client=client, models=["a", "b"])["id"] == "gen_1"
    client = SimpleNamespace(models=FakeImageModels(quota, quota))
    with pytest.raises(media.MediaError, match="billing-enabled"):
        media.generate_image(job, "x", client=client, models=["a", "b"])
    client = SimpleNamespace(models=FakeImageModels(image_response(None)))
    with pytest.raises(media.MediaError, match="wouldn't make that picture"):
        media.generate_image(job, "a real celebrity", client=client, models=["a"])
    with pytest.raises(media.MediaError, match="Tell me what"):
        media.generate_image(job, "  ", client=client)
    assert [f.name for f in (job / "assets").glob("gen_*")] == ["gen_1.png"]  # failures saved nothing more


def test_image_model_chain_prefers_the_env_choice(monkeypatch):
    monkeypatch.delenv("GEMINI_IMAGE_MODEL", raising=False)
    assert media.image_model_chain()[0] == "gemini-3.1-flash-lite-image"
    monkeypatch.setenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
    chain = media.image_model_chain()
    assert chain[0] == "gemini-2.5-flash-image" and len(chain) == len(set(chain)) == 3


# ----------------------------------------------------------------------------- resolving references
def test_resolve_image_handles_every_kind_of_reference(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    att = assets.save_attachment(job, "mine.png", png_bytes())
    ctx = ctx_for(job, att)
    path, extra = media.resolve_image(job, ctx, "att_1")
    assert path == att.path and extra == {"origin": "attachment", "title": "mine.png"}
    patch_search(monkeypatch, ROWS)
    media.search_images(job, "k", count=2)
    monkeypatch.setattr(fetch, "fetch_image", lambda url, max_bytes=0: png_bytes())
    path, extra = media.resolve_image(job, ctx, "res_1")
    assert path.is_file() and extra["credit"] == "Image: site1.com" and extra["origin"] == "web"
    path, extra = media.resolve_image(job, ctx, "https://www.example.org/pic.jpg")
    assert extra["credit"] == "Image: example.org" and path.name.startswith("url_")
    gen = assets.assets_dir(job) / "gen_1.png"
    gen.write_bytes(png_bytes())
    assert media.resolve_image(job, ctx, "gen_1")[1]["origin"] == "generated"


@pytest.mark.parametrize("ref", ["att_9", "res_7", "gen_4", "../etc/passwd", "file:///c:/x.png", "att_1/../x", ""])
def test_resolve_image_rejects_bad_references(tmp_path, ref):
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
def test_web_picture_gets_a_credit_line(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    patch_search(monkeypatch, ROWS)
    found = run_tool(ctx_for(job), "search_web_images", {"query": "kohli"})
    assert [r["id"] for r in found["results"]][:2] == ["res_1", "res_2"]
    ctx = ctx_for(job)
    assert ctx.cards == []  # cards belong to the context that ran the search
    c = ctx_for(job)
    run_tool(c, "search_web_images", {"query": "kohli again", "count": 3})
    assert c.cards[0]["type"] == "choices" and c.cards[0]["items"][0]["caption"].startswith("res_5")
    monkeypatch.setattr(fetch, "fetch_image", lambda url, max_bytes=0: png_bytes())
    out = run_tool(ctx_for(job), "place_image", {"image": "res_2", "start": 20.0, "end": 23.0})
    assert out["ok"] and slot(job, 3)["asset"]["credit"] == "Image: site2.com"


def test_overlapping_placement_fails_fast_with_the_free_ranges(tmp_path):
    job = make_job(tmp_path)
    att = assets.save_attachment(job, "p.png", png_bytes())
    before = (job / "plan.json").read_text()
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 3.0})
    assert "overlaps slot #1" in out["error"] and "6.1-13.9s" in out["error"] and "18.1-30s" in out["error"]
    assert (job / "plan.json").read_text() == before and not snapshots.can_undo(job)
    out = run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "slot_id": 99})
    assert "No B-roll with id 99" in out["error"]
    assert "Unknown layout" in run_tool(ctx_for(job, att), "place_image", {"image": "att_1", "start": 20, "layout": "x"})["error"]


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
                                                         "start": 20.0, "end": 40.0})
    assert "overlaps" in out["error"] or "only" in out["error"] or "under" in out["error"]
    assert (job / "plan.json").read_text() == before


def test_every_media_tool_is_registered_and_declared():
    for name in ("search_web_images", "generate_image", "place_image", "place_video_file", "place_youtube_clip"):
        assert name in tools.REGISTRY
    assert {d.name for d in tools.declarations()} == set(tools.REGISTRY)
    for name in ("place_image", "place_video_file", "place_youtube_clip"):
        assert tools.REGISTRY[name].mutating


def test_generate_image_tool_shows_the_picture_and_reports_failures(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    def fake_generate(j, p):
        (assets.assets_dir(j) / "gen_1.png").write_bytes(png_bytes())
        return {"id": "gen_1", "path": "assets/gen_1.png"}

    monkeypatch.setattr(media, "generate_image", fake_generate)
    ctx = ctx_for(job)
    assert run_tool(ctx, "generate_image", {"prompt": "a bat"})["image_id"] == "gen_1"
    assert ctx.cards == [{"type": "image", "path": "assets/gen_1.png", "caption": "gen_1"}]

    def refuse(j, p):
        raise media.MediaError("Gemini wouldn't make that picture. I can search the web instead.")

    monkeypatch.setattr(media, "generate_image", refuse)
    assert "search the web" in run_tool(ctx_for(job), "generate_image", {"prompt": "lebron"})["error"]


def test_cleanup_removes_assets_but_the_chat_survives(tmp_path):
    from src.cleanup import purge_job_temp

    job = make_job(tmp_path)
    assets.save_attachment(job, "p.png", png_bytes())
    (job / "chat.json").write_text("[]")
    purge_job_temp(job, root=tmp_path)
    assert not (job / "assets").exists() and (job / "chat.json").is_file()
