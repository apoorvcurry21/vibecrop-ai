"""Chat cards (layout picker, search results, previews) and the slot-preview tool."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src import workflow
from src.assistant import agent, assets, media_tools, store, tools
from src.assistant.tools import ToolContext, run_tool

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def make_job(root: Path, real_media=False) -> Path:
    job = root / "jobs" / "20260101_000000_ui"
    (job / "clips").mkdir(parents=True)
    items = []
    for i, start in ((1, 2.0), (2, 14.0)):
        clip = job / "clips" / f"clip_{i:02d}.mp4"
        if real_media:
            ff("-f", "lavfi", "-i", "color=c=green:s=1280x720:r=30:d=8", "-pix_fmt", "yuv420p", str(clip))
        else:
            clip.write_bytes(b"x")
        items.append({"id": i, "asset_type": "video", "timestamp_start": start, "timestamp_end": start + 4,
                      "search_query": f"q{i}", "alternative_queries": [], "english_meaning": f"says {i}",
                      "layout": "overlay", "asset": {"status": "ok", "path": f"clips/clip_{i:02d}.mp4",
                                                     "clip_duration": 8.0, "title": "T"}})
    (job / "plan.json").write_text(json.dumps({"video_topic": "UI", "video_duration": 30.0, "items": items}))
    meta = {"language": "hi", "source_name": "demo.mp4"}
    if real_media:
        src = root / "speaker.mp4"
        ff("-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=30:d=30", "-pix_fmt", "yuv420p", str(src))
        meta["source_video"] = str(src)
    (job / "job.json").write_text(json.dumps(meta))
    return job


def test_layout_mockups_exist_and_are_phone_shaped():
    for name in ("overlay", "split", "pip", "text"):
        f = assets.STATIC_DIR / f"layout_{name}.png"
        assert f.is_file(), f
        w, h = Image.open(f).size
        assert 1.6 < h / w < 1.9  # roughly 9:16
    assert [c[0] for c in media_tools.LAYOUT_CHOICES] == ["overlay", "split", "pip"]


def test_ask_layout_emits_a_picker_card_and_the_model_is_told_to_wait(tmp_path):
    ctx = ToolContext(job=make_job(tmp_path))
    out = run_tool(ctx, "ask_layout", {})
    assert out["shown"] is True and "wait" not in out.get("error", "")
    assert ctx.cards == [{"type": "layouts", "text_example": False}]
    run_tool(ctx, "ask_layout", {"include_text_example": True})
    assert ctx.cards[1]["text_example"] is True
    assert not ctx.changed  # showing options changes nothing, so no page refresh and no undo entry


def test_system_prompt_teaches_when_to_ask_for_the_layout():
    assert "ask_layout" in agent.SYSTEM_PROMPT and "preview_slot" in agent.SYSTEM_PROMPT
    assert "ask_layout" in tools.REGISTRY and not tools.REGISTRY["ask_layout"].mutating


@needs_ffmpeg
def test_preview_slot_makes_a_still_of_the_final_look(tmp_path):
    job = make_job(tmp_path, real_media=True)
    workflow.set_pip(job, 1, pos="bottom", scale=0.5)
    ctx = ToolContext(job=job)
    out = run_tool(ctx, "preview_slot", {"slot_id": 1})
    assert out["ok"] is True
    card = ctx.cards[0]
    img = Image.open(job / card["path"]).convert("RGB")
    assert card["type"] == "image" and img.size[0] == 360 and img.size[1] > 600
    assert img.getpixel((180, 10))[0] > 200 and img.getpixel((180, 10))[1] < 80  # top: the red speaker
    w = img.size[0]
    assert img.getpixel((w // 2, int(img.size[1] * 0.72)))[1] > 100  # the green pip box sits low (position "bottom")
    assert not list((job / "assets").glob("_still*"))  # temp render removed
    assert "No B-roll with id 9" in run_tool(ToolContext(job=job), "preview_slot", {"slot_id": 9})["error"]
    (job / "clips" / "clip_02.mp4").unlink()
    assert "no clip" in run_tool(ToolContext(job=job), "preview_slot", {"slot_id": 2})["error"]


# ----------------------------------------------------------------------------- the panel renders cards
def open_chat(tmp_path, monkeypatch, messages):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    (job / "chat.json").write_text(json.dumps(messages))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.session_state["chat_open"] = True
    at.run()
    return job, at


def test_layout_picker_shows_three_tappable_examples_and_tapping_sends_a_message(tmp_path, monkeypatch):
    messages = [{"role": "user", "text": "add this picture"},
                {"role": "assistant", "text": "How should it look?", "cards": [{"type": "layouts", "text_example": True}]}]
    job, at = open_chat(tmp_path, monkeypatch, messages)
    assert not at.exception, [e.value for e in at.exception]
    keys = [b.key for b in at.button if b.key and b.key.startswith("lay_")]
    assert len(keys) == 3 and [k.rsplit("_", 1)[1] for k in keys] == ["overlay", "split", "pip"]
    assert len(at.get("image")) >= 4  # three layout examples + the text example
    seen = {}

    def fake_stream(job_, history, text_, attachments=None, **kw):
        seen["text"] = text_
        yield {"type": "done", "text": "ok", "changed": False, "cards": []}

    monkeypatch.setattr(agent, "stream_turn", fake_stream)
    at.button(key=keys[2]).click().run()
    assert seen["text"] == "Use the small box (picture-in-picture) layout"
    assert [m["role"] for m in store.load(job)][-2:] == ["user", "assistant"]


def test_search_results_show_thumbnails_with_use_buttons_and_missing_files_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    a = assets.assets_dir(job)
    for n in (1, 2):
        Image.new("RGB", (80, 60), "blue").save(a / f"res_{n}_thumb.jpg")
    cards = [{"type": "choices", "items": [
        {"id": f"res_{n}", "path": f"assets/res_{n}_thumb.jpg", "caption": f"res_{n} · site.com"} for n in (1, 2, 3)]},
        {"type": "image", "path": "assets/gone.png", "caption": "purged after a render"}]
    (job / "chat.json").write_text(json.dumps([{"role": "assistant", "text": "Found these", "cards": cards}]))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.session_state["chat_open"] = True
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    use = [b for b in at.button if b.key and b.key.startswith("use_")]
    assert [b.key.rsplit("_", 2)[1] + "_" + b.key.rsplit("_", 1)[1] for b in use] == ["res_1", "res_2"]  # res_3 had no file
    seen = {}

    def fake_stream(job_, history, text_, attachments=None, **kw):
        seen["text"] = text_
        yield {"type": "done", "text": "placed", "changed": False, "cards": []}

    monkeypatch.setattr(agent, "stream_turn", fake_stream)
    use[1].click().run()
    assert seen["text"] == "Use res_2"


def test_cards_from_a_live_turn_are_shown_and_saved_with_the_message(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    Image.new("RGB", (80, 60), "red").save(assets.assets_dir(job) / "gen_1.png")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.session_state["chat_open"] = True
    at.run()

    def fake_stream(job_, history, text_, attachments=None, **kw):
        card = {"type": "image", "path": "assets/gen_1.png", "caption": "gen_1"}
        yield {"type": "card", "card": card}
        yield {"type": "text", "text": "Here you go."}
        yield {"type": "done", "text": "Here you go.", "changed": False, "cards": [card]}

    monkeypatch.setattr(agent, "stream_turn", fake_stream)
    at.chat_input[0].set_value("make me a picture").run()
    assert not at.exception, [e.value for e in at.exception]
    assert store.load(job)[-1]["cards"] == [{"type": "image", "path": "assets/gen_1.png", "caption": "gen_1"}]
    assert len(at.get("image")) >= 1


def test_the_chat_toggle_lives_in_a_fragment_so_closing_does_not_redraw_the_page():
    """Closing used to rerun the whole review page (every slot and video player), which felt slow. The open/close
    controls must stay inside the shell fragment, and the panel must not be a second, nested fragment.
    (AppTest cannot simulate fragment-scoped reruns, so this guards the structure; verify the speed in a browser.)"""
    from src.assistant import shell, ui

    assert hasattr(shell.shell, "__wrapped__")  # decorated with st.fragment
    assert not hasattr(ui.chat_panel, "__wrapped__")
    assert shell.shell.__wrapped__.__module__ == "src.assistant.shell"
    app_src = Path(APP).read_text(encoding="utf-8")
    assert "chat_shell(str(job))" in app_src and "chat_open_btn" not in app_src  # app.py no longer owns the toggle
