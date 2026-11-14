"""The chat assistant: tools, history store, attachments, the Gemini function-calling loop (mocked) and the panel."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors, types
from streamlit.testing.v1 import AppTest

from src import config, workflow
from src.assistant import agent, assets, snapshots, store, tools
from src.assistant.tools import ToolContext, find_phrase, run_tool

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def make_job(root: Path) -> Path:
    """Two 4 s slots (2-6 and 14-18) in a 30 s video, with a word-level transcript."""
    job = root / "jobs" / "20260101_000000_chat"
    (job / "clips").mkdir(parents=True)
    items = []
    for i, start in ((1, 2.0), (2, 14.0)):
        (job / "clips" / f"clip_{i:02d}.mp4").write_bytes(f"clip-{i}".encode())
        items.append({"id": i, "asset_type": "video", "timestamp_start": start, "timestamp_end": start + 4,
                      "search_query": f"q{i}", "alternative_queries": [], "spoken_text": "", "english_meaning":
                      f"says {i}", "reasoning": "", "layout": "overlay", "ai_layout": "overlay",
                      "asset": {"status": "ok", "path": f"clips/clip_{i:02d}.mp4", "clip_duration": 9.0,
                                "title": f"Title {i}"}})
    (job / "plan.json").write_text(json.dumps({"video_topic": "Cricket", "video_duration": 30.0, "items": items}))
    (job / "job.json").write_text(json.dumps({"language": "hi", "source_name": "demo.mp4", "style": "overlay"}))
    words = [("so", 0.0), ("virat", 7.0), ("koli", 7.4), ("is", 7.8), ("the", 8.0), ("ronaldo", 8.3), ("of", 8.9),
             ("cricket", 9.2)]
    (job / "transcript.json").write_text(json.dumps({
        "segments": [{"id": 0, "start": 0.0, "end": 6.0, "text": "so hello everyone"},
                     {"id": 1, "start": 7.0, "end": 10.0, "text": "virat koli is the ronaldo of cricket"}],
        "words": [{"word": w, "start": s, "end": s + 0.3} for w, s in words]}))
    return job


def ctx_for(job: Path) -> ToolContext:
    return ToolContext(job=job)


def slot(job, i):
    return next(x for x in workflow.load_plan(job)["items"] if x["id"] == i)


# ----------------------------------------------------------------------------- tools
def test_declarations_cover_every_tool_and_are_valid_for_gemini():
    decls = tools.declarations()
    assert {d.name for d in decls} == set(tools.REGISTRY)
    pip = next(d for d in decls if d.name == "set_pip")
    assert pip.parameters.required == ["slot_id"] and "position" in pip.parameters.properties


def test_unknown_tool_and_missing_arguments_are_reported_not_raised(tmp_path):
    job = make_job(tmp_path)
    assert "Unknown tool" in run_tool(ctx_for(job), "format_disk", {})["error"]
    assert "Missing required" in run_tool(ctx_for(job), "set_layout", {"slot_id": 1})["error"]
    assert "whole number" in run_tool(ctx_for(job), "set_layout", {"slot_id": "x", "layout": "pip"})["error"]
    assert not snapshots.can_undo(job)  # rejected before any snapshot


def test_a_mutating_tool_changes_the_plan_marks_it_changed_and_is_undoable(tmp_path):
    job = make_job(tmp_path)
    ctx = ctx_for(job)
    out = run_tool(ctx, "set_layout", {"slot_id": 1, "layout": "pip"})
    assert out["ok"] and out["slot"]["layout"] == "pip" and ctx.changed and snapshots.can_undo(job)
    assert slot(job, 1)["pip"] == {"pos": "top-right", "scale": 0.6}
    assert run_tool(ctx, "undo", {}) == {"ok": True, "undone": "Changing slot 1 layout"}
    assert slot(job, 1)["layout"] == "overlay" and run_tool(ctx, "undo", {})["ok"] is False


def test_a_failing_tool_leaves_the_project_exactly_as_it_was(tmp_path):
    job = make_job(tmp_path)
    before = (job / "plan.json").read_text()
    ctx = ctx_for(job)
    out = run_tool(ctx, "set_timing", {"slot_id": 1, "start": 10.0, "end": 15.0})  # runs into slot 2
    assert "cannot end after" in out["error"] and not ctx.changed
    assert (job / "plan.json").read_text() == before and not snapshots.can_undo(job)
    assert "No B-roll with id 9" in run_tool(ctx, "set_layout", {"slot_id": 9, "layout": "split"})["error"]


def test_unexpected_exceptions_are_contained_and_rolled_back(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    before = (job / "plan.json").read_text()

    def boom(*a, **k):
        workflow.set_layout(job, 1, "split")  # a half-finished edit...
        raise RuntimeError("network died")

    monkeypatch.setattr(workflow, "regenerate", boom)
    out = run_tool(ctx_for(job), "regenerate", {"slot_id": 1})
    assert out == {"error": "RuntimeError: network died"}
    assert (job / "plan.json").read_text() == before  # ...is rolled back


def test_swap_tool_snapshots_both_clips_so_undo_restores_them(tmp_path):
    job = make_job(tmp_path)
    run_tool(ctx_for(job), "swap_slots", {"slot_a": 1, "slot_b": 2})
    assert (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-2"
    run_tool(ctx_for(job), "undo", {})
    assert (job / "clips" / "clip_01.mp4").read_bytes() == b"clip-1"


def test_get_plan_lists_slots_in_time_order_with_free_gaps(tmp_path):
    job = make_job(tmp_path)
    plan = run_tool(ctx_for(job), "get_plan", {})
    assert [s["id"] for s in plan["slots"]] == [1, 2] and plan["free_gaps"] == [[6.1, 13.9], [18.1, 30.0]]
    assert plan["video_duration"] == 30.0 and plan["style"] == "overlay" and plan["can_undo"] is False
    summary = json.loads(tools.plan_summary(job))
    assert summary["video_topic"] == "Cricket" and len(summary["slots"]) == 2


def test_find_phrase_matches_loose_spellings_and_returns_the_time(tmp_path):
    job = make_job(tmp_path)
    out = run_tool(ctx_for(job), "find_in_transcript", {"phrase": "Virat Kohli"})
    assert out["matches"][0]["start"] == 7.0 and out["matches"][0]["heard_as"] == "virat koli"
    assert run_tool(ctx_for(job), "find_in_transcript", {"phrase": "zzzzzz"})["matches"] == []
    assert find_phrase([], "x") == [] and find_phrase([{"word": "a", "start": 0, "end": 1}], "  ") == []


def test_get_transcript_filters_by_range_and_truncates_long_ones(tmp_path):
    job = make_job(tmp_path)
    assert [s["text"] for s in run_tool(ctx_for(job), "get_transcript", {"start": 6.5, "end": 12})["segments"]] \
        == ["virat koli is the ronaldo of cricket"]
    long = {"segments": [{"id": i, "start": i, "end": i + 1, "text": "word " * 20} for i in range(500)], "words": []}
    (job / "transcript.json").write_text(json.dumps(long))
    out = run_tool(ctx_for(job), "get_transcript", {})
    assert out["truncated"] is True and 0 < len(out["segments"]) < 500


def test_text_framing_pip_and_style_tools(tmp_path):
    job = make_job(tmp_path)
    ctx = ctx_for(job)
    t = run_tool(ctx, "add_text", {"text": "GOAT", "start": 7, "end": 10, "position": "bottom", "color": "yellow",
                                   "size": "xl"})["text"]
    assert (t["id"], t["pos"], t["color"], t["size"]) == (1, "bottom", "yellow", "xl")
    assert run_tool(ctx, "update_text", {"text_id": 1, "text": "GOAT!!", "color": "red"})["text"]["color"] == "red"
    assert run_tool(ctx, "remove_text", {"text_id": 1})["ok"] and workflow.load_plan(job)["texts"] == []
    z = run_tool(ctx, "set_framing", {"slot_id": 1, "preset": "zoom_in"})["framing"]
    assert z["start"]["z"] == 1.0 and z["end"]["z"] == 1.6
    z = run_tool(ctx, "set_framing", {"slot_id": 1, "zoom_end": 2.0, "pan_x_end": 0.2})["framing"]
    assert z["end"]["z"] == 2.0 and z["end"]["x"] == 0.2 and z["start"]["z"] == 1.0
    assert "Unknown preset" in run_tool(ctx, "set_framing", {"slot_id": 1, "preset": "spin"})["error"]
    p = run_tool(ctx, "set_pip", {"slot_id": 2, "position": "bottom-left", "size": 0.4})["slot"]
    assert p["layout"] == "pip" and p["pip"] == {"pos": "bottom-left", "scale": 0.4}
    run_tool(ctx, "apply_style", {"style": "split"})
    assert (slot(job, 1)["layout"], slot(job, 2)["layout"]) == ("split", "pip")  # pip survives a style change
    assert "Unknown video style" in run_tool(ctx, "apply_style", {"style": "disco"})["error"]


def test_remove_restore_and_delete_slot_tools(tmp_path):
    job = make_job(tmp_path)
    ctx = ctx_for(job)
    run_tool(ctx, "remove_slot", {"slot_id": 1})
    assert slot(job, 1)["enabled"] is False and run_tool(ctx, "get_plan", {})["slots"][0]["removed"] is True
    run_tool(ctx, "restore_slot", {"slot_id": 1})
    assert slot(job, 1)["enabled"] is True
    run_tool(ctx, "delete_slot", {"slot_id": 2})
    assert not (job / "clips" / "clip_02.mp4").exists()
    run_tool(ctx, "undo", {})
    assert (job / "clips" / "clip_02.mp4").read_bytes() == b"clip-2"


# ----------------------------------------------------------------------------- history store + attachments
def test_store_roundtrip_cap_and_corruption(tmp_path):
    job = tmp_path / "j"
    job.mkdir()
    assert store.load(job) == []
    store.append(job, "user", "hi", files=[{"id": "att_1", "name": "a.png", "kind": "image"}])
    msgs = store.append(job, "assistant", "hello", cards=[{"type": "image", "path": "x.png"}])
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[0]["files"][0]["id"] == "att_1"
    for n in range(store.MAX_MESSAGES + 20):
        store.append(job, "user", f"m{n}")
    assert len(store.load(job)) == store.MAX_MESSAGES and store.load(job)[-1]["text"] == f"m{store.MAX_MESSAGES + 19}"
    (job / "chat.json").write_text("{not json")
    assert store.load(job) == []
    (job / "chat.json").write_text(json.dumps([{"role": "system", "text": "x"}, {"role": "user", "text": "ok"}, 5]))
    assert [m["text"] for m in store.load(job)] == ["ok"]


def test_attachments_are_renamed_typed_and_validated(tmp_path):
    job = tmp_path / "j"
    job.mkdir()
    a = assets.save_attachment(job, "../../evil name.PNG", b"img")
    b = assets.save_attachment(job, "clip.mov", b"vid")
    assert (a.id, a.kind, a.path.name, a.name) == ("att_1", "image", "att_1.png", "evil name.PNG")
    assert (b.id, b.kind, b.path.name) == ("att_2", "video", "att_2.mov")
    assert a.path.parent == job / "assets" and not (tmp_path / "evil name.PNG").exists()
    for bad in (("virus.exe", b"x"), ("empty.png", b""), ("noext", b"x")):
        with pytest.raises(assets.AssetError):
            assets.save_attachment(job, *bad)
    assert assets.load_attachment(job, "att_1").path == a.path
    assert assets.load_attachment(job, "../att_1") is None and assets.load_attachment(job, "att_9") is None


def test_vision_copy_is_small_and_bad_files_are_skipped(tmp_path):
    from PIL import Image

    big = tmp_path / "big.png"
    Image.new("RGB", (3000, 2000), "red").save(big)
    jpeg = assets.vision_jpeg(big)
    assert jpeg and Image.open(__import__("io").BytesIO(jpeg)).size[0] == assets.VISION_SIDE
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    assert assets.vision_jpeg(bad) is None


# ----------------------------------------------------------------------------- the Gemini loop (mocked client)
def text(t):
    return types.Part(text=t)


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def chunk(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))])


class FakeModels:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def generate_content_stream(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return iter(step)


class FakeClient:
    def __init__(self, *script):
        self.models = FakeModels(script)


def run_agent(job, client, message="hi", attachments=None, models=("m1",), history=None):
    return list(agent.stream_turn(job, history or [], message, attachments, client=client, models=list(models),
                                  sleep=lambda s: None))


def kinds(events):
    return [e["type"] for e in events]


def test_plain_reply_streams_text_and_changes_nothing(tmp_path):
    job = make_job(tmp_path)
    ev = run_agent(job, FakeClient([chunk(text("Hello ")), chunk(text("there!"))]))
    assert kinds(ev) == ["text", "text", "done"]
    assert ev[-1] == {"type": "done", "text": "Hello there!", "changed": False, "cards": []}


def test_tool_round_trip_edits_the_plan_and_feeds_the_result_back(tmp_path):
    job = make_job(tmp_path)
    client = FakeClient([chunk(text("On it. "), call("set_layout", slot_id=1, layout="split"))],
                        [chunk(text("Slot 1 is now split."))])
    ev = run_agent(job, client)
    assert kinds(ev) == ["text", "tool", "text", "done"]
    assert ev[1]["label"] == "Changing slot 1 layout..."
    assert ev[-1]["changed"] is True and ev[-1]["text"] == "On it. Slot 1 is now split."
    assert slot(job, 1)["layout"] == "split"
    second = client.models.calls[1]["contents"]
    assert second[-2].role == "model" and second[-2].parts[1].function_call.name == "set_layout"
    resp = second[-1].parts[0].function_response
    assert resp.name == "set_layout" and resp.response["ok"] is True


def test_a_tool_error_goes_back_to_the_model_which_can_recover(tmp_path):
    job = make_job(tmp_path)
    client = FakeClient([chunk(call("set_timing", slot_id=1, start=10, end=15))],
                        [chunk(call("set_timing", slot_id=1, start=3, end=6))],
                        [chunk(text("Moved it to 3-6s."))])
    ev = run_agent(job, client)
    first_resp = client.models.calls[1]["contents"][-1].parts[0].function_response.response
    assert "cannot end after" in first_resp["error"]
    assert (slot(job, 1)["timestamp_start"], slot(job, 1)["timestamp_end"]) == (3.0, 6.0)
    assert ev[-1]["changed"] is True and ev[-1]["text"] == "Moved it to 3-6s."


def test_silent_model_after_edits_still_gets_a_summary(tmp_path):
    job = make_job(tmp_path)
    ev = run_agent(job, FakeClient([chunk(call("remove_slot", slot_id=2))], [chunk()]))
    assert ev[-1]["text"] == "Done: Removing slot 2." and slot(job, 2)["enabled"] is False


def test_a_runaway_tool_loop_stops_at_the_round_limit(tmp_path):
    job = make_job(tmp_path)
    ev = run_agent(job, FakeClient(*[[chunk(call("get_plan"))] for _ in range(agent.MAX_ROUNDS + 2)]))
    assert kinds(ev).count("tool") == agent.MAX_ROUNDS and "stopped here" in ev[-1]["text"]


def test_transient_errors_retry_then_fall_back_to_the_next_model(tmp_path):
    job = make_job(tmp_path)
    overloaded = errors.ServerError(503, {"error": {"message": "overloaded"}})
    client = FakeClient(overloaded, overloaded, overloaded, [chunk(text("from the backup"))])
    ev = run_agent(job, client, models=("main", "backup"))
    assert [c["model"] for c in client.models.calls] == ["main", "main", "main", "backup"]
    assert ev[-1]["text"] == "from the backup"


def test_everything_failing_gives_a_friendly_error_event(tmp_path):
    job = make_job(tmp_path)
    bad = errors.ClientError(403, {"error": {"message": "key revoked"}})
    ev = run_agent(job, FakeClient(bad, bad), models=("a", "b"))
    assert kinds(ev) == ["error", "done"] and "couldn't reach the AI" in ev[0]["text"]


def test_a_model_without_thinking_levels_is_retried_without_them(tmp_path):
    job = make_job(tmp_path)
    bad = errors.ClientError(400, {"error": {"message": "thinking level is not supported by this model"}})
    client = FakeClient(bad, [chunk(text("ok"))])
    ev = run_agent(job, client)
    assert client.models.calls[0]["config"].thinking_config is not None
    assert client.models.calls[1]["config"].thinking_config is None and ev[-1]["text"] == "ok"


def test_missing_api_key_is_reported_as_an_error_event(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "make_client", lambda *a, **k: (_ for _ in ()).throw(config.ConfigError("GEMINI_API_KEY is not set.")))
    ev = list(agent.stream_turn(job, [], "hi"))
    assert kinds(ev) == ["error", "done"] and "GEMINI_API_KEY" in ev[0]["text"]


def test_history_and_attachments_reach_the_model(tmp_path):
    from PIL import Image

    job = make_job(tmp_path)
    Image.new("RGB", (40, 30), "blue").save(job / "pic.png")
    att = assets.save_attachment(job, "kohli.png", (job / "pic.png").read_bytes())
    history = [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi!"},
               {"role": "user", "text": "", "files": [{"id": "att_0", "name": "old.png", "kind": "image"}]}]
    client = FakeClient([chunk(text("nice pic"))])
    run_agent(job, client, "use this", [att], history=history)
    call_ = client.models.calls[0]
    roles = [c.role for c in call_["contents"]]
    assert roles == ["user", "model", "user", "user"] and "old.png" in call_["contents"][2].parts[0].text
    last = call_["contents"][-1].parts
    assert last[0].text == "use this" and last[2].inline_data.mime_type == "image/jpeg"
    system = call_["config"].system_instruction
    assert "att_1 (image, 'kohli.png')" in system and '"video_topic":"Cricket"' in system


def test_the_model_cannot_reach_files_through_tools():
    for t in tools.REGISTRY.values():
        assert not [p for p in t.params if "path" in p or "file" in p], t.name


# ----------------------------------------------------------------------------- the panel (AppTest, agent mocked)
def open_app(tmp_path, monkeypatch, opened=False):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    if opened:
        at.session_state["chat_open"] = True
    at.run()
    return job, at


def test_closed_state_is_just_a_floating_button_and_opens_the_panel(tmp_path, monkeypatch):
    job, at = open_app(tmp_path, monkeypatch)
    assert not at.exception and len(at.chat_input) == 0
    at.button(key="chat_open_btn").click().run()
    assert not at.exception and len(at.chat_input) == 1 and at.session_state["chat_open"] is True
    assert {b.label for b in at.button} >= {"↩ Undo", "✕"} and at.button(key="chat_undo").disabled
    at.button(key="chat_close").click().run()
    assert len(at.chat_input) == 0 and at.session_state["chat_open"] is False


def test_sending_a_message_streams_the_reply_and_saves_the_chat(tmp_path, monkeypatch):
    job, at = open_app(tmp_path, monkeypatch, opened=True)

    def fake_stream(job_, history, text_, attachments=None, **kw):
        yield {"type": "tool", "label": "Thinking..."}
        yield {"type": "text", "text": "Sure, "}
        yield {"type": "text", "text": "done."}
        yield {"type": "done", "text": "Sure, done.", "changed": False, "cards": []}

    monkeypatch.setattr(agent, "stream_turn", fake_stream)
    at.chat_input[0].set_value("make slot 1 bigger").run()
    assert not at.exception, [e.value for e in at.exception]
    saved = store.load(job)
    assert [(m["role"], m["text"]) for m in saved] == [("user", "make slot 1 bigger"), ("assistant", "Sure, done.")]
    assert "Sure, done." in " ".join(m.value for m in at.markdown)


def test_a_turn_that_changes_the_plan_refreshes_the_page_and_enables_undo(tmp_path, monkeypatch):
    job, at = open_app(tmp_path, monkeypatch, opened=True)

    def fake_stream(job_, history, text_, attachments=None, **kw):
        snapshots.take(job_, "Changing slot 1 layout")
        workflow.set_layout(job_, 1, "split")
        yield {"type": "text", "text": "Slot 1 is split now."}
        yield {"type": "done", "text": "Slot 1 is split now.", "changed": True, "cards": []}

    monkeypatch.setattr(agent, "stream_turn", fake_stream)
    at.chat_input[0].set_value("split slot 1").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.button(key="chat_undo").disabled  # the page re-ran: Undo is now available
    at.button(key="chat_undo").click().run()
    assert slot(job, 1)["layout"] == "overlay"
    assert store.load(job)[-1]["text"] == "Undone: Changing slot 1 layout."


def test_chips_send_a_ready_made_message(tmp_path, monkeypatch):
    job, at = open_app(tmp_path, monkeypatch, opened=True)
    seen = {}

    def fake_stream(job_, history, text_, attachments=None, **kw):
        seen["text"] = text_
        yield {"type": "done", "text": "ok", "changed": False, "cards": []}

    monkeypatch.setattr(agent, "stream_turn", fake_stream)
    at.button(key="chat_s1").click().run()
    assert seen["text"] == "Show me the plan"


def test_chat_model_chain_prefers_the_fast_model_and_never_repeats(monkeypatch):
    monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "big")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "small,big")
    assert config.chat_model_chain() == ["gemini-3.1-flash-lite", "big", "small"]
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "big")
    assert config.chat_model_chain() == ["big", "small"]


# ----------------------------------------------------------------------------- slow / broken API must never hang or crash the panel
def test_a_timeout_skips_straight_to_the_next_model_without_retrying_the_slow_one(tmp_path):
    import httpx

    job = make_job(tmp_path)
    client = FakeClient(httpx.ReadTimeout("slow"), [chunk(text("from the backup"))])
    ev = run_agent(job, client, models=("slow", "backup"))
    assert [c["model"] for c in client.models.calls] == ["slow", "backup"]  # one try on the slow model, not three
    assert ev[-1]["text"] == "from the backup"


def test_every_model_timing_out_gives_a_friendly_error_not_a_crash(tmp_path):
    import httpx

    job = make_job(tmp_path)
    ev = run_agent(job, FakeClient(httpx.ReadTimeout("x"), httpx.ConnectError("y")), models=("a", "b"))
    assert kinds(ev) == ["error", "done"] and "couldn't reach the AI" in ev[0]["text"]


def test_a_timeout_in_the_middle_of_the_answer_is_reported(tmp_path):
    import httpx

    def stream():
        yield chunk(text("Partial "))
        raise httpx.ReadTimeout("stalled")

    ev = run_agent(make_job(tmp_path), FakeClient(stream()))
    assert kinds(ev) == ["text", "error", "done"] and ev[-1]["text"] == "Partial "


def test_an_unexpected_bug_inside_a_turn_becomes_an_error_event(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    monkeypatch.setattr(tools, "run_tool", lambda *a, **k: (_ for _ in ()).throw(KeyError("boom")))
    ev = run_agent(job, FakeClient([chunk(call("get_plan"))]))
    assert kinds(ev)[-2:] == ["error", "done"] and "Something went wrong (KeyError)" in ev[-2]["text"]


def test_the_chat_uses_a_short_timeout_but_the_planner_keeps_the_long_one(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(agent, "make_client", lambda timeout_ms=None: seen.setdefault("t", timeout_ms) and
                        SimpleNamespace(models=FakeModels([[chunk(text("hi"))]])))
    list(agent.stream_turn(make_job(tmp_path), [], "hi", models=["m"]))
    assert seen["t"] == config.CHAT_TIMEOUT_MS == 30_000 < config.GEMINI_TIMEOUT_MS
