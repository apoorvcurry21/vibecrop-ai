"""Caption style presets, word emphasis (ASS output, workflow, chat tools, preview pixels, the review-page panel)."""
import io
import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import workflow
from src.assistant.tools import ToolContext, plan_view, run_tool
from src.captions import styles
from src.rendering import ass, preview
from tests.test_chat_agent import make_job

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")

CHUNK = {"id": 0, "start": 10.0, "end": 11.6, "words": [
    {"text": "You", "start": 10.0, "end": 10.3}, {"text": "will", "start": 10.4, "end": 10.7},
    {"text": "never", "start": 10.8, "end": 11.2}, {"text": "quit", "start": 11.2, "end": 11.6}]}


def words_of(n, per=0.4):
    return {"id": 0, "start": 0.0, "end": n * per, "words": [{"text": f"w{i}", "start": i * per, "end": i * per + 0.3}
                                                              for i in range(n)]}


# ---- styles ---------------------------------------------------------------------------------------------------------

def test_every_preset_is_complete_and_unknown_ids_fall_back_to_classic():
    need = {"label", "about", "font", "size", "size_deva", "bold", "outline", "shadow", "uppercase", "max_words",
            "primary", "active", "pop", "box", "margin_v"}
    for sid, st in styles.STYLES.items():
        assert need <= set(st), sid
        assert len(st["active"]) == 6 and len(st["primary"]) == 6
        assert ("box_color" in st) == st["box"]
    assert styles.get_style("nope") is styles.STYLES["classic"] and styles.get_style(None) is styles.STYLES["classic"]


def test_caption_settings_default_and_drop_garbage():
    assert styles.caption_settings({}) == {"style": "classic", "emphasis": []}
    got = styles.caption_settings({"captions": {"style": "zzz", "emphasis": [{"start": 1, "end": 2}, "junk", {"a": 1}]}})
    assert got["style"] == "classic" and got["emphasis"] == [{"start": 1, "end": 2}]


def test_fit_chunks_splits_evenly_and_leaves_short_chunks_alone():
    out = styles.fit_chunks([words_of(4)], 3)
    assert [len(c["words"]) for c in out] == [2, 2]  # not 3 + 1
    assert [len(c["words"]) for c in styles.fit_chunks([words_of(7)], 3)] == [3, 2, 2]
    assert [len(c["words"]) for c in styles.fit_chunks([words_of(3), words_of(2)], 3)] == [3, 2]
    assert [c["id"] for c in styles.fit_chunks([words_of(4), words_of(4)], 3)] == [0, 1, 2, 3]
    assert out[0]["start"] == 0.0 and out[0]["end"] == pytest.approx(0.7)  # words w0, w1
    assert out[1]["start"] == pytest.approx(0.8) and out[1]["end"] == pytest.approx(1.5)  # words w2, w3


def test_colour_and_scale_validation():
    assert styles.normalize_color("red") == "FF3B30" and styles.normalize_color("#abcdef") == "ABCDEF"
    assert styles.normalize_color(None) is None and styles.normalize_color("  ") is None
    with pytest.raises(styles.CaptionStyleError):
        styles.normalize_color("not-a-colour")
    assert styles.normalize_scale(5) == 1.8 and styles.normalize_scale(0.2) == 1.0 and styles.normalize_scale(None) == 1.35
    with pytest.raises(styles.CaptionStyleError):
        styles.normalize_scale("big")


def test_emphasis_matches_by_the_middle_of_the_word():
    emph = [{"start": 10.8, "end": 11.2, "color": "FF3B30", "scale": 1.4}]
    assert styles.emphasis_for(CHUNK["words"][2], emph) is emph[0]
    assert styles.emphasis_for(CHUNK["words"][1], emph) is None
    assert styles.emphasis_for({"text": "x", "start": 10.9, "end": 11.0}, emph) is emph[0]  # rewritten words work too


# ---- ASS output ------------------------------------------------------------------------------------------------------

def test_classic_events_are_unchanged():
    ev = ass.caption_events([CHUNK])
    assert ev[1].endswith("You {\\c&H0000FFFF&}will{\\c&H00FFFFFF&} never quit")
    assert ev == ass.caption_events([CHUNK], style_id="classic", emphasis=[])
    assert "Style: Active" not in ass.header() and "Layer, Start" in ass.header()


def test_bold_pop_is_uppercase_green_and_pops():
    line = ass.caption_events([CHUNK], style_id="bold_pop")[1]
    assert "WILL" in line and "will" not in line
    assert "\\c&H0014FF39&" in line  # 39FF14 in BGR
    assert "\\fscx82\\fscy82\\t(0,90,\\fscx100\\fscy100)" in line
    assert "Style: CaptionLatin,Impact,104" in ass.header("bold_pop")


def test_karaoke_box_adds_a_second_layer_only_the_spoken_word_shows_in():
    ev = ass.caption_events([CHUNK], style_id="karaoke_box")
    assert len(ev) == 8  # base + box per word
    base, box = ev[2], ev[3]
    assert base.startswith("Dialogue: 1,") and ",CaptionLatin," in base and "\\c&H" not in base
    assert box.startswith("Dialogue: 2,") and ",ActiveLatin," in box  # a different layer, or libass moves it up
    assert box.count("\\alpha&HFF&") == 3 and "never" in box
    assert "Style: ActiveLatin," in ass.header("karaoke_box") and "Style: ActiveDevaSplit," in ass.header("karaoke_box")


def test_emphasis_adds_colour_and_size_and_both_layers_agree_on_size():
    emph = [{"start": 10.8, "end": 11.2, "color": "FF3B30", "scale": 1.4}]
    base = ass.caption_events([CHUNK], emphasis=emph)[0]
    assert "{\\c&H00303BFF&\\fscx140\\fscy140}never{\\r}" in base
    ev = ass.caption_events([CHUNK], style_id="karaoke_box", emphasis=emph)
    assert "\\fscx140\\fscy140}never" in ev[0] and "\\fscx140\\fscy140}never" in ev[1]  # same width -> box lines up
    size_only = ass.caption_events([CHUNK], emphasis=[{"start": 10.8, "end": 11.2, "color": None, "scale": 1.6}])[0]
    assert "{\\fscx160\\fscy160}never{\\r}" in size_only  # size only: no colour tag in it


def test_emphasis_survives_a_rewritten_caption_with_a_different_word_count():
    rewritten = {"id": 0, "start": 10.0, "end": 11.6, "words": [
        {"text": "Tum", "start": 10.0, "end": 10.5}, {"text": "kabhi", "start": 10.5, "end": 11.0},
        {"text": "nahi", "start": 11.0, "end": 11.3}, {"text": "haroge", "start": 11.3, "end": 11.6}]}
    out = ass.caption_events([rewritten], emphasis=[{"start": 10.8, "end": 11.2, "color": "FF3B30", "scale": 1.0}])[0]
    assert "{\\c&H00303BFF&}kabhi" not in out and "{\\c&H00303BFF&}nahi" in out  # by time: only "nahi" is inside


# ---- real render of a caption still ------------------------------------------------------------------------------------

def _pixels(png):
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB")
    return im, im.getdata()


@needs_ffmpeg
def test_stills_show_the_style_colours():
    from PIL import Image  # noqa: F401
    emph = [{"start": 10.8, "end": 11.2, "color": "FF3B30", "scale": 1.4}]
    classic = _pixels(preview.caption_frame_png(CHUNK, "classic", emph, active=1))[1]
    assert any(abs(r - 255) < 12 and abs(g - 255) < 12 and b < 40 for r, g, b in classic)  # the yellow spoken word
    assert any(r > 230 and 30 < g < 90 and 20 < b < 80 for r, g, b in classic)  # the red emphasised word
    pop = _pixels(preview.caption_frame_png(CHUNK, "bold_pop", None, active=1))[1]
    assert any(g > 230 and r < 80 and b < 60 for r, g, b in pop)  # neon green
    box = _pixels(preview.caption_frame_png(CHUNK, "karaoke_box", None, active=1))[1]
    assert any(r > 230 and 190 < g < 235 and b < 40 for r, g, b in box)  # the yellow box


@needs_ffmpeg
def test_the_box_sits_on_the_word_not_above_the_line():
    """libass pushes same-layer events apart; on its own layer the box must land on the text line."""
    im, _ = _pixels(preview.caption_frame_png(CHUNK, "karaoke_box", None, active=1, width=1080))
    w, h = im.size
    px = im.load()
    yellow = [y for y in range(h) for x in range(0, w, 3) if (lambda p: p[0] > 230 and 190 < p[1] < 235 and p[2] < 40)(px[x, y])]
    white = [y for y in range(h) for x in range(0, w, 3) if min(px[x, y]) > 235]
    assert yellow and white
    assert min(white) - 40 < min(yellow) < max(white) and max(yellow) <= max(white) + 40


# ---- workflow --------------------------------------------------------------------------------------------------------

def test_set_style_and_emphasis_are_saved_replaced_and_validated(tmp_path):
    job = make_job(tmp_path)
    assert workflow.set_caption_style(job, "bold_pop") == "bold_pop"
    with pytest.raises(workflow.WorkflowError):
        workflow.set_caption_style(job, "comic_sans")
    a = workflow.add_emphasis(job, 7.0, 7.3, color="red", text="virat")
    assert a["id"] == 1 and a["color"] == "FF3B30" and a["scale"] == 1.0
    b = workflow.add_emphasis(job, 7.4, 7.7, scale=1.5)
    assert b["id"] == 2 and b["color"] is None
    again = workflow.add_emphasis(job, 7.01, 7.31, color="#00ff00", scale=1.2)  # same words: replaced, not stacked
    assert again["id"] == 1 and again["color"] == "00FF00"
    cap = workflow.load_plan(job)["captions"]
    assert cap["style"] == "bold_pop" and [e["id"] for e in cap["emphasis"]] == [1, 2]
    with pytest.raises(workflow.WorkflowError, match="Nothing to change"):
        workflow.add_emphasis(job, 1.0, 2.0, scale=1.0)
    with pytest.raises(workflow.WorkflowError, match="colour"):
        workflow.add_emphasis(job, 1.0, 2.0, color="chartreuse-ish")
    with pytest.raises(workflow.WorkflowError):
        workflow.add_emphasis(job, 40.0, 41.0, color="red")  # past the end of the 30 s video
    assert workflow.remove_emphasis(job, 1) == 1
    with pytest.raises(workflow.WorkflowError):
        workflow.remove_emphasis(job, 99)
    assert workflow.remove_emphasis(job) == 1 and workflow.load_plan(job)["captions"]["emphasis"] == []


def test_emphasis_count_is_capped(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    monkeypatch.setattr(styles, "MAX_EMPHASIS", 2)
    workflow.add_emphasis(job, 1.0, 1.5, color="red")
    workflow.add_emphasis(job, 2.0, 2.5, color="red")
    with pytest.raises(workflow.WorkflowError, match="At most 2"):
        workflow.add_emphasis(job, 3.0, 3.5, color="red")


# ---- chat tools ------------------------------------------------------------------------------------------------------

def test_chat_sets_the_style_highlights_words_and_it_all_undoes(tmp_path):
    job = make_job(tmp_path)
    ctx = ToolContext(job=job)
    assert run_tool(ctx, "set_caption_style", {"style": "karaoke_box"})["style"] == "karaoke_box"
    assert "Unknown caption style" in run_tool(ctx, "set_caption_style", {"style": "neon"})["error"]
    out = run_tool(ctx, "highlight_words", {"phrase": "cricket", "color": "yellow", "size": 1.5})
    assert out["ok"] and out["found"] == 1 and out["highlighted"][0]["at"] == 9.2
    view = plan_view(job)["captions"]
    assert view["style"] == "karaoke_box" and view["highlights"][0]["text"] == "cricket"
    assert set(view["styles"]) == set(styles.STYLES)
    run_tool(ctx, "undo", {})
    assert plan_view(job)["captions"]["highlights"] == []


def test_highlight_defaults_to_red_and_a_bit_bigger_and_handles_misses(tmp_path):
    job = make_job(tmp_path)
    ctx = ToolContext(job=job)
    out = run_tool(ctx, "highlight_words", {"phrase": "virat koli"})
    assert out["ok"] and (out["color"], out["size"]) == ("red", 1.25)
    entry = workflow.load_plan(job)["captions"]["emphasis"][0]
    assert entry["color"] == "FF3B30" and entry["scale"] == 1.25 and entry["start"] == 7.0 and entry["end"] == pytest.approx(7.7)
    miss = run_tool(ctx, "highlight_words", {"phrase": "zebra"})
    assert "Could not find" in miss["error"]
    assert len(workflow.load_plan(job)["captions"]["emphasis"]) == 1  # the failed call changed nothing


def test_highlight_occurrences_all_first_last_nth(tmp_path):
    job = make_job(tmp_path)
    words = [{"word": w, "start": s, "end": s + 0.3} for w, s in
             [("never", 1.0), ("give", 1.5), ("never", 3.0), ("quit", 3.5), ("never", 6.0)]]
    (job / "transcript.json").write_text(json.dumps({"segments": [], "words": words}))
    ctx = ToolContext(job=job)

    def run(**kw):
        run_tool(ctx, "clear_highlights", {})
        out = run_tool(ctx, "highlight_words", {"phrase": "never", "color": "red", **kw})
        return out, [e["start"] for e in workflow.load_plan(job)["captions"]["emphasis"]]

    out, starts = run()
    assert out["found"] == 3 and starts == [1.0, 3.0, 6.0]
    assert run(occurrence="first")[1] == [1.0] and run(occurrence="last")[1] == [6.0]
    assert run(occurrence="nth", which=2)[1] == [3.0]
    assert "only 3" in run(occurrence="nth", which=9)[0]["error"]
    assert run_tool(ctx, "clear_highlights", {"highlight_id": 77})["error"]
    assert run_tool(ctx, "clear_highlights", {})["removed"] == 0


def test_highlight_finds_words_in_a_cached_english_caption_rewrite(tmp_path):
    job = make_job(tmp_path)
    (job / "captions_english.json").write_text(json.dumps([{"id": 0, "start": 7.0, "end": 8.0, "words": [
        {"text": "Namaste", "start": 5.0, "end": 5.4}, {"text": "dosto", "start": 5.5, "end": 5.9}]}]))
    out = run_tool(ToolContext(job=job), "highlight_words", {"phrase": "namaste", "color": "red"})
    assert out["ok"] and out["highlighted"][0]["at"] == 5.0  # not in the spoken transcript, only in the rewrite


# ---- the review-page panel ---------------------------------------------------------------------------------------------

def _open_page(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["job_dir"] = str(job)
    at.run()
    return job, at


def test_panel_changes_the_style_and_applies_a_highlight(tmp_path, monkeypatch):
    job, at = _open_page(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="cap_style").value == "classic"
    at.radio(key="cap_style").set_value("bold_pop").run()
    assert not at.exception and workflow.load_plan(job)["captions"]["style"] == "bold_pop"
    assert at.get("image")  # the caption preview still
    assert [b.label for b in at.button if b.label in ("Apply to this line", "Clear all highlights")] == [
        "Apply to this line", "Clear all highlights"]
    assert next(b for b in at.button if b.label == "Clear all highlights").disabled  # nothing highlighted yet
    workflow.add_emphasis(job, 7.0, 7.3, color="red", text="virat")
    at.run()
    assert not next(b for b in at.button if b.label == "Clear all highlights").disabled
    next(b for b in at.button if b.label == "Clear all highlights").click().run()
    assert not at.exception and workflow.load_plan(job)["captions"]["emphasis"] == []


# ---- the real export -----------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_render_uses_the_chosen_style(tmp_path, monkeypatch):
    import subprocess

    from src.rendering import pipeline as rendering
    job = make_job(tmp_path)
    for p in (job / "clips").glob("*.mp4"):
        p.unlink()
    plan = workflow.load_plan(job)
    plan["items"] = []
    (job / "plan.json").write_text(json.dumps(plan))
    workflow.set_caption_style(job, "karaoke_box")
    workflow.add_emphasis(job, 7.0, 7.3, color="red", text="virat")
    main = tmp_path / "main.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=0x2b2f3a:s=1080x1920:r=30:d=12", "-f", "lavfi", "-i", "sine=frequency=440:d=12",
                    "-shortest", "-pix_fmt", "yuv420p", str(main)], check=True)
    rendering.run(job, main, caption_mode="original")
    ass_text = (job / "overlays.ass").read_text(encoding="utf-8")
    assert "Style: ActiveLatin," in ass_text and ",ActiveLatin," in ass_text and "\\c&H00303BFF&" in ass_text
    assert (job / "final.mp4").stat().st_size > 1000
