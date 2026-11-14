"""Sound effects: catalog, library, fetching (YouTube/by name, mocked network, real ffmpeg), workflow, chat tools, the
audio mix in the export, and the review-page panel."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import config, workflow
from src.assistant import snapshots, ytclip
from src.assistant.tools import ToolContext, plan_view, run_tool
from src.rendering import compose
from src.rendering import pipeline as rendering
from src.rendering.compose import Overlay, Sound, build_audio_graph, render
from src.sfx import fetch, library
from src.sfx.catalog import CATALOG
from src.sourcing import youtube
from tests.test_chat_agent import make_job

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def tone(path: Path, freq=1000, seconds=1.0, volume=0.5):
    ff("-f", "lavfi", "-i", f"sine=frequency={freq}:d={seconds}", "-af", f"volume={volume}", str(path))
    return path


@pytest.fixture
def sfx(tmp_path, monkeypatch):
    """An empty library in a temp folder with a few real (generated) sounds."""
    d = tmp_path / "sfx"
    monkeypatch.setenv("VIBECROP_SFX_DIR", str(d))
    d.mkdir()
    for lib_id, name, aliases, freq in (("fah", "Fah", ["fahhh"], 1000), ("vine_boom", "Vine boom", ["boom"], 200),
                                        ("anime_wow", "Anime wow", ["wow anime"], 700),
                                        ("owen_wilson_wow", "Owen Wilson wow", ["wow"], 800)):
        f = tone(d / f"{lib_id}.wav", freq, volume=3.0).with_suffix(".mp3")
        ff("-i", str(d / f"{lib_id}.wav"), str(f))
        (d / f"{lib_id}.wav").unlink()
        library.register({"id": lib_id, "name": name, "aliases": aliases, "file": f.name, "duration": 1.0,
                          "source": "catalog", "url": ""})
    return d


# ---- catalog ---------------------------------------------------------------------------------------------------------

def test_catalog_is_about_fifty_unique_well_formed_entries():
    assert 45 <= len(CATALOG) <= 70
    assert len({c["id"] for c in CATALOG}) == len(CATALOG)
    for c in CATALOG:
        assert re.fullmatch(r"[a-z0-9_]+", c["id"]) and c["name"] and c["query"].strip()
        assert isinstance(c["aliases"], list) and all(isinstance(a, str) and a for a in c["aliases"])
    assert {"fah", "vine_boom", "bruh", "airhorn"} <= {c["id"] for c in CATALOG}


def test_sfx_folder_and_models_are_never_committed():
    ignored = subprocess.run(["git", "check-ignore", "sfx/fah.mp3", "models/x.rnnn"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent.parent).stdout.split()
    assert "sfx/fah.mp3" in ignored and "models/x.rnnn" in ignored
    code = subprocess.run(["git", "check-ignore", "src/sfx/catalog.py"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent.parent)
    assert code.returncode == 1  # the code under src/sfx is NOT ignored


# ---- library ---------------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_find_matches_names_aliases_and_close_spellings(sfx):
    assert library.find("fah")[0]["id"] == "fah" and library.find("fah")[0]["score"] == 1.0
    assert library.find("FAHHH")[0]["id"] == "fah"  # alias
    assert library.find("faaah")[0]["id"] == "fah"  # fuzzy
    assert library.find("vine boom")[0]["id"] == "vine_boom"
    assert library.find("boom")[0]["id"] == "vine_boom"
    assert library.find("fah sound")[0]["id"] == "fah"  # the name is inside a longer request
    assert library.find("zebra crossing") == [] and library.find("") == []


@needs_ffmpeg
def test_ambiguous_wow_returns_both_and_a_missing_file_is_ignored(sfx):
    ids = [h["id"] for h in library.find("wow")]
    assert set(ids) == {"anime_wow", "owen_wilson_wow"}
    (sfx / "fah.mp3").unlink()
    assert library.find("fah") == [] and "Fah" not in library.names()  # listed in the index, gone from disk


@needs_ffmpeg
def test_register_is_atomic_replaces_and_ids_stay_unique(sfx):
    assert library.unique_id("fah") == "fah_2" and library.unique_id("new") == "new"
    before = len(library.load())
    library.register({**library.get("fah"), "name": "Fah (renamed)"})
    assert len(library.load()) == before and library.get("fah")["name"] == "Fah (renamed)"
    with pytest.raises(library.SfxError):
        library.register({"id": "ghost", "name": "Ghost", "file": "ghost.mp3"})
    assert not list(sfx.glob("*.tmp"))
    (sfx / "index.json").write_text("{not json")
    assert library.load() == []  # a corrupt index never crashes the app


# ---- processing and fetching -------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_process_audio_trims_silence_normalises_and_caps(tmp_path):
    loud = tone(tmp_path / "t.wav", seconds=1.2, volume=0.2)
    padded = tmp_path / "padded.wav"
    ff("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=0.6", "-i", str(loud), "-f", "lavfi", "-i",
       "anullsrc=r=44100:cl=stereo:d=0.6", "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(padded))
    out = tmp_path / "out.mp3"
    length = fetch.process_audio(padded, out)
    assert 1.15 <= length <= 1.4  # 2.4 s in, the 0.6 s silences at both ends are gone
    assert fetch._max_volume_db(out) == pytest.approx(fetch.PEAK_DB, abs=1.5)
    assert fetch.process_audio(padded, tmp_path / "short.mp3", max_s=0.5) == pytest.approx(0.5, abs=0.1)
    silent = tmp_path / "silent.wav"
    ff("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=1", str(silent))
    with pytest.raises(fetch.FetchError):
        fetch.process_audio(silent, tmp_path / "x.mp3")
    junk = tmp_path / "junk.m4a"
    junk.write_bytes(b"not audio")
    with pytest.raises(fetch.FetchError):
        fetch.process_audio(junk, tmp_path / "y.mp3")


def _fake_download(tmp_path, calls=None):
    def download(cand, start, end, stem):
        if calls is not None:
            calls.append((cand.get("id"), round(start, 2), round(end, 2)))
        return tone(Path(str(stem) + ".wav"), seconds=max(end - start, 0.5), volume=0.4)
    return download


@needs_ffmpeg
def test_from_youtube_uses_the_given_time_and_saves_to_the_library(sfx, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ytclip, "video_info", lambda url: {"id": "abcdefghijk", "url": "https://www.youtube.com/watch?v=abcdefghijk",
                                                            "title": "Funny sounds", "channel": "C", "duration": 300.0})
    monkeypatch.setattr(youtube, "download_audio_section", _fake_download(tmp_path, calls))
    monkeypatch.setattr(ytclip, "find_window", lambda *a, **k: pytest.fail("a given time needs no AI"))
    e = fetch.from_youtube("https://youtu.be/abcdefghijk", from_time="1:05", length=2, name="my fah")
    assert calls == [("abcdefghijk", 65.0, 67.5)]
    assert e["id"] == "yt_abcdefghijk_65" and e["name"] == "my fah" and e["source"] == "youtube"
    assert e["file"].startswith("_added/") and library.available(e) and library.get(e["id"])


@needs_ffmpeg
def test_from_youtube_asks_gemini_to_listen_when_no_time_is_given(sfx, tmp_path, monkeypatch):
    calls, seen = [], {}
    monkeypatch.setattr(ytclip, "video_info", lambda url: {"id": "abcdefghijk", "url": "u", "title": "T", "channel": "C",
                                                            "duration": 300.0})
    monkeypatch.setattr(youtube, "download_audio_section", _fake_download(tmp_path, calls))

    def window(info, about, length, client=None, models=None, sleep=None, sound=False):
        seen.update(about=about, length=length, sound=sound)
        return 42.0, "the fah is clean here"

    monkeypatch.setattr(ytclip, "find_window", window)
    e = fetch.from_youtube("https://youtu.be/abcdefghijk", about="the fah sound")
    assert seen == {"about": "the fah sound", "length": 3.0, "sound": True} and calls[0][1] == 42.0
    assert e["why"] == "the fah is clean here"
    with pytest.raises(fetch.FetchError, match="which sound"):
        fetch.from_youtube("https://youtu.be/abcdefghijk")  # long video, no time, no description
    with pytest.raises(fetch.FetchError, match="only 300s"):
        fetch.from_youtube("https://youtu.be/abcdefghijk", from_time="9:00")
    monkeypatch.setattr(ytclip, "video_info", lambda url: (_ for _ in ()).throw(ytclip.YtError("not a youtube link")))
    with pytest.raises(fetch.FetchError, match="not a youtube link"):
        fetch.from_youtube("nope")


@needs_ffmpeg
def test_from_name_prefers_short_results_and_falls_through_failures(sfx, tmp_path, monkeypatch):
    calls = []
    cands = [{"id": "long", "title": "10 min compilation", "duration": 100.0, "url": "u/long", "channel": "c"},
             {"id": "bad", "title": "blocked", "duration": 3.0, "url": "u/bad", "channel": "c"},
             {"id": "good", "title": "Fah sound", "duration": 2.0, "url": "u/good", "channel": "c"}]
    monkeypatch.setattr(youtube, "search_candidates", lambda q, skip_ids=frozenset(), limit=10, min_s=30, max_s=1200: (
        calls.append(("search", q, min_s, max_s)) or cands))

    def dl(cand, start, end, stem):
        if cand["id"] == "bad":
            raise RuntimeError("HTTP 403")
        calls.append(("dl", cand["id"], start, end))
        return tone(Path(str(stem) + ".wav"), seconds=2.0)

    monkeypatch.setattr(youtube, "download_audio_section", dl)
    monkeypatch.setattr(ytclip, "find_window", lambda *a, **k: pytest.fail("short results are used from 0:00"))
    e = fetch.from_name("fah")
    assert calls[0] == ("search", "fah sound effect", 0.5, fetch.NAME_SEARCH_MAX_S)  # sounds are short: no 30 s minimum
    assert ("dl", "good", 0.0, 2.0) in calls and e["id"] == "web_fah" and e["source"] == "web"
    assert library.find("fah")[0]["id"] == "fah"  # the catalog 'fah' still wins by exact name
    again = fetch.from_name("fah")
    assert again["id"] == "web_fah_2"  # never overwrites another sound


@needs_ffmpeg
def test_from_name_with_only_long_results_asks_gemini_to_listen(sfx, tmp_path, monkeypatch):
    monkeypatch.setattr(youtube, "search_candidates", lambda *a, **k: [
        {"id": "long", "title": "compilation", "duration": 90.0, "url": "u/long", "channel": "c"}])
    monkeypatch.setattr(ytclip, "find_window", lambda info, about, length, **k: (30.0, "here") if k.get("sound") else pytest.fail())
    calls = []
    monkeypatch.setattr(youtube, "download_audio_section", _fake_download(tmp_path, calls))
    fetch.from_name("sad violin")
    assert calls == [("long", 30.0, 33.5)]
    monkeypatch.setattr(youtube, "search_candidates", lambda *a, **k: [])
    with pytest.raises(fetch.FetchError, match="couldn't find"):
        fetch.from_name("zzz")
    with pytest.raises(fetch.FetchError):
        fetch.from_name("   ")


# ---- workflow --------------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_workflow_sounds_validate_sort_and_remove(sfx, tmp_path):
    job = make_job(tmp_path)
    b = workflow.add_sound(job, "fah", 12.0)
    a = workflow.add_sound(job, "vine_boom", 3.0, volume=1.5)
    assert (a["id"], b["id"]) == (2, 1) and a["volume"] == 1.5 and b["volume"] == workflow.DEFAULT_SOUND_VOLUME
    assert [s["id"] for s in workflow.load_plan(job)["sounds"]] == [2, 1]  # in time order
    assert workflow.add_sound(job, "fah", 3.0)["id"] == 3  # the same sound again, even at the same time: fine
    for bad in (dict(lib_id="ghost", start=1), dict(lib_id="fah", start=-1), dict(lib_id="fah", start=31.0),
                dict(lib_id="fah", start=1, volume=3), dict(lib_id="fah", start=1, volume="loud")):
        with pytest.raises(workflow.WorkflowError):
            workflow.add_sound(job, bad["lib_id"], bad["start"], bad.get("volume"))
    upd = workflow.update_sound(job, 1, start=2.0, volume=0.3)
    assert (upd["start"], upd["volume"]) == (2.0, 0.3) and workflow.load_plan(job)["sounds"][0]["id"] in (1, 2, 3)
    assert [s["start"] for s in workflow.load_plan(job)["sounds"]] == sorted(s["start"] for s in workflow.load_plan(job)["sounds"])
    with pytest.raises(workflow.WorkflowError):
        workflow.update_sound(job, 99, start=1)
    assert workflow.remove_sound(job, 1) == 1
    with pytest.raises(workflow.WorkflowError):
        workflow.remove_sound(job, 1)
    assert workflow.remove_sound(job) == 2 and workflow.load_plan(job)["sounds"] == []


@needs_ffmpeg
def test_sounds_are_capped(sfx, tmp_path, monkeypatch):
    job = make_job(tmp_path)
    monkeypatch.setattr(workflow, "MAX_SOUNDS", 2)
    workflow.add_sound(job, "fah", 1.0)
    workflow.add_sound(job, "fah", 2.0)
    with pytest.raises(workflow.WorkflowError, match="At most 2"):
        workflow.add_sound(job, "fah", 3.0)


# ---- chat tools ------------------------------------------------------------------------------------------------------

def _talk_job(tmp_path):
    job = make_job(tmp_path)
    words = [("bhai", 1.0), ("kya", 1.4), ("bhai", 4.0), ("sun", 4.5), ("bhai", 9.0)]
    (job / "transcript.json").write_text(json.dumps({"segments": [], "words": [
        {"word": w, "start": s, "end": s + 0.3} for w, s in words]}))
    return job


@needs_ffmpeg
def test_chat_adds_a_sound_at_every_occurrence_of_a_phrase_and_it_undoes(sfx, tmp_path):
    job = _talk_job(tmp_path)
    ctx = ToolContext(job=job)
    out = run_tool(ctx, "add_sound", {"sound": "fah", "when_phrase": "bhai"})
    assert out["ok"] and out["from"] == "library" and out["matches_found"] == 3
    assert [p["at"] for p in out["placed"]] == [1.0, 4.0, 9.0]
    assert ctx.cards == [{"type": "sound", "lib_id": "fah", "caption": "Fah"}]
    assert [s["start"] for s in plan_view(job)["sounds"]] == [1.0, 4.0, 9.0]
    run_tool(ctx, "undo", {})
    assert plan_view(job)["sounds"] == []


@needs_ffmpeg
def test_chat_occurrence_align_offset_and_volume(sfx, tmp_path):
    job = _talk_job(tmp_path)
    ctx = ToolContext(job=job)

    def starts(**kw):
        run_tool(ctx, "remove_sound", {})
        run_tool(ctx, "add_sound", {"sound": "fah", "when_phrase": "bhai", **kw})
        return [s["start"] for s in workflow.load_plan(job)["sounds"]]

    assert starts(occurrence="first") == [1.0] and starts(occurrence="last") == [9.0]
    assert starts(occurrence="nth", which=2) == [4.0]
    assert starts(occurrence="first", align="end") == [1.3] and starts(occurrence="first", offset=-0.5) == [0.5]
    assert starts(occurrence="first", offset=-5) == [0.0]  # never before the start of the video
    assert "only 3" in run_tool(ctx, "add_sound", {"sound": "fah", "when_phrase": "bhai", "occurrence": "nth", "which": 7})["error"]
    run_tool(ctx, "remove_sound", {})
    run_tool(ctx, "add_sound", {"sound": "fah", "at_time": 5, "volume": 1.4})
    assert workflow.load_plan(job)["sounds"][0]["volume"] == 1.4


@needs_ffmpeg
def test_chat_needs_a_time_and_validates_before_any_download(sfx, tmp_path, monkeypatch):
    job = _talk_job(tmp_path)
    monkeypatch.setattr(fetch, "from_name", lambda *a, **k: pytest.fail("must not download before the time is valid"))
    ctx = ToolContext(job=job)
    assert "when the sound plays" in run_tool(ctx, "add_sound", {"sound": "not in library"})["error"]
    assert "Could not find" in run_tool(ctx, "add_sound", {"sound": "not in library", "when_phrase": "zebra"})["error"]
    assert plan_view(job)["sounds"] == []


@needs_ffmpeg
def test_chat_fetches_unknown_sounds_by_name_and_reports_close_matches(sfx, tmp_path, monkeypatch):
    job = _talk_job(tmp_path)
    fetched = []

    def from_name(name, client=None, log=print):
        fetched.append(name)
        (config.sfx_dir() / "_added").mkdir(exist_ok=True)
        ff("-f", "lavfi", "-i", "sine=frequency=500:d=1", str(config.sfx_dir() / "_added" / "web_nokia.mp3"))
        return library.register({"id": "web_nokia", "name": name, "aliases": [], "file": "_added/web_nokia.mp3",
                                 "duration": 1.0, "source": "web", "url": "u"})

    monkeypatch.setattr(fetch, "from_name", from_name)
    ctx = ToolContext(job=job)
    out = run_tool(ctx, "add_sound", {"sound": "nokia ringtone", "at_time": 2})
    assert out["ok"] and out["from"].startswith("youtube search") and fetched == ["nokia ringtone"]
    assert library.get("web_nokia") and plan_view(job)["sounds"][0]["name"] == "nokia ringtone"
    assert run_tool(ctx, "add_sound", {"sound": "wow", "at_time": 2.5})["sound"] == "Owen Wilson wow"  # exact alias wins
    library.register({**library.get("owen_wilson_wow"), "aliases": []})  # now "wow" fits both names equally well
    out = run_tool(ctx, "add_sound", {"sound": "wow", "at_time": 3})
    assert out["from"] == "library" and len(out["also_matches"]) == 1 and "note" in out  # anime wow vs owen wilson wow
    assert fetched == ["nokia ringtone"]  # the library match did not hit the network
    monkeypatch.setattr(fetch, "from_name", lambda *a, **k: (_ for _ in ()).throw(fetch.FetchError("nothing found")))
    assert run_tool(ctx, "add_sound", {"sound": "qwertyuiop", "at_time": 1})["error"] == "nothing found"
    assert len(plan_view(job)["sounds"]) == 3  # the failed call changed nothing


@needs_ffmpeg
def test_chat_sound_from_a_youtube_link(sfx, tmp_path, monkeypatch):
    job = _talk_job(tmp_path)
    seen = {}

    def from_youtube(url, about="", from_time=None, length=None, name=None, client=None, log=print):
        seen.update(url=url, about=about, from_time=from_time, length=length, name=name)
        ff("-f", "lavfi", "-i", "sine=frequency=900:d=1", str(config.sfx_dir() / "yt_x.mp3"))
        e = library.register({"id": "yt_x", "name": name or about, "aliases": [], "file": "yt_x.mp3", "duration": 1.0,
                              "source": "youtube", "url": url, "window_start": 42.0})
        return {**e, "why": "clean"}

    monkeypatch.setattr(fetch, "from_youtube", from_youtube)
    out = run_tool(ToolContext(job=job), "add_sound_from_youtube", {
        "url": "https://youtu.be/abcdefghijk", "from_time": "0:42", "length": 2, "when_phrase": "kya", "name": "ding dong"})
    assert out["ok"] and out["placed"][0]["at"] == 1.4 and out["starts_at_in_video"] == 42.0 and out["picked_because"] == "clean"
    assert seen == {"url": "https://youtu.be/abcdefghijk", "about": "", "from_time": "0:42", "length": 2.0, "name": "ding dong"}
    monkeypatch.setattr(fetch, "from_youtube", lambda *a, **k: (_ for _ in ()).throw(fetch.FetchError("private video")))
    assert run_tool(ToolContext(job=job), "add_sound_from_youtube", {"url": "u", "at_time": 1})["error"] == "private video"


@needs_ffmpeg
def test_chat_list_update_and_remove(sfx, tmp_path):
    job = _talk_job(tmp_path)
    ctx = ToolContext(job=job)
    listed = run_tool(ctx, "list_sounds", {})
    assert listed["count"] == 4 and {"id": "fah", "name": "Fah"} in listed["sounds"]
    assert [s["id"] for s in run_tool(ctx, "list_sounds", {"query": "boom"})["sounds"]] == ["vine_boom"]
    run_tool(ctx, "add_sound", {"sound": "fah", "at_time": 2})
    sid = plan_view(job)["sounds"][0]["id"]
    assert "Give a new start" in run_tool(ctx, "update_sound", {"sound_id": sid})["error"]
    out = run_tool(ctx, "update_sound", {"sound_id": sid, "start": 6, "volume": 1.2})
    assert out["sound"]["start"] == 6.0 and out["sound"]["volume"] == 1.2
    assert run_tool(ctx, "remove_sound", {"sound_id": 99})["error"]
    assert run_tool(ctx, "remove_sound", {"sound_id": sid})["removed"] == 1
    assert snapshots.can_undo(job)


# ---- the audio mix in the export -------------------------------------------------------------------------------------

def test_audio_graph_delays_scales_and_mixes_without_renormalising():
    g = build_audio_graph([Sound(Path("a.mp3"), 3.25, 0.8), Sound(Path("b.mp3"), 0.0, 1.5)], 2, True, 8.0)
    assert "[0:a:0]aresample=44100,aformat=channel_layouts=stereo[a0]" in g
    assert "[2:a:0]" in g and "volume=0.800,adelay=3250|3250[s0]" in g
    assert "[3:a:0]" in g and "volume=1.500,adelay=0|0[s1]" in g
    assert "[a0][s0][s1]amix=inputs=3:duration=first:normalize=0" in g and g.endswith("alimiter=limit=0.97[aout]")
    silent = build_audio_graph([Sound(Path("a.mp3"), 1.0, 1.0)], 1, False, 8.0)
    assert silent.startswith("anullsrc=r=44100:cl=stereo,atrim=0:8.000[a0]")  # a silent speaker video is mixed onto silence


def test_without_sounds_the_original_audio_mapping_is_kept(tmp_path, monkeypatch):
    seen = {}

    def fake(cmd, cwd, total_s, progress):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"x")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(compose, "_run_ffmpeg", fake)
    monkeypatch.setattr(compose, "require_binary", lambda n: n)
    render(tmp_path / "main.mp4", [], tmp_path / "o.mp4")
    assert seen["cmd"][seen["cmd"].index("-map") + 3] == "0:a:0?" and "[aout]" not in " ".join(seen["cmd"])
    render(tmp_path / "main.mp4", [], tmp_path / "o.mp4", sounds=[Sound(tmp_path / "missing.mp3", 1.0)])  # missing file: ignored
    assert "[aout]" not in " ".join(seen["cmd"])


def _mean_db(video: Path, t0: float, t1: float, highpass: int = 800) -> float:
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-ss", str(t0), "-t", str(t1 - t0), "-i", str(video), "-af",
                        f"highpass=f={highpass},volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    return float(re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr).group(1))


@needs_ffmpeg
def test_the_export_has_the_sound_at_the_right_moment_and_nowhere_else(sfx, tmp_path):
    main = tmp_path / "main.mp4"  # 8 s, speaker audio is a low 100 Hz tone (the 800 Hz highpass in _mean_db removes it)
    ff("-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=30:d=8", "-f", "lavfi", "-i", "sine=frequency=100:d=8",
       "-shortest", "-pix_fmt", "yuv420p", str(main))
    snd = Sound(library.path_of(library.get("fah")), 3.0, 0.8)  # a loud 1 kHz tone, 1 s long
    out = render(main, [], tmp_path / "out.mp4", sounds=[snd])
    inside, before, after = _mean_db(out, 3.1, 3.9), _mean_db(out, 1.0, 2.5), _mean_db(out, 5.0, 7.0)
    assert inside > before + 15 and inside > after + 15  # only the 3-4 s window carries the 1 kHz effect
    assert _mean_db(out, 0.5, 1.5, highpass=100) > -35  # the speaker's own audio is still there
    assert float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
                                capture_output=True, text=True).stdout) == pytest.approx(8.0, abs=0.15)  # not longer


@needs_ffmpeg
def test_overlapping_effects_do_not_clip_and_a_silent_speaker_still_gets_sound(sfx, tmp_path):
    silent = tmp_path / "silent.mp4"
    ff("-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=30:d=6", "-pix_fmt", "yuv420p", str(silent))
    s = library.path_of(library.get("fah"))
    out = render(silent, [], tmp_path / "o.mp4", sounds=[Sound(s, 1.0, 2.0), Sound(s, 1.0, 2.0), Sound(s, 1.0, 2.0)])
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(out), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    peak = float(re.search(r"max_volume:\s*(-?[\d.]+) dB", r.stderr).group(1))
    assert -12 < peak <= 0.0  # three overlapping +6 dB effects: loud, but limited
    assert _mean_db(out, 1.1, 1.9) > _mean_db(out, 3.5, 5.5) + 15


@needs_ffmpeg
def test_pipeline_mixes_planned_sounds_and_skips_missing_ones(sfx, tmp_path):
    job = make_job(tmp_path)
    for p in (job / "clips").glob("*.mp4"):
        p.unlink()
    plan = workflow.load_plan(job)
    plan["items"] = []
    (job / "plan.json").write_text(json.dumps(plan))
    workflow.add_sound(job, "fah", 3.0)
    workflow.add_sound(job, "vine_boom", 5.0)
    (sfx / "vine_boom.mp3").unlink()
    main = tmp_path / "main.mp4"
    ff("-f", "lavfi", "-i", "color=c=gray:s=1080x1920:r=30:d=8", "-f", "lavfi", "-i", "sine=frequency=100:d=8",
       "-shortest", "-pix_fmt", "yuv420p", str(main))
    logs = []
    out = rendering.run(job, main, log=logs.append)
    assert any("missing from the sound library" in m for m in logs) and any("1 sound effects" in m for m in logs)
    assert _mean_db(out, 3.1, 3.9) > _mean_db(out, 1.0, 2.5) + 15


# ---- the review-page panel -------------------------------------------------------------------------------------------

def _open(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = make_job(tmp_path)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["job_dir"] = str(job)
    at.run()
    return job, at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


@needs_ffmpeg
def test_panel_adds_edits_and_deletes_sounds(sfx, tmp_path, monkeypatch):
    job, at = _open(tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]
    assert any("Sound effects (0)" in e.label for e in at.expander)
    at.selectbox(key="sfx_pick").set_value("vine_boom")
    at.number_input(key="sfx_when").set_value(4.5)
    _button(at, "Add").click().run()
    assert not at.exception
    s = workflow.load_plan(job)["sounds"][0]
    assert (s["lib_id"], s["start"], s["volume"]) == ("vine_boom", 4.5, workflow.DEFAULT_SOUND_VOLUME)
    assert at.get("audio"), "each placed sound can be listened to"
    ver = f"{s['id']}_{s['start']}_{s['volume']}"
    at.slider(key=f"snd_vol_{ver}").set_value(1.3).run()
    assert not at.exception and workflow.load_plan(job)["sounds"][0]["volume"] == 1.3
    s = workflow.load_plan(job)["sounds"][0]
    at.number_input(key=f"snd_at_{s['id']}_{s['start']}_{s['volume']}").set_value(7.0).run()
    assert workflow.load_plan(job)["sounds"][0]["start"] == 7.0
    s = workflow.load_plan(job)["sounds"][0]
    _button(at, "Delete").click().run()
    assert not at.exception and workflow.load_plan(job)["sounds"] == []


def test_panel_explains_an_empty_library(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_SFX_DIR", str(tmp_path / "nothing-here"))
    job, at = _open(tmp_path, monkeypatch)
    assert not at.exception
    assert any("build_sfx_library" in i.value for i in at.info)


# ---- phrase search accuracy (found with a live Gemini run: "hai" was matching "bhai") ----------------------------------

def test_short_words_must_match_exactly_but_longer_names_stay_fuzzy():
    from src.assistant.tools import find_phrase
    words = [{"word": w, "start": float(i), "end": i + 0.3} for i, w in
             enumerate(["bhai", "kya", "kar", "raha", "hai", "bhai", "Kohli", "the"])]
    assert [h["start"] for h in find_phrase(words, "bhai")] == [0.0, 5.0]  # not "hai" at 4.0
    assert [h["start"] for h in find_phrase(words, "hai")] == [4.0]  # and "hai" does not find "bhai"
    assert [h["start"] for h in find_phrase(words, "koli")] == [6.0]  # a 4-letter name still tolerates a spelling slip
    assert find_phrase(words, "tha") == [] and [h["start"] for h in find_phrase(words, "kar")] == [2.0]
