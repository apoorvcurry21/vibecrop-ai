"""Talking-head cleanup: voice chain, denoise, pauses, retake detection (Groq + Gemini mocked), the cut pass with a real
ffmpeg, transcript remapping, and how the workflow and the upload screen use it."""
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src import config, workflow
from src.talkinghead import denoise, pipeline, retakes, silence, voice
from src.transcription.audio import probe

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def W(text, start, end):
    return {"word": text, "start": start, "end": end}


def words_from(text, step=0.4, dur=0.3):
    return [W(t, i * step, i * step + dur) for i, t in enumerate(text.split())]


def speechlike(path: Path, bursts, total, noise=0.0, freq=300):
    """Tone bursts (the 'words') at the given (start, end) times in `total` seconds, optionally over pink noise."""
    gate = "+".join(f"between(t,{a},{b})" for a, b in bursts)
    src = f"aevalsrc='0.4*sin(2*PI*{freq}*t)*({gate})':s=48000:d={total}"  # quoted: the commas are not filter separators
    if not noise:
        ff("-f", "lavfi", "-i", src, str(path))
        return path
    ff("-f", "lavfi", "-i", src, "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={noise}:d={total}:r=48000",
       "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0", str(path))
    return path


def loudness(path: Path) -> float:
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)[-1])


def mean_db(path: Path, t0=None, t1=None) -> float:
    seek = ["-ss", str(t0), "-t", str(t1 - t0)] if t0 is not None else []
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", *seek, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    return float(re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr).group(1))


# ---- voice chain ---------------------------------------------------------------------------------------------------

def test_voice_chain_has_every_stage_in_order_with_its_constants():
    f = voice.voice_filter()
    stages = [s.split("=")[0] for s in f.split(",")]
    assert stages == ["highpass", "equalizer", "equalizer", "acompressor", "loudnorm", "aresample"]
    assert f"highpass=f={voice.HIGHPASS_HZ}" in f and f"loudnorm=I={voice.TARGET_LUFS}" in f
    assert f"equalizer=f={voice.MUD_HZ}" in f and f"g={voice.MUD_GAIN_DB}" in f and f"f={voice.PRESENCE_HZ}" in f
    assert f"ratio={voice.COMP_RATIO}" in f and f"aresample={voice.OUTPUT_RATE}" in f


@needs_ffmpeg
def test_voice_chain_levels_a_quiet_and_a_loud_voice_to_the_target(tmp_path):
    for name, vol in (("quiet", 0.03), ("loud", 0.9)):
        src = tmp_path / f"{name}.wav"
        ff("-f", "lavfi", "-i", "sine=frequency=300:d=8", "-af", f"volume={vol}", str(src))
        out = tmp_path / f"{name}_out.wav"
        ff("-i", str(src), "-af", voice.voice_filter(), str(out))
        assert loudness(out) == pytest.approx(voice.TARGET_LUFS, abs=2.0), name
        assert probe(out)["duration"] == pytest.approx(8.0, abs=0.3)
        rate = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(out)],
                              capture_output=True, text=True).stdout.strip()
        assert rate == str(voice.OUTPUT_RATE)


# ---- denoise ---------------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_denoise_falls_back_to_afftdn_when_the_model_cannot_be_downloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(denoise.requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    logs = []
    src = speechlike(tmp_path / "in.wav", [(0.5, 1.5)], 2, noise=0.05)
    method = denoise.denoise(src, tmp_path / "out.wav", logs.append)
    assert method == denoise.AFFTDN and (tmp_path / "out.wav").stat().st_size > 1000
    assert any("simpler filter" in m for m in logs) and not (tmp_path / "models" / denoise.MODEL_NAME).exists()


def test_model_download_is_validated_cached_and_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_MODELS_DIR", str(tmp_path / "m"))
    calls = []
    monkeypatch.setattr(denoise.requests, "get", lambda url, timeout: calls.append(url) or SimpleNamespace(
        content=b"x" * 150_000, raise_for_status=lambda: None))
    got = denoise.ensure_model(lambda m: None)
    assert got == tmp_path / "m" / denoise.MODEL_NAME and got.stat().st_size == 150_000
    assert denoise.ensure_model(lambda m: None) == got and len(calls) == 1  # cached: no second download
    assert not list((tmp_path / "m").glob("*.part"))
    got.unlink()
    monkeypatch.setattr(denoise.requests, "get", lambda *a, **k: SimpleNamespace(content=b"<html>404", raise_for_status=lambda: None))
    assert denoise.ensure_model(lambda m: None) is None  # an error page is not a model


@needs_ffmpeg
def test_arnndn_removes_background_noise_when_the_model_is_available(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_MODELS_DIR", str(tmp_path / "models"))
    if denoise.ensure_model(lambda m: None) is None:
        pytest.skip("the RNNoise model could not be downloaded (offline)")
    src = speechlike(tmp_path / "in.wav", [(1.0, 2.0)], 3, noise=0.08)
    out = tmp_path / "out.wav"
    assert denoise.denoise(src, out) == denoise.ARNNDN
    assert mean_db(out, 2.3, 2.9) < mean_db(src, 2.3, 2.9) - 8  # a noise-only stretch is much quieter


# ---- pauses ----------------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_silencedetect_finds_the_gaps_including_the_start_and_the_end(tmp_path):
    f = speechlike(tmp_path / "s.wav", [(0.8, 1.6), (3.6, 4.4)], 6.0)
    found = silence.detect(f, 6.0)
    assert len(found) == 3
    assert found[0][0] == 0.0 and found[0][1] == pytest.approx(0.8, abs=0.1)
    assert found[1] == pytest.approx((1.6, 3.6), abs=0.1) and found[2][1] == pytest.approx(6.0, abs=0.1)


def test_cuts_keep_a_pad_ignore_short_pauses_and_never_touch_words():
    words = [W("a", 1.0, 1.5), W("b", 4.0, 4.5)]
    cuts = silence.cuts_from_silence([(1.5, 4.0), (6.0, 6.4)], words, 10.0)
    assert cuts == [(pytest.approx(1.68), pytest.approx(3.82))]  # a 2.5 s pause keeps 0.18 s on both sides; 0.4 s stays
    # silence at the very start and end goes entirely, except the pad beside speech
    cuts = silence.cuts_from_silence([(0.0, 1.0), (4.5, 10.0)], words, 10.0)
    assert cuts[0] == (0.0, pytest.approx(0.82)) and cuts[1] == (pytest.approx(4.68), 10.0)
    # a word that the detector placed inside a "silence" (quiet speech) splits the cut and is never removed
    cuts = silence.cuts_from_silence([(1.5, 6.0)], [W("quiet", 3.0, 3.4)], 10.0)
    assert len(cuts) == 2 and cuts[0][1] <= 3.0 - silence.WORD_GUARD_S + 1e-9 and cuts[1][0] >= 3.4 + silence.WORD_GUARD_S - 1e-9
    assert silence.cuts_from_silence([(1.0, 1.7)], [], 10.0) == [(pytest.approx(1.18), pytest.approx(1.52))]  # 0.7 s -> 0.36 s
    assert silence.cuts_from_silence([(1.0, 1.55)], [], 10.0) == []  # 0.55 s is natural rhythm: left alone


# ---- retake detection ----------------------------------------------------------------------------------------------

TEXT = ("Hello everyone, today I want to talk about habits. The best habit is waking up. "
        "The best habit is waking up early, because mornings are calm. It helps you you you focus.")
RETAKE = (9, 14)  # "The best habit is waking up." (the failed first take)
STUTTER = (28, 30)  # "you you you"; the cut must leave the last "you", i.e. words 28-29


def test_numbered_transcript_marks_pauses():
    w = [W("a", 0, 0.2), W("b", 0.3, 0.5), W("c", 1.5, 1.8)]
    assert retakes.numbered_transcript(w) == "[0] a [1] b | [2] c"


def test_sanitize_drops_garbage_merges_overlaps_and_caps_size():
    raw = [{"from": 5, "to": 8, "reason": "retake"}, {"from": 7, "to": 10, "reason": "stutter"},
           {"from": "x", "to": 2}, {"from": 3, "to": 1}, {"from": 0, "to": 99}, {"from": 1, "to": 1, "reason": "bogus"},
           {"from": 0, "to": 70}, "junk", None]
    out = retakes.sanitize(raw, 20)
    assert out == [{"from": 1, "to": 1, "reason": "retake"}, {"from": 5, "to": 10, "reason": "retake"}]
    assert retakes.sanitize({"not": "a list"}, 5) == []


def test_stutters_cut_every_copy_but_the_last():
    w = words_from("It helps you you you focus on goals")
    assert retakes.stutter_candidates(w) == [{"from": 2, "to": 3, "reason": "stutter"}]  # you you [you]
    w2 = words_from("so I I think the the best way way")
    assert [(c["from"], c["to"]) for c in retakes.stutter_candidates(w2)] == [(1, 1), (4, 4), (7, 7)]
    assert retakes.stutter_candidates(words_from("one two three four five")) == []
    far = [W("no", 0, 0.3), W("no", 2.0, 2.3)]  # repeated, but seconds apart: that is not a stutter
    assert retakes.stutter_candidates(far) == []
    two = words_from("bye bye now see you see you later")  # a two-word unit
    assert (3, 4) in [(c["from"], c["to"]) for c in retakes.stutter_candidates(two)]


def test_a_bad_stutter_proposal_is_repaired_so_the_word_survives():
    w = words_from("It helps you you you focus")
    assert retakes.repair(w, {"from": 2, "to": 4, "reason": "stutter"}) == {"from": 2, "to": 3, "reason": "stutter"}
    assert retakes.repair(w, {"from": 0, "to": 1, "reason": "stutter"}) is None  # not a repetition at all
    keep = {"from": 0, "to": 5, "reason": "retake"}
    assert retakes.repair(w, keep) is keep


def test_restart_candidates_find_the_failed_first_take_only_at_sentence_boundaries():
    w = words_from(TEXT)
    got = retakes.restart_candidates(w)
    assert got == [{"from": RETAKE[0], "to": RETAKE[1], "reason": "retake"}]
    midway = words_from("I said that we want to go and then we want to go home")  # same words, not a sentence start
    assert retakes.restart_candidates(midway) == []
    far = words_from("Go home now. " + "filler " * 30 + "Go home now.")
    assert retakes.restart_candidates(far) == []  # too far apart to be a retake


def test_to_time_cuts_snaps_to_words_and_stays_clear_of_neighbours():
    w = [W("a", 0.0, 0.5), W("b", 0.6, 1.0), W("c", 1.02, 1.4), W("d", 1.5, 1.9)]
    (cut,) = retakes.to_time_cuts(w, [{"from": 1, "to": 2, "reason": "retake"}], 2.0)
    assert cut["start"] == pytest.approx(0.55) and cut["end"] == pytest.approx(1.45) and cut["text"] == "b c"
    (cut,) = retakes.to_time_cuts(w, [{"from": 0, "to": 0, "reason": "stutter"}], 2.0)
    assert cut["start"] == 0.0 and cut["end"] == pytest.approx(0.55)
    (cut,) = retakes.to_time_cuts(w, [{"from": 3, "to": 3, "reason": "retake"}], 1.95)
    assert cut["end"] == pytest.approx(1.94)  # stops just short of the end of the video, never past it


class FakeGroq:
    def __init__(self, cuts=None, text=None, error=None):
        self.cuts, self.text, self.error, self.calls = cuts, text, error, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls += 1
        assert kw["temperature"] == 0 and kw["response_format"] == {"type": "json_object"}
        if self.error:
            raise RuntimeError(self.error)
        body = self.text if self.text is not None else json.dumps({"cuts": self.cuts})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])


@pytest.fixture
def gemini(monkeypatch):
    """Gemini's verdict, scripted: approve(ids) -> the call returns those; also records what it was shown."""
    state = SimpleNamespace(approve=None, fail=None, prompts=[])

    def fake(client, models, history, sleep, system, schema):
        state.prompts.append(history[0].parts[0].text)
        if state.fail:
            raise retakes.GeminiError(state.fail)
        ids = state.approve if state.approve is not None else list(range(10))
        return json.dumps({"approved": ids, "notes": "ok"}), "m"

    monkeypatch.setattr(retakes, "generate_with_fallback", fake)
    state.client = object()
    return state


def run_find(words, groq, gemini, duration=30.0):
    return retakes.find_retakes(words, "en", duration, groq_client=groq, gemini_client=gemini.client,
                                sleep=lambda s: None, log=lambda m: None)


def test_groq_and_the_code_propose_gemini_approves_and_only_those_are_cut(gemini):
    w = words_from(TEXT)
    groq = FakeGroq(cuts=[{"from": RETAKE[0], "to": RETAKE[1], "reason": "retake"},
                          {"from": STUTTER[0], "to": STUTTER[1], "reason": "stutter"}])  # Groq's stutter cuts all three
    res = run_find(w, groq, gemini)
    assert groq.calls == 2  # two Groq runs, merged
    assert [(c["reason"], c["text"]) for c in res["cuts"]] == [("retake", "The best habit is waking up."), ("stutter", "you you")]
    assert res["proposed"] == 2 and res["approved"] == 2 and res["warnings"] == []
    prompt = gemini.prompts[0]
    assert "restarts: yes" in prompt and "[REMOVE: The best habit is waking up.]" in prompt  # the hint Gemini needs
    assert "[REMOVE: you you]" in prompt and "restarts: no" not in prompt.split("Proposal 1")[0]


def test_nothing_is_cut_that_gemini_did_not_approve(gemini):
    w = words_from(TEXT)
    gemini.approve = [1]  # only the stutter
    res = run_find(w, FakeGroq(cuts=[]), gemini)
    assert [c["reason"] for c in res["cuts"]] == ["stutter"] and res["proposed"] == 2 and res["approved"] == 1
    gemini.approve = []
    assert run_find(w, FakeGroq(cuts=[]), gemini)["cuts"] == []


def test_when_gemini_fails_nothing_is_cut_and_the_user_is_told(gemini):
    gemini.fail = "503 overloaded"
    res = run_find(words_from(TEXT), FakeGroq(cuts=[{"from": 11, "to": 16, "reason": "retake"}]), gemini)
    assert res["cuts"] == [] and "nothing was cut without a second check" in res["warnings"][0]


def test_when_groq_fails_the_code_found_candidates_still_get_checked(gemini):
    res = run_find(words_from(TEXT), FakeGroq(error="rate limited"), gemini)
    assert [c["reason"] for c in res["cuts"]] == ["retake", "stutter"]
    assert "Groq was not available" in res["warnings"][0] and "rate limited" in res["warnings"][0]


def test_groq_sending_nonsense_is_survivable(gemini):
    w = words_from(TEXT)
    for body in ("I could not find anything", "```json\n{\"cuts\": []}\n```", "[1, 2, 3]", '{"cuts": "none"}'):
        res = run_find(w, FakeGroq(text=body), gemini)
        assert len(res["cuts"]) == 2  # the code-found candidates carry on whatever Groq said
    assert retakes._json_object("```json\n{\"cuts\": []}\n```") == {"cuts": []}
    with pytest.raises(retakes.RetakeError):
        retakes._json_object("nope")


def test_too_many_cuts_is_distrusted_and_short_transcripts_are_skipped(gemini, monkeypatch):
    w = words_from(TEXT)
    monkeypatch.setattr(retakes, "MAX_REMOVED_FRACTION", 0.05)  # the two cuts (about 3 s of 30 s) are now "too much"
    res = run_find(w, FakeGroq(cuts=[]), gemini)
    assert res["cuts"] == [] and "more than 40%" in res["warnings"][0]
    monkeypatch.setattr(retakes, "MAX_REMOVED_FRACTION", 0.40)
    assert len(run_find(w, FakeGroq(cuts=[]), gemini)["cuts"]) == 2
    n = len(gemini.prompts)
    assert run_find(words_from("too short"), FakeGroq(cuts=[]), gemini)["cuts"] == [] and len(gemini.prompts) == n


# ---- time maths and the transcript -------------------------------------------------------------------------------------

def test_keep_segments_merge_cuts_and_drop_slivers():
    assert pipeline.keep_segments(10, [(2, 3), (2.5, 4), (8, 12)]) == [(0, 2), (4, 8)]
    assert pipeline.keep_segments(10, []) == [(0, 10)] and pipeline.keep_segments(10, [(0, 10)]) == []
    assert pipeline.keep_segments(10, [(1, 5), (5.05, 9.95)]) == [(0, 1)]  # 0.05 s slivers are dropped
    assert pipeline.merge_cuts([(5, 6), (1, 2), (2, 3)], 10) == [(1, 3), (5, 6)]


def test_remap_shifts_words_drops_cut_ones_and_rebuilds_the_segments_from_the_words():
    """The real failure this guards: Whisper's segment edges disagree with its word times, so shifting its segments
    produced 'The The ...' and lost the word 'It'. Segments are now rebuilt from the cleaned words."""
    words = [W("Hello", 0.5, 1.0), W("there.", 1.0, 1.5), W("Say", 3.6, 4.0), W("this.", 4.0, 4.5), W("Say", 4.6, 5.0),
             W("this", 5.0, 5.4), W("It", 8.0, 8.4), W("works.", 8.4, 9.0)]
    segments = [{"id": 0, "start": 0.0, "end": 4.55, "text": "Hello there. Say this."},  # edge 4.55: inside "Say"(4.6)? no
                {"id": 1, "start": 4.55, "end": 8.1, "text": "Say this It"}, {"id": 2, "start": 8.1, "end": 9.0, "text": "works."}]
    tr = {"language": "en", "text": "", "duration": 10.0, "words": words, "segments": segments}
    keep = pipeline.keep_segments(10.0, [(0.0, 0.4), (1.6, 4.55), (5.5, 7.9)])
    out = pipeline.remap_transcript(tr, keep, 5.7)
    assert [w["word"] for w in out["words"]] == ["Hello", "there.", "Say", "this", "It", "works."]
    assert out["words"][0]["start"] == pytest.approx(0.1) and out["words"][2]["start"] == pytest.approx(1.25)  # 4.6 s minus 0.4 + 2.95 s cut before it
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(out["words"], out["words"][1:]))
    assert out["text"] == "Hello there. Say this It works." and out["duration"] == 5.7 and out["video_duration"] == 5.7
    assert [s["text"] for s in out["segments"]] == ["Hello there.", "Say this It works."]
    assert [s["id"] for s in out["segments"]] == [0, 1]


def test_resegment_splits_at_sentence_ends_pauses_and_long_runs():
    w = [W("one", 0, 0.3), W("two.", 0.3, 0.6), W("three", 0.7, 1.0), W("four", 2.5, 2.8), W("five?", 2.8, 3.0)]
    assert [s["text"] for s in pipeline.resegment(w)] == ["one two.", "three", "four five?"]  # . and a 1.5 s pause
    assert [s["text"] for s in pipeline.resegment([W("नमस्ते।", 0, 1), W("दोस्तों", 1, 2)], "hi")] == ["नमस्ते।", "दोस्तों"]
    many = [W(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(65)]
    assert [len(s["text"].split()) for s in pipeline.resegment(many)] == [30, 30, 5]
    assert pipeline.resegment([W("你", 0, 0.2), W("好", 0.2, 0.4)], "zh")[0]["text"] == "你好"  # no spaces in Chinese/Japanese
    assert pipeline.resegment([]) == []


def test_cut_graph_has_one_piece_per_keep_and_the_voice_chain_after_the_concat():
    g = pipeline.build_cut_graph([(0.0, 1.5), (2.0, 3.25)])
    assert "[0:v]trim=start=0.000:end=1.500,setpts=PTS-STARTPTS[v0]" in g
    assert "[1:a]atrim=start=2.000:end=3.250,asetpts=PTS-STARTPTS[a1]" in g
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vc][ac]" in g and g.endswith(f"[ac]{voice.voice_filter()}[aout]")
    long = pipeline.build_cut_graph([(i, i + 0.5) for i in range(400)])
    assert len(long) > 32_000  # more than a Windows command line allows: this is why the graph goes through a file


# ---- the whole thing, with a real ffmpeg --------------------------------------------------------------------------------

SCRIPT = [("Hello", 0.5, 1.0), ("there", 1.1, 1.6),  # then a 2 s pause
          ("Say", 3.6, 4.0), ("this", 4.1, 4.5), ("line.", 4.6, 5.0),  # the failed first take
          ("Say", 5.2, 5.6), ("this", 5.7, 6.1), ("line,", 6.2, 6.6), ("again.", 6.7, 7.1)]
TOTAL = 8.0


@pytest.fixture
def raw_video(tmp_path):
    audio = speechlike(tmp_path / "a.wav", [(a, b) for _, a, b in SCRIPT], TOTAL, noise=0.004)
    video = tmp_path / "raw.mp4"
    ff("-f", "lavfi", "-i", f"color=c=0x2b2f3a:s=270x480:r=30:d={TOTAL}", "-i", str(audio), "-shortest",
       "-pix_fmt", "yuv420p", str(video))
    return video


def scripted_transcript(path, language):
    return {"language": language, "model": "m", "duration": TOTAL, "text": "",
            "words": [W(t, a, b) for t, a, b in SCRIPT],
            "segments": [{"id": 0, "start": 0.4, "end": 7.2, "text": "Hello there Say this line. Say this line, again."}]}


@needs_ffmpeg
def test_clean_cuts_pauses_and_the_failed_take_keeps_the_picture_in_sync_and_levels_the_voice(tmp_path, raw_video, gemini,
                                                                                         monkeypatch):
    monkeypatch.setenv("VIBECROP_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.setattr(denoise.requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    job = tmp_path / "job"
    job.mkdir()
    res = pipeline.clean(job, raw_video, "en", log=lambda m: None, transcribe_fn=scripted_transcript,
                         groq_client=FakeGroq(cuts=[]), gemini_client=gemini.client, sleep=lambda s: None)
    rep = res["report"]
    clean = res["video"]
    assert clean == job / "source_clean.mp4" and clean.is_file()
    assert [(r["reason"], r["text"]) for r in rep["retakes"]] == [("retake", "Say this line.")]
    assert rep["denoise"] == denoise.AFFTDN and any("simpler noise filter" in w for w in rep["warnings"])
    assert rep["original_duration"] == pytest.approx(8.0, abs=0.1)
    assert 3.0 < rep["removed_s"] < 5.5 and rep["clean_duration"] == pytest.approx(8.0 - rep["removed_s"], abs=0.15)
    assert rep["pauses_cut"] >= 2 and rep["removed_retakes_s"] == pytest.approx(1.4, abs=0.3)
    info = probe(clean)
    assert info["has_audio"] and info["duration"] == pytest.approx(rep["clean_duration"], abs=0.15)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-of", "compact", str(clean)],
                             capture_output=True, text=True).stdout
    assert "width=270" in streams and "height=480" in streams and "codec_type=audio" in streams
    assert loudness(clean) == pytest.approx(voice.TARGET_LUFS, abs=2.5)

    t = res["transcript"]
    assert [w["word"] for w in t["words"]] == ["Hello", "there", "Say", "this", "line,", "again."]  # first take gone
    assert t["video"] == "source_clean.mp4" and t["duration"] == pytest.approx(info["duration"], abs=0.2)
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(t["words"], t["words"][1:]))
    again = t["words"][-1]  # the speech really is where the transcript says: the tone is loud at "again", quiet in the gaps
    assert mean_db(clean, again["start"] + 0.05, again["end"] - 0.05) > -30
    assert not any((job / n).exists() for n in ("raw.wav", "denoised.wav", "audio.flac", "cut_graph.txt"))
    assert json.loads((job / "cleanup_report.json").read_text(encoding="utf-8"))["removed_s"] == rep["removed_s"]


@needs_ffmpeg
def test_clean_without_any_ai_still_cuts_pauses_and_reports_why(tmp_path, raw_video, gemini, monkeypatch):
    monkeypatch.setenv("VIBECROP_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.setattr(denoise.requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    gemini.fail = "no key"
    job = tmp_path / "job"
    job.mkdir()
    res = pipeline.clean(job, raw_video, "en", log=lambda m: None, transcribe_fn=scripted_transcript,
                         groq_client=FakeGroq(error="no key"), gemini_client=gemini.client, sleep=lambda s: None)
    assert res["report"]["retakes"] == [] and res["report"]["removed_retakes_s"] == 0
    assert len(res["transcript"]["words"]) == len(SCRIPT)  # nothing was cut that was not approved
    assert any("Retake cuts skipped" in w for w in res["report"]["warnings"])
    assert res["report"]["removed_pauses_s"] > 2.0 and res["video"].is_file()


def test_clean_refuses_videos_without_sound_or_over_the_limit(tmp_path, monkeypatch):
    from src.transcription.audio import AudioExtractionError
    silent = tmp_path / "silent.mp4"
    ff("-f", "lavfi", "-i", "color=c=black:s=64x64:d=2", "-pix_fmt", "yuv420p", str(silent))
    with pytest.raises(AudioExtractionError, match="no audio"):
        pipeline.clean(tmp_path, silent, "en")
    loud = tmp_path / "loud.mp4"
    ff("-f", "lavfi", "-i", "color=c=black:s=64x64:d=3", "-f", "lavfi", "-i", "sine=d=3", "-shortest", "-pix_fmt", "yuv420p", str(loud))
    monkeypatch.setenv("MAX_VIDEO_SECONDS", "2")
    with pytest.raises(AudioExtractionError, match="limit"):
        pipeline.clean(tmp_path, loud, "en")


# ---- how the workflow uses it ----------------------------------------------------------------------------------------

def fake_clean(job_dir, source, language, log=print, **kw):
    clean = job_dir / "source_clean.mp4"
    clean.write_bytes(b"clean-video")
    (job_dir / "cleanup_report.json").write_text(json.dumps({"original_duration": 20, "clean_duration": 15}))
    return {"video": clean, "report": {}, "transcript": {"language": language, "words": [W("hi", 0, 0.3)], "segments": [],
                                                          "text": "hi", "duration": 15.0, "video_duration": 15.0}}


def patch_workflow(monkeypatch):
    calls = SimpleNamespace(clean=0, transcribe=0)

    def clean(*a, **k):
        calls.clean += 1
        return fake_clean(*a, **k)

    def transcribe(video, language, job_dir=None, **k):
        calls.transcribe += 1
        (job_dir / "transcript.json").write_text(json.dumps({"words": [W("raw", 0, 1)], "segments": [], "duration": 5.0,
                                                             "video_duration": 5.0}))
    monkeypatch.setattr(workflow.talkinghead, "clean", clean)
    monkeypatch.setattr(workflow.transcription, "run", transcribe)
    monkeypatch.setattr(workflow.recommend, "recommend_count", lambda t: {"count": 3, "reason": "r", "source": "fallback"})
    return calls


def test_analyse_with_cleanup_points_the_job_at_the_clean_video_once(tmp_path, monkeypatch):
    calls = patch_workflow(monkeypatch)
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"raw")
    job = workflow.create_job(src, "en", root=tmp_path, clean=True)
    assert workflow.read_job_meta(job)["clean_talking_head"] is True
    workflow.analyse(job, log=lambda m: None)
    meta = workflow.read_job_meta(job)
    assert meta["cleaned"] and meta["source_video"] == str(job / "source_clean.mp4") and meta["original_video"] == str(src)
    assert meta["video_duration"] == 15.0 and calls.clean == 1 and calls.transcribe == 0
    assert json.loads((job / "transcript.json").read_text(encoding="utf-8"))["words"][0]["word"] == "hi"
    workflow.analyse(job, log=lambda m: None)
    assert calls.clean == 1  # a second analyse does not clean again
    assert workflow.read_cleanup_report(job) == {"original_duration": 20, "clean_duration": 15}


def test_without_the_cleanup_nothing_changes(tmp_path, monkeypatch):
    calls = patch_workflow(monkeypatch)
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"raw")
    job = workflow.create_job(src, "en", root=tmp_path)
    workflow.analyse(job, log=lambda m: None)
    meta = workflow.read_job_meta(job)
    assert calls.clean == 0 and calls.transcribe == 1 and not meta.get("cleaned") and meta["source_video"] == str(src)
    assert workflow.read_cleanup_report(job) is None


def test_toggling_the_cleanup_goes_back_to_the_original_and_forward_again(tmp_path, monkeypatch):
    calls = patch_workflow(monkeypatch)
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"raw")
    job = workflow.create_job(src, "en", root=tmp_path, clean=True)
    workflow.analyse(job, log=lambda m: None)
    (job / "plan.json").write_text("{}")
    workflow.set_cleanup(job, False)
    meta = workflow.read_job_meta(job)
    assert not meta["clean_talking_head"] and not meta["cleaned"] and meta["source_video"] == str(src)
    assert not (job / "source_clean.mp4").exists() and not (job / "cleanup_report.json").exists()
    assert not (job / "transcript.json").exists() and not (job / "plan.json").exists() and "suggested_brolls" not in meta
    workflow.analyse(job, log=lambda m: None)
    assert calls.transcribe == 1 and calls.clean == 1
    workflow.set_cleanup(job, True)
    workflow.analyse(job, log=lambda m: None)
    assert calls.clean == 2 and workflow.read_job_meta(job)["cleaned"]
    workflow.set_cleanup(job, True)  # already on: nothing happens
    assert (job / "transcript.json").exists() and calls.clean == 2


def test_changing_the_language_of_a_cleaned_job_transcribes_the_clean_video_without_cleaning_again(tmp_path, monkeypatch):
    calls = patch_workflow(monkeypatch)
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"raw")
    job = workflow.create_job(src, "en", root=tmp_path, clean=True)
    workflow.analyse(job, log=lambda m: None)
    workflow.set_language(job, "hi")
    workflow.analyse(job, log=lambda m: None)
    assert calls.clean == 1 and calls.transcribe == 1
    assert workflow.read_job_meta(job)["source_video"] == str(job / "source_clean.mp4")


def test_the_cleanup_report_survives_the_post_render_purge(tmp_path):
    from src.cleanup import purge_job_temp
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    for n in ("final.mp4", "cleanup_report.json", "source_clean.mp4", "raw.wav", "source.mp4"):
        (job / n).write_bytes(b"x")
    purge_job_temp(job, root=tmp_path)
    assert sorted(p.name for p in job.iterdir()) == ["cleanup_report.json", "final.mp4"]


# ---- the upload screen -----------------------------------------------------------------------------------------------

def fake_analysis(job_dir, log=print):
    workflow.write_job_meta(job_dir, suggested_brolls=4, suggested_reason="x", suggested_source="gemini", video_duration=30.0)
    return {"count": 4, "reason": "x", "source": "gemini"}


def test_the_checkbox_sets_the_flag_and_toggling_it_reanalyses_the_draft(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    video = tmp_path / "my.mp4"
    video.write_bytes(b"x")
    seen = []
    monkeypatch.setattr(workflow, "analyse", lambda job_dir, log=print: seen.append(
        bool(workflow.read_job_meta(job_dir).get("clean_talking_head"))) or fake_analysis(job_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and at.checkbox(key="clean_th").value is False
    at.checkbox(key="clean_th").set_value(True).run()
    at.text_input[0].set_value(str(video)).run()
    assert not at.exception and seen == [True]
    draft = Path(at.session_state["draft_job"])
    assert workflow.read_job_meta(draft)["clean_talking_head"] is True
    at.checkbox(key="clean_th").set_value(False).run()  # change of mind: back to the original, analysed again
    assert not at.exception and seen == [True, False]
    assert workflow.read_job_meta(draft)["clean_talking_head"] is False and not workflow.read_job_meta(draft).get("cleaned")


def test_the_report_is_shown_after_analysis_and_on_the_review_page(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = tmp_path / "jobs" / "20260101_000000_x"
    job.mkdir(parents=True)
    (job / "plan.json").write_text(json.dumps({"video_topic": "t", "video_duration": 15.0, "items": []}))
    (job / "job.json").write_text(json.dumps({"language": "en", "source_name": "a.mp4", "cleaned": True,
                                              "clean_talking_head": True, "style": "overlay"}))
    (job / "cleanup_report.json").write_text(json.dumps({
        "original_duration": 125.0, "clean_duration": 101.0, "removed_pauses_s": 18.2, "removed_retakes_s": 5.8,
        "denoise": "arnndn", "pauses_cut": 7, "warnings": ["Retake cuts skipped: Gemini was busy."],
        "retakes": [{"start": 12.5, "end": 15.0, "reason": "false_start", "text": "so what I wanted"}],
        "seconds": {"denoise": 2.1, "cut": 9.0}}))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Talking-head cleanup report" in e.label for e in at.expander)
    assert any("2:05 to 1:41" in s.value and "18.2 s" in s.value for s in at.success)
    assert any("Gemini was busy" in w.value for w in at.warning)
    assert any("so what I wanted" in m.value for m in at.markdown)


def test_a_cleaned_job_whose_video_was_purged_is_not_offered_the_original(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBECROP_WORKDIR", str(tmp_path))
    job = tmp_path / "jobs" / "20260101_000000_x"
    job.mkdir(parents=True)
    (job / "plan.json").write_text(json.dumps({"video_topic": "t", "video_duration": 15.0, "items": []}))
    (job / "job.json").write_text(json.dumps({"language": "en", "source_name": "a.mp4", "cleaned": True, "style": "overlay",
                                              "source_video": str(job / "source_clean.mp4"), "original_video": str(tmp_path / "a.mp4")}))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception and any("cleaned video was deleted" in w.value for w in at.warning)
    assert not any("original video file for this job was not found" in (t.label or "") for t in at.text_input)
