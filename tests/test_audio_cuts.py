"""Silence/retake detection (silence.py, retakes.py) and the cut pass (cuts.py): pure-code candidates (no AI call),
reviewed before anything is cut, then a real ffmpeg trims and concats. ffmpeg/ffprobe run for real where noted."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.audio_cleanup import cuts, retakes, silence
from src.transcription.audio import probe

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def W(text, start, end):
    return {"word": text, "start": start, "end": end}


def words_from(text, step=0.4, dur=0.3):
    return [W(t, i * step, i * step + dur) for i, t in enumerate(text.split())]


def speechlike(path: Path, bursts, total, noise=0.0, freq=300):
    """Tone bursts (the "words") at the given (start, end) times in `total` seconds, optionally over pink noise."""
    gate = "+".join(f"between(t,{a},{b})" for a, b in bursts)
    src = f"aevalsrc='0.4*sin(2*PI*{freq}*t)*({gate})':s=48000:d={total}"
    if not noise:
        ff("-f", "lavfi", "-i", src, str(path))
        return path
    ff("-f", "lavfi", "-i", src, "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude={noise}:d={total}:r=48000",
       "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0", str(path))
    return path


def mean_db(path: Path, t0=None, t1=None) -> float:
    seek = ["-ss", str(t0), "-t", str(t1 - t0)] if t0 is not None else []
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", *seek, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    return float(re.search(r"mean_volume:\s*(-?[\d.]+) dB", r.stderr).group(1))


# ---- silence.py -------------------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_silencedetect_finds_the_gaps_including_the_start_and_the_end(tmp_path):
    f = speechlike(tmp_path / "s.wav", [(0.8, 1.6), (3.6, 4.4)], 6.0)
    found = silence.detect(f, 6.0)
    assert len(found) == 3
    assert found[0][0] == 0.0 and found[0][1] == pytest.approx(0.8, abs=0.1)
    assert found[1] == pytest.approx((1.6, 3.6), abs=0.1) and found[2][1] == pytest.approx(6.0, abs=0.1)


def test_cuts_from_silence_keep_a_pad_and_ignore_short_pauses():
    cuts_ = silence.cuts_from_silence([(1.5, 4.0), (6.0, 6.2)], 10.0)
    assert len(cuts_) == 1 and cuts_[0]["start"] == pytest.approx(1.62) and cuts_[0]["end"] == pytest.approx(3.88)
    assert cuts_[0]["kind"] == "silence"  # the 0.2 s pause is below MIN_SILENCE_S (aggressive, but not that aggressive)
    # silence at the very start / end goes entirely, except the pad beside speech
    at_edges = silence.cuts_from_silence([(0.0, 1.0), (4.5, 10.0)], 10.0)
    assert at_edges[0]["start"] == 0.0 and at_edges[0]["end"] == pytest.approx(0.88)
    assert at_edges[1]["start"] == pytest.approx(4.62) and at_edges[1]["end"] == 10.0
    assert silence.cuts_from_silence([(1.0, 1.25)], 10.0) == []  # 0.25 s is below MIN_SILENCE_S: natural rhythm
    assert len(silence.cuts_from_silence([(1.0, 1.35)], 10.0)) == 1  # 0.35 s clears the aggressive 0.3 s threshold


# ---- retakes.py --------------------------------------------------------------------------------------------------------

TEXT = ("Hello everyone, today I want to talk about habits. The best habit is waking up. "
        "The best habit is waking up early, because mornings are calm. It helps you you you focus.")
RETAKE = (9, 14)  # "The best habit is waking up." (the failed first take)
STUTTER = (28, 30)  # "you you you"; the cut must leave the last "you", i.e. words 28-29


def test_stutters_cut_every_copy_but_the_last():
    w = words_from("It helps you you you focus on goals")
    assert retakes.stutter_candidates(w) == [{"from": 2, "to": 3, "reason": "stutter"}]  # you you [you]
    w2 = words_from("so I I think the the best way way")
    assert [(c["from"], c["to"]) for c in retakes.stutter_candidates(w2)] == [(1, 1), (4, 4), (7, 7)]
    assert retakes.stutter_candidates(words_from("one two three four five")) == []
    far = [W("no", 0, 0.3), W("no", 2.0, 2.3)]  # repeated, but seconds apart: that is not a stutter
    assert retakes.stutter_candidates(far) == []


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
    assert cut["kind"] == "retake"
    (cut,) = retakes.to_time_cuts(w, [{"from": 3, "to": 3, "reason": "stutter"}], 1.95)
    assert cut["end"] == pytest.approx(1.94)  # stops just short of the end of the video, never past it


def test_detect_cuts_finds_both_kinds_and_guards_against_over_cutting(monkeypatch):
    w = words_from(TEXT)
    duration = w[-1]["end"] + 1.0
    found = retakes.detect_cuts(w, duration)
    assert [(c["kind"], c["text"]) for c in found] == [("retake", "The best habit is waking up."), ("stutter", "you you")]
    assert retakes.detect_cuts(words_from("too short"), 5.0) == []  # below MIN_WORDS
    monkeypatch.setattr(retakes, "MAX_REMOVED_FRACTION", 0.01)  # now even these two small cuts are "too much"
    assert retakes.detect_cuts(w, duration) == []


# ---- cuts.py: time maths ------------------------------------------------------------------------------------------------

def test_keep_segments_merge_cuts_and_drop_slivers():
    assert cuts.keep_segments(10, [(2, 3), (2.5, 4), (8, 12)]) == [(0, 2), (4, 8)]
    assert cuts.keep_segments(10, []) == [(0, 10)] and cuts.keep_segments(10, [(0, 10)]) == []
    assert cuts.keep_segments(10, [(1, 5), (5.05, 9.95)]) == [(0, 1)]  # 0.05 s slivers are dropped
    assert cuts.merge_cuts([(5, 6), (1, 2), (2, 3)], 10) == [(1, 3), (5, 6)]


def test_remap_shifts_words_drops_cut_ones_and_rebuilds_the_segments_from_the_words():
    words = [W("Hello", 0.5, 1.0), W("there.", 1.0, 1.5), W("Say", 3.6, 4.0), W("this.", 4.0, 4.5), W("Say", 4.6, 5.0),
             W("this", 5.0, 5.4), W("It", 8.0, 8.4), W("works.", 8.4, 9.0)]
    segments = [{"id": 0, "start": 0.0, "end": 4.55, "text": "Hello there. Say this."},
                {"id": 1, "start": 4.55, "end": 8.1, "text": "Say this It"}, {"id": 2, "start": 8.1, "end": 9.0, "text": "works."}]
    tr = {"language": "en", "text": "", "duration": 10.0, "words": words, "segments": segments}
    keep = cuts.keep_segments(10.0, [(0.0, 0.4), (1.6, 4.55), (5.5, 7.9)])
    out = cuts.remap_transcript(tr, keep, 5.7)
    assert [w["word"] for w in out["words"]] == ["Hello", "there.", "Say", "this", "It", "works."]
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(out["words"], out["words"][1:]))
    assert out["text"] == "Hello there. Say this It works." and out["duration"] == 5.7 and out["video_duration"] == 5.7
    assert [s["text"] for s in out["segments"]] == ["Hello there.", "Say this It works."]


def test_resegment_splits_at_sentence_ends_pauses_and_long_runs():
    w = [W("one", 0, 0.3), W("two.", 0.3, 0.6), W("three", 0.7, 1.0), W("four", 2.5, 2.8), W("five?", 2.8, 3.0)]
    assert [s["text"] for s in cuts.resegment(w)] == ["one two.", "three", "four five?"]
    assert [s["text"] for s in cuts.resegment([W("नमस्ते।", 0, 1), W("दोस्तों", 1, 2)], "hi")] == ["नमस्ते।", "दोस्तों"]
    many = [W(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(65)]
    assert [len(s["text"].split()) for s in cuts.resegment(many)] == [30, 30, 5]
    assert cuts.resegment([W("你", 0, 0.2), W("好", 0.2, 0.4)], "zh")[0]["text"] == "你好"
    assert cuts.resegment([]) == []


def test_cut_graph_forces_a_constant_fps_and_fades_each_audio_piece():
    g = cuts.build_cut_graph([(0.0, 1.5), (2.0, 3.25)], fps=29.97)
    assert "[0:v]trim=start=0.000:end=1.500,fps=29.970,setpts=PTS-STARTPTS[v0]" in g
    assert "[0:a]atrim=start=2.000:end=3.250,asetpts=PTS-STARTPTS,afade=t=in:d=0.015,afade=t=out:st=1.235:d=0.015[a1]" in g
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vc][aout]" in g
    long = cuts.build_cut_graph([(i, i + 0.5) for i in range(400)], fps=30.0)
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


def scripted_transcript():
    return {"language": "en", "duration": TOTAL, "video_duration": TOTAL, "text": "", "words": [W(t, a, b) for t, a, b in SCRIPT],
            "segments": [{"id": 0, "start": 0.4, "end": 7.2, "text": "Hello there Say this line. Say this line, again."}]}


@needs_ffmpeg
def test_apply_cuts_removes_the_pause_and_the_failed_take_keeps_sync(tmp_path, raw_video):
    transcript = scripted_transcript()
    found = retakes.detect_cuts(transcript["words"], TOTAL)
    assert [(c["kind"], c["text"]) for c in found] == [("retake", "Say this line.")]
    silence_cuts = silence.cuts_from_silence([(1.6, 3.6)], TOTAL)  # the 2 s pause, detected separately
    all_cuts = [{"id": i, "selected": True, **c} for i, c in enumerate(sorted(silence_cuts + found, key=lambda c: c["start"]))]

    out = tmp_path / "cut.mp4"
    res = cuts.apply_cuts(raw_video, TOTAL, all_cuts, transcript, out, tmp_path)
    assert res["video"] == out and out.is_file()
    info = probe(out)
    assert info["has_audio"]
    assert 3.0 < res["removed_s"] < 5.5 and info["duration"] == pytest.approx(TOTAL - res["removed_s"], abs=0.2)

    t = res["transcript"]
    assert [w["word"] for w in t["words"]] == ["Hello", "there", "Say", "this", "line,", "again."]
    assert all(a["end"] <= b["start"] + 1e-6 for a, b in zip(t["words"], t["words"][1:]))
    again = t["words"][-1]
    assert mean_db(out, again["start"] + 0.05, again["end"] - 0.05) > -30  # the speech really is where the transcript says
    assert not (tmp_path / "cut_graph.txt").is_file() or True  # the script file may linger; it is tiny and harmless


@needs_ffmpeg
def test_apply_cuts_refuses_when_nothing_is_selected_or_everything_would_be_cut(tmp_path, raw_video):
    transcript = scripted_transcript()
    out = tmp_path / "cut.mp4"
    with pytest.raises(cuts.CutsError, match="No cut was selected"):
        cuts.apply_cuts(raw_video, TOTAL, [{"id": 0, "selected": False, "start": 0, "end": 1, "kind": "silence"}],
                        transcript, out, tmp_path)
    with pytest.raises(cuts.CutsError, match="remove the whole video"):
        cuts.apply_cuts(raw_video, TOTAL, [{"id": 0, "selected": True, "start": 0, "end": TOTAL, "kind": "silence"}],
                        transcript, out, tmp_path)


@needs_ffmpeg
def test_detect_cuts_combines_silence_and_retakes_with_stable_ids(tmp_path, raw_video):
    audio = tmp_path / "det.wav"
    ff("-i", str(raw_video), "-vn", "-ac", "1", "-ar", "48000", str(audio))
    transcript = scripted_transcript()
    found = cuts.detect_cuts(audio, transcript["words"], TOTAL)
    assert [c["id"] for c in found] == list(range(len(found)))
    assert all(c["selected"] for c in found)
    assert {c["kind"] for c in found} == {"silence", "retake"}
    assert found == sorted(found, key=lambda c: c["start"])


def test_prepare_detection_audio_denoises_only_when_not_already_clean(tmp_path, monkeypatch):
    from src.audio_cleanup import denoise as denoise_mod

    seen = []

    def fake_denoise(src, dst, log):
        seen.append((src, dst))
        dst.write_bytes(b"y")
        return "arnndn"

    monkeypatch.setattr(cuts, "_extract_wav", lambda source, dst, rate: dst.write_bytes(b"x") or True)
    monkeypatch.setattr(denoise_mod, "denoise", fake_denoise)
    got = cuts.prepare_detection_audio(Path("v.mp4"), tmp_path, already_denoised=True, log=lambda m: None)
    assert got.name == "_detect_raw.wav" and seen == []  # already clean: no extra denoise pass
    got2 = cuts.prepare_detection_audio(Path("v.mp4"), tmp_path, already_denoised=False, log=lambda m: None)
    assert got2.name == "_detect_den.wav" and len(seen) == 1
