"""Audio cleanup: voice chain, denoise, the polish pass with a real ffmpeg (picture copied, nothing cut), and how the
workflow and the upload screen use it."""
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src import config, workflow
from src.audio_cleanup import denoise, pipeline, voice
from src.transcription.audio import probe

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args], check=True)


def W(text, start, end):
    return {"word": text, "start": start, "end": end}


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


# ---- the whole thing, with a real ffmpeg --------------------------------------------------------------------------------

SCRIPT = [("Hello", 0.5, 1.0), ("there", 1.1, 1.6), ("Say", 3.6, 4.0), ("this", 4.1, 4.5), ("line.", 4.6, 5.0)]
TOTAL = 6.0


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
            "segments": [{"id": 0, "start": 0.4, "end": 5.0, "text": "Hello there Say this line."}]}


def codec_of(path, kind):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries", "stream=codec_name",
                           "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()


@needs_ffmpeg
def test_clean_only_cleans_the_audio_nothing_is_cut_and_the_picture_is_copied(tmp_path, raw_video, monkeypatch):
    monkeypatch.setenv("VIBECROP_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.setattr(denoise.requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    job = tmp_path / "job"
    job.mkdir()
    res = pipeline.clean(job, raw_video, "en", log=lambda m: None, transcribe_fn=scripted_transcript)
    rep, clean = res["report"], res["video"]
    assert clean == job / "source_clean.mp4" and clean.is_file()
    assert rep["denoise"] == denoise.AFFTDN and any("simpler noise filter" in w for w in rep["warnings"])
    assert rep["original_duration"] == pytest.approx(TOTAL, abs=0.1)
    assert rep["clean_duration"] == pytest.approx(rep["original_duration"], abs=0.15)  # nothing was cut
    assert not any(k in rep for k in ("removed_s", "retakes", "pauses_cut"))
    info = probe(clean)
    assert info["has_audio"] and info["duration"] == pytest.approx(TOTAL, abs=0.15)
    assert codec_of(clean, "v:0") == codec_of(raw_video, "v:0") == "h264"  # the picture was copied, not re-encoded
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "compact", str(clean)],
                             capture_output=True, text=True).stdout
    assert "width=270" in streams and "height=480" in streams
    assert loudness(clean) == pytest.approx(voice.TARGET_LUFS, abs=2.5)

    t = res["transcript"]  # the same words at the same times: there is nothing to shift
    assert [(w["word"], w["start"], w["end"]) for w in t["words"]] == [(x, a, b) for x, a, b in SCRIPT]
    assert t["video"] == "source_clean.mp4" and t["duration"] == pytest.approx(info["duration"], abs=0.01)
    assert t["video_duration"] == t["duration"]
    assert mean_db(clean, 4.65, 4.95) > -30  # the speech is where the transcript says (the tone is loud at "line.")
    assert not any((job / n).exists() for n in ("raw.wav", "denoised.wav", "audio.flac"))
    assert json.loads((job / "cleanup_report.json").read_text(encoding="utf-8"))["clean_duration"] == rep["clean_duration"]


def test_polish_command_copies_the_picture_unless_told_otherwise():
    copy = pipeline.build_polish_cmd(Path("in.mp4"), Path("den.wav"), Path("out.mp4"), copy_video=True)
    assert copy[copy.index("-c:v") + 1] == "copy" and voice.voice_filter() in copy
    assert copy[copy.index("-map") + 1] == "0:v:0" and "1:a:0" in copy  # picture from the video, sound from the denoised audio
    redo = pipeline.build_polish_cmd(Path("in.mp4"), Path("den.wav"), Path("out.mp4"), copy_video=False)
    assert redo[redo.index("-c:v") + 1] != "copy"


def test_polish_falls_back_to_re_encoding_when_the_picture_cannot_be_copied(tmp_path, monkeypatch):
    out = tmp_path / "out.mp4"
    attempts = []

    def fake_run(cmd, **kw):
        copy = cmd[cmd.index("-c:v") + 1] == "copy"
        attempts.append(copy)
        if copy:
            return SimpleNamespace(returncode=1, stderr="could not write header")
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline.polish_video(tmp_path / "in.mkv", tmp_path / "den.wav", out)
    assert attempts == [True, False] and out.is_file()
    monkeypatch.setattr(pipeline.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=1, stderr="boom"))
    with pytest.raises(pipeline.CleanupError, match="boom"):
        pipeline.polish_video(tmp_path / "in.mkv", tmp_path / "den.wav", out)


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
    monkeypatch.setattr(workflow.audio_cleanup, "clean", clean)
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
        "original_duration": 101.0, "clean_duration": 101.0, "denoise": "arnndn",
        "warnings": ["Used the simpler noise filter."], "seconds": {"denoise": 2.1, "polish": 1.0}}))
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["job_dir"] = str(job)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Audio cleanup report" in e.label for e in at.expander)
    assert any("1:41" in s.value and "arnndn" in s.value and "Nothing was cut" in s.value for s in at.success)
    assert any("simpler noise filter" in w.value for w in at.warning)


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
