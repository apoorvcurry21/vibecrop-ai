"""Phase 1 tests. Groq is mocked, so no API key or network is needed. FFmpeg is real."""
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from src.transcription import pipeline
from src.transcription.audio import AudioExtractionError, extract_audio, probe
from src.transcription.groq_whisper import TranscriptionError, normalize, transcribe

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

FAKE = {
    "text": " Hello world. ",
    "duration": 2.0,
    "words": [{"word": " Hello", "start": 0.0, "end": 0.5}, {"word": "world.", "start": 0.55, "end": 1.2345}],
    "segments": [{"id": 0, "start": 0.0, "end": 1.2345, "text": " Hello world."}],
}


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


@pytest.fixture
def clip(tmp_path):
    """2-second test video with a sine-wave audio track."""
    out = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-shortest", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out


def test_normalize_handles_dicts_and_objects():
    as_obj = SimpleNamespace(
        text=FAKE["text"], duration=FAKE["duration"],
        words=[SimpleNamespace(**w) for w in FAKE["words"]],
        segments=[SimpleNamespace(**s) for s in FAKE["segments"]],
    )
    for resp in (FAKE, as_obj):
        out = normalize(resp, "en", "m")
        assert out["text"] == "Hello world."
        assert [w["word"] for w in out["words"]] == ["Hello", "world."]
        assert out["words"][1]["end"] == pytest.approx(1.2345, abs=0.001)
        assert out["segments"][0]["text"] == "Hello world."


def test_rejects_bad_language(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"x")
    with pytest.raises(TranscriptionError):
        transcribe(audio, "english", client=FakeClient(FAKE))


def test_rejects_empty_words(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"x")
    with pytest.raises(TranscriptionError):
        transcribe(audio, "en", client=FakeClient({"text": "", "words": [], "segments": []}))


@needs_ffmpeg
def test_extract_audio_is_16k_mono_flac(clip, tmp_path):
    out = extract_audio(clip, tmp_path / "audio.flac")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels", "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["codec_name"], stream["sample_rate"], stream["channels"]) == ("flac", "16000", 1)


@needs_ffmpeg
def test_pipeline_writes_transcript(clip, tmp_path):
    client = FakeClient(FAKE)
    out = pipeline.run(clip, "en", job_dir=tmp_path / "job", client=client)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["video"] == "clip.mp4" and data["language"] == "en"
    assert len(data["words"]) == 2
    call = client.calls[0]
    assert call["timestamp_granularities"] == ["word", "segment"]
    assert call["response_format"] == "verbose_json" and call["language"] == "en"


@needs_ffmpeg
def test_pipeline_rejects_video_without_audio(tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x180:rate=10:duration=1", "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    assert probe(silent)["has_audio"] is False
    with pytest.raises(AudioExtractionError):
        pipeline.run(silent, "en", job_dir=tmp_path / "job", client=FakeClient(FAKE))
