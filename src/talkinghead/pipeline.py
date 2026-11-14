"""Talking-head cleanup: raw video -> a cleaned video (+ its transcript) that becomes the job's source.

    1. extract the audio, remove background noise (arnndn)
    2. transcribe the DENOISED audio with Groq Whisper (cleaner audio = better words and timestamps)
    3. find long pauses (silencedetect) and stutters / retakes (Groq proposes, Gemini verifies)
    4. ONE ffmpeg pass: cut everything found, polish the voice (EQ, compressor, loudness), encode source_clean.mp4
    5. shift the transcript's timestamps onto the shorter timeline (no second transcription)

The caller (workflow) points the job at the returned video and transcript. Nothing here touches plan.json.
"""
from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from src import config, perf
from src.rendering.compose import video_encoder_args
from src.talkinghead import denoise, retakes, silence, voice
from src.transcription.audio import AudioExtractionError, extract_audio, probe

MIN_KEEP_S = 0.12  # a kept piece shorter than this is dropped (it would only be a click)
REPORT_NAME = "cleanup_report.json"
CLEAN_NAME = "source_clean.mp4"
CRF = 17  # an intermediate: the final export encodes again, so keep this one high quality


class CleanupError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- time maths
def merge_cuts(cuts: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted((max(0.0, a), min(duration, b)) for a, b in cuts if b > a):
        if out and a <= out[-1][1] + 1e-6:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def keep_segments(duration: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The parts of [0, duration] that stay once `cuts` are removed (pieces under MIN_KEEP_S are dropped)."""
    keep, cur = [], 0.0
    for a, b in merge_cuts(cuts, duration):
        if a - cur >= MIN_KEEP_S:
            keep.append((cur, a))
        cur = b
    if duration - cur >= MIN_KEEP_S:
        keep.append((cur, duration))
    return keep


def _remap(t: float, keep: list[tuple[float, float]]) -> float | None:
    """Where original time `t` lands on the cut timeline (None if it falls in a removed part)."""
    offset = 0.0
    for a, b in keep:
        if a - 1e-6 <= t <= b + 1e-6:
            return offset + min(max(t, a), b) - a
        offset += b - a
    return None


SENTENCE_END = (".", "?", "!", "।", "…")  # includes the Devanagari danda
SEGMENT_GAP_S = 0.8  # a pause this long also ends a segment
SEGMENT_MAX_WORDS = 30
NO_SPACE_LANGUAGES = {"ja", "zh"}


def resegment(words: list[dict], language: str = "") -> list[dict]:
    """Segments built from the (already cut) words: a new one after sentence-ending punctuation, a long pause, or too
    many words. Whisper's own segments cannot be shifted onto a cut timeline: their edges do not line up with the
    word times, so a word near an edge lands in the wrong segment."""
    glue = "" if language in NO_SPACE_LANGUAGES else " "
    segments: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            segments.append({"id": len(segments), "start": cur[0]["start"], "end": cur[-1]["end"],
                             "text": glue.join(w["word"] for w in cur)})
            cur.clear()

    for i, w in enumerate(words):
        if cur and (w["start"] - cur[-1]["end"] >= SEGMENT_GAP_S or len(cur) >= SEGMENT_MAX_WORDS):
            flush()
        cur.append(w)
        if w["word"].rstrip().endswith(SENTENCE_END):
            flush()
    flush()
    return segments


def remap_transcript(transcript: dict, keep: list[tuple[float, float]], new_duration: float) -> dict:
    """Shift every word onto the cut timeline (words that were cut disappear) and rebuild the segments from them."""
    words = []
    for w in transcript.get("words", []):
        mid = (w["start"] + w["end"]) / 2
        piece = next(((x, y) for x, y in keep if x - 1e-6 <= mid <= y + 1e-6), None)
        if piece is None:
            continue
        ns = _remap(max(w["start"], piece[0]), keep)  # a word that sticks out of its piece is clamped into it
        ne = _remap(min(w["end"], piece[1]), keep)
        words.append({**w, "start": round(ns, 3), "end": round(max(ne, ns + 0.01), 3)})
    segments = resegment(words, transcript.get("language", ""))
    return {**transcript, "duration": round(new_duration, 3), "video_duration": round(new_duration, 3),
            "words": words, "segments": segments, "text": " ".join(x["text"] for x in segments)}


# ---------------------------------------------------------------------------------------------------- the cut pass
def build_cut_graph(keep: list[tuple[float, float]]) -> str:
    """trim + concat every kept piece of the video (input 0) and the denoised audio (input 1), then the voice chain."""
    parts, joined = [], []
    for i, (a, b) in enumerate(keep):
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[1:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
        joined.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(joined)}concat=n={len(keep)}:v=1:a=1[vc][ac]")
    parts.append(f"[ac]{voice.voice_filter()}[aout]")
    return ";".join(parts)


def cut_video(source: Path, audio: Path, keep: list[tuple[float, float]], out: Path, work: Path,
              progress: Callable[[float], None] | None = None) -> None:
    """Write `out`: the kept pieces of `source` (picture) with the matching pieces of `audio` (denoised sound).
    The graph goes through a script file: hundreds of pieces would overflow the Windows command-line limit."""
    graph = work / "cut_graph.txt"
    graph.write_text(build_cut_graph(keep), encoding="utf-8")
    cmd = [config.require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-i", str(audio),
           "-filter_complex_script", str(graph), "-map", "[vc]", "-map", "[aout]",
           *video_encoder_args("veryfast", CRF, hw=True), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise CleanupError(f"FFmpeg could not cut the video: {r.stderr.strip()[-500:]}")


# ---------------------------------------------------------------------------------------------------- the pipeline
@contextmanager
def _timed(job_dir: Path, timings: dict, name: str):
    t0 = time.perf_counter()
    with perf.stage(job_dir, f"cleanup:{name}"):
        yield
    timings[name] = round(time.perf_counter() - t0, 2)


def _measure(cuts: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in cuts)


def clean(job_dir: Path, source: Path, language: str, log: Callable[[str], None] = print,
          transcribe_fn: Callable[[Path, str], dict] | None = None, groq_client: Any = None,
          gemini_client: Any = None, sleep: Callable[[float], None] | None = None) -> dict:
    """Run the whole cleanup. Returns {"video": Path, "transcript": dict, "report": dict} and writes the report file.
    Writes only temporary files and `source_clean.mp4` into `job_dir`."""
    job_dir = Path(job_dir)
    info = probe(source)
    if not info["has_audio"]:
        raise AudioExtractionError("This video has no audio track to clean.")
    limit = config.max_video_seconds()
    if info["duration"] > limit:
        raise AudioExtractionError(
            f"Video is {info['duration']:.0f}s; the limit is {limit:.0f}s (set MAX_VIDEO_SECONDS to change it).")
    duration = info["duration"]
    warnings: list[str] = []
    timings: dict[str, float] = {}
    temps = [job_dir / n for n in ("raw.wav", "denoised.wav", "audio.flac", "cut_graph.txt")]


    try:
        with _timed(job_dir, timings, "denoise"):
            log("Removing background noise...")
            raw, den = temps[0], temps[1]
            if not _extract_wav(source, raw):
                raise AudioExtractionError("Could not read the audio of this video.")
            method = denoise.denoise(raw, den, log)
        if method != denoise.ARNNDN:
            warnings.append("Used the simpler noise filter (the noise-removal model was not available).")

        with _timed(job_dir, timings, "transcribe"):
            log("Transcribing the cleaned audio (Groq Whisper)...")
            flac = extract_audio(den, temps[2])
            if transcribe_fn is not None:
                transcript = transcribe_fn(flac, language)
            else:
                from src.transcription.groq_whisper import transcribe
                transcript = transcribe(flac, language)

        words = transcript["words"]
        with _timed(job_dir, timings, "silence"):
            log("Finding long pauses...")
            silent = silence.detect(den, duration)
            silence_cuts = silence.cuts_from_silence(silent, words, duration)

        with _timed(job_dir, timings, "retakes"):
            found = retakes.find_retakes(words, language, duration, groq_client=groq_client,
                                         gemini_client=gemini_client, sleep=sleep, log=log)
        warnings += found["warnings"]
        retake_cuts = [(c["start"], c["end"]) for c in found["cuts"]]

        keep = keep_segments(duration, silence_cuts + retake_cuts)
        if not keep:
            raise CleanupError("Nothing would be left of this video after cutting. Try it without cleanup.")
        out = job_dir / CLEAN_NAME
        with _timed(job_dir, timings, "cut"):
            log(f"Cutting and polishing the voice ({len(keep)} pieces)...")
            cut_video(source, den, keep, out, job_dir)
        new_duration = probe(out)["duration"]
        transcript = remap_transcript(transcript, keep, new_duration)
        transcript["video"] = out.name

        merged_retakes = merge_cuts(retake_cuts, duration)
        removed_retake = _measure(merged_retakes)
        removed_total = duration - _measure(keep)
        report = {
            "original_duration": round(duration, 2), "clean_duration": round(new_duration, 2),
            "removed_s": round(removed_total, 2), "removed_retakes_s": round(removed_retake, 2),
            "removed_pauses_s": round(max(removed_total - removed_retake, 0.0), 2), "pauses_cut": len(silence_cuts),
            "retakes": [{"start": c["start"], "end": c["end"], "reason": c["reason"], "text": c["text"]}
                        for c in found["cuts"]],
            "retakes_proposed": found["proposed"], "retakes_approved": found["approved"],
            "denoise": method, "pieces": len(keep), "warnings": warnings, "seconds": timings,
        }
        (job_dir / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"video": out, "transcript": transcript, "report": report}
    finally:
        for t in temps:
            t.unlink(missing_ok=True)


def _extract_wav(source: Path, dst: Path) -> bool:
    r = subprocess.run([config.require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-vn",
                        "-ac", "1", "-ar", str(denoise.SAMPLE_RATE), str(dst)], capture_output=True, text=True)
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0
