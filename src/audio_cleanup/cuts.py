"""Cut silences and retakes out of a video: time maths, the ffmpeg pass, and transcript remapping.

A list of candidate cuts (silence.py + retakes.py) is shown to the user to review before anything is touched (see
workflow.detect_cut_candidates / apply_cuts). Only once they confirm does `apply_cuts` here actually cut.

Two bugs sank the retired version of this feature: "4 s stutters" and clicks at cut points. Both are addressed here:
every video piece is forced to a constant frame rate before it is trimmed (concat drifts when a piece's timing
does not line up exactly to a frame), and every audio piece gets a short fade in/out instead of a hard edge."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from src import config
from src.config import require_binary
from src.rendering.compose import video_encoder_args

MIN_KEEP_S = 0.12  # a kept piece shorter than this is dropped (it would only be a click)
FADE_S = 0.015  # tiny audio fade at each cut boundary: long enough to kill a click, short enough to be inaudible
CRF = 17  # an intermediate: the final export encodes again, so keep this one high quality
DEFAULT_FPS = 30.0  # used only if ffprobe's own answer cannot be parsed


class CutsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- detecting candidates
def _extract_wav(source: Path, dst: Path, sample_rate: int) -> bool:
    r = subprocess.run([require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-vn",
                        "-ac", "1", "-ar", str(sample_rate), str(dst)], capture_output=True, text=True)
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def prepare_detection_audio(video: Path, work: Path, already_denoised: bool,
                            log: Callable[[str], None] = print) -> Path:
    """A short-lived mono WAV to run silence-detection on. If `video`'s own audio is already denoised (the "Clean
    my audio" step ran first), it is used as-is; otherwise a throwaway copy is denoised first so background noise
    does not hide where the real pauses are. The caller deletes the returned file when done."""
    from src.audio_cleanup import denoise

    raw = work / "_detect_raw.wav"
    if not _extract_wav(video, raw, denoise.SAMPLE_RATE):
        raise CutsError("Could not read the audio of this video.")
    if already_denoised:
        return raw
    den = work / "_detect_den.wav"
    denoise.denoise(raw, den, log)
    raw.unlink(missing_ok=True)
    return den


def detect_cuts(detection_audio: Path, words: list[dict], duration: float) -> list[dict]:
    """Every candidate cut (silence + retakes), sorted, with a stable "id" and "selected": True by default."""
    from src.audio_cleanup import retakes, silence

    found = silence.detect_cuts(detection_audio, duration) + retakes.detect_cuts(words, duration)
    found.sort(key=lambda c: c["start"])
    return [{"id": i, "selected": True, **c} for i, c in enumerate(found)]


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


SENTENCE_END = (".", "?", "!", "।", "…")
SEGMENT_GAP_S = 0.8
SEGMENT_MAX_WORDS = 30
NO_SPACE_LANGUAGES = {"ja", "zh"}


def resegment(words: list[dict], language: str = "") -> list[dict]:
    """Segments built from the (already cut) words: a new one after sentence-ending punctuation, a long pause, or
    too many words. Whisper's own segments cannot be shifted onto a cut timeline: their edges do not line up with
    the word times, so a word near an edge would land in the wrong segment."""
    glue = "" if language in NO_SPACE_LANGUAGES else " "
    segments: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            segments.append({"id": len(segments), "start": cur[0]["start"], "end": cur[-1]["end"],
                             "text": glue.join(w["word"] for w in cur)})
            cur.clear()

    for w in words:
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
        ns = _remap(max(w["start"], piece[0]), keep)
        ne = _remap(min(w["end"], piece[1]), keep)
        words.append({**w, "start": round(ns, 3), "end": round(max(ne, ns + 0.01), 3)})
    segments = resegment(words, transcript.get("language", ""))
    return {**transcript, "duration": round(new_duration, 3), "video_duration": round(new_duration, 3),
            "words": words, "segments": segments, "text": " ".join(x["text"] for x in segments)}


# ---------------------------------------------------------------------------------------------------- the cut pass
def detect_source_fps(video: Path) -> float:
    r = subprocess.run([require_binary("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=r_frame_rate", "-of", "default=nw=1:nk=1", str(video)],
                       capture_output=True, text=True)
    try:
        num, _, den = r.stdout.strip().partition("/")
        fps = float(num) / float(den or 1)
        return fps if fps > 0 else DEFAULT_FPS
    except ValueError:
        return DEFAULT_FPS


def build_cut_graph(keep: list[tuple[float, float]], fps: float) -> str:
    """trim + concat every kept piece. The video is forced to a constant `fps` first (a variable-rate source is
    what caused the retired version's multi-second stutters once several pieces were concatenated); the audio gets
    a short fade at both ends of every piece instead of a hard edge (what caused the audible clicks)."""
    parts, joined = [], []
    for i, (a, b) in enumerate(keep):
        fade_out_at = max(0.0, (b - a) - FADE_S)
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},fps={fps:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={FADE_S},afade=t=out:st={fade_out_at:.3f}:d={FADE_S}[a{i}]")
        joined.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(joined)}concat=n={len(keep)}:v=1:a=1[vc][aout]")
    return ";".join(parts)


def cut_video(source: Path, keep: list[tuple[float, float]], out: Path, work: Path) -> None:
    """Write `out`: the kept pieces of `source` (picture and sound both come from it - whatever cleanup already
    ran is preserved as-is). The graph goes through a script file: hundreds of pieces would overflow the Windows
    command-line limit."""
    fps = detect_source_fps(source)
    graph = work / "cut_graph.txt"
    graph.write_text(build_cut_graph(keep, fps), encoding="utf-8")
    cmd = [require_binary("ffmpeg"), "-nostdin", "-y", "-loglevel", "error", "-i", str(source),
           "-filter_complex_script", str(graph), "-map", "[vc]", "-map", "[aout]",
           *video_encoder_args("veryfast", CRF, hw=True), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.is_file():
        raise CutsError(f"FFmpeg could not cut the video: {r.stderr.strip()[-500:]}")


def apply_cuts(source: Path, duration: float, cuts: list[dict], transcript: dict, out: Path, work: Path) -> dict:
    """Cut `source` down to the selected candidates and remap `transcript` onto the new timeline."""
    from src.transcription.audio import probe

    ranges = [(c["start"], c["end"]) for c in cuts if c.get("selected", True)]
    if not ranges:
        raise CutsError("No cut was selected.")
    keep = keep_segments(duration, merge_cuts(ranges, duration))
    if not keep:
        raise CutsError("The selected cuts would remove the whole video.")
    cut_video(source, keep, out, work)
    new_duration = probe(out)["duration"]
    new_transcript = remap_transcript(transcript, keep, new_duration)
    return {"video": out, "transcript": new_transcript, "removed_s": round(duration - new_duration, 2)}
