"""Does a downloaded clip actually show what the speaker is talking about?

The clip window is picked blindly (youtube.pick_window), so some clips are off-topic. After a download we cut two small
frames out of the clip and ask a light Gemini model "does this fit this moment?". A "no" makes the sourcing code try
the next candidate video.

It must never make things worse, so every problem (no key, quota, timeout, unreadable frames) means "unknown" and the
clip is kept. After a couple of failures in a row the check pauses for a minute instead of slowing every slot down."""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from src import config
from src.config import require_binary
from src.gemini_client import GeminiError, generate_with_fallback, make_client

CHECK_TIMEOUT_MS = 20_000  # a stalled answer must not hold a slot up
MAX_MODELS = 2  # the first two models of the chat chain (both light and fast)
FRAME_POINTS = (0.25, 0.7)  # where in the clip the two frames are taken (fraction of its length)
FRAME_WIDTH = 480
FAILURES_BEFORE_PAUSE = 2
PAUSE_S = 60.0
MAX_RETRY_WAIT_S = 2.0

SYSTEM_PROMPT = ("You are a fair assistant video editor checking B-roll footage for a short vertical reel. "
                 "You are shown two frames from one candidate clip.")


class Verdict(BaseModel):
    match: bool  # true if the clip plausibly illustrates the moment
    reason: str  # at most 12 words


def build_prompt(meaning: str, query: str) -> str:
    return (f"The speaker is saying (in English): {meaning or '(not available)'}\n"
            f"The search used to find this clip: {query}\n\n"
            "Does the clip plausibly illustrate that moment? It does not have to be exact. Answer match=false only if "
            "it is clearly unrelated, or it is a title card, logo, black screen, slideshow, or screen full of text, or "
            "just a person talking to a camera when the moment is not about that person. Give a short reason.")


def extract_frames(clip: Path, duration: float) -> list[bytes]:
    """Small JPEG frames from the clip (empty list if ffmpeg cannot read it)."""
    frames = []
    for point in FRAME_POINTS:
        r = subprocess.run(
            [require_binary("ffmpeg"), "-nostdin", "-loglevel", "error", "-ss", f"{max(duration * point, 0):.2f}",
             "-i", str(clip), "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-f", "image2pipe",
             "-vcodec", "mjpeg", "-"], capture_output=True)
        if r.returncode == 0 and r.stdout:
            frames.append(r.stdout)
    return frames


class Checker:
    """One shared instance serves all the parallel downloads (see `default_checker`)."""

    def __init__(self, client: Any = None, models: list[str] | None = None,
                 sleep: Callable[[float], None] | None = None, clock: Callable[[], float] = time.monotonic):
        self._client, self._models = client, models
        self._sleep = sleep or (lambda s: time.sleep(min(s, MAX_RETRY_WAIT_S)))
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._paused_until = 0.0

    def paused(self) -> bool:
        return self._clock() < self._paused_until

    def _fail(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= FAILURES_BEFORE_PAUSE:
                self._paused_until, self._failures = self._clock() + PAUSE_S, 0

    def check(self, clip: Path, duration: float, meaning: str, query: str) -> tuple[bool | None, str]:
        """(True, '') it fits, (False, why) it does not, (None, '') unknown: keep the clip."""
        if self.paused():
            return None, ""
        try:
            frames = extract_frames(clip, duration)
            if not frames:
                return None, ""
            from google.genai import types

            parts = [types.Part(text=build_prompt(meaning, query))]
            parts += [types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]
            client = self._client or make_client(CHECK_TIMEOUT_MS)
            models = self._models or config.chat_model_chain()[:MAX_MODELS]
            text, _ = generate_with_fallback(client, models, [types.Content(role="user", parts=parts)],
                                             self._sleep, SYSTEM_PROMPT, Verdict)
            verdict = Verdict.model_validate_json(text)
        except (GeminiError, config.ConfigError, ValueError, OSError):
            self._fail()
            return None, ""
        with self._lock:
            self._failures = 0
        return (True, "") if verdict.match else (False, " ".join(verdict.reason.split())[:120])


_default: Checker | None = None
_default_lock = threading.Lock()


def default_checker() -> Checker | None:
    """The shared checker, or None when RELEVANCE_CHECK is switched off."""
    global _default
    if not config.relevance_enabled():
        return None
    with _default_lock:
        if _default is None:
            _default = Checker()
        return _default
