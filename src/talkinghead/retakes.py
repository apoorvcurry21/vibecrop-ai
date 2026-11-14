"""Stutters, repeated words and retakes: Groq proposes cuts from the transcript, Gemini double-checks each one.

Two models on purpose. Groq is fast and finds candidates; a wrong cut deletes real content, so Gemini then reads each
proposal in context and approves only genuine stutters / retakes / false starts. Nothing is cut that Gemini did
not approve, and every failure (no key, API down, nonsense output) just skips this step with a warning: the pipeline
never stops because of it. Cuts snap to word boundaries."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from pydantic import BaseModel

from src import config
from src.gemini_client import GeminiError, content, generate_with_fallback, make_client

PAUSE_MARK_S = 0.5  # a gap this long between words is shown to the models as "|"
MAX_CUT_WORDS = 60  # one proposal removing more than this is almost certainly wrong
MAX_REMOVED_FRACTION = 0.40  # more than this of the video in retake cuts: distrust the models and cut none
MIN_WORDS = 6
MAX_BACKOFF_S = 2.0
REASONS = ("stutter", "retake", "false_start")
CUT_PAD_S = 0.05

GROQ_SYSTEM = """You edit raw talking-head recordings. You get a transcript as numbered words; "|" marks a pause of \
half a second or more. Find the parts the speaker clearly wants gone:
- stutter: a word or short phrase repeated by accident ("I I I think", "the the").
- retake: a sentence or phrase said, then said again (better) right after. Remove the EARLIER attempt(s) and keep \
the last complete one.
- false_start: a sentence abandoned midway and then restarted.
Do NOT remove: anything said only once, repetition used for emphasis, filler words, ordinary pauses. Be \
conservative: when unsure, leave it in. Word numbers are inclusive. Cuts must not overlap.
Reply with JSON only: {"cuts": [{"from": <first word number>, "to": <last word number>, "reason": "stutter" | \
"retake" | "false_start"}]}. If there is nothing to cut, reply {"cuts": []}."""

GEMINI_SYSTEM = """You are the second check on cuts proposed for a talking-head transcript. Each proposal shows some words before, the words to REMOVE, and some words after, plus a computed hint "restarts": whether the words right after the removed ones begin with the same opening words again.
How to decide:
- retake: the speaker said a line, then said it again right after (usually longer or better). The removed words are the EARLIER attempt. If "restarts" is yes, this is almost certainly a retake: APPROVE it, even when the second take is longer or slightly different, and even though the removed words look like a complete sentence (that is exactly what a retake is: a complete first attempt that the speaker replaced).
- stutter: a word or short phrase repeated by accident right next to itself: approve.
- false_start: a sentence abandoned midway (it breaks off or trails away) and then restarted: approve.
REJECT when the removed words carry content that is said nowhere else, or are an intentional repeat for emphasis, or removing them would leave the next words without their context. When in doubt about a retake that does not restart, reject. Reply with JSON: {"approved": [the ids you approve], "notes": "one short sentence"}."""


class RetakeError(RuntimeError):
    pass


class Verdict(BaseModel):
    approved: list[int]
    notes: str = ""


def numbered_transcript(words: list[dict]) -> str:
    """'[0] so [1] hello | [2] everyone ...' (a | marks a pause)."""
    parts = []
    for i, w in enumerate(words):
        if i and w["start"] - words[i - 1]["end"] >= PAUSE_MARK_S:
            parts.append("|")
        parts.append(f"[{i}] {w['word']}")
    return " ".join(parts)


def _json_object(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise RetakeError("The AI did not answer with JSON.") from None
        try:
            data = json.loads(m.group(0))
        except ValueError:
            raise RetakeError("The AI did not answer with valid JSON.") from None
    if not isinstance(data, dict):
        raise RetakeError("The AI answer has the wrong shape.")
    return data


def sanitize(raw: Any, n_words: int) -> list[dict]:
    """Valid, sorted, non-overlapping proposals {from, to, reason}. Anything malformed is dropped."""
    out = []
    for c in raw if isinstance(raw, list) else []:
        try:
            a, b = int(c["from"]), int(c["to"])
        except (KeyError, TypeError, ValueError):
            continue
        reason = c.get("reason") if c.get("reason") in REASONS else "retake"
        if 0 <= a <= b < n_words and b - a + 1 <= MAX_CUT_WORDS:
            out.append({"from": a, "to": b, "reason": reason})
    out.sort(key=lambda c: c["from"])
    merged: list[dict] = []
    for c in out:
        if merged and c["from"] <= merged[-1]["to"]:  # overlapping: keep the first, extend it
            merged[-1]["to"] = max(merged[-1]["to"], c["to"])
            continue
        merged.append(c)
    return [c for c in merged if c["to"] - c["from"] + 1 <= MAX_CUT_WORDS]


def propose(words: list[dict], language: str, client: Any = None, model: str | None = None) -> list[dict]:
    """Groq's candidate cuts (unverified)."""
    try:
        if client is None:
            from groq import Groq

            client = Groq(api_key=config.get_secret("GROQ_API_KEY"), max_retries=2)
        resp = client.chat.completions.create(
            model=model or config.groq_llm_model(), temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": GROQ_SYSTEM},
                      {"role": "user", "content": f"Language: {language}\nTranscript:\n{numbered_transcript(words)}"}])
        data = _json_object(resp.choices[0].message.content)
    except config.ConfigError as exc:
        raise RetakeError(str(exc)) from exc
    except RetakeError:
        raise
    except Exception as exc:  # noqa: BLE001 - network, rate limit, model gone
        raise RetakeError(f"Groq could not check for retakes: {str(exc)[:160]}") from exc
    return sanitize(data.get("cuts"), len(words))


def restarts(words: list[dict], a: int, b: int, k: int = 3) -> bool:
    """True if the words right after the removed range [a, b] open with the same first words again (a restart)."""
    n = min(k, b - a + 1)
    first = [_norm_word(w["word"]) for w in words[a:a + n]]
    after = [_norm_word(w["word"]) for w in words[b + 1:b + 1 + n]]
    return len(after) == n and first == after and all(first)


def _norm_word(word: str) -> str:
    return re.sub(r"[^\wऀ-ॿ]+", "", word.lower())


def _context(words: list[dict], a: int, b: int, span: int = 8) -> str:
    before = " ".join(w["word"] for w in words[max(0, a - span):a])
    gone = " ".join(w["word"] for w in words[a:b + 1])
    after = " ".join(w["word"] for w in words[b + 1:b + 1 + span])
    return f"{before} [REMOVE: {gone}] {after}".strip()


def _quick_sleep(seconds: float) -> None:
    """Gemini's retry back-off is 5, 10, 15 s: fine for a plan, too slow here. This step can simply be skipped."""
    time.sleep(min(seconds, MAX_BACKOFF_S))


def verify(words: list[dict], proposals: list[dict], client: Any = None, models: list[str] | None = None,
           sleep: Callable[[float], None] | None = None) -> list[dict]:
    """Gemini's approval of each proposal; returns only the approved ones."""
    if not proposals:
        return []
    lines = [f"Proposal {i} ({c['reason']}, restarts: {'yes' if restarts(words, c['from'], c['to']) else 'no'}): "
             f"{_context(words, c['from'], c['to'])}" for i, c in enumerate(proposals)]
    prompt = ("Full transcript for context:\n" + " ".join(w["word"] for w in words)
              + "\n\nProposals:\n" + "\n".join(lines))
    try:
        text, _ = generate_with_fallback(client or make_client(config.CHAT_TIMEOUT_MS), models or config.chat_model_chain(),
                                         [content("user", prompt)], sleep or _quick_sleep, GEMINI_SYSTEM, Verdict)
        verdict = Verdict.model_validate_json(text)
    except (GeminiError, config.ConfigError, ValueError) as exc:
        raise RetakeError(f"Gemini could not double-check the cuts: {str(exc)[:160]}") from exc
    ok = set(verdict.approved)
    return [c for i, c in enumerate(proposals) if i in ok]


def to_time_cuts(words: list[dict], cuts: list[dict], duration: float) -> list[dict]:
    """Word-number cuts -> time ranges, snapped just outside the removed words and clear of the kept neighbours."""
    out = []
    for c in cuts:
        a, b = c["from"], c["to"]
        lo = words[a - 1]["end"] if a > 0 else 0.0
        hi = words[b + 1]["start"] if b + 1 < len(words) else duration
        start = min(max(lo + 0.01, words[a]["start"] - CUT_PAD_S), words[a]["start"])
        end = max(min(hi - 0.01, words[b]["end"] + CUT_PAD_S), words[b]["end"])
        out.append({"start": round(start, 3), "end": round(min(end, duration), 3), "reason": c["reason"],
                    "text": " ".join(w["word"] for w in words[a:b + 1])})
    return out


# ---------------------------------------------------------------------------------------------- code-found candidates
# An LLM misses things now and then (even at temperature 0, Groq's answers differ between runs), so simple,
# precise patterns are searched in plain code too. Gemini still has to approve every candidate.
SENTENCE_END = (".", "?", "!", "।", "…")
STUTTER_GAP_S = 0.8  # repeats further apart than this are not a stutter
RESTART_WORDS = 3  # a restart = the same first 3 words again
RESTART_MAX_SPAN = 25  # the failed take is at most this many words
RESTART_BOUNDARY_GAP_S = 0.25


def _same(words: list[dict], i: int, j: int, length: int) -> bool:
    if i + length > len(words) or j + length > len(words):
        return False
    a, b = [_norm_word(w["word"]) for w in words[i:i + length]], [_norm_word(w["word"]) for w in words[j:j + length]]
    return a == b and all(a)


def stutter_at(words: list[dict], i: int) -> tuple[int, int] | None:
    """(from, to) of a stutter starting at word i: a 1-3 word unit said 2+ times in a row; everything but the LAST
    repetition is cut, so the speaker's words stay. None if there is no stutter at i."""
    for length in (1, 2, 3):
        reps = 1
        while (_same(words, i, i + reps * length, length)
               and words[i + reps * length]["start"] - words[i + reps * length - 1]["end"] < STUTTER_GAP_S):
            reps += 1
        if reps >= 2:
            return i, i + (reps - 1) * length - 1
    return None


def stutter_candidates(words: list[dict]) -> list[dict]:
    out, i = [], 0
    while i < len(words) - 1:
        hit = stutter_at(words, i)
        if hit:
            out.append({"from": hit[0], "to": hit[1], "reason": "stutter"})
            i = hit[1] + 1
        else:
            i += 1
    return out


def _starts_sentence(words: list[dict], i: int) -> bool:
    return i == 0 or words[i - 1]["word"].rstrip().endswith(SENTENCE_END) or         words[i]["start"] - words[i - 1]["end"] >= RESTART_BOUNDARY_GAP_S


def restart_candidates(words: list[dict]) -> list[dict]:
    """A line that begins again right after a failed first attempt: the same first words come back within a few
    words, at a sentence boundary. The earlier attempt [i, j-1] is the candidate cut."""
    out, i, n = [], 0, len(words)
    while i < n - RESTART_WORDS:
        found = None
        if _starts_sentence(words, i):
            for j in range(i + RESTART_WORDS, min(n, i + RESTART_MAX_SPAN + 1)):
                if _starts_sentence(words, j) and _same(words, i, j, RESTART_WORDS):
                    found = j
                    break
        if found:
            out.append({"from": i, "to": found - 1, "reason": "retake"})
            i = found
        else:
            i += 1
    return out


def repair(words: list[dict], c: dict) -> dict | None:
    """Make a proposal safe. A stutter cut must leave the last repetition (the models sometimes cut every copy and
    the word vanishes); a stutter that is not really a repetition is dropped."""
    if c["reason"] != "stutter":
        return c
    hit = stutter_at(words, c["from"])
    return {"from": hit[0], "to": hit[1], "reason": "stutter"} if hit else None


def _groq_candidates(words: list[dict], language: str, client: Any, runs: int = 2) -> tuple[list[dict], list[str]]:
    """Groq's proposals from `runs` parallel asks, merged (two runs differ, and what either one finds is checked)."""
    from concurrent.futures import ThreadPoolExecutor

    found: list[dict] = []
    errors: list[str] = []

    def ask(_):
        try:
            return propose(words, language, client=client), None
        except RetakeError as exc:
            return [], str(exc)

    with ThreadPoolExecutor(max_workers=runs) as pool:
        for got, err in pool.map(ask, range(runs)):
            found += got
            if err:
                errors.append(err)
    return found, errors


def candidates(words: list[dict], groq_found: list[dict]) -> list[dict]:
    """Groq's proposals (repaired) plus the code-found ones, de-duplicated, sorted, overlaps merged."""
    pool = [x for x in (repair(words, c) for c in groq_found) if x] + stutter_candidates(words) + restart_candidates(words)
    unique = {(c["from"], c["to"]): c for c in pool}.values()
    return sanitize(list(unique), len(words))


def find_retakes(words: list[dict], language: str, duration: float, groq_client: Any = None,
                 gemini_client: Any = None, sleep: Callable[[float], None] | None = None,
                 log: Callable[[str], None] = print) -> dict:
    """{"cuts": [{start, end, reason, text}], "warnings": [...], "proposed": n, "approved": n}. Never raises."""
    result: dict = {"cuts": [], "warnings": [], "proposed": 0, "approved": 0}
    if len(words) < MIN_WORDS:
        return result
    log("Looking for stutters and retakes (Groq)...")
    groq_found, errors = _groq_candidates(words, language, groq_client)
    if errors and not groq_found:
        result["warnings"].append(f"Groq was not available for retake detection ({errors[0]}); only exact repeats "
                                  "were looked for.")
    proposals = candidates(words, groq_found)
    result["proposed"] = len(proposals)
    if not proposals:
        return result
    try:
        log(f"Checking {len(proposals)} possible cut(s) (Gemini)...")
        approved = sanitize(verify(words, proposals, client=gemini_client, sleep=sleep), len(words))
    except RetakeError as exc:
        result["warnings"].append(f"Retake cuts skipped, nothing was cut without a second check: {exc}")
        return result
    result["approved"] = len(approved)
    cuts = to_time_cuts(words, approved, duration)
    if duration > 0 and sum(c["end"] - c["start"] for c in cuts) > MAX_REMOVED_FRACTION * duration:
        result["warnings"].append("The AI wanted to cut more than 40% of the video as retakes, which is not plausible, "
                                  "so no retakes were cut.")
        return result
    result["cuts"] = cuts
    return result
