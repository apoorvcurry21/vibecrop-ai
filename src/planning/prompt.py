"""The rigid editor prompt. Rule numbers are referenced in docs/tests; keep them stable."""
from __future__ import annotations

VIDEO_MIN_S, VIDEO_MAX_S = 3.0, 5.0
MIN_GAP_S = 0.5
MAX_GAP_S = 25.0  # longest stretch of the video allowed without any B-roll
INTRO_SAFE_S = 1.0

SYSTEM_PROMPT = f"""You are the editing brain of a short-form video editor (Instagram Reels / YouTube Shorts).
You receive the timestamped transcript of a raw talking-head video and plan the B-roll that will be laid over the speaker.

RULES
1. COUNT AND TYPE. Return exactly the number of B-roll items requested. Every item must have asset_type "video". Choose footage that exists as real video clips: motion, action, emotion, atmosphere, scenes, events, named people in action.
2. MEANING OVER WORDS (no metaphor-blindness). Visualize what the speaker MEANS, not the literal words.
   - "I panicked and hit a wall" -> "stressed anxious person overwhelmed at desk", NEVER a brick wall.
   - "He is a shark in business" -> "confident businessman negotiating deal", NEVER a shark.
   - "My career was on fire" -> "successful professional celebrating achievement", NEVER flames.
   Only be literal when the speaker literally names a real person, place, event or object.
3. LANGUAGE. The transcript may be Hindi, Hinglish, or English words spelled in Devanagari (for example "बैलन डियोर" means "Ballon d'Or"). Infer the correct names and terms from context. ALL search keywords, queries and English fields must be in English, Latin script, with proper names spelled correctly.
4. SEARCH QUERY. search_query is ONE string of 3 to 6 plain words, concrete and filmable, written the way you would type into YouTube. Plain words only: no punctuation or apostrophes (write "Ballon dOr" as "Ballon d Or"). No lone abstract words such as "success" or "motivation". Prefer specific subjects ("Rafael Nadal Roland Garros trophy celebration"). Give exactly 2 alternative_queries, each a single string approaching the same idea from a different angle.
5. TIMING. Use seconds from the transcript timestamps. Start where the relevant spoken words begin and end where they finish. Each item lasts {VIDEO_MIN_S:g}-{VIDEO_MAX_S:g} seconds. Items must never overlap, keep at least {MIN_GAP_S:g} seconds between them, spread evenly across the whole video (never leave a longer stretch without a B-roll than the "Longest allowed stretch" stated in the request, even if you must place a lighter illustration there), start no earlier than {INTRO_SAFE_S:g} seconds (the speaker owns the opening), and end before the video ends.
6. ALIGNMENT. Each B-roll must illustrate what the speaker is saying at exactly that moment.
7. FIELDS. spoken_text = the transcript words covered (keep the original script). english_meaning = a short English rendering of them. reasoning = one sentence on why the visual fits the meaning. video_topic = one-line English summary of the whole video.
8. Reply with JSON matching the schema and nothing else."""


HYBRID_RULE = (
    '9. LAYOUT. Give every item a "layout". "overlay" = the B-roll fills the whole screen (best for short, punchy, '
    'high-impact or emotional visuals and fast action). "split" = the B-roll takes the top half while the speaker '
    "stays visible in the bottom half (best for longer, explanatory or supporting visuals, or when seeing the "
    "speaker's reaction matters). Use a real mix of both across the video, and never more than 3 items in a row "
    "with the same layout."
)


def get_system_prompt(hybrid: bool = False) -> str:
    """The editor prompt. `hybrid` appends rule 9 (per-item layout); the other styles do not ask Gemini about layout."""
    return SYSTEM_PROMPT + "\n" + HYBRID_RULE if hybrid else SYSTEM_PROMPT


def max_gap_for(total: int, duration: float) -> float:
    """Longest stretch allowed without B-roll: 25 s, relaxed for small counts (few items must spread wide)."""
    return max(MAX_GAP_S, 2.0 * duration / max(total, 1))


def feasible_max(duration: float) -> int:
    """Most video B-rolls that physically fit: an intro-safe start, then n shortest clips with a gap between.

    Used to warn about / clamp counts that the planner could never satisfy (e.g. 15 clips in a 30 s reel)."""
    n = int((duration - INTRO_SAFE_S + MIN_GAP_S) // (VIDEO_MIN_S + MIN_GAP_S))
    return max(1, n)


MAX_HOOK_LEN = 60
HOOK_REQUEST = (
    "Also fill `hooks` with exactly 3 DIFFERENT opening lines for the first 3 seconds of the reel, the text on screen "
    "that makes a viewer keep watching instead of scrolling. Each is 3 to 7 words, true to what this video really says "
    "(no false promises), and uses a curiosity gap, a bold claim or a direct question. Write them in Latin script in "
    "the speaker's own language style (English, or Hinglish if the speaker talks Hindi). No emojis, no hashtags, no "
    "quotation marks."
)


def clean_hooks(raw: list[str] | None) -> list[str]:
    """At most 3 distinct, non-empty, reasonably short opening lines (quotes stripped, spaces collapsed)."""
    out: list[str] = []
    for h in raw or []:
        h = " ".join(str(h).split()).strip("\"'“”‘’ ")
        if h and len(h) <= MAX_HOOK_LEN and h.lower() not in {x.lower() for x in out}:
            out.append(h)
    return out[:3]


def build_user_prompt(transcript: dict, total: int, notes: str = "") -> str:
    duration = float(transcript.get("video_duration") or transcript.get("duration") or 0.0)
    lines = [f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in transcript["segments"]]
    notes = " ".join((notes or "").split())
    notes_block = (
        "CREATOR NOTES (written by the person who made this video; treat them as correct and use them: "
        "exact names and spellings, topics, things they want shown or avoided. They override the "
        f"transcript wherever the transcript misheard something):\n{notes}\n\n" if notes else ""
    )
    return (
        f"Spoken language: {transcript.get('language', 'unknown')}\n"
        f"Video duration: {duration:.2f} seconds\n"
        f"Plan exactly {total} B-roll items.\n"
        f"Longest allowed stretch without B-roll: {max_gap_for(total, duration):.0f} seconds.\n"
        f"{HOOK_REQUEST}\n\n"
        f"{notes_block}"
        "TRANSCRIPT\n" + "\n".join(lines)
    )
