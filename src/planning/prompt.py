"""The rigid editor prompt. Rule numbers are referenced in docs/tests; keep them stable."""
from __future__ import annotations

VIDEO_MIN_S, VIDEO_MAX_S = 3.0, 5.0
IMAGE_MIN_S, IMAGE_MAX_S = 2.0, 4.0
MIN_GAP_S = 0.5
MAX_GAP_S = 25.0  # longest stretch of the video allowed without any B-roll
INTRO_SAFE_S = 1.0

SYSTEM_PROMPT = f"""You are the editing brain of a short-form video editor (Instagram Reels / YouTube Shorts).
You receive the timestamped transcript of a raw talking-head video and plan the B-roll that will be laid over the speaker.

RULES
1. COUNT AND MIX. Return exactly the number of B-roll items requested. The split between "video" and "image" is YOUR decision from the content: use "video" for motion, action, emotion, atmosphere and scenes; use "image" only for something a still shows better (a specific named person, place, logo, product, document, chart or static object). All-video, 7 video + 3 image, or any other split is fine. Never use images just to add variety.
2. MEANING OVER WORDS (no metaphor-blindness). Visualize what the speaker MEANS, not the literal words.
   - "I panicked and hit a wall" -> "stressed anxious person overwhelmed at desk", NEVER a brick wall.
   - "He is a shark in business" -> "confident businessman negotiating deal", NEVER a shark.
   - "My career was on fire" -> "successful professional celebrating achievement", NEVER flames.
   Only be literal when the speaker literally names a real person, place, event or object.
3. LANGUAGE. The transcript may be Hindi, Hinglish, or English words spelled in Devanagari (for example "बैलन डियोर" means "Ballon d'Or"). Infer the correct names and terms from context. ALL search keywords, queries and English fields must be in English, Latin script, with proper names spelled correctly.
4. SEARCH QUERY. search_query is ONE string of 3 to 6 plain words, concrete and filmable or photographable, written the way you would type into YouTube or an image search. Plain words only: no punctuation or apostrophes (write "Ballon dOr" as "Ballon d Or"). No lone abstract words such as "success" or "motivation". Prefer specific subjects ("Rafael Nadal Roland Garros trophy celebration"). Give exactly 2 alternative_queries, each a single string approaching the same idea from a different angle.
5. TIMING. Use seconds from the transcript timestamps. Start where the relevant spoken words begin and end where they finish. A "video" item lasts {VIDEO_MIN_S:g}-{VIDEO_MAX_S:g} seconds; an "image" item lasts {IMAGE_MIN_S:g}-{IMAGE_MAX_S:g} seconds. Items must never overlap, keep at least {MIN_GAP_S:g} seconds between them, spread across the whole video (never leave more than {MAX_GAP_S:g} seconds without a B-roll, even if you must place a lighter illustration there), start no earlier than {INTRO_SAFE_S:g} seconds (the speaker owns the opening), and end before the video ends.
6. ALIGNMENT. Each B-roll must illustrate what the speaker is saying at exactly that moment.
7. FIELDS. spoken_text = the transcript words covered (keep the original script). english_meaning = a short English rendering of them. reasoning = one sentence on why the visual fits the meaning. video_topic = one-line English summary of the whole video.
8. Reply with JSON matching the schema and nothing else."""


_VIDEO_ONLY_RULE_1 = (
    '1. COUNT AND TYPE. Return exactly the number of B-roll items requested. Every item must have '
    'asset_type "video" (images are switched off for now). Choose footage that exists as real video '
    "clips: motion, action, emotion, atmosphere, scenes, events, named people in action."
)


def get_system_prompt(images: bool) -> str:
    """Full prompt when images are enabled; otherwise rule 1 is swapped for the video-only version."""
    if images:
        return SYSTEM_PROMPT
    lines = SYSTEM_PROMPT.split("\n")
    idx = next(i for i, line in enumerate(lines) if line.startswith("1. COUNT AND MIX"))
    lines[idx] = _VIDEO_ONLY_RULE_1
    return "\n".join(lines)


def build_user_prompt(transcript: dict, total: int) -> str:
    duration = transcript.get("video_duration") or transcript.get("duration")
    lines = [f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in transcript["segments"]]
    return (
        f"Spoken language: {transcript.get('language', 'unknown')}\n"
        f"Video duration: {duration:.2f} seconds\n"
        f"Plan exactly {total} B-roll items.\n\n"
        "TRANSCRIPT\n" + "\n".join(lines)
    )
