"""The chat assistant's brain: a Gemini function-calling loop.

`stream_turn` is a generator of small event dicts the UI renders as they arrive:
  {"type": "text", "text": ...}      a chunk of the reply (streamed, so the first words show within ~1 s)
  {"type": "tool", "label": ...}     a tool is running ("Swapping slots 2 and 4...")
  {"type": "card", "card": {...}}    something to show in the chat (thumbnails, layout picker, ...)
  {"type": "error", "text": ...}     the AI could not be reached
  {"type": "done", "text", "changed", "cards"}   always last

It runs synchronously inside the Streamlit script, so edits to plan.json are naturally one at a time."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterator

from src import config
from src.assistant import assets, tools
from src.assistant.tools import Attachment, ToolContext
from src.gemini_client import TRANSIENT_CODES, GeminiError, make_client

MAX_ROUNDS = 8  # tool-call rounds per message; a runaway loop stops here
HISTORY_MESSAGES = 16  # earlier chat messages sent as context (text only)

SYSTEM_PROMPT = """You are the editing assistant inside VibeCrop AI, a mini video editor that turns a talking-head \
video into a vertical 9:16 reel with B-roll on top of the speaker. You edit the user's project by calling tools; the \
user can also use the buttons on the page, so the project may change between messages.

Language: reply in the language the user writes (English, Hindi or Hinglish), short and friendly, like a helpful \
editor sitting next to them. No long lists, no markdown headings.

How the project works:
- A "slot" is one B-roll shown over the speaker from `start` to `end` seconds. Slots never overlap and the first \
second belongs to the speaker.
- Layouts: "overlay" (clip fills the screen), "split" (clip on the top half, speaker on the bottom half), "pip" \
(small picture-in-picture box over the speaker; position and size are adjustable).
- Text overlays are separate from slots and can sit anywhere on the timeline.

Rules:
- Use ids and times only from get_plan / find_in_transcript / get_transcript. Never invent them. When the user says \
"where I say X", call find_in_transcript first.
- Edits apply instantly and the user has an Undo button, so just do what was asked; do not ask for permission for \
normal edits. Ask a short question only when the request is genuinely ambiguous.
- After editing, say in one or two lines what you changed (slot ids, times, layout).
- If a tool returns an error, explain it plainly and fix it if you can (for example pick another time range from \
free_gaps). Never claim something is done unless a tool call succeeded.
- You cannot re-plan the whole video or export it; tell the user to use the page's buttons for that.

Pictures and clips:
- Pictures or videos the user attaches arrive as att_1, att_2... (you can see the pictures). Use place_image / place_video_file with those ids. Never ask the user for a file path.
- "Put this where I say X": find_in_transcript for X, then place the media starting about 0.3 s before that word, for roughly 3 seconds (or as long as the user asked). "Wherever it fits": read get_transcript and choose a moment that matches what the picture or clip shows.
- A YouTube link: place_youtube_clip with the user's time as from_time if they gave one, otherwise describe the moment in `about`. Default clip length 5 seconds unless the user says otherwise.
- "Find / get me a picture of ...": search_web_images, then place the best match if they said where, else let them choose. "Make / generate a picture": generate_image (it may fail for real people or on this account: then offer search_web_images). Web pictures get a small credit line in the video automatically.
- If the user asks you to add a NEW picture or clip and has not said how it should look, and you have not asked yet in this chat, call ask_layout, end your reply with one short question and wait for their tap. Do not ask when they already said (full screen / split / small box / corner...), when replacing the clip of an existing slot, or after they have chosen once (reuse that choice unless they change it).
- Layouts: full screen = overlay, a small box = pip with a position (top-right by default), split = top half.
- After placing something the user cares about seeing, you may call preview_slot.

Current project (may be stale after your own edits; call get_plan for a fresh copy):
"""


def system_prompt(job: Path, attachments: list[Attachment]) -> str:
    extra = ""
    if attachments:
        extra = "\n\nFiles the user attached to this message: " + "; ".join(
            f"{a.id} ({a.kind}, '{a.name}')" for a in attachments)
    return SYSTEM_PROMPT + tools.plan_summary(job) + extra


def build_config(system: str, thinking: bool = True) -> Any:
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=tools.declarations())],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.4,
        thinking_config=types.ThinkingConfig(thinking_level="low") if thinking else None,
    )


def history_contents(history: list[dict]) -> list:
    """Earlier chat turns as plain text (tool calls of old turns are not replayed; the plan summary has the state)."""
    from google.genai import types

    out = []
    for m in history[-HISTORY_MESSAGES:]:
        text = (m.get("text") or "").strip()
        for f in m.get("files") or []:
            text += f"\n[attached {f.get('kind', 'file')} {f.get('id', '')} '{f.get('name', '')}']"
        if text.strip():
            out.append(types.Content(role="user" if m["role"] == "user" else "model",
                                     parts=[types.Part(text=text.strip())]))
    return out


def user_content(text: str, attachments: list[Attachment]):
    """The new message. Pictures go to the model as small JPEGs so it can see what they show."""
    from google.genai import types

    parts = [types.Part(text=text.strip() or "(no text, only the attached file)")]
    for a in attachments:
        if a.kind == "image" and (jpeg := assets.vision_jpeg(a.path)):
            parts.append(types.Part(text=f"[picture {a.id}, file '{a.name}']"))
            parts.append(types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
    return types.Content(role="user", parts=parts)


def _chunk_parts(chunk: Any) -> list:
    cands = getattr(chunk, "candidates", None) or []
    content = getattr(cands[0], "content", None) if cands else None
    return list(getattr(content, "parts", None) or [])


def open_stream(client: Any, models: list[str], contents: list, config_: Any,
                sleep: Callable[[float], None]) -> Iterator[Any]:
    """Stream one model response, retrying transient API errors and falling back to the next model. Errors that
    happen before the first chunk are retried; after that they surface as GeminiError."""
    import httpx
    from google.genai import errors

    last: Exception | None = None
    for model in models:
        cfg = config_
        for attempt in range(1, 4):
            try:
                it = iter(client.models.generate_content_stream(model=model, contents=contents, config=cfg))
                first = next(it, None)
            except httpx.HTTPError as exc:  # a timeout or a dropped connection: don't wait again, try the next model
                last = exc
                break
            except errors.APIError as exc:
                last = exc
                code = getattr(exc, "code", None)
                if code == 400 and cfg is config_ and "think" in str(exc).lower():
                    cfg = config_.model_copy(update={"thinking_config": None})  # this model has no thinking levels
                    continue
                if code in TRANSIENT_CODES and attempt < 3:
                    sleep(2 * attempt)
                    continue
                break  # try the next model
            try:
                if first is not None:
                    yield first
                yield from it
            except (errors.APIError, httpx.HTTPError) as exc:
                raise GeminiError(f"Gemini request failed: {exc}") from exc
            return
    raise GeminiError(f"Gemini request failed: {_describe(last)}")


def _describe(exc: Exception | None) -> str:
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        return "the AI took too long to answer"
    return str(exc) if exc else "unknown error"


def stream_turn(job: Path, history: list[dict], user_text: str, attachments: list[Attachment] | None = None,
                client: Any = None, models: list[str] | None = None,
                sleep: Callable[[float], None] = time.sleep,
                log: Callable[[str], None] = lambda m: None) -> Iterator[dict]:
    attachments = attachments or []
    ctx = ToolContext(job=job, attachments={a.id: a for a in attachments}, log=log)
    try:
        client = client or make_client(config.CHAT_TIMEOUT_MS)
        models = models or config.chat_model_chain()
        contents = history_contents(history) + [user_content(user_text, attachments)]
        cfg = build_config(system_prompt(job, attachments))
    except (config.ConfigError, GeminiError) as exc:
        yield {"type": "error", "text": str(exc)}
        yield {"type": "done", "text": "", "changed": False, "cards": []}
        return

    from google.genai import types

    reply: list[str] = []
    done_labels: list[str] = []
    cards: list[dict] = []
    finished = False
    try:
        for _ in range(MAX_ROUNDS):
            round_parts, calls = [], []
            for chunk in open_stream(client, models, contents, cfg, sleep):
                for part in _chunk_parts(chunk):
                    round_parts.append(part)
                    if getattr(part, "function_call", None):
                        calls.append(part.function_call)
                    elif getattr(part, "text", None) and not getattr(part, "thought", False):
                        reply.append(part.text)
                        yield {"type": "text", "text": part.text}
            if not calls:
                finished = True
                break
            contents.append(types.Content(role="model", parts=round_parts))
            responses = []
            for call in calls:
                args = dict(call.args or {})
                label = tools.status_label(call.name, args)
                yield {"type": "tool", "label": label}
                result = tools.run_tool(ctx, call.name, args)
                if "error" not in result and call.name in tools.REGISTRY and tools.REGISTRY[call.name].mutating:
                    done_labels.append(tools.REGISTRY[call.name].label(args))
                responses.append(types.Part.from_function_response(name=call.name, response=result))
                for card in ctx.cards:
                    cards.append(card)
                    yield {"type": "card", "card": card}
                ctx.cards.clear()
            contents.append(types.Content(role="user", parts=responses))
    except GeminiError as exc:
        yield {"type": "error", "text": f"I couldn't reach the AI just now. {exc}"}
        yield {"type": "done", "text": "".join(reply), "changed": ctx.changed, "cards": cards}
        return
    except Exception as exc:  # noqa: BLE001 - whatever goes wrong, the chat panel must keep working
        yield {"type": "error", "text": f"Something went wrong ({type(exc).__name__}). Your project is unchanged "
                                        "by this step; try again or rephrase."}
        yield {"type": "done", "text": "".join(reply), "changed": ctx.changed, "cards": cards}
        return

    text = "".join(reply).strip()
    if not finished:
        note = "\n\n(I stopped here to avoid going in circles. Tell me what to do next.)"
        text += note
        yield {"type": "text", "text": note}
    elif not text:
        text = ("Done: " + "; ".join(done_labels) + ".") if done_labels else "Okay."
        yield {"type": "text", "text": text}
    yield {"type": "done", "text": text, "changed": ctx.changed, "cards": cards}
