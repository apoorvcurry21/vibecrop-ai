"""Streamlit chat panel. Imported only when the panel is opened (see app.py).

The panel is a fragment: sending a message, clicking a chip or Undo reruns only the chat, never the whole review
page. The page is refreshed (one full rerun) only after a turn that actually changed the project."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.assistant import agent, assets, media_tools, snapshots, store, style
from src.assistant.tools import Attachment

VISIBLE_MESSAGES = 50  # older ones stay in chat.json but are not drawn
BOX_HEIGHT = 440
SUGGESTIONS = ("What can you do?", "Show me the plan", "Add some text on screen", "Swap two of the B-rolls")
FILE_TYPES = ["png", "jpg", "jpeg", "webp", "mp4", "mov", "m4v"]


def _close() -> None:
    st.session_state["chat_open"] = False


def _suggest(text: str) -> None:
    st.session_state["chat_pending"] = text


def _is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - older Streamlit / no theme info: assume light
        return False


def _render_cards(job: Path, cards: list[dict], key: str = "live") -> None:
    """Pictures the assistant shows: one generated/attached image, or a row of search results with a Use button."""
    for ci, card in enumerate(cards or []):
        if card.get("type") == "image" and (job / card.get("path", "x")).is_file():
            st.image(str(job / card["path"]), caption=card.get("caption"), width=220)
        elif card.get("type") == "layouts":
            _render_layouts(card, key, ci)
        elif card.get("type") == "choices":
            items = [i for i in card.get("items", []) if (job / i.get("path", "x")).is_file()]
            for row in range(0, len(items), 3):
                for col, item in zip(st.columns(3), items[row:row + 3]):
                    with col:
                        st.image(str(job / item["path"]), caption=item.get("caption"), width="stretch")
                        col.button("Use this", key=f"use_{key}_{ci}_{item['id']}", on_click=_suggest,
                                   args=(f"Use {item['id']}",), width="stretch")


def _render_layouts(card: dict, key: str, ci: int) -> None:
    """Illustrated examples of the layouts; tapping one sends 'Use the ... layout' as the user's next message."""
    for col, (lid, name, blurb, say) in zip(st.columns(3), media_tools.LAYOUT_CHOICES):
        with col:
            st.image(str(assets.STATIC_DIR / f"layout_{lid}.png"), width="stretch")
            st.markdown(f"**{name}**")
            st.caption(blurb)
            st.button("Use this", key=f"lay_{key}_{ci}_{lid}", on_click=_suggest, args=(say,), width="stretch")
    if card.get("text_example"):
        st.image(str(assets.STATIC_DIR / "layout_text.png"),
                 caption="Text on top of the video (tell me the words, where and what colour)", width=150)


def _render_message(job: Path, m: dict, key: str = "live") -> None:
    with st.chat_message(m["role"]):
        for f in m.get("files") or []:
            st.caption(f"📎 {f.get('name', 'file')}")
        if m.get("text"):
            st.markdown(m["text"])
        _render_cards(job, m.get("cards"), key)


def _save_files(job: Path, files: list) -> tuple[list[Attachment], list[str]]:
    saved, problems = [], []
    for f in files or []:
        try:
            saved.append(assets.save_attachment(job, f.name, f.getvalue()))
        except assets.AssetError as exc:
            problems.append(str(exc))
    return saved, problems


def _run_turn(job: Path, box, text: str, attachments: list[Attachment], problems: list[str]) -> bool:
    """Show the user's message, stream the assistant's reply, save both. Returns True if the project changed."""
    history = store.load(job)
    files = [{"id": a.id, "name": a.name, "kind": a.kind} for a in attachments]
    store.append(job, "user", text, files=files)
    state: dict = {"changed": False, "cards": [], "error": None, "final": ""}
    with box:
        _render_message(job, {"role": "user", "text": text, "files": files})
        for p in problems:
            st.warning(p)
        with st.chat_message("assistant"):
            status = st.empty()
            status.caption("Thinking...")

            def tokens():
                for ev in agent.stream_turn(job, history, text, attachments):
                    kind = ev["type"]
                    if kind == "text":
                        status.empty()
                        yield ev["text"]
                    elif kind == "tool":
                        status.caption(ev["label"])
                    elif kind == "card":
                        state["cards"].append(ev["card"])
                    elif kind == "error":
                        state["error"] = ev["text"]
                    elif kind == "done":
                        state["changed"], state["final"] = ev["changed"], ev["text"]

            streamed = st.write_stream(tokens())
            status.empty()
            if state["error"]:
                st.error(state["error"])
            _render_cards(job, state["cards"], key="new")
    reply = (streamed if isinstance(streamed, str) else state["final"]) or state["error"] or ""
    store.append(job, "assistant", reply.strip(), cards=state["cards"])
    return state["changed"]


@st.fragment
def chat_panel(job_str: str) -> None:
    job = Path(job_str)
    title, undo_col, close_col = st.columns([5, 2, 1], vertical_alignment="center")
    title.markdown("**AI assistant**")
    last = snapshots.last_label(job)
    if undo_col.button("↩ Undo", key="chat_undo", disabled=last is None, help=f"Undo: {last}" if last else
                       "Nothing to undo yet"):
        undone = snapshots.undo(job)
        store.append(job, "assistant", f"Undone: {undone}.")
        st.rerun()  # the plan changed on disk: refresh the whole page
    close_col.button("✕", key="chat_close", on_click=_close, help="Close")

    box = st.container(height=BOX_HEIGHT, border=False)
    messages = store.load(job)
    with box:
        if not messages:
            st.caption("Tell me what to change, in English or Hinglish. I can swap or retime B-rolls, change "
                       "layouts, add text, and more. You can attach pictures too.")
            cols = st.columns(2)
            for i, s in enumerate(SUGGESTIONS):
                cols[i % 2].button(s, key=f"chat_s{i}", on_click=_suggest, args=(s,), width="stretch")
        for n, m in enumerate(messages[-VISIBLE_MESSAGES:]):
            _render_message(job, m, key=f"h{n}")

    prompt = st.chat_input("Ask me to edit your video...", key="chat_input", accept_file="multiple",
                           file_type=FILE_TYPES)
    pending = st.session_state.pop("chat_pending", None)
    text = (prompt.text if prompt else pending) or ""
    files = list(prompt.files) if prompt else []
    if not (text.strip() or files):
        return
    attachments, problems = _save_files(job, files)
    if _run_turn(job, box, text.strip(), attachments, problems):
        st.rerun()  # the project changed: refresh the timeline and slots (one full rerun per turn)


def mount(job: Path) -> None:
    """Draw the open panel (the closed state, a floating button, lives in app.py)."""
    st.markdown(style.panel_css(_is_dark()), unsafe_allow_html=True)
    with st.container(key="chat_panel"):
        chat_panel(str(job))
