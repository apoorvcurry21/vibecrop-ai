"""The chat's floating button and panel, wrapped in ONE fragment.

Opening and closing the chat therefore reruns only this small fragment, never the whole review page (which would
redraw every slot and video player and feel slow). Light on purpose: the heavy chat code (ui.py) is only imported
once the panel is opened. Edits made by the chat still refresh the page, via an explicit st.rerun() in ui.py."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.assistant.style import FAB_CSS


def _open() -> None:
    st.session_state["chat_open"] = True


@st.fragment
def shell(job_str: str) -> None:
    if st.session_state.get("chat_open"):
        from src.assistant import ui  # imported only once the panel is opened

        ui.mount(Path(job_str))
        return
    st.markdown(FAB_CSS, unsafe_allow_html=True)
    with st.container(key="chat_fab"):
        st.button("💬", key="chat_open_btn", on_click=_open, help="Open the AI assistant")
