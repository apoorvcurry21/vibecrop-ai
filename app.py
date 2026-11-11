"""VibeCrop AI - Streamlit mini-editor.  Run:  streamlit run app.py"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="VibeCrop AI", page_icon="🎬", layout="wide")

# Guard: running with the wrong Python (not the project's .venv) is the most common setup mistake.
_missing = [m for m in ("groq", "google.genai", "yt_dlp", "dotenv") if importlib.util.find_spec(m) is None]
if _missing:
    st.title("VibeCrop AI")
    st.error(f"Missing Python packages: {', '.join(_missing)}. This app is running with `{sys.executable}`, "
             "which is not the project's virtual environment.")
    st.write("Close this app (Ctrl+C in its terminal) and start it with the project's Python instead:")
    st.code(f'cd "{Path(__file__).resolve().parent}"\n.\\.venv\\Scripts\\python.exe -m streamlit run app.py\n'
            "# or just double-click run_app.bat")
    st.stop()

from src import config, workflow  # noqa: E402 - imported after the dependency guard on purpose
from src.captions.rewrite import MODES  # noqa: E402
from src.cleanup import BUDGET_GB, fmt_bytes, over_budget, workdir_usage_bytes  # noqa: E402


def open_job(path: Path | None) -> None:
    """Remember the open job in the session and in the URL, so a page reload comes back to it."""
    if path is None:
        st.session_state.pop("job_dir", None)
        st.query_params.pop("job", None)
    else:
        st.session_state["job_dir"] = str(path)
        st.query_params["job"] = path.name


def job_dir() -> Path | None:
    raw = st.session_state.get("job_dir")
    if not raw and st.query_params.get("job"):
        # Only ever resolve to a folder directly inside <workdir>/jobs (the URL is user-controlled).
        candidate = workflow.jobs_root() / Path(st.query_params["job"]).name
        if candidate.is_dir():
            st.session_state["job_dir"] = raw = str(candidate)
    return Path(raw) if raw and Path(raw).is_dir() else None


def fail(exc: Exception) -> None:
    st.error(str(exc))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    spoken = st.selectbox(
        "Spoken language", list(workflow.SPOKEN_LANGUAGES),
        format_func=lambda c: f"{workflow.SPOKEN_LANGUAGES[c]} ({c})", key="spoken",
    )
    captions_on = st.checkbox("Burn in captions", value=True)
    caption_mode = st.radio("Caption language", list(MODES), format_func=MODES.get, disabled=not captions_on)
    auto_clean = st.checkbox(
        "Delete temporary clips after render", value=True,
        help="Keeps only final.mp4 and the JSON files. Untick it to try another caption language without re-downloading.",
    )
    usage = workdir_usage_bytes()
    st.metric("Temp storage used", fmt_bytes(usage), help=f"Folder: {config.workdir_root()}")
    if over_budget():
        st.warning(f"Over the {BUDGET_GB:g} GB budget. Delete old job folders.")

    jobs = workflow.list_jobs()
    if jobs:
        st.divider()
        chosen = st.selectbox(
            "Open an earlier job", jobs, index=None, placeholder="Choose a job...",
            format_func=lambda p: p.name + ("" if (p / "plan.json").is_file() else "  (unfinished)"))
        if chosen and st.button("Open job"):
            open_job(chosen)
            st.rerun()
    if job_dir() and st.button("Start a new video"):
        open_job(None)
        st.rerun()

st.title("VibeCrop AI")
st.caption("Upload a talking-head video, review the AI's B-roll plan, then export a captioned 9:16 reel.")

job = job_dir()

# ---------------------------------------------------------------- step 1: video in
if job is None:
    st.subheader("1. Choose your video")
    up = st.file_uploader("Upload a raw video (about 1-2 minutes)", type=["mp4", "mov", "mkv", "m4v"])
    local = st.text_input("...or the path of a video already on this PC", placeholder=r"D:\vids\my_video.mp4")
    if st.button("Build B-roll plan", type="primary", disabled=not (up or local.strip())):
        try:
            if up:
                tmp = config.workdir_root() / "uploads" / Path(up.name).name
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(up.getbuffer())
                new = workflow.create_job(tmp, spoken, copy_into_job=True)
                tmp.unlink(missing_ok=True)
            else:
                new = workflow.create_job(local.strip().strip('"'), spoken)
            open_job(new)
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - show any setup problem to the user
            fail(exc)
    st.stop()

# ---------------------------------------------------------------- step 2: build plan (resumable)
plan_ready = (job / "plan.json").is_file()
meta = workflow.read_job_meta(job)
if not plan_ready:
    st.subheader("2. Building your plan")
    st.write(f"Video: **{meta.get('source_name', job.name)}**, language: **{meta.get('language')}**")
    with st.status("Transcribing, planning and downloading clips (a few minutes)...", expanded=True) as status:
        try:
            workflow.run_pipeline(job, log=lambda m: status.write(m))
            status.update(label="Plan ready", state="complete")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            status.update(label="Something went wrong", state="error")
            fail(exc)
            st.info("Fix the problem (for example a missing API key in .env), then reload this page: "
                    "the job resumes from the step that failed.")
    st.stop()

# ---------------------------------------------------------------- step 3: review
plan = workflow.load_plan(job)
items = plan["items"]
ready = sum(1 for i in items if (i.get("asset") or {}).get("status") == "ok" and i.get("enabled") is not False)
final = job / "final.mp4"

st.subheader("2. Review the plan")
st.write(f"**{plan.get('video_topic', '')}**")
st.caption(f"{ready} of {len(items)} B-roll clips ready. Edit a search query, press Regenerate for a different "
           "video, or remove a slot (the speaker stays on screen there).")

if not any((job / (i.get("asset") or {}).get("path", "x")).is_file() for i in items) and final.is_file():
    st.info("This job has been rendered and its temporary clips were cleaned up.")

for item in items:
    asset = item.get("asset") or {}
    removed = item.get("enabled") is False
    clip = job / asset.get("path", "") if asset.get("path") else None
    with st.container(border=True):
        left, mid, right = st.columns([2, 4, 2], vertical_alignment="top")
        with left:
            if clip and clip.is_file() and not removed:
                st.video(str(clip))
            elif removed:
                st.caption("Removed: the speaker stays on screen.")
            elif asset.get("status") == "failed":
                st.warning("No clip found for this query. Edit it and search again.")
            else:
                st.caption("No preview available.")
        with mid:
            st.markdown(f"**#{item['id']}**  ·  {item['timestamp_start']:.1f}s to {item['timestamp_end']:.1f}s")
            st.caption(f"Says: {item.get('english_meaning', '')}")
            new_q = st.text_input("Search query", value=item["search_query"], key=f"q{item['id']}",
                                  disabled=removed)
            if asset.get("status") == "ok":
                st.caption(f"From: {asset.get('channel', '?')} - {asset.get('title', '')[:70]}")
        with right:
            changed = new_q.strip() != item["search_query"]
            if st.button("Search with this query", key=f"s{item['id']}", disabled=removed or not changed):
                with st.spinner("Searching YouTube..."):
                    try:
                        workflow.set_query(job, item["id"], new_q)
                    except Exception as exc:  # noqa: BLE001
                        fail(exc)
                st.rerun()
            if st.button("Regenerate asset", key=f"r{item['id']}", disabled=removed):
                with st.spinner("Finding a different clip..."):
                    workflow.regenerate(job, item["id"])
                st.rerun()
            if st.button("Restore" if removed else "Remove slot", key=f"d{item['id']}"):
                workflow.set_enabled(job, item["id"], removed)
                st.rerun()

# ---------------------------------------------------------------- step 4: render
st.subheader("3. Export")
if workflow.source_video_path(job) is None:
    path = st.text_input("The original video file for this job was not found. Enter its path:",
                         placeholder=r"D:\vids\my_video.mp4")
    if path.strip() and Path(path.strip().strip('"')).is_file():
        workflow.write_job_meta(job, source_video=path.strip().strip('"'))
        st.rerun()
else:
    if st.button("Approve and render", type="primary"):
        with st.status("Rendering your reel...", expanded=True) as status:
            try:
                _, freed = workflow.render_final(
                    job, caption_mode if captions_on else None, auto_clean=auto_clean,
                    log=lambda m: status.write(m),
                )
                status.update(label="Done" + (f" - freed {fmt_bytes(freed)}" if freed else ""), state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label="Render failed", state="error")
                fail(exc)

if final.is_file():
    st.success(f"Final video: {final}")
    st.video(str(final))
