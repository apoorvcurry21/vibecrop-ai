"""VibeCrop AI - Streamlit mini-editor.  Run:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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
from src.captions import styles as caption_styles  # noqa: E402
from src.captions.chunks import chunk_words  # noqa: E402
from src.captions.rewrite import MODES  # noqa: E402
from src.cleanup import BUDGET_GB, fmt_bytes, over_budget, workdir_usage_bytes  # noqa: E402
from src.planning.layouts import LAYOUTS, STYLE_LABELS, STYLES  # noqa: E402
from src.planning.prompt import feasible_max  # noqa: E402
from src.rendering import framing as framing_mod  # noqa: E402
from src.rendering import preview  # noqa: E402
from src.rendering.compose import DEFAULT_FACE_Y, PIP_MAX_SCALE, PIP_MIN_SCALE, normalize_pip  # noqa: E402
from src.sfx import library as sfx_library  # noqa: E402


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


LAYOUT_LABELS = {"overlay": "Overlay (full screen)", "split": "Split (b-roll on top, facecam below)",
                 "pip": "Picture-in-picture (small box over you)"}
LAYOUT_COLORS = {"overlay": "#3b82f6", "split": "#10b981", "pip": "#f59e0b"}
PIP_POSITION_LABELS = {"top-left": "Top left", "top": "Top", "top-right": "Top right", "center": "Center",
                       "bottom-left": "Bottom left", "bottom": "Bottom", "bottom-right": "Bottom right"}


@st.cache_data(show_spinner=False, max_entries=256)
def cached_frame(clip: str, mtime: float, framing_json: str, layout: str, duration: float, at: str,
                 clip_len: float | None, pip_json: str = "null") -> bytes:
    return preview.frame_png(Path(clip), json.loads(framing_json), layout, duration, at, clip_len,
                             pip=json.loads(pip_json))


@st.cache_data(show_spinner=False, max_entries=64)
def cached_source_view(clip: str, mtime: float, framing_json: str, layout: str, pip_json: str = "null") -> bytes:
    return preview.source_view_png(Path(clip), json.loads(framing_json), layout, pip=json.loads(pip_json))


@st.cache_data(show_spinner=False, max_entries=128)
def cached_thumb(clip: str, mtime: float) -> bytes:
    return preview.thumb_png(Path(clip))


@st.cache_data(show_spinner=False, ttl=30)
def cached_usage() -> int:
    """The temp-folder size walks every file in the workdir, so it must not run on every rerun."""
    return workdir_usage_bytes()


@st.fragment
def slot_player(clip: str, mtime: float, item_id: int) -> None:
    """A still with a play button; the real video player mounts only when asked. A page with many <video>
    elements makes the browser (and every rerun) heavy, and most slots are never played."""
    state = f"vid{item_id}_{mtime}"
    box = st.empty()
    if not st.session_state.get(state):
        with box.container():
            try:
                st.image(cached_thumb(clip, mtime), use_container_width=True)
            except Exception:  # noqa: BLE001 - a missing still must never break the list
                st.caption("No still available.")
            if st.button("▶ Play", key=f"play_{state}"):
                st.session_state[state] = True
    if st.session_state.get(state):  # the click above lands here in the same run: the still is replaced by the player
        box.video(clip)


def rerun_fragment() -> None:
    """Rerun just the fragment this is called from. (A script run that is not a fragment rerun, such as a test's
    first click, does not allow the scope, so it falls back to a full rerun.)"""
    try:
        st.rerun(scope="fragment")
    except st.errors.StreamlitAPIException:
        st.rerun()


@st.fragment
def sfx_panel(job: Path) -> None:
    """Sound effects on the video: listen, move, change the volume, delete, add from the library. A fragment, so
    every click here reruns only this panel. (The chat can also add sounds by name, phrase or YouTube link.)"""
    plan = workflow.load_plan(job)
    sounds = plan.get("sounds", [])
    duration = float(plan.get("video_duration") or 0.0) or 600.0
    library = [e for e in sfx_library.load() if sfx_library.available(e)]
    if not library:
        st.info("The sound library is empty. Download it once with `python scripts/build_sfx_library.py` (about 53 "
                "meme and utility sounds, kept on this PC only). Meanwhile you can still ask the chat for any sound "
                "by name or YouTube link.")
    st.caption("Sounds play in the exported video only (the previews above are silent). Volume 1 = as recorded; "
               "new sounds start a little under the voice.")
    for s in sounds:
        entry = sfx_library.get(s["lib_id"])
        ver = f"{s['id']}_{s['start']}_{s['volume']}"  # widgets follow what is saved (e.g. after a chat change)
        c1, c2, c3, c4 = st.columns([3, 2, 3, 1], vertical_alignment="center")
        with c1:
            st.markdown(f"**{s.get('label') or s['lib_id']}**")
            if entry is not None and sfx_library.available(entry):
                st.audio(str(sfx_library.path_of(entry)))
            else:
                st.caption("Missing from the library: it will be skipped.")
        with c2:
            at = st.number_input("At (seconds)", 0.0, duration, float(s["start"]), 0.1, key=f"snd_at_{ver}")
        with c3:
            vol = st.slider("Volume", 0.0, workflow.MAX_SOUND_VOLUME, float(s["volume"]), 0.05, key=f"snd_vol_{ver}")
        with c4:
            gone = st.button("Delete", key=f"snd_del_{ver}")
        if gone:
            workflow.remove_sound(job, s["id"])
            rerun_fragment()
        if abs(at - s["start"]) > 1e-6 or abs(vol - s["volume"]) > 1e-6:
            try:
                workflow.update_sound(job, s["id"], start=at, volume=vol)
            except workflow.WorkflowError as exc:
                fail(exc)
            rerun_fragment()
    if library:
        st.markdown("**Add a sound**")
        a1, a2, a3 = st.columns([3, 2, 1], vertical_alignment="bottom")
        with a1:
            pick = st.selectbox("Sound", [e["id"] for e in library], key="sfx_pick",
                                format_func=lambda i: next(e["name"] for e in library if e["id"] == i))
        with a2:
            when = st.number_input("At (seconds)", 0.0, duration, 0.0, 0.1, key="sfx_when")
        with a3:
            if st.button("Add", key="sfx_add", type="primary"):
                try:
                    workflow.add_sound(job, pick, when)
                except workflow.WorkflowError as exc:
                    fail(exc)
                rerun_fragment()
    if sounds and st.button("Remove all sounds", key="sfx_clear"):
        workflow.remove_sound(job)
        rerun_fragment()


@st.cache_data(show_spinner=False, max_entries=96)
def cached_caption_still(style_id: str, chunk_json: str, emphasis_json: str, active: int) -> bytes:
    return preview.caption_frame_png(json.loads(chunk_json), style_id, json.loads(emphasis_json), active=active,
                                     width=300)


@st.fragment
def captions_panel(job: Path) -> None:
    """Pick the caption style and make words stand out. A fragment: every click here reruns only this panel."""
    plan = workflow.load_plan(job)
    cap = caption_styles.caption_settings(plan)
    style = st.radio("Caption style", caption_styles.style_ids(), index=caption_styles.style_ids().index(cap["style"]),
                     format_func=lambda s: caption_styles.STYLES[s]["label"], horizontal=True, key="cap_style")
    if style != cap["style"]:
        workflow.set_caption_style(job, style)
        cap = caption_styles.caption_settings(workflow.load_plan(job))
    st.caption(caption_styles.STYLES[style]["about"])

    try:
        words = workflow.read_transcript_words(job)
    except (OSError, ValueError):
        st.info("There is no transcript for this job, so there is nothing to highlight.")
        return
    chunks = caption_styles.fit_chunks(chunk_words(words), caption_styles.STYLES[style]["max_words"])
    if not chunks:
        return
    emphasis = cap["emphasis"]
    st.markdown("**Make words stand out** (another colour and/or bigger)")
    idx = st.selectbox("Caption line", range(len(chunks)), key=f"cap_line_{style}",
                       format_func=lambda i: f"{chunks[i]['start']:.1f}s  ·  " + " ".join(w["text"] for w in chunks[i]["words"]))
    chunk = chunks[idx]
    marked = [j for j, w in enumerate(chunk["words"]) if caption_styles.emphasis_for(w, emphasis)]
    ver = hashlib.md5(json.dumps([style, idx, emphasis]).encode()).hexdigest()[:6]  # widgets follow what is saved
    picked = st.pills("Words", list(range(len(chunk["words"]))), selection_mode="multi", default=marked,
                      format_func=lambda j: chunk["words"][j]["text"], key=f"cap_words_{ver}") or []
    c1, c2 = st.columns(2)
    with c1:
        use_color = st.checkbox("Change colour", value=True, key=f"cap_usecolor_{ver}")
        color = st.color_picker("Colour", "#FF3B30", key=f"cap_color_{ver}", disabled=not use_color)
    with c2:
        scale = st.select_slider("Size", [1.0, 1.2, 1.4, 1.6, 1.8], value=1.4, key=f"cap_scale_{ver}",
                                 format_func=lambda s: "normal" if s == 1.0 else f"{s:g}x")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Apply to this line", key=f"cap_apply_{ver}", type="primary"):
            try:
                for j, w in enumerate(chunk["words"]):
                    hit = caption_styles.emphasis_for(w, emphasis)
                    if j in picked:
                        workflow.add_emphasis(job, w["start"], w["end"], color if use_color else None, scale,
                                              text=w["text"])
                    elif hit is not None:
                        workflow.remove_emphasis(job, hit["id"])
            except workflow.WorkflowError as exc:
                fail(exc)
            rerun_fragment()
    with b2:
        if st.button("Clear this line", key=f"cap_clear_{ver}", disabled=not marked):
            for j in marked:
                hit = caption_styles.emphasis_for(chunk["words"][j], emphasis)
                if hit is not None:
                    workflow.remove_emphasis(job, hit["id"])
            rerun_fragment()
    with b3:
        if st.button("Clear all highlights", key=f"cap_clearall_{ver}", disabled=not emphasis):
            workflow.remove_emphasis(job)
            rerun_fragment()
    try:
        st.image(cached_caption_still(style, json.dumps(chunk), json.dumps(emphasis), picked[0] if picked else 0),
                 caption="Preview (the word being spoken is marked by the style)")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Preview unavailable: {exc}")
    if emphasis:
        st.caption(f"{len(emphasis)} highlighted: " + ", ".join(e.get("text") or f"{e['start']:.1f}s" for e in emphasis[:12])
                   + (" ..." if len(emphasis) > 12 else ""))


def flash(*messages: str) -> None:
    """Remember messages (e.g. swap warnings) across the st.rerun() that follows an action."""
    st.session_state.setdefault("flash", []).extend(messages)


def timeline_strip(items: list[dict], duration: float) -> None:
    """All slots as coloured blocks along the video: position and length are to scale."""
    if duration <= 0 or not items:
        return
    blocks = []
    for it in items:
        length = it["timestamp_end"] - it["timestamp_start"]
        left, width = 100 * it["timestamp_start"] / duration, 100 * length / duration
        removed = it.get("enabled") is False
        color = "#9ca3af" if removed else LAYOUT_COLORS.get(it.get("layout", "overlay"), "#3b82f6")
        title = f"#{it['id']} {it['timestamp_start']:.1f}-{it['timestamp_end']:.1f}s, {it.get('layout', 'overlay')}"
        blocks.append(
            f'<div title="{title}" style="position:absolute;left:{left:.2f}%;width:{max(width, 0.8):.2f}%;top:4px;'
            f'height:34px;background:{color};opacity:{0.45 if removed else 0.95};border-radius:5px;color:#fff;'
            f'font:600 12px sans-serif;display:flex;align-items:center;justify-content:center;'
            f'overflow:hidden">{it["id"]}</div>')
    step = 10 if duration > 40 else 5
    ticks = "".join(
        f'<span style="position:absolute;left:{100 * t / duration:.2f}%;top:40px;font:11px sans-serif;opacity:.65;'
        f'transform:translateX(-50%)">{t}s</span>' for t in range(0, int(duration) + 1, step))
    legend = " ".join(f'<span style="color:{c}">&#9632;</span> {n}' for n, c in LAYOUT_COLORS.items())
    st.html(f'<div style="position:relative;height:60px;border-radius:6px;background:rgba(128,128,128,.15)">'
            f'{"".join(blocks)}{ticks}</div><div style="font:12px sans-serif;opacity:.8;margin-top:2px">{legend}'
            f' &nbsp; (the speaker is on screen in the gaps)</div>')


def _set_state(**values) -> None:
    st.session_state.update(values)


@st.fragment
def slot_editor(job: Path, item_id: int) -> None:
    """Layout, timing, swap and zoom/pan keyframes for one slot. It is a fragment: moving a slider reruns only
    this editor, not the whole page. Actions that change other slots call st.rerun() (the full page)."""
    plan = workflow.load_plan(job)
    items = sorted(plan["items"], key=lambda i: i["timestamp_start"])  # timeline order (the chat can add slots in between)
    item = next(i for i in items if i["id"] == item_id)
    pos = items.index(item)
    asset = item.get("asset") or {}
    clip = job / asset["path"] if asset.get("path") else None
    start, end = item["timestamp_start"], item["timestamp_end"]
    saved = framing_mod.normalize(item.get("framing"))
    layout = item.get("layout", "overlay")
    saved_pip = normalize_pip(item.get("pip"))
    # Widget keys carry a fingerprint of what is saved, so after a swap / reset / save the widgets show the new
    # saved values instead of stale ones.
    ver = hashlib.md5(json.dumps([saved, layout, saved_pip, start, end, asset.get("path")]).encode()).hexdigest()[:6]

    def k(name: str) -> str:
        return f"{name}{item_id}_{ver}"

    # ---- layout + order
    c1, c2, c3 = st.columns([4, 2, 2])
    with c1:
        new_layout = st.radio("Layout", list(LAYOUTS), index=LAYOUTS.index(layout), format_func=LAYOUT_LABELS.get,
                              horizontal=True, key=k("lay"))
        if new_layout != layout:
            workflow.set_layout(job, item_id, new_layout)
            st.rerun()
        if layout == "pip":
            pc1, pc2 = st.columns(2)
            with pc1:
                new_pos = st.selectbox("Position", list(PIP_POSITION_LABELS), index=list(PIP_POSITION_LABELS).index(
                    saved_pip["pos"]), format_func=PIP_POSITION_LABELS.get, key=k("pippos"))
            with pc2:
                new_scale = st.slider("Size", PIP_MIN_SCALE, PIP_MAX_SCALE, float(saved_pip["scale"]), 0.05,
                                      key=k("pipscale"))
            if (new_pos, round(new_scale, 2)) != (saved_pip["pos"], saved_pip["scale"]):
                workflow.set_pip(job, item_id, pos=new_pos, scale=new_scale)
                st.rerun()
    with c2:
        if st.button("Swap with previous", key=k("up"), disabled=pos == 0,
                     help="Trade clips with the slot before this one (timings stay where they are)."):
            flash(*workflow.swap_slots(job, item_id, items[pos - 1]["id"]))
            st.rerun()
    with c3:
        if st.button("Swap with next", key=k("down"), disabled=pos == len(items) - 1,
                     help="Trade clips with the slot after this one (timings stay where they are)."):
            flash(*workflow.swap_slots(job, item_id, items[pos + 1]["id"]))
            st.rerun()

    # ---- timing
    lim = workflow.timing_limits(job, item_id)
    lo, hi = min(lim["min_start"], start), max(lim["max_end"], end)
    t_start, t_end = st.slider("When it plays (seconds): drag the ends", lo, hi, (start, end), step=0.1,
                               key=k("time"))
    clip_len = lim["max_len"]
    too_long = clip_len is not None and t_end - t_start > clip_len + 0.05
    if too_long:
        st.warning(f"The downloaded clip is only {clip_len:.1f}s long. Shorten the slot, or fetch a longer clip "
                   "(this may pick a different video).")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Save timing", key=k("savet"), disabled=too_long or (t_start, t_end) == (start, end)):
            try:
                workflow.set_timing(job, item_id, t_start, t_end)
                st.rerun()
            except workflow.WorkflowError as exc:
                fail(exc)
    with b2:
        if st.button("Fetch longer clip", key=k("fetch"), disabled=not too_long):
            with st.spinner("Downloading a longer clip..."):
                try:
                    workflow.set_timing(job, item_id, t_start, t_end, refetch=True)
                except Exception as exc:  # noqa: BLE001
                    fail(exc)
            st.rerun()
    for col, label, delta in ((b3, "Nudge -0.1s", -0.1), (b4, "Nudge +0.1s", 0.1)):
        with col:
            if st.button(label, key=k(f"nudge{delta}"), help="Move the whole slot a little earlier / later."):
                try:
                    workflow.set_timing(job, item_id, start + delta, end + delta)
                    st.rerun()
                except workflow.WorkflowError as exc:
                    fail(exc)

    # ---- framing (zoom / pan keyframes)
    if not (clip and clip.is_file()):
        st.caption("No clip to frame yet.")
        return
    st.markdown("**Framing: zoom and pan** (from the start keyframe to the end keyframe over the slot)")
    cur = {}
    for col, name in zip(st.columns(2), ("start", "end")):
        with col:
            st.caption(f"{name.capitalize()} keyframe")
            cur[name] = {
                "z": st.slider("Zoom", 1.0, framing_mod.MAX_ZOOM, float(saved[name]["z"]), 0.05, key=k(f"z{name}")),
                "x": st.slider("Pan left / right", 0.0, 1.0, float(saved[name]["x"]), 0.01, key=k(f"x{name}")),
                "y": st.slider("Pan up / down", 0.0, 1.0, float(saved[name]["y"]), 0.01, key=k(f"y{name}")),
            }

    def preset(z_start: float, z_end: float) -> dict:
        vals = {}
        for name, z in (("start", z_start), ("end", z_end)):
            vals.update({k(f"z{name}"): z, k(f"x{name}"): 0.5, k(f"y{name}"): 0.5})
        return vals

    p1, p2, p3, p4, p5 = st.columns(5)
    for col, label, vals in ((p1, "Zoom in", preset(1.0, 1.6)), (p2, "Zoom out", preset(1.6, 1.0)),
                             (p3, "Hold", preset(1.0, 1.0)), (p4, "Hold zoomed", preset(1.4, 1.4))):
        with col:
            st.button(label, key=k("p" + label), on_click=_set_state, kwargs=vals, use_container_width=True)
    unsaved = framing_mod.normalize(cur) != saved
    with p5:
        if st.button("Save framing", key=k("savef"), type="primary", disabled=not unsaved):
            workflow.set_framing(job, item_id, cur)
            st.rerun()
    if unsaved:
        st.caption("Unsaved framing changes: press Save framing to keep them (the preview below already shows them).")

    # ---- preview
    view = st.radio("Preview as", ["9:16 (as exported)", "16:9 (whole clip, crop outlined)"], horizontal=True,
                    key=f"view{item_id}")
    framing_json, mtime = json.dumps(framing_mod.normalize(cur)), clip.stat().st_mtime
    pip_json = json.dumps(saved_pip) if new_layout == "pip" else "null"
    half = {"split": " (top half)", "pip": " (the small box)"}.get(new_layout, "")
    try:
        if view.startswith("9:16"):
            a, b = st.columns(2)
            with a:
                st.image(cached_frame(str(clip), mtime, framing_json, new_layout, end - start, "start", clip_len,
                                      pip_json), caption="First frame of the slot" + half)
            with b:
                st.image(cached_frame(str(clip), mtime, framing_json, new_layout, end - start, "end", clip_len,
                                      pip_json), caption="Last frame of the slot" + half)
        else:
            st.image(cached_source_view(str(clip), mtime, framing_json, new_layout, pip_json),
                     caption="Yellow solid box = what shows at the start; blue dashed = at the end. "
                             "The dimmed part is cut off.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Preview unavailable: {exc}")
    if st.button("Play preview with the speaker", key=k("play"),
                 help="A quick real render of just this slot, exactly as the final video will look."):
        with st.spinner("Rendering preview..."):
            try:
                face_y = float(workflow.read_job_meta(job).get("split_face_y", DEFAULT_FACE_Y))
                out = preview.render_clip_preview(
                    clip, workflow.source_video_path(job), start, end, new_layout, framing_mod.normalize(cur),
                    face_y, job / "previews" / f"slot_{item_id}.mp4", pip=saved_pip if new_layout == "pip" else None)
                st.session_state[f"playing{item_id}"] = str(out)
            except Exception as exc:  # noqa: BLE001
                fail(exc)
    playing = st.session_state.get(f"playing{item_id}")
    if playing and Path(playing).is_file():
        st.video(playing)


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
    usage = cached_usage()
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
def draft_job() -> Path | None:
    raw = st.session_state.get("draft_job")
    return Path(raw) if raw and Path(raw).is_dir() else None


def mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def cleanup_summary(job: Path) -> None:
    """What the audio cleanup did: the length, how the noise was removed, and anything that was skipped."""
    report = workflow.read_cleanup_report(job)
    if not report:
        return
    st.success(f"Audio cleaned ({mmss(report['clean_duration'])}): background noise removed with {report['denoise']}, "
               "voice evened out. Nothing was cut.")
    for w in report.get("warnings", []):
        st.warning(w)
    st.caption("Taking the cleanup off (the checkbox above) goes back to the original video. Steps took: "
               + ", ".join(f"{k} {v:g} s" for k, v in report.get("seconds", {}).items()))


def run_analysis(draft: Path) -> None:
    """Transcribe the video and ask Gemini how many B-rolls suit it; preset the count slider with the answer.
    With the cleanup ticked it first removes background noise and evens out the voice (a bit longer)."""
    cleaning = bool(workflow.read_job_meta(draft).get("clean_talking_head"))
    with st.status("Cleaning the recording, then analysing it..." if cleaning else
                   "Analysing your video (transcribing, then asking the AI how many B-rolls suit it)...",
                   expanded=True) as status:
        try:
            suggestion = workflow.analyse(draft, log=lambda m: status.write(m))
            st.session_state["count"] = suggestion["count"]
            st.session_state.pop("analysis_failed", None)
            status.update(label=f"Analysed: the AI suggests {suggestion['count']} B-rolls", state="complete")
        except Exception as exc:  # noqa: BLE001
            st.session_state["analysis_failed"] = True
            status.update(label="Analysis failed", state="error")
            fail(exc)


if job is None:
    st.subheader("1. Choose your video")
    clean = st.checkbox(
        "Clean the audio (remove background noise, even out the voice)", key="clean_th",
        help="Removes background noise and evens out the voice, then works on the cleaned video. Nothing is cut: "
             "edit out pauses and retakes before uploading. The picture is copied as it is. "
             "The original is kept; untick this to go back to it.")
    up = st.file_uploader("Upload a raw video (about 1-2 minutes)", type=["mp4", "mov", "mkv", "m4v"])
    local = st.text_input("...or the path of a video already on this PC", placeholder=r"D:\vids\my_video.mp4")
    local_path = local.strip().strip('"')
    sig = ("upload", up.name, up.size) if up else (("path", local_path) if local_path else None)

    draft = draft_job()
    if draft and st.session_state.get("draft_sig") != sig:  # another video (or none): the old draft is obsolete
        workflow.discard_draft(draft)
        for key in ("draft_job", "draft_sig", "analysis_failed", "count"):
            st.session_state.pop(key, None)
        draft = None
    if sig and draft is None and st.session_state.get("failed_sig") != sig:
        try:
            if up:
                draft = workflow.create_job_from_upload(up.name, up.getbuffer(), spoken, clean=clean)
            else:
                draft = workflow.create_job(local_path, spoken, clean=clean)
            st.session_state.update(draft_job=str(draft), draft_sig=sig)
            st.session_state.pop("failed_sig", None)
            run_analysis(draft)
        except Exception as exc:  # noqa: BLE001 - show any setup problem to the user
            st.session_state["failed_sig"] = sig  # do not retry on every rerun
            fail(exc)
    elif draft and workflow.read_job_meta(draft).get("language") != spoken:  # language changed: re-transcribe
        workflow.set_language(draft, spoken)
        run_analysis(draft)
    elif draft and bool(workflow.read_job_meta(draft).get("clean_talking_head")) != clean:  # cleanup toggled
        workflow.set_cleanup(draft, clean)
        run_analysis(draft)
    elif draft and st.session_state.get("analysis_failed"):
        if st.button("Retry analysis"):
            run_analysis(draft)

    draft_meta = workflow.read_job_meta(draft) if draft else {}
    if draft and draft_meta.get("cleaned"):
        cleanup_summary(draft)
    st.subheader("2. Pick a video style")
    style = st.radio("Video style", list(STYLES), format_func=STYLE_LABELS.get, key="style",
                     label_visibility="collapsed")

    st.subheader("3. Tell the AI what you want")
    slider_kw = {} if "count" in st.session_state else {"value": config.total_brolls()}  # else the AI's suggestion
    count = st.slider("How many B-rolls?", min_value=1, max_value=config.MAX_BROLLS, key="count",
                      help="1 to 15. Short videos look better with fewer (about one per 8-10 seconds). "
                           "The AI suggests a number after you add a video; you can change it.", **slider_kw)
    if draft_meta.get("suggested_brolls"):
        tag = "AI suggests" if draft_meta.get("suggested_source") == "gemini" else "Suggested"
        st.caption(f"{tag} **{draft_meta['suggested_brolls']}** for this {draft_meta.get('video_duration', 0):.0f}s "
                   f"video: {draft_meta.get('suggested_reason', '')}")
    else:
        st.caption("Add a video above and the AI will suggest how many B-rolls suit it.")
    if draft_meta.get("video_duration") and count > feasible_max(draft_meta["video_duration"]):
        st.warning(f"Only about {feasible_max(draft_meta['video_duration'])} B-rolls physically fit in this video "
                   "(3 s minimum each). Asking for more may make planning fail.")
    notes = st.text_area(
        "Notes for the AI (optional)", height=110,
        placeholder="e.g. The player is Truls Moregard (Swedish table tennis). Show fast rallies and crowd "
                    "reactions. Avoid podcasts and interviews.",
        help="Anything that helps: correct spellings of names the speech recognition may mishear, the topic, "
             "what you want to see, what to avoid. The AI treats this as correct and also uses it to spell "
             "names right in the captions.")
    if st.button("Build B-roll plan", type="primary", disabled=draft is None):
        try:
            workflow.start_planning(draft, count, notes, style)
            open_job(draft)
            for key in ("draft_job", "draft_sig", "analysis_failed", "count"):
                st.session_state.pop(key, None)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            fail(exc)
    st.stop()

# ---------------------------------------------------------------- step 2: build plan (resumable)
meta = workflow.read_job_meta(job)
if workflow.needs_pipeline(job):
    st.subheader("Building your plan")
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
items = sorted(plan["items"], key=lambda i: i["timestamp_start"])  # timeline order
ready = sum(1 for i in items if (i.get("asset") or {}).get("status") == "ok" and i.get("enabled") is not False)
final = job / "final.mp4"

st.subheader("3. Review the plan")
st.write(f"**{plan.get('video_topic', '')}**")
st.caption(f"{ready} of {len(items)} B-roll clips ready. Edit a search query, press Regenerate for a different "
           "video, or remove a slot (the speaker stays on screen there).")

if not any((job / (i.get("asset") or {}).get("path", "x")).is_file() for i in items) and final.is_file():
    st.info("This job has been rendered and its temporary clips were cleaned up.")

with st.expander("Change the number of B-rolls or add notes, then re-plan"):
    st.caption("This replaces the current plan and its downloaded clips (the transcript is kept).")
    re_count = st.slider("How many B-rolls?", 1, config.MAX_BROLLS,
                         value=config.clamp_brolls(meta.get("total_brolls") or len(items)), key="re_count")
    fits = feasible_max(float(plan.get("video_duration") or 0))
    if re_count > fits:
        st.warning(f"Only about {fits} B-rolls physically fit in this video (3 s minimum each). "
                   "Asking for more may make planning fail.")
    re_notes = st.text_area("Notes for the AI (optional)", value=meta.get("notes", ""), key="re_notes", height=90,
                            placeholder="Names, spellings, topic, what to show or avoid...")
    if st.button("Re-plan with these settings"):
        workflow.replan(job, re_count, re_notes)
        st.rerun()

for message in st.session_state.pop("flash", []):
    st.warning(message)

if workflow.read_cleanup_report(job):
    with st.expander("Audio cleanup report"):
        cleanup_summary(job)

with st.expander("Video style (overlay, split or hybrid)"):
    cur_style = meta.get("style", "overlay")
    new_style = st.radio("Style", list(STYLES), index=STYLES.index(cur_style) if cur_style in STYLES else 0,
                         format_func=STYLE_LABELS.get, key=f"restyle_{cur_style}")
    if st.button("Apply this style to all slots", disabled=new_style == cur_style,
                 help="Resets every slot's layout. Hybrid restores the AI's own mix. You can still change any "
                      "single slot below."):
        workflow.apply_style(job, new_style)
        st.rerun()
    saved_face = float(meta.get("split_face_y", DEFAULT_FACE_Y))
    face = st.slider("Facecam focus in split slots (0 = top of the speaker video, 1 = bottom)", 0.0, 1.0,
                     saved_face, 0.05, key="face_y",
                     help="In a split slot the speaker fills the bottom half, cropped from the full-height video. "
                          "Move this so the face stays in view.")
    if abs(face - saved_face) > 1e-9:
        workflow.set_split_face_y(job, face)

timeline_strip(sorted(items, key=lambda i: i["timestamp_start"]), float(plan.get("video_duration") or 0))

for item in items:
    asset = item.get("asset") or {}
    removed = item.get("enabled") is False
    clip = job / asset.get("path", "") if asset.get("path") else None
    with st.container(border=True):
        left, mid, right = st.columns([2, 4, 2], vertical_alignment="top")
        with left:
            if clip and clip.is_file() and not removed:
                slot_player(str(clip), clip.stat().st_mtime, item["id"])
            elif removed:
                st.caption("Removed: the speaker stays on screen.")
            elif asset.get("status") == "failed":
                st.warning("No clip found for this query. Edit it and search again.")
            else:
                st.caption("No preview available.")
        with mid:
            st.markdown(f"**#{item['id']}**  ·  {item['timestamp_start']:.1f}s to {item['timestamp_end']:.1f}s"
                        f"  ·  {LAYOUT_LABELS.get(item.get('layout', 'overlay'), '').split(' (')[0]}")
            st.caption(f"Says: {item.get('english_meaning', '')}")
            # The key carries the saved query, so after a swap / re-search the box shows the new saved text.
            qkey = hashlib.md5(item["search_query"].encode()).hexdigest()[:6]
            new_q = st.text_input("Search query", value=item["search_query"], key=f"q{item['id']}_{qkey}",
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
        if not removed and clip and clip.is_file():
            if st.toggle("Edit layout, timing, order and framing (zoom / pan)", key=f"edit{item['id']}"):
                slot_editor(job, item["id"])

# ---------------------------------------------------------------- captions: style + highlighted words
with st.expander("Captions: style and highlighted words"):
    captions_panel(job)

# ---------------------------------------------------------------- sound effects
with st.expander(f"Sound effects ({len(plan.get('sounds', []))})"):
    sfx_panel(job)

# ---------------------------------------------------------------- step 4: render
st.subheader("4. Export")
if workflow.source_video_path(job) is None and meta.get("cleaned"):
    st.warning("This job used the talking-head cleanup, and the cleaned video was deleted after the render to save "
               "space. The plan only fits the cleaned video, so it cannot be rendered again from the original. "
               "Start a new video (untick 'Delete temporary clips after render' to be able to render again).")
elif workflow.source_video_path(job) is None:
    path = st.text_input("The original video file for this job was not found. Enter its path:",
                         placeholder=r"D:\vids\my_video.mp4")
    if path.strip() and Path(path.strip().strip('"')).is_file():
        workflow.write_job_meta(job, source_video=path.strip().strip('"'))
        st.rerun()
else:
    if st.button("Approve and render", type="primary"):
        with st.status("Rendering your reel...", expanded=True) as status:
            bar = status.progress(0.0, text="Preparing...")
            try:
                _, freed = workflow.render_final(
                    job, caption_mode if captions_on else None, auto_clean=auto_clean,
                    log=lambda m: status.write(m),
                    progress=lambda f: bar.progress(f, text=f"Encoding video: {f:.0%}"),
                )
                bar.empty()
                status.update(label="Done" + (f" - freed {fmt_bytes(freed)}" if freed else ""), state="complete")
            except Exception as exc:  # noqa: BLE001
                status.update(label="Render failed", state="error")
                fail(exc)

if final.is_file():
    st.success(f"Final video: {final}")
    st.video(str(final))


# ---------------------------------------------------------------- AI assistant (floating button -> right-hand panel)
from src.assistant.shell import shell as chat_shell  # noqa: E402 - a light import; the chat code loads on open

chat_shell(str(job))  # a fragment: opening / closing the chat never reruns the whole page
