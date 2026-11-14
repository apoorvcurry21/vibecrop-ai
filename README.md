# VibeCrop AI

An automated AI video editor for short-form content (Instagram Reels / YouTube Shorts, 1-2 minutes).
Upload a raw talking-head video. VibeCrop AI transcribes it, plans contextual B-rolls
(1 to 15, your choice), lets you review and tweak the plan, then renders a vertical 9:16
video with burned-in captions.

Built on a zero-budget stack: free cloud APIs, `yt-dlp`, FFmpeg and Streamlit.

## Pipeline

1. **Transcription** - Groq API (Whisper) gives word- and sentence-level timestamps.
2. **Planning** - Gemini (free tier) reads the transcript and outputs a strict JSON timeline of 1-15 B-roll slots (default 10).
3. **Sourcing** - `yt-dlp --download-sections` pulls 3-5 s clips; DuckDuckGo / Unsplash supply images.
4. **Rendering** - FFmpeg center-crops to 9:16, mutes B-roll audio, overlays on the main track, burns captions.
5. **UI** - Streamlit dashboard to edit queries / regenerate assets before export; temp files are purged after render.

## Roadmap

- [x] Phase 1: Audio extraction and Groq transcription
- [x] Phase 2: Gemini editor (context and planning)
- [x] Phase 3: Media sourcing and smart 9:16 formatting (video only; images are deferred)
- [x] Phase 4: Streamlit mini-editor and caption export
- [x] Phase 5: AI count suggestion, video styles (overlay / split / hybrid), timeline editing, zoom/pan framing

## Setup

Requirements: Python 3.10+, FFmpeg on your PATH (`ffmpeg -version` must work).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then add your Groq and Gemini keys
```

## Phase 1: transcribe a video

```
python -m src.transcription.pipeline path\to\video.mp4 --language en
```

Extracts 16 kHz mono FLAC audio, sends it to Groq Whisper (`whisper-large-v3-turbo` by default,
override with `WHISPER_MODEL`), and saves `transcript.json` (segments + per-word timestamps) in a
fresh job folder under `VIBECROP_WORKDIR`. The language is chosen by the user (ISO code, e.g. `en`, `hi`).

On Colab, add `GROQ_API_KEY` / `GEMINI_API_KEY` under Secrets (Notebook access on); they are picked up automatically.

Tests (no API key needed, Groq is mocked): `pip install -r requirements-dev.txt && pytest`

## Phase 2: plan the B-rolls

```
python -m src.planning.pipeline D:\vibecrop_tmp\jobs\<job folder>
```

Gemini (`GEMINI_MODEL`, default `gemini-3.5-flash`, with `GEMINI_FALLBACK_MODELS` if it is overloaded)
reads `transcript.json` and writes `plan.json`: 1-15 B-rolls (`TOTAL_BROLLS`, default 10). Gemini decides the video/image mix
from the content. Each item has `asset_type`, `timestamp_start`, `timestamp_end`, a `search_query` and two
`alternative_queries`. The prompt forbids literal readings of metaphors ("hit a wall" -> "stressed anxious person").
Python snaps times to spoken-word boundaries and rejects plans with overlaps, wrong durations (video 3-5 s,
image 2-4 s), bad queries or gaps over 25 s, then asks Gemini to fix them (up to 3 attempts).
The plan is a `draft` until the user approves it in the Phase 4 UI.

## Phase 3: source clips and render

```
python -m src.sourcing.pipeline  D:\vibecrop_tmp\jobs\<job>
python -m src.rendering.pipeline D:\vibecrop_tmp\jobs\<job> D:\vids\video.mp4
```

Sourcing searches YouTube with yt-dlp, skips very short/long/live results, and downloads only a 3-6 s
section (about 25% into the video) at up to 1080p, video-only, so B-roll is silent by construction.
If a download fails it tries the next result, then the alternative queries. Clip metadata (channel, URL,
window) is written back into `plan.json`.

Rendering (`final.mp4`, 1080x1920, 30 fps) center-crops each clip to 9:16 without stretching, lays it over the
speaker only during its slot, never uses the clip's audio, and burns a small "Video: <channel>" credit line.
A slot whose clip failed simply keeps the speaker on screen.

Known limits: the clip window is chosen blindly, so some clips are off-topic (use the Phase 4 "Regenerate");
some YouTube videos carry their own burned-in captions or watermarks; footage is third-party and copyrighted
(credited, but you are responsible for how you publish it); YouTube may block downloads from cloud IPs.
Images are off by default (`ENABLE_IMAGES=1` lets Gemini plan them; image sourcing is not built yet).

## Phase 4: the Streamlit editor

```
streamlit run app.py        # then open http://localhost:8501
```

1. Upload a video (or type a path to one on disk) and pick the spoken language. The app **analyses it
   straight away**: it transcribes the speech and Gemini suggests how many B-rolls suit this reel (shown as
   "AI suggests N: reason", and the slider is preset to it). Then choose a **video style** (see below), adjust
   **how many B-rolls** (1 to 15; you can change the AI's number), and optionally add **notes for the AI**
   (correct names and spellings, topic, what to show or avoid; the notes also fix name spellings in the
   captions). Press **Build B-roll plan** (plan -> download clips; the transcript from the analysis is reused;
   resumable if something fails). The review screen has a "re-plan" box to change the count or notes later.
   If Gemini is unavailable the suggestion falls back to about one B-roll per 8 seconds.
2. **Review**: every B-roll has a playable preview, its timing, what the speaker says, and an editable
   search query. **Search with this query** fetches a new clip, **Regenerate asset** picks a different video
   for the same query, **Remove slot** keeps the speaker on screen there.
3. **Edit a slot** (switch on "Edit layout, timing, order and framing" on its row):
   - **Layout** per slot: *overlay* (the clip fills the screen), *split* (see styles) or *picture-in-picture*
     (a small box over you; pick its position and size).
   - **Timing**: drag the start/end, or nudge by 0.1 s. A slot cannot overlap its neighbours, start before 1 s
     or run longer than its downloaded clip (the app downloads a few seconds of headroom; **Fetch longer
     clip** re-downloads, which may pick a different video).
   - **Order**: *Swap with previous / next* trades the clips between slots (timings and layouts stay).
   - **Framing (zoom and pan)**: a start and an end keyframe, each with zoom (1x to 3x) and pan left/right/up/down;
     the clip animates smoothly between them (Ken Burns). Presets: Zoom in, Zoom out, Hold.
     **Preview as 9:16** shows the first and last frame exactly as exported; **16:9** shows the whole source
     clip with the crop outlined (solid = start, dashed = end, dimmed = cut off). **Play preview with the
     speaker** renders a quick real clip of that slot with the same code as the final export.
4. **Approve and render**: burns captions (2-4 word chunks, active word in yellow) in the original language,
   English, or Roman Hindi (Gemini rewrites the text; timings are kept), then deletes the temporary clips and
   audio, keeping only `final.mp4` and the JSON files. The sidebar shows temp storage against a 32 GB budget.
   Cleanup only ever touches job folders inside `VIBECROP_WORKDIR/jobs`.

### AI assistant (chat)

Press the round **💬 button** (bottom right) on the review screen: a chat panel opens on the right and the page
makes room for it. You can do everything above by talking to it, in English or Hinglish, and a few things the
buttons cannot do:

- *"swap B-roll 2 and 4"*, *"make slot 3 split"*, *"zoom in slowly on slot 1"*, *"move slot 2 to 14-17 s"*
- **Attach a picture** (paperclip): *"put this as overlay when I say Virat Kohli"*. It finds where the phrase is
  spoken (spelling-tolerant), and adds the picture there. An attached video works too: it cuts the part you choose.
- **Paste a YouTube link**: *"cut 5 s of the football part and put it where it fits"*. Say a time (`2:15`) to use
  it, otherwise Gemini watches the video and picks the moment. Only the needed seconds are downloaded.
- *"get me a picture of ..."*: web image search with thumbnails and a **Use this** button. Web pictures get a small
  "Image: site.com" credit line in the video.
- *"generate a picture of ..."*: Gemini image generation (see the note below).
- **Text on the video**: *"big yellow GOAT at the top from 7 s to 9 s"* (position, size, colour, font).
- When you add a picture or clip without saying how it should look, it shows **illustrated examples** of the three
  layouts to tap: *full screen*, *split* (picture on top, you below) or *small box* (picture-in-picture, with a
  position and size).

**No timing rules in the chat:** you can put things at 0 s, make them as short as you like, or lay them over other slots (the later-starting one is drawn on top, and the assistant tells you when one covers another). Only the video's length and a clip's own length limit it. The sliders in the editor keep their guard rails.

Edits apply instantly and **Undo** (top of the panel, or just say "undo") reverts the last ten changes, including
replaced clips. The chat history is saved per job in `chat.json` (a few KB); attached/generated pictures and undo
snapshots are temporary and deleted with the other temp files after a render.

The chat uses `GEMINI_CHAT_MODEL` (default `gemini-3.1-flash-lite`, about 3 s per step; `gemini-3.5-flash` is
smarter but takes 10-20 s) and gives up on a stalled model after 30 s, then tries the next one.

> **Image generation needs a billing-enabled Gemini key.** On the free tier every Gemini image model answers
> "quota exceeded"; the assistant then says so and offers a web image search instead. Set `GEMINI_IMAGE_MODEL` to
> pick a model. Real people and brands are often refused by the model, so web search is the better route for those.

### Video styles

| Style | Look |
|---|---|
| **Overlay** | The b-roll covers the whole 9:16 frame while your voice keeps running (the original look). |
| **Split** | The b-roll fills the top half, your facecam the bottom half, with a thin white border between. Captions sit on the border and the credit just above it. "Facecam focus" chooses which part of the speaker video is shown. |
| **Hybrid** | A mix. Gemini picks a layout per b-roll (short punchy visuals full screen, longer explanatory ones split; never more than 3 in a row) and you can change any slot. |

"Apply this style to all slots" (review screen) resets every slot to a style. Output is always 1080x1920.

Windows uses Nirmala UI for Devanagari captions; on Linux/Colab install `fonts-noto-core`.
CLI equivalent of the render step: `python -m src.rendering.pipeline <job> <video> --captions hinglish`.

## Project layout

```
src/
  transcription/   # Phase 1
  planning/        # Phase 2
  sourcing/        # Phase 3
  rendering/       # Phase 3-5 (compose.py = layouts, framing.py = zoom/pan maths, preview.py = editor previews)
  assistant/       # chat: agent.py (Gemini tool-calling loop), tools.py + media_tools.py (what it can do),
                   #       snapshots.py (undo), ytclip.py / media.py / fetch.py (clips and pictures), ui.py (panel)
scripts/make_layout_mockups.py   # redraws the layout example pictures shown in the chat
```
