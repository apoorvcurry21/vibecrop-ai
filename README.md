# VibeCrop AI

An automated AI video editor for short-form content (Instagram Reels / YouTube Shorts, 1-2 minutes).
Upload a raw talking-head video. VibeCrop AI transcribes it, plans 10 contextual B-rolls
(6 video clips + 4 images), lets you review and tweak the plan, then renders a vertical 9:16
video with burned-in captions.

Built on a zero-budget stack: free cloud APIs, `yt-dlp`, FFmpeg and Streamlit.

## Pipeline

1. **Transcription** - Groq API (Whisper) gives word- and sentence-level timestamps.
2. **Planning** - Gemini (free tier) reads the transcript and outputs a strict JSON timeline of 10 B-roll slots.
3. **Sourcing** - `yt-dlp --download-sections` pulls 3-5 s clips; DuckDuckGo / Unsplash supply images.
4. **Rendering** - FFmpeg center-crops to 9:16, mutes B-roll audio, overlays on the main track, burns captions.
5. **UI** - Streamlit dashboard to edit queries / regenerate assets before export; temp files are purged after render.

## Roadmap

- [x] Phase 1: Audio extraction and Groq transcription
- [x] Phase 2: Gemini editor (context and planning)
- [ ] Phase 3: Media sourcing and smart 9:16 formatting
- [ ] Phase 4: Streamlit mini-editor and caption export

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
reads `transcript.json` and writes `plan.json`: 10 B-rolls (`TOTAL_BROLLS`). Gemini decides the video/image mix
from the content. Each item has `asset_type`, `timestamp_start`, `timestamp_end`, a `search_query` and two
`alternative_queries`. The prompt forbids literal readings of metaphors ("hit a wall" -> "stressed anxious person").
Python snaps times to spoken-word boundaries and rejects plans with overlaps, wrong durations (video 3-5 s,
image 2-4 s), bad queries or gaps over 25 s, then asks Gemini to fix them (up to 3 attempts).
The plan is a `draft` until the user approves it in the Phase 4 UI.

## Project layout

```
src/
  transcription/   # Phase 1
  planning/        # Phase 2
  sourcing/        # Phase 3
  rendering/       # Phase 3-4
```
