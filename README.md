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

- [ ] Phase 1: Audio extraction and Groq transcription
- [ ] Phase 2: Gemini editor (context and planning)
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

## Project layout

```
src/
  transcription/   # Phase 1
  planning/        # Phase 2
  sourcing/        # Phase 3
  rendering/       # Phase 3-4
```
