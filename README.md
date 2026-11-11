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
- [x] Phase 3: Media sourcing and smart 9:16 formatting (video only; images are deferred)
- [x] Phase 4: Streamlit mini-editor and caption export

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

1. Upload a video (or type a path to one on disk), pick the spoken language, press **Build B-roll plan**
   (transcribe -> plan -> download clips; resumable if something fails).
2. **Review**: every B-roll has a playable preview, its timing, what the speaker says, and an editable
   search query. **Search with this query** fetches a new clip, **Regenerate asset** picks a different video
   for the same query, **Remove slot** keeps the speaker on screen there.
3. **Approve and render**: burns captions (2-4 word chunks, active word in yellow) in the original language,
   English, or Roman Hindi (Gemini rewrites the text; timings are kept), then deletes the temporary clips and
   audio, keeping only `final.mp4` and the JSON files. The sidebar shows temp storage against a 32 GB budget.
   Cleanup only ever touches job folders inside `VIBECROP_WORKDIR/jobs`.

Windows uses Nirmala UI for Devanagari captions; on Linux/Colab install `fonts-noto-core`.
CLI equivalent of the render step: `python -m src.rendering.pipeline <job> <video> --captions hinglish`.

## Project layout

```
src/
  transcription/   # Phase 1
  planning/        # Phase 2
  sourcing/        # Phase 3
  rendering/       # Phase 3-4
```
