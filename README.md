# Manga Story Video Studio

Automated end-to-end studio for converting manga and manhwa chapters into cinematic, voiceover-narrated videos (Shorts/Reels 9:16 or Landscape 16:9).

---

## ✨ Features
- 🌐 **Interactive Web UI**: Enter any manga chapter URL, configure voice and pacing, track live progress, and preview/download videos.
- ⚡ **Multi-Chapter Automation**: Automatically crawls and processes multiple chapters in sequence, then stitches them into a single merged master video.
- 🎯 **Smart Character & Action Framing**: Slices long manhwa vertical strips, prioritizes people/faces/actions, and eliminates empty text boxes with centered framing and ambient background fill.
- 🎙️ **Story Narrator Voiceover**: Powered by `edge-tts` with selectable voices (Christopher, Guy, Aria, Andrew) and customizable speed (`1.25x`, `1.5x`).
- ⏱️ **Frame-by-Frame Cut Transitions**: Displays each relevant comic panel steadily with crisp synchronization to the voiceover lines.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.9+
- FFmpeg (`brew install ffmpeg`)
- Tesseract OCR (`brew install tesseract`)

### 2. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Web UI
```bash
python app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser!

---

## 💻 CLI Usage

### Generate Single or Multi-Chapter Videos from URL
```bash
python story_pipeline.py https://www.mgeko.cc/reader/en/return-of-the-mount-hua-sect-chapter-1-eng-li/
```

### Options:
- **`--chapters`**: Number of chapters to process (e.g. `1`, `3`, `5`, or `all`).
- **`--voice`**: Select narration voice (e.g. `en-US-ChristopherNeural`, `en-US-GuyNeural`, `en-US-AriaNeural`).
- **`--speed`**: Voice & image pace (`+25%`, `+50%`).
- **`--aspect-ratio`**: `16:9` or `9:16`.
- **`--output`**: Target output directory (default: `output/`).
- **`--use-llm`**: Use an LLM (OpenAI, requires `API_KEY` in `.env`) to generate the narration script instead of the built-in regex/OCR generator. Off by default — no API calls unless passed.

---

## 🤖 LLM-Based Narration (optional)

By default, narration scripts are generated with a local OCR + regex heuristic (no API calls, no cost). To use a real LLM for higher-quality, more general narration:

1. Add your OpenAI key to `.env`:
   ```
   API_KEY=sk-...
   ```
2. Optionally override the model (default is the inexpensive `gpt-4o-mini`):
   ```
   LLM_MODEL=gpt-4o-mini
   ```
3. Pass `--use-llm` on the CLI, or `"use_llm": true` in a `/api/generate` request, or `"use_llm": true` in a batch job (see below).

Each chapter makes exactly **one** LLM API call (not per-panel), and the pipeline falls back to the regex generator automatically if the call fails or no key is set.

---

## 📦 Batch Automation

To process a list of manga/manhwa series unattended, use `run_batch.py` with a JSON job manifest.

1. Copy `jobs.example.json` to `jobs.json` and edit it:
   ```json
   [
     {
       "url": "https://www.mgeko.cc/reader/en/some-series-chapter-1/",
       "chapters": 3,
       "voice": "en-US-ChristopherNeural",
       "speed": 1.25,
       "aspect_ratio": "9:16",
       "use_llm": false
     }
   ]
   ```
2. Run the batch:
   ```bash
   python run_batch.py jobs.json
   ```
   Jobs run sequentially; each one's chapters, voice, speed, aspect ratio, and LLM usage are independently configurable.
3. A safety cap limits `chapters` per job to 5 by default (prevents an accidental huge/expensive run). Raise it explicitly if needed:
   ```bash
   python run_batch.py jobs.json --max-chapters-per-job 10
   ```
4. Results (success/failure per job, output paths, timing) are written to `batch_results.json` (override with `--log`).
