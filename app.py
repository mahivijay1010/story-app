#!/usr/bin/env python3
"""
Manga Story Video Generator - Web UI
Runs a modern web interface for automated multi-chapter video generation.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from story_pipeline import (
    discover_all_chapters,
    extract_manga_name,
    process_chapter_urls,
    process_multi_chapters,
    process_single_chapter,
)


app = FastAPI(title="Manga Story Video Studio")

# Global state for background job tracking
current_job = {
    "status": "idle",
    "message": "Ready to generate",
    "progress": 0,
    "chapter_videos": [],
    "merged_video": "",
    "script_file": "",
    "logs": [],
}


class GenerateRequest(BaseModel):
    url: Optional[str] = None
    urls: Optional[list[str]] = None
    max_chapters: int = 1
    chapter_start: Optional[float] = None
    chapter_end: Optional[float] = None
    voice: str = "am_michael"
    speed: float = 1.0
    aspect_ratio: str = "9:16"
    use_llm: bool = True
    merged_only: bool = True
    enable_ai_context: bool = True


def log_progress(msg: str, pct: int):
    global current_job
    current_job["message"] = msg
    current_job["progress"] = pct
    current_job["logs"].append(msg)
    print(f"[{pct}%] {msg}")


async def run_pipeline_task(req: GenerateRequest):
    global current_job
    current_job["status"] = "processing"
    current_job["progress"] = 5
    current_job["logs"] = []
    current_job["chapter_videos"] = []
    current_job["merged_video"] = ""
    current_job["script_file"] = ""

    speed_rate = f"{int(round((req.speed - 1.0) * 100)):+d}%"

    try:
        log_progress("Initializing story pipeline...", 10)

        aspect_slug = req.aspect_ratio.replace(":", "x")

        if req.urls:
            manga_slug = extract_manga_name(req.urls[0])
            manga_work_dir = Path("workspace") / manga_slug / aspect_slug
            manga_out_dir = Path("output") / manga_slug / aspect_slug

            chapter_videos, merged_video = await process_chapter_urls(
                chapter_urls=req.urls,
                voice=req.voice,
                speed_rate=speed_rate,
                aspect_ratio=req.aspect_ratio,
                use_llm=req.use_llm,
                enable_ai_context=req.enable_ai_context,
                output_dir=manga_out_dir,
                workspace_dir=manga_work_dir,
                merged_only=req.merged_only,
                progress_callback=log_progress,
            )
        else:
            manga_slug = extract_manga_name(req.url)
            manga_work_dir = Path("workspace") / manga_slug / aspect_slug
            manga_out_dir = Path("output") / manga_slug / aspect_slug

            chapter_videos, merged_video = await process_multi_chapters(
                start_url=req.url,
                max_chapters=req.max_chapters,
                voice=req.voice,
                speed_rate=speed_rate,
                aspect_ratio=req.aspect_ratio,
                use_llm=req.use_llm,
                enable_ai_context=req.enable_ai_context,
                output_dir=manga_out_dir,
                workspace_dir=manga_work_dir,
                merged_only=req.merged_only,
                chapter_start=req.chapter_start,
                chapter_end=req.chapter_end,
                progress_callback=log_progress,
            )

        current_job["chapter_videos"] = [] if req.merged_only else [
            str(v.relative_to(Path("output"))) if v.is_relative_to(Path("output")) else v.name
            for v in chapter_videos
        ]
        current_job["merged_video"] = (
            str(merged_video.relative_to(Path("output")))
            if merged_video.is_relative_to(Path("output"))
            else merged_video.name
        )
        script_file = merged_video.with_name("all_chapters_script.txt")
        current_job["script_file"] = (
            str(script_file.relative_to(Path("output")))
            if script_file.exists() and script_file.is_relative_to(Path("output"))
            else ""
        )
        current_job["status"] = "completed"
        log_progress("Video generation finished successfully!", 100)

    except Exception as exc:
        current_job["status"] = "error"
        current_job["message"] = f"Error: {exc}"
        log_progress(f"Failed with error: {exc}", 100)


@app.post("/api/discover")
async def api_discover(data: dict):
    url = data.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Missing URL")
    try:
        chapters = discover_all_chapters(url)
        return {"count": len(chapters), "chapters": chapters[:50]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/generate")
async def api_generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    global current_job
    if current_job["status"] == "processing":
        return {"status": "already_running", "message": "A job is already in progress"}

    if not req.url and not req.urls:
        raise HTTPException(status_code=400, detail="Either 'url' or 'urls' must be provided")
    if req.url and req.urls:
        raise HTTPException(status_code=400, detail="'url' and 'urls' are mutually exclusive")

    background_tasks.add_task(run_pipeline_task, req)
    return {"status": "started", "message": "Video generation started in background"}


@app.get("/api/status")
async def api_status():
    global current_job
    return current_job


@app.get("/api/videos/{filepath:path}")
async def get_video(filepath: str):
    file_path = Path("output") / filepath
    if not file_path.is_file():
        matches = list(Path("output").rglob(Path(filepath).name))
        if matches:
            file_path = matches[0]
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    media_type = "text/plain; charset=utf-8" if file_path.suffix == ".txt" else "video/mp4"
    return FileResponse(file_path, media_type=media_type)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manga Story Video Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0b10;
            --bg-card: rgba(22, 24, 38, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-blue: #3b82f6;
            --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #3b82f6 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.15) 0%, transparent 40%);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            max-width: 1080px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }

        .header {
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }

        .badge {
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid rgba(139, 92, 246, 0.4);
            color: #c4b5fd;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        h1 {
            font-size: 38px;
            font-weight: 800;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 16px;
            max-width: 600px;
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 32px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 14px;
            font-weight: 600;
            color: #cbd5e1;
        }

        .field-hint {
            font-size: 12.5px;
            color: #8890a4;
            margin: -2px 0 0 0;
        }

        .range-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .range-row input {
            flex: 1;
            min-width: 0;
        }

        .range-sep {
            color: #8890a4;
            font-size: 14px;
            font-weight: 600;
            flex-shrink: 0;
        }

        input, select {
            background: rgba(15, 17, 26, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #fff;
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 15px;
            outline: none;
            transition: all 0.2s ease;
        }

        input:focus, select:focus, textarea:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        textarea {
            background: rgba(15, 17, 26, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #fff;
            padding: 14px 18px;
            border-radius: 12px;
            font-size: 15px;
            outline: none;
            transition: all 0.2s ease;
            font-family: inherit;
            resize: vertical;
        }

        .toggle-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .mode-btn {
            flex: 1;
            min-width: 220px;
            background: rgba(15, 17, 26, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #cbd5e1;
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .mode-btn.active {
            background: var(--accent-gradient);
            color: white;
            border-color: transparent;
        }

        .checkbox-group {
            display: flex;
            align-items: center;
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
            font-weight: 500;
            color: #cbd5e1;
            cursor: pointer;
        }

        .checkbox-label input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #8b5cf6;
            cursor: pointer;
        }

        .btn-primary {
            background: var(--accent-gradient);
            border: none;
            color: white;
            padding: 16px 28px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 12px;
            box-shadow: 0 10px 25px rgba(139, 92, 246, 0.35);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(139, 92, 246, 0.5);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Progress Box */
        .progress-box {
            display: none;
            flex-direction: column;
            gap: 16px;
            margin-top: 24px;
            background: rgba(10, 11, 16, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            font-weight: 600;
        }

        .progress-bar-bg {
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 9999px;
            overflow: hidden;
        }

        .progress-bar-fill {
            height: 100%;
            width: 0%;
            background: var(--accent-gradient);
            border-radius: 9999px;
            transition: width 0.4s ease;
        }

        .log-terminal {
            background: #050608;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 14px;
            font-family: monospace;
            font-size: 13px;
            color: #38bdf8;
            height: 140px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.6;
        }

        /* Video Showcase */
        .video-card {
            display: none;
            flex-direction: column;
            gap: 20px;
            align-items: center;
        }

        video {
            max-width: 400px;
            width: 100%;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
        }

        .download-btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            color: #f8fafc;
            padding: 12px 24px;
            border-radius: 10px;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: background 0.2s ease;
        }

        .download-btn:hover {
            background: rgba(255, 255, 255, 0.16);
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <span class="badge">AI Storyteller Studio</span>
            <h1>Manga to Story Video Generator</h1>
            <p class="subtitle">Enter any chapter URL to automatically scrape, slice character panels, synthesize voiceovers, and generate a video.</p>
        </header>

        <section class="card">
            <form id="generateForm" onsubmit="event.preventDefault(); startGeneration();">
                <div class="form-grid">
                    <div class="form-group full-width url-mode-toggle">
                        <label>URL Mode</label>
                        <div class="toggle-row">
                            <button type="button" id="modeSingleBtn" class="mode-btn active" onclick="setUrlMode('single')">Single URL (discover N chapters forward)</button>
                            <button type="button" id="modeMultiBtn" class="mode-btn" onclick="setUrlMode('multi')">Multiple URLs (one video per URL + merged)</button>
                        </div>
                    </div>

                    <div class="form-group full-width" id="singleUrlGroup">
                        <label for="urlInput">Manga Series or Chapter URL</label>
                        <input type="url" id="urlInput" value="https://www.mgeko.cc/manga/sl-ragnorak/" placeholder="https://www.mgeko.cc/manga/series-name/  or  .../reader/en/series-chapter-1-eng-li/">
                        <p class="field-hint">Paste the series page (all chapters, starts from Chapter 1) or a specific chapter page (starts from there).</p>
                    </div>

                    <div class="form-group full-width" id="multiUrlGroup" style="display: none;">
                        <label for="urlsInput">Manga Chapter URLs (one per line)</label>
                        <textarea id="urlsInput" rows="5" placeholder="https://www.mgeko.cc/reader/en/.../chapter-1-eng-li/&#10;https://www.mgeko.cc/reader/en/.../chapter-2-eng-li/&#10;https://www.mgeko.cc/reader/en/.../chapter-3-eng-li/"></textarea>
                    </div>

                    <div class="form-group" id="chaptersGroup">
                        <label for="chaptersInput">Number of Chapters</label>
                        <select id="chaptersInput" onchange="onChaptersModeChange()">
                            <option value="1">1 Chapter</option>
                            <option value="2">2 Chapters</option>
                            <option value="3">3 Chapters</option>
                            <option value="5">5 Chapters</option>
                            <option value="10" selected>10 Chapters</option>
                            <option value="20">20 Chapters</option>
                            <option value="50">50 Chapters</option>
                            <option value="custom">Custom Range...</option>
                        </select>
                    </div>

                    <div class="form-group" id="chapterRangeGroup" style="display: none;">
                        <label for="chapterStartInput">Chapter Range (e.g. 10 to 20)</label>
                        <div class="range-row">
                            <input type="number" id="chapterStartInput" placeholder="Start (e.g. 10)" min="0" step="1">
                            <span class="range-sep">to</span>
                            <input type="number" id="chapterEndInput" placeholder="End (e.g. 20)" min="0" step="1">
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="voiceSelect">AI Story Voice</label>
                        <select id="voiceSelect">
                            <optgroup label="Kokoro - runs locally, open source (best quality)">
                                <option value="am_michael" selected>Michael (Deep Cinematic Narrator)</option>
                                <option value="am_fenrir">Fenrir (Intense / Action)</option>
                                <option value="am_adam">Adam (Warm Storyteller)</option>
                                <option value="bm_george">George (British Narrator)</option>
                                <option value="bm_fable">Fable (British Storyteller)</option>
                                <option value="af_heart">Heart (Expressive Female)</option>
                                <option value="af_bella">Bella (Warm Female)</option>
                            </optgroup>
                            <optgroup label="Microsoft neural voices - online fallback">
                                <option value="en-US-AndrewMultilingualNeural">Andrew (Warm Storyteller)</option>
                                <option value="en-US-BrianMultilingualNeural">Brian (Casual / Sincere)</option>
                                <option value="en-US-ChristopherNeural">Christopher (Authority)</option>
                                <option value="en-US-GuyNeural">Guy (Action Hero)</option>
                                <option value="en-US-AvaMultilingualNeural">Ava (Expressive Female)</option>
                                <option value="en-US-AriaNeural">Aria (Confident Female)</option>
                            </optgroup>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="speedSelect">Voiceover & Frame Speed</label>
                        <select id="speedSelect">
                            <option value="1.0" selected>1.0x Natural (recommended)</option>
                            <option value="1.1">1.1x Brisk</option>
                            <option value="1.25">1.25x Dynamic Pacing</option>
                            <option value="1.5">1.5x Fast Paced</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="formatSelect">Video Format</label>
                        <select id="formatSelect">
                            <option value="9:16" selected>9:16 Vertical (YouTube Story Recap / Shorts / Reels / TikTok)</option>
                            <option value="16:9">16:9 Landscape</option>
                        </select>
                    </div>

                    <div class="form-group full-width checkbox-group" style="display: flex; flex-direction: column; gap: 8px;">
                        <label class="checkbox-label">
                            <input type="checkbox" id="useLlmInput" checked>
                            📖 Story Narrator (reads the whole chapter, then narrates it in third person — needs API_KEY)
                        </label>
                        <label class="checkbox-label">
                            <input type="checkbox" id="aiContextInput" checked>
                            ✨ Add AI Context Images & Badges (bespoke cinematic character & scene visuals)
                        </label>
                        <label class="checkbox-label">
                            <input type="checkbox" id="mergedOnlyInput" checked>
                            Only keep the merged video (skip saving individual per-chapter videos)
                        </label>
                    </div>

                    <button type="submit" id="generateBtn" class="btn-primary full-width">
                        ⚡ Generate Story Video
                    </button>
                </div>
            </form>

            <div id="progressBox" class="progress-box">
                <div class="progress-header">
                    <span id="statusMessage">Starting pipeline...</span>
                    <span id="progressPercent">0%</span>
                </div>
                <div class="progress-bar-bg">
                    <div id="progressBarFill" class="progress-bar-fill"></div>
                </div>
                <div id="logTerminal" class="log-terminal">Waiting for process to start...</div>
            </div>
        </section>

        <section id="videoSection" class="card video-card">
            <h2>🎬 Generated Story Video</h2>
            <video id="videoPlayer" controls autoplay muted style="max-width: 720px; width: 100%; border-radius: 16px; border: 1px solid var(--border-color);"></video>
            <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center;">
                <a id="downloadMergedBtn" class="download-btn" href="#" download>
                    ⬇️ Download Master Video
                </a>
                <a id="downloadScriptBtn" class="download-btn" href="#" download style="display: none;">
                    📝 Download Narration Script
                </a>
            </div>
        </section>
    </div>

    <script>
        let pollTimer = null;
        let urlMode = 'single';

        function setUrlMode(mode) {
            urlMode = mode;
            document.getElementById('modeSingleBtn').classList.toggle('active', mode === 'single');
            document.getElementById('modeMultiBtn').classList.toggle('active', mode === 'multi');
            document.getElementById('singleUrlGroup').style.display = mode === 'single' ? 'flex' : 'none';
            document.getElementById('multiUrlGroup').style.display = mode === 'multi' ? 'flex' : 'none';
            document.getElementById('chaptersGroup').style.display = mode === 'single' ? 'flex' : 'none';
            onChaptersModeChange();
        }

        function onChaptersModeChange() {
            const isCustom = urlMode === 'single' && document.getElementById('chaptersInput').value === 'custom';
            document.getElementById('chapterRangeGroup').style.display = isCustom ? 'flex' : 'none';
        }

        async function startGeneration() {
            const voice = document.getElementById('voiceSelect').value;
            const speed = parseFloat(document.getElementById('speedSelect').value);
            const aspect_ratio = document.getElementById('formatSelect').value;
            const merged_only = document.getElementById('mergedOnlyInput').checked;
            const enable_ai_context = document.getElementById('aiContextInput').checked;
            const use_llm = document.getElementById('useLlmInput').checked;

            let payload = { voice, speed, aspect_ratio, merged_only, enable_ai_context, use_llm };

            if (urlMode === 'multi') {
                const urls = document.getElementById('urlsInput').value
                    .split('\\n')
                    .map(u => u.trim())
                    .filter(u => u.length > 0);
                if (urls.length === 0) {
                    alert("Enter at least one chapter URL.");
                    return;
                }
                payload.urls = urls;
            } else {
                const url = document.getElementById('urlInput').value.trim();
                if (!url) {
                    alert("Enter a chapter URL.");
                    return;
                }
                payload.url = url;

                const chaptersValue = document.getElementById('chaptersInput').value;
                if (chaptersValue === 'custom') {
                    const startVal = document.getElementById('chapterStartInput').value;
                    const endVal = document.getElementById('chapterEndInput').value;
                    if (startVal === '' || endVal === '') {
                        alert("Enter both a start and end chapter number for the custom range.");
                        return;
                    }
                    if (parseFloat(startVal) > parseFloat(endVal)) {
                        alert("Start chapter must be less than or equal to end chapter.");
                        return;
                    }
                    payload.chapter_start = parseFloat(startVal);
                    payload.chapter_end = parseFloat(endVal);
                } else {
                    payload.max_chapters = parseInt(chaptersValue);
                }
            }

            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            btn.innerText = "⏳ Generating Video...";

            document.getElementById('progressBox').style.display = 'flex';
            document.getElementById('videoSection').style.display = 'none';

            try {
                const resp = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();

                if (!resp.ok) {
                    throw new Error(data.detail || 'Request failed');
                }

                if (pollTimer) clearInterval(pollTimer);
                pollTimer = setInterval(checkStatus, 1500);
            } catch (err) {
                alert("Failed to start generation: " + err);
                btn.disabled = false;
                btn.innerText = "⚡ Generate Story Video";
            }
        }

        async function checkStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();

                document.getElementById('statusMessage').innerText = data.message;
                document.getElementById('progressPercent').innerText = data.progress + "%";
                document.getElementById('progressBarFill').style.width = data.progress + "%";

                if (data.logs && data.logs.length) {
                    const terminal = document.getElementById('logTerminal');
                    terminal.innerText = data.logs.join('\\n');
                    terminal.scrollTop = terminal.scrollHeight;
                }

                if (data.status === 'completed') {
                    clearInterval(pollTimer);
                    document.getElementById('generateBtn').disabled = false;
                    document.getElementById('generateBtn').innerText = "⚡ Generate Another Video";

                    if (data.merged_video) {
                        const videoSrc = `/api/videos/${data.merged_video}`;
                        const player = document.getElementById('videoPlayer');
                        player.src = videoSrc;
                        document.getElementById('downloadMergedBtn').href = videoSrc;
                        const scriptBtn = document.getElementById('downloadScriptBtn');
                        if (data.script_file) {
                            scriptBtn.href = `/api/videos/${data.script_file}`;
                            scriptBtn.style.display = '';
                        } else {
                            scriptBtn.style.display = 'none';
                        }
                        document.getElementById('videoSection').style.display = 'flex';
                    }
                } else if (data.status === 'error') {
                    clearInterval(pollTimer);
                    document.getElementById('generateBtn').disabled = false;
                    document.getElementById('generateBtn').innerText = "⚡ Generate Story Video";
                    alert(data.message);
                }
            } catch (err) {
                console.error("Status check failed", err);
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Manga Story Video Studio on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
