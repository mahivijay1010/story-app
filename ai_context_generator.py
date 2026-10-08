#!/usr/bin/env python3
"""
AI Context Generator for Manga & Manhwa Story Videos
Generates bespoke cinematic character portraits, scene shifts, and cosmic lore visuals
using OpenAI image models (gpt-image-1-mini / gpt-image-1), with stylish glassmorphic
lower-third badges for character introductions and dramatic scene shifts.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI


# Color schemes for different badge types
BADGE_STYLES = {
    "CHARACTER": {
        "tag_bg": (147, 51, 234, 235),      # Royal purple
        "border": (168, 85, 247, 210),
        "accent": (192, 132, 252, 255),
        "tag_label": "✦ CHARACTER INTRO",
    },
    "SCENE": {
        "tag_bg": (14, 165, 233, 235),      # Electric blue
        "border": (56, 189, 248, 210),
        "accent": (125, 211, 252, 255),
        "tag_label": "✦ SCENE SHIFT",
    },
    "LORE": {
        "tag_bg": (217, 119, 6, 235),       # Amber gold
        "border": (245, 158, 11, 210),
        "accent": (252, 211, 77, 255),
        "tag_label": "✦ STORY LORE",
    },
    "BATTLE": {
        "tag_bg": (220, 38, 38, 235),       # Crimson red
        "border": (239, 68, 68, 210),
        "accent": (252, 165, 165, 255),
        "tag_label": "✦ BATTLE CONFLICT",
    },
    "CLIMAX": {
        "tag_bg": (124, 58, 237, 235),      # Deep violet
        "border": (139, 92, 246, 210),
        "accent": (216, 180, 254, 255),
        "tag_label": "✦ EPIC CLIMAX",
    },
}


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API_KEY or OPENAI_API_KEY in environment (.env)")
    return OpenAI(api_key=api_key)


def generate_ai_image(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
    retries: int = 2,
) -> Path:
    """
    Generates a bespoke cinematic illustration using OpenAI's gpt-image models
    and writes the image directly to output_path.
    """
    if output_path.exists() and output_path.stat().st_size > 10000:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_openai_client()

    # Determine optimal size for aspect ratio
    # 16:9 -> 1536x1024, 9:16 -> 1024x1536
    size = "1536x1024" if aspect_ratio == "16:9" else "1024x1536"

    models_to_try = ["gpt-image-1-mini", "gpt-image-1", "gpt-image-1.5"]
    last_err = None

    for model_name in models_to_try:
        for attempt in range(retries):
            try:
                resp = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    n=1,
                    size=size,
                )
                b64_data = resp.data[0].b64_json
                if b64_data:
                    img_bytes = base64.b64decode(b64_data)
                elif resp.data[0].url:
                    from urllib.request import urlopen
                    with urlopen(resp.data[0].url, timeout=30) as r:
                        img_bytes = r.read()
                else:
                    raise ValueError("No b64_json or url in image generation response")

                output_path.write_bytes(img_bytes)
                return output_path
            except Exception as exc:
                last_err = exc
                time.sleep(2)

    raise RuntimeError(f"Failed to generate AI image after trying models {models_to_try}: {last_err}")


def render_context_badge(
    base_image_path: Path,
    output_path: Path,
    badge_type: str,
    title: str,
    subtitle: str,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    """
    Composites a sleek glassmorphic lower-third context badge onto the image
    with glowing accent borders, category pill tag, and clean typography.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img = Image.open(base_image_path).convert("RGBA")
    
    # Scale and center-crop to target dimensions
    target_aspect = width / height
    img_aspect = img.width / img.height
    if img_aspect > target_aspect:
        new_h = height
        new_w = int(img.width * (height / img.height))
    else:
        new_w = width
        new_h = int(img.height * (width / img.width))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    canvas = img_resized.crop((left, top, left + width, top + height))

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    # Subtle bottom vignette gradient for contrast
    grad_h = int(height * 0.35)
    for y in range(grad_h):
        alpha = int(((y / grad_h) ** 1.8) * 190)
        draw_ov.line([(0, height - grad_h + y), (width, height - grad_h + y)], fill=(6, 8, 18, alpha))

    # Badge style configuration
    style = BADGE_STYLES.get(badge_type.upper(), BADGE_STYLES["CHARACTER"])
    
    # Card layout
    bx = 64
    bh = 148
    by = height - bh - 60
    bw = min(int(width * 0.52), 760)

    # Glassmorphic card surface
    card = Image.new("RGBA", (bw, bh), (10, 12, 24, 225))
    card_draw = ImageDraw.Draw(card)
    card_draw.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=16, outline=style["border"], width=2)
    card_draw.rounded_rectangle([0, 0, 6, bh - 1], radius=3, fill=style["accent"])
    overlay.paste(card, (bx, by), card)

    # Fonts
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFPro.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    font_tag = None
    font_title = None
    font_sub = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font_tag = ImageFont.truetype(fp, 17)
                font_title = ImageFont.truetype(fp, 34)
                font_sub = ImageFont.truetype(fp, 19)
                break
            except Exception:
                continue
    if not font_title:
        font_tag = font_title = font_sub = ImageFont.load_default()

    draw = ImageDraw.Draw(overlay)

    # Category Pill
    tag_text = style["tag_label"]
    tag_w = 200
    tag_h = 28
    draw.rounded_rectangle([bx + 22, by + 16, bx + 22 + tag_w, by + 16 + tag_h], radius=14, fill=style["tag_bg"])
    draw.text((bx + 34, by + 21), tag_text, fill=(255, 255, 255), font=font_tag)

    # Title with crisp drop shadow
    tx = bx + 22
    ty = by + 54
    draw.text((tx + 2, ty + 2), title.upper(), fill=(0, 0, 0, 200), font=font_title)
    draw.text((tx, ty), title.upper(), fill=(255, 255, 255, 255), font=font_title)

    # Subtitle
    sx = bx + 22
    sy = by + 102
    draw.text((sx + 1, sy + 1), subtitle, fill=(0, 0, 0, 150), font=font_sub)
    draw.text((sx, sy), subtitle, fill=(203, 213, 225, 240), font=font_sub)

    final_img = Image.alpha_composite(canvas, overlay).convert("RGB")
    final_img.save(output_path, quality=95)
    return output_path


def render_enhanced_segment(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    width: int = 1920,
    height: int = 1080,
    has_zoom: bool = True,
) -> None:
    """
    Renders an audio/video segment for a beat with subtle cinematic slow zoom (Ken Burns)
    that brings the bespoke illustration to life.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 30
    total_frames = max(int(duration * fps), 30)

    if has_zoom:
        # Slow cinematic push-in (zoom 1.00 to 1.05)
        vf = (
            f"[0:v]scale={width}:{height},zoompan="
            f"z='min(zoom+0.0005,1.05)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex", vf,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "19",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-t", f"{duration:.3f}",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "19",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-t", f"{duration:.3f}",
            str(output_path),
        ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def identify_and_generate_ai_context(
    beats: list[dict],
    chapter_title: str,
    ch_work_dir: Path,
    aspect_ratio: str = "16:9",
    max_images: int = 6,
    width: int = 1920,
    height: int = 1080,
    progress_callback=None,
) -> dict[str, Path]:
    """
    Analyzes narration beats using an LLM to select key character introductions,
    dramatic scene shifts, and climactic moments, then generates bespoke AI illustrations
    with glassmorphic context badges.
    Returns mapping of {beat_id: badged_image_path}.
    """
    # Image generation is OpenAI-only (DeepSeek has no image API), so this whole
    # feature needs an OpenAI key regardless of which provider narrates.
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {}

    ai_dir = ch_work_dir / "ai_context"
    ai_dir.mkdir(parents=True, exist_ok=True)

    client = get_openai_client()

    # Build beat manifest
    beat_lines = [f"{b['id']}: {b['narration']}" for b in beats]
    manifest = "\n".join(beat_lines)

    prompt = (
        f"You are a cinematic anime director creating a high-impact video recap for a manhwa/manga "
        f"chapter titled \"{chapter_title}\".\n\n"
        f"Below are the chapter's narration beats:\n{manifest}\n\n"
        f"Select between 3 and {max_images} beats that represent the most important CHARACTER INTRODUCTIONS, "
        f"DRAMATIC SCENE SHIFTS, COSMIC LORE REVELATIONS, or CLIMACTIC MOMENTS that would benefit from a "
        f"bespoke full-screen cinematic illustration with a lower-third context badge.\n\n"
        f"Respond with ONLY a valid JSON array of objects with these exact keys:\n"
        f"[\n"
        f"  {{\n"
        f"    \"beat_id\": \"<exact beat_id e.g. beat_005>\",\n"
        f"    \"badge_type\": \"CHARACTER\" | \"SCENE\" | \"LORE\" | \"BATTLE\" | \"CLIMAX\",\n"
        f"    \"title\": \"<Short uppercase title e.g. SUNG SUHO>\",\n"
        f"    \"subtitle\": \"<Concise context e.g. Heir to the Shadow Monarch | Student>\",\n"
        f"    \"prompt\": \"<Detailed cinematic anime illustration prompt in vibrant manhwa art style with dramatic lighting, 8k>\"\n"
        f"  }}\n"
        f"]"
    )

    try:
        completion = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000,
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
        plan = json.loads(raw)
    except Exception as exc:
        print(f"[WARN] Failed to analyze beats for AI context: {exc}")
        return {}

    enhanced_map: dict[str, Path] = {}

    for idx, item in enumerate(plan, start=1):
        bid = item.get("beat_id")
        btype = item.get("badge_type", "CHARACTER")
        btitle = item.get("title", "STORY EVENT")
        bsub = item.get("subtitle", "")
        bprompt = item.get("prompt", "")

        if not bid or not bprompt:
            continue

        raw_path = ai_dir / f"{bid}_raw.png"
        badged_path = ai_dir / f"{bid}_badged.png"

        try:
            if progress_callback:
                progress_callback(f"Generating bespoke AI visual {idx}/{len(plan)}: {btitle}...", 50)
            
            # Generate or reuse raw AI image
            if not raw_path.exists() or raw_path.stat().st_size < 10000:
                generate_ai_image(bprompt, raw_path, aspect_ratio=aspect_ratio)

            # Render or reuse badged image
            if not badged_path.exists() or badged_path.stat().st_size < 10000:
                render_context_badge(
                    raw_path,
                    badged_path,
                    badge_type=btype,
                    title=btitle,
                    subtitle=bsub,
                    width=width,
                    height=height,
                )

            enhanced_map[bid] = badged_path
        except Exception as exc:
            print(f"[WARN] Failed generating AI visual for {bid}: {exc}")
            continue

    return enhanced_map

