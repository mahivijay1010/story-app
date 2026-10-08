#!/usr/bin/env python3
"""
Applies bespoke AI-generated context images and stylish lower-third badges
to Solo Leveling: Ragnarok (sl-ragnorak) Chapter 0 and Chapter 1,
then reassembles the chapter videos and merged master story video.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from ai_context_generator import (
    generate_ai_image,
    render_context_badge,
    render_enhanced_segment,
)
from story_pipeline import concat_video_segments, get_audio_duration


SL_CHAPTER_0_CONTEXTS = [
    {
        "beat_id": "beat_005",
        "badge_type": "CHARACTER",
        "title": "SUNG SUHO",
        "subtitle": "Heir to the Shadow Monarch | The Leveling Dream",
        "prompt": (
            "Solo Leveling style anime illustration of young protagonist Sung Suho standing "
            "amidst dark purple shadow mist with glowing violet eyes, holographic leveling up system "
            "interface glowing in the air, dark fantasy, cinematic lighting, highly detailed 8k"
        ),
    },
    {
        "beat_id": "beat_006",
        "badge_type": "BATTLE",
        "title": "THE TRIAL OF SHADOWS",
        "subtitle": "Endless Battles | Level Up Quest",
        "prompt": (
            "Dynamic action anime illustration of a Korean hunter slashing through demonic shadow "
            "beasts with glowing purple energy daggers, sparks and shadowy tendrils flying, "
            "Solo Leveling manhwa art style, intense battle composition, 8k"
        ),
    },
    {
        "beat_id": "beat_007",
        "badge_type": "CLIMAX",
        "title": "SECRET QUEST: THE POWERLESS",
        "subtitle": "The Final Boss | Monarch Trial",
        "prompt": (
            "Terrifying colossal shadow demon titan boss towering over a cracked volcanic arena, "
            "blood red and purple magical lightning, tiny hunter silhouette facing the titan, "
            "Solo Leveling final boss scale, epic dark fantasy art, 8k"
        ),
    },
    {
        "beat_id": "beat_010",
        "badge_type": "LORE",
        "title": "SOLO LEVELING: RAGNAROK",
        "subtitle": "The Legend Awakens | Succession of Shadows",
        "prompt": (
            "Legendary Shadow Monarch silhouette hovering majestically high above the glowing night skyline "
            "of modern Seoul, dark purple aurora Borealis ripping open the heavens, shadow dragons "
            "flying in the clouds, epic cinematic anime cover art, 8k"
        ),
    },
]


SL_CHAPTER_1_CONTEXTS = [
    {
        "beat_id": "beat_003",
        "badge_type": "LORE",
        "title": "THE OUTER GODS",
        "subtitle": "Supreme Beings of the Multiverse | Celestial Realm",
        "prompt": (
            "Solo Leveling style celestial illustration of colossal cosmic Outer Gods and Supreme Beings "
            "floating in deep outer space, creating and shattering galaxies with purple and gold divine energy, "
            "epic cosmic fantasy art, breathtaking scale, 8k"
        ),
    },
    {
        "beat_id": "beat_011",
        "badge_type": "SCENE",
        "title": "WAR OF DIMENSIONS",
        "subtitle": "Gathering of Supreme Beings | Multiverse Invasion",
        "prompt": (
            "Massive cosmic dimensional portals tearing open across the deep void of space, armies of celestial "
            "beings and cosmic entities preparing to invade, glowing dark violet and gold space rifts, "
            "cinematic sci-fi fantasy anime art, 8k"
        ),
    },
    {
        "beat_id": "beat_035",
        "badge_type": "CHARACTER",
        "title": "SUNG JINWOO",
        "subtitle": "The Shadow Monarch | Protector of the Human Realm",
        "prompt": (
            "Masterpiece anime illustration of Sung Jinwoo in glorious dark monarch armor, glowing deep purple "
            "shadow aura, loyal shadow army kneeling behind him in deep space, glowing purple eyes, "
            "supreme sovereign of the dead, Solo Leveling art style, 8k"
        ),
    },
    {
        "beat_id": "beat_060",
        "badge_type": "LORE",
        "title": "THE ERA OF HUNTERS",
        "subtitle": "Awakened Society | Modern Seoul",
        "prompt": (
            "Bustling futuristic modern Seoul boulevard with glowing holographic Hunter Association banners, "
            "elite Korean hunters wearing stylish tactical combat gear walking among citizens, "
            "glass skyscrapers, crisp vibrant manhwa art style, 8k"
        ),
    },
    {
        "beat_id": "beat_078",
        "badge_type": "CHARACTER",
        "title": "SUNG SUHO",
        "subtitle": "The Sealed Heir | Korea University",
        "prompt": (
            "Handsome young Korean college student Sung Suho wearing modern casual streetwear with backpack "
            "walking through university campus, faint subtle purple shadow ripples under his footsteps, "
            "clean aesthetic webtoon manhwa illustration, 8k"
        ),
    },
    {
        "beat_id": "beat_098",
        "badge_type": "SCENE",
        "title": "DUNGEON BREAK",
        "subtitle": "Gate Outbreak | Seoul University Campus",
        "prompt": (
            "Dramatic anime disaster scene of a glowing blue and crimson dungeon gate violently ripping open "
            "the ceiling of a modern university building, shattering glass and concrete, red lightning and "
            "dark dungeon fog pouring through, high-octane cinematic composition, 8k"
        ),
    },
    {
        "beat_id": "beat_0122",  # Will map to beat_122
        "badge_type": "BATTLE",
        "title": "DEMONIC BEAST INVASION",
        "subtitle": "Shadow Hyenas | Classroom Ambush",
        "prompt": (
            "Fierce, terrifying fanged shadow hyena demon with glowing crimson eyes snarling amidst "
            "university classroom debris, black smoke and purple sparks, Solo Leveling monster art, "
            "intense suspenseful action, 8k"
        ),
    },
    {
        "beat_id": "beat_0136",  # Will map to beat_136
        "badge_type": "CLIMAX",
        "title": "THE SHADOW MONARCH AWAKENS",
        "subtitle": "Sung Suho | Awakening of the Successor",
        "prompt": (
            "Epic climax illustration of young hero Sung Suho standing bold and fearless before the monster, "
            "brilliant royal blue and violet shadow monarch lightning erupting around his fists and eyes, "
            "shadows answering his call, Solo Leveling masterpiece art, 8k"
        ),
    },
]


def process_chapter_enhancements(
    series_slug: str = "sl-ragnorak",
    aspect_ratio: str = "16:9",
    ch_num: int = 0,
    context_configs: list[dict] = None,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    aspect_slug = aspect_ratio.replace(":", "x")
    ch_str = f"{ch_num:03d}"
    work_dir = Path("workspace") / series_slug / aspect_slug / f"chapter_{ch_str}"
    out_dir = Path("output") / series_slug / aspect_slug
    
    audio_dir = work_dir / "audio"
    segments_dir = work_dir / "segments"
    ai_dir = work_dir / "ai_context"
    ai_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"Enhancing Chapter {ch_num} with AI Context Images")
    print(f"==========================================")

    for cfg in context_configs:
        raw_bid = cfg["beat_id"]
        # Normalize beat_id format (e.g. beat_0122 -> beat_122)
        m = re.match(r"beat_0*(\d+)", raw_bid)
        if m:
            beat_id = f"beat_{int(m.group(1)):03d}"
        else:
            beat_id = raw_bid

        audio_file = audio_dir / f"{beat_id}.mp3"
        segment_file = segments_dir / f"{beat_id}.mp4"

        if not audio_file.exists():
            print(f"  [WARN] Audio {audio_file.name} not found, skipping {beat_id}")
            continue

        duration = get_audio_duration(audio_file)
        
        raw_img_path = ai_dir / f"{beat_id}_raw.png"
        badged_img_path = ai_dir / f"{beat_id}_badged.png"

        print(f"  --> Processing {beat_id}: [{cfg['badge_type']}] {cfg['title']}")
        
        # 1. Generate bespoke AI illustration
        if not raw_img_path.exists() or raw_img_path.stat().st_size < 10000:
            print(f"      Calling OpenAI API for {cfg['title']}...")
            generate_ai_image(cfg["prompt"], raw_img_path, aspect_ratio=aspect_ratio)
            print(f"      Generated AI image: {raw_img_path.name}")
        else:
            print(f"      Reusing cached AI image: {raw_img_path.name}")

        # 2. Render stylish lower-third context badge
        if not badged_img_path.exists() or badged_img_path.stat().st_size < 10000:
            render_context_badge(
                raw_img_path,
                badged_img_path,
                badge_type=cfg["badge_type"],
                title=cfg["title"],
                subtitle=cfg["subtitle"],
                width=width,
                height=height,
            )
            print(f"      Rendered glassmorphic badge onto: {badged_img_path.name}")
        else:
            print(f"      Reusing badged image: {badged_img_path.name}")

        # 3. Render video segment with subtle cinematic slow zoom
        print(f"      Rendering cinematic video segment for {beat_id} ({duration:.2f}s)...")
        render_enhanced_segment(
            badged_img_path,
            audio_file,
            segment_file,
            duration=duration,
            width=width,
            height=height,
            has_zoom=True,
        )

    # 4. Re-concatenate all segments for this chapter
    all_segments = sorted(segments_dir.glob("beat_*.mp4"))
    def beat_num(p: Path) -> int:
        match = re.search(r"beat_(\d+)", p.name)
        return int(match.group(1)) if match else 0
    all_segments.sort(key=beat_num)

    chapter_video = out_dir / f"chapter_{ch_str}_video.mp4"
    print(f"\n  Assembling enhanced Chapter {ch_num} video ({len(all_segments)} segments)...")
    concat_video_segments(all_segments, chapter_video)
    print(f"  [SUCCESS] Written enhanced chapter video: {chapter_video} ({chapter_video.stat().st_size // (1024*1024)}MB)")
    return chapter_video


def reassemble_all(series_slug: str = "sl-ragnorak", aspect_ratio: str = "16:9"):
    aspect_slug = aspect_ratio.replace(":", "x")
    out_dir = Path("output") / series_slug / aspect_slug
    
    # Process Chapter 0
    ch0_vid = process_chapter_enhancements(
        series_slug=series_slug,
        aspect_ratio=aspect_ratio,
        ch_num=0,
        context_configs=SL_CHAPTER_0_CONTEXTS,
    )

    # Process Chapter 1
    ch1_vid = process_chapter_enhancements(
        series_slug=series_slug,
        aspect_ratio=aspect_ratio,
        ch_num=1,
        context_configs=SL_CHAPTER_1_CONTEXTS,
    )

    # Merge Chapter 0 and Chapter 1 into master story video
    merged_output = out_dir / "all_chapters_merged.mp4"
    print(f"\n==========================================")
    print(f"Merging Chapter 0 and Chapter 1 into Master Video")
    print(f"==========================================")
    concat_video_segments([ch0_vid, ch1_vid], merged_output)
    print(f"[COMPLETE] Master story video written to: {merged_output} ({merged_output.stat().st_size // (1024*1024)}MB)")


if __name__ == "__main__":
    reassemble_all()
