#!/usr/bin/env python3
"""
Automated Manhwa Story Video Pipeline
- Discovers and scrapes chapters from mgeko.cc (or similar manga sites).
- Slices continuous strips into clean individual panels.
- Prioritizes character/people/action panels and filters out empty text bubbles.
- Automatically generates story voiceover scripts from panel dialogue & context.
- Synthesizes accelerated (1.25x) neural voiceover with edge-tts.
- Renders 1080x1920 frame-by-frame videos with ambient blurred background fill.
- Concatenates chapter videos and merges multi-chapter series into a single master video.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import cv2
import edge_tts
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
import pytesseract

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


def resolve_llm_provider() -> Optional[tuple[str, Optional[str], str]]:
    """
    Picks the text-generation provider from whatever keys are in .env.

    DeepSeek is preferred when its key is present: it is much cheaper for the long
    chapter-comprehension prompts and its API is OpenAI-compatible, so only the
    base URL and model name differ. Returns (api_key, base_url, model), or None
    when no key is configured at all.
    """
    deepseek = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek:
        return (
            deepseek,
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            os.environ.get("LLM_MODEL", "deepseek-chat"),
        )
    openai_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return openai_key, None, LLM_MODEL
    return None


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.mgeko.cc/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
}


def extract_manga_name(url: str) -> str:
    """
    Extracts a clean, human-readable manga series slug from any manga reader URL.
    Works with mgeko, asuracomics, flamecomics, reaperscans, manganato, webtoons, etc.
    """
    from urllib.parse import urlparse
    path = urlparse(url).path.lower().strip('/')

    # 1. Webtoons style: /en/fantasy/tower-of-god/season-1-ep-0/
    m_wt = re.search(r'/(?:en|fr|id|th|zh|es)/[a-z0-9\-]+/([a-z0-9\-]+)/', '/' + path + '/')
    if m_wt and m_wt.group(1) not in ['viewer', 'episode', 'ep', 'chapter']:
        return m_wt.group(1)

    # 2. Strip chapter / episode numbers from path
    clean_path = re.sub(r'-(?:chapter|ch|ep|episode|season)-?\d+.*', '', path)

    # 3. /reader/en/manga-slug
    m1 = re.search(r'/reader/[a-z]{2}/([a-z0-9\-]+?)(?:-(?:mg\d+|raw|eng|li|fix|scan))*$', '/' + clean_path)
    if m1:
        return m1.group(1)

    # 4. /(?:series|manga|comics|comic)/manga-slug
    m2 = re.search(r'/(?:series|manga|comics|comic)/([a-z0-9\-]+)', '/' + clean_path)
    if m2:
        slug = m2.group(1)
        return re.sub(r'^\d+-', '', slug)

    parts = [p for p in clean_path.split('/') if p and not re.match(r'^(?:reader|en|viewer|comic|comics|series|manga)$', p)]
    if parts:
        slug = parts[0]
        slug = re.sub(r'^\d+-', '', slug)
        return slug
    return "manga_story"


class ChapterParser(HTMLParser):
    """
    Universal HTML parser that extracts manga reader strip images across
    diverse manga reader themes (Madara, MangaReader, MangaStream, Webtoons, custom readers).
    """
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.in_reader = False
        self.reader_div_depth = 0
        self.image_urls: list[str] = []
        self.all_found_urls: list[str] = []
        self.chapter_options: list[tuple[str, str]] = []
        self.raw_script_data: list[str] = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        tag_id = attr_dict.get("id", "").lower()
        tag_cls = attr_dict.get("class", "").lower()

        # Detect reader container boundaries
        reader_markers = [
            "chapter-reader", "readerarea", "reading-content", "entry-content",
            "reader-area", "comic-page", "chapter-content", "container-chapter-reader",
            "img-container", "manga-read", "viewer"
        ]
        if any(m in tag_id or m in tag_cls for m in reader_markers):
            self.in_reader = True
            self.reader_div_depth += 1
        elif self.in_reader and tag in ["div", "section", "article", "main"]:
            self.reader_div_depth += 1

        if tag == "script":
            self.in_script = True

        if tag == "option" and "value" in attr_dict:
            self.chapter_options.append((attr_dict["value"], ""))

        # Check all possible image source attributes
        if tag == "img":
            candidate_src = (
                attr_dict.get("src")
                or attr_dict.get("data-src")
                or attr_dict.get("data-original")
                or attr_dict.get("data-lazy-src")
                or attr_dict.get("data-url")
                or attr_dict.get("data-full-url")
                or ""
            )
            if candidate_src:
                full_url = urljoin(self.base_url, candidate_src.strip())
                # Exclude UI icons, logos, social badges, ads
                lower_url = full_url.lower()
                ignore_keywords = [
                    "logo", "banner", "icon", "loading", "avatar", "ads",
                    "vline", "credits", "discord", "patreon", "footer",
                    "header", "gravatar", "favicon", "button"
                ]
                if not any(k in lower_url for k in ignore_keywords):
                    self.all_found_urls.append(full_url)
                    if self.in_reader:
                        self.image_urls.append(full_url)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False
        if self.in_reader and tag in ["div", "section", "article", "main"]:
            self.reader_div_depth -= 1
            if self.reader_div_depth <= 0:
                self.in_reader = False

    def handle_data(self, data):
        if self.in_script and data.strip():
            self.raw_script_data.append(data.strip())


def fetch_html(url: str, retries: int = 3) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": origin + "/",
        "Origin": origin,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise last_exc


def parse_chapter_number(text_or_url: str) -> Optional[float]:
    """Extracts a numeric chapter identifier (e.g. 1, 12.5) from title or URL."""
    clean = text_or_url.lower()
    # Pattern: chapter-12, ch.12.5, episode-5, ep-02, ch_100
    m = re.search(r'(?:chapter|ch|ep|episode)[-_/ ]*(\d+(?:[-.]\d+)?)', clean)
    if m:
        try:
            return float(m.group(1).replace('-', '.'))
        except ValueError:
            pass
    # Standalone number at end of url
    m2 = re.search(r'[-/](\d+(?:\.\d+)?)/?$', clean)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass
    return None


def discover_all_chapters(start_url: str) -> list[dict[str, str]]:
    """
    Discovers all available chapters from a given chapter URL or series URL.
    Works universally across dropdown options, link lists, and reader scripts.
    """
    html = fetch_html(start_url)

    # A series landing page (no chapter number of its own) sometimes only lists
    # its most recent chapters and links to a separate, more complete listing -
    # follow that when present so a series URL doesn't silently miss early chapters.
    if parse_chapter_number(start_url) is None:
        m = re.search(r'href=[\"\']([^\"\']*all-chapters[^\"\']*)[\"\']', html, flags=re.IGNORECASE)
        if m:
            all_chapters_url = urljoin(start_url, m.group(1))
            if all_chapters_url.rstrip("/").lower() != start_url.rstrip("/").lower():
                try:
                    html = fetch_html(all_chapters_url)
                except Exception:
                    pass

    chapters = []
    seen_urls = set()

    # If the start URL is itself a chapter-reader page, make sure it's included -
    # a series index/listing page (no chapter number of its own) has no such entry,
    # its real chapters all come from the link scan below.
    start_num = parse_chapter_number(start_url)
    clean_start = start_url.rstrip("/").lower()
    seen_urls.add(clean_start)
    seen_urls.add(clean_start + "/")

    if start_num is not None:
        chapters.append({
            "url": start_url,
            "title": f"Chapter {int(start_num) if start_num.is_integer() else start_num}",
            "number": start_num,
        })

    # 1. Search dropdown <option value="...">
    options = re.findall(r'<option\s+[^>]*value=[\"\']([^\"\']+)[\"\'][^>]*>([^<]*)</option>', html, flags=re.IGNORECASE)
    for val, title in options:
        val_clean = val.strip()
        if not val_clean or val_clean.startswith("#") or val_clean.startswith("javascript:"):
            continue
        full_url = urljoin(start_url, val_clean)
        clean_full = full_url.rstrip("/").lower()
        if clean_full not in seen_urls:
            ch_num = parse_chapter_number(full_url) or parse_chapter_number(title)
            if ch_num is not None:
                seen_urls.add(clean_full)
                seen_urls.add(clean_full + "/")
                display_title = title.strip() or f"Chapter {int(ch_num) if ch_num.is_integer() else ch_num}"
                chapters.append({
                    "url": full_url,
                    "title": display_title,
                    "number": ch_num,
                })

    # 2. Search <a href="..."> chapter links. Inner text isn't captured here -
    # chapter list pages often nest divs/spans inside the <a> before any text
    # (e.g. mgeko's series page), which a "text with no nested tags" pattern
    # would silently fail to match at all. The chapter number is parsed from
    # the URL itself instead, which is reliable on its own.
    hrefs = re.findall(r'<a\s+[^>]*href=[\"\']([^\"\']+)[\"\']', html, flags=re.IGNORECASE)
    for href in hrefs:
        href_clean = href.strip()
        if not href_clean or href_clean.startswith("#") or href_clean.startswith("javascript:"):
            continue
        full_url = urljoin(start_url, href_clean)
        clean_full = full_url.rstrip("/").lower()
        if clean_full not in seen_urls:
            ch_num = parse_chapter_number(full_url)
            if ch_num is not None:
                seen_urls.add(clean_full)
                seen_urls.add(clean_full + "/")
                chapters.append({
                    "url": full_url,
                    "title": f"Chapter {int(ch_num) if ch_num.is_integer() else ch_num}",
                    "number": ch_num,
                })

    # Sort naturally in ascending order (Chapter 1 -> Chapter N)
    chapters.sort(key=lambda c: c["number"])
    return chapters


def score_character_presence(image_path: Path) -> tuple[float, bool]:
    """
    Computes a character / person presence score for an image panel:
    - Analyzes skin tone coverage in YCrCb color space (faces/hands/bodies)
    - Measures HSV color saturation and vibrancy (filtering out monochromatic text boxes)
    - Detects flat background / speech bubble dominance (pure white/black text boxes)
    Returns: (score, is_text_heavy)
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0, True
        h, w = img.shape[:2]
        total_pixels = h * w
        if total_pixels == 0:
            return 0.0, True

        # Convert to YCrCb for robust skin detection
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # Skin tone range in YCrCb: Cr in [133, 175], Cb in [80, 125]
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 80]), np.array([255, 175, 125]))
        skin_pixels = cv2.countNonZero(skin_mask)
        skin_ratio = skin_pixels / float(total_pixels)

        # Convert to HSV to measure colorfulness/saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        color_pixels = np.count_nonzero(sat > 35)
        color_pixel_ratio = color_pixels / float(total_pixels)

        # Flat background / text box detection (pure white or pure black)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flat_white = np.count_nonzero(gray > 240)
        flat_black = np.count_nonzero(gray < 20)
        flat_ratio = (flat_white + flat_black) / float(total_pixels)

        # If it's mostly flat white/black and very little skin/color, it's a text bubble / empty box
        is_text_heavy = (flat_ratio > 0.50 and skin_ratio < 0.05 and color_pixel_ratio < 0.20) or (flat_ratio > 0.70 and skin_ratio < 0.08)

        # Weighted score: high skin tone (person/face) + color richness - penalty for flat white/black
        score = (skin_ratio * 350.0) + (color_pixel_ratio * 100.0) - (flat_ratio * 60.0)
        return float(score), is_text_heavy
    except Exception:
        return 0.0, False


def scrape_chapter_images(chapter_url: str, output_dir: Path) -> list[Path]:
    """
    Scrapes all high-resolution manga strip images from any reader page.
    Handles standard DOM images, lazy-loaded sources, and embedded script arrays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    html = fetch_html(chapter_url)
    parser = ChapterParser(chapter_url)
    parser.feed(html)

    # 1. Collect from parser
    urls = list(parser.image_urls)
    if not urls:
        urls = list(parser.all_found_urls)

    # 2. Extract embedded script JSON arrays if no images found in DOM
    if not urls:
        import json
        for script in parser.raw_script_data:
            for match in re.findall(r'\"images\"\s*:\s*(\[[^\]]+\])', script):
                try:
                    arr = json.loads(match)
                    if isinstance(arr, list):
                        urls.extend([urljoin(chapter_url, str(u)) for u in arr if isinstance(u, str)])
                except Exception:
                    pass

    # 3. Regex fallback for direct image URLs on reader CDNs
    if not urls:
        raw_matches = re.findall(r'(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp)(?:\?[^\s"\'<>]*)?)', html)
        for u in raw_matches:
            if not any(k in u.lower() for k in ["logo", "banner", "icon", "avatar", "ads"]):
                urls.append(u)

    # Deduplicate preserving order
    seen = set()
    deduped_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped_urls.append(u)

    downloaded = []
    from urllib.parse import urlparse
    parsed = urlparse(chapter_url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": chapter_url,
        "Origin": f"{parsed.scheme}://{parsed.netloc}",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    for idx, url in enumerate(deduped_urls, start=1):
        suffix = Path(url.split("?")[0]).suffix or ".jpg"
        file_path = output_dir / f"page-{idx:03d}{suffix}"
        if not file_path.exists() or file_path.stat().st_size == 0:
            try:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=30) as resp:
                    file_path.write_bytes(resp.read())
            except Exception:
                continue
        if file_path.exists() and file_path.stat().st_size > 5000:
            downloaded.append(file_path)

    return downloaded


def extract_panel_text(panel_path: Path) -> str:
    try:
        with Image.open(panel_path) as im:
            txt = pytesseract.image_to_string(im).strip()
            return " ".join(txt.split())
    except Exception:
        return ""


COMMON_VOCAB = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he",
    "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about",
    "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "sect", "mount", "hua", "great", "sword", "saint", "demon", "demons", "chunma", "beggar",
    "temple", "shaolin", "wudang", "province", "fallen", "martial", "dead", "alive", "hundred", "years", "ten",
    "hunter", "dungeon", "system", "level", "rank", "mana", "magic", "skill", "quest", "gate", "monster",
    "shadow", "monarch", "king", "emperor", "lord", "dragon", "beast", "strike", "blade", "power", "strength",
    "awakened", "reborn", "regressed", "reincarnation", "master", "disciple", "clan", "family", "leader",
    "technique", "aura", "qi", "cultivation", "world", "domain", "realm", "tower", "god", "devil", "hero"
}

ORDINALS_MAP = {
    '1st': 'first', '2nd': 'second', '3rd': 'third', '4th': 'fourth', '5th': 'fifth',
    '6th': 'sixth', '7th': 'seventh', '8th': 'eighth', '9th': 'ninth', '10th': 'tenth',
    '11th': 'eleventh', '12th': 'twelfth', '13th': 'thirteenth', '14th': 'fourteenth',
    '15th': 'fifteenth', '16th': 'sixteenth', '17th': 'seventeenth', '18th': 'eighteenth',
    '19th': 'nineteenth', '20th': 'twentieth', '21st': 'twenty-first', '22nd': 'twenty-second',
    '23rd': 'twenty-third', '24th': 'twenty-fourth', '25th': 'twenty-fifth', '30th': 'thirtieth',
    '40th': 'fortieth', '50th': 'fiftieth', '100th': 'hundredth'
}

NUMBERS_MAP = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
    12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',
    17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty', 21: 'twenty-one',
    22: 'twenty-two', 23: 'twenty-three', 24: 'twenty-four', 25: 'twenty-five',
    30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty', 70: 'seventy',
    80: 'eighty', 90: 'ninety', 100: 'one hundred', 1000: 'one thousand'
}

TRANSLATOR_PATTERNS = [
    r'\b(?:asura|flame\s*scans|manga\s*stream|reaper\s*scans|void\s*scans|mgeko|bato)\b',
    r'\b(?:discord\.gg|patreon|ko-fi|paypal|t\.me\/)\b',
    r'\b(?:translated\s*by|typesetter|proofreader|cleaner|redrawer|raw\s*provider)\b',
    r'\b(?:tl\s*:|pr\s*:|ed\s*:|ts\s*:|rd\s*:)\b',
    r'\b(?:visit\s*(?:our)?\s*site|read\s*at|join\s*our\s*discord)\b',
    r'\b(?:chapter\s*\d+|end\s*of\s*chapter|to\s*be\s*continued)\b',
    r'\b(?:all\s*rights\s*reserved|do\s*not\s*re-?upload)\b',
]


def is_translator_note(text: str) -> bool:
    if not text:
        return True
    t_lower = text.lower()
    for pat in TRANSLATOR_PATTERNS:
        if re.search(pat, t_lower):
            return True
    return False


VOWELS_SET = set('aeiouyAEIOUY')


def is_clean_english_word(w: str) -> bool:
    if not w:
        return False
    if re.search(r'[A-Za-z][\.\-][A-Za-z]', w):
        return False
    w_clean = re.sub(r'[^a-zA-Z]', '', w)
    if not w_clean:
        return False
    if len(w_clean) == 1:
        return w_clean.lower() in ('a', 'i')
    # Manga lettering is almost entirely uppercase, so casing says nothing about
    # whether a word is real - only the vowel/consonant shape does.
    if re.search(r'[a-z][A-Z]', w_clean):
        return False
    if not any(c in VOWELS_SET for c in w_clean):
        return False
    if re.search(r'[^aeiouyAEIOUY]{4,}', w_clean):
        return False
    w_lower = w_clean.lower()
    if w_lower in COMMON_VOCAB:
        return True
    vowel_ratio = sum(1 for c in w_clean if c in VOWELS_SET) / len(w_clean)
    return 0.20 <= vowel_ratio <= 0.75


def sanitize_narration(raw_text: str) -> str:
    if not raw_text:
        return ""
    t = raw_text
    t = re.sub(r'\b(?:\w+\-)+\w+\b', ' ', t)
    for ord_key, ord_val in ORDINALS_MAP.items():
        t = re.sub(rf'\b{ord_key}\b', ord_val, t, flags=re.IGNORECASE)
    for num_key, num_val in sorted(NUMBERS_MAP.items(), key=lambda x: -len(str(x[0]))):
        t = re.sub(rf'\b{num_key}\b', num_val, t)

    t = re.sub(r'\b(?:sex|sexes)\b', 'sects', t, flags=re.IGNORECASE)
    t = re.sub(r'\b10-grade\b', 'Ten Great', t, flags=re.IGNORECASE)
    t = re.sub(r'\b10 Great\b', 'Ten Great', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(?:[A-Za-z]\-)+[A-Za-z]\b', ' ', t)
    t = re.sub(r'\b(?:[A-Za-z][\.\s]){2,}[A-Za-z]?\b', ' ', t)
    t = re.sub(r'\d+\s*\%', ' ', t)
    # OCR reads the pronoun "I" in manga lettering as a pipe or slash, so recover it
    # before the symbol strip below removes those characters outright.
    t = re.sub(r'(^|\s)[\|\/](\s|$)', r'\1I\2', t)
    t = re.sub(r'[\=\>\<\|\~\%\$\\\@\^\*\_\+\#\{\}\[\]\/\"\`\—\–0-9]+', ' ', t)
    t = re.sub(r'\?[A-Za-z0-9\s\!]+', '?', t)
    t = re.sub(r'\!+[A-Za-z0-9\s\?]+', '!', t)
    t = re.sub(r'[\?\!]+', lambda m: '?' if '?' in m.group(0) else '!', t)
    t = re.sub(r'\.{2,}', '...', t)

    words = t.split()
    valid_words = []
    for w in words:
        if is_clean_english_word(w):
            clean_w = re.sub(r'[^a-zA-Z0-9\'\.\,\?\!\-]', '', w)
            if clean_w:
                valid_words.append(clean_w)

    # Four words is the shortest thing that reads as a narration line; below that
    # OCR fragments ("Awl IT") survive the per-word checks but are meaningless.
    if len(valid_words) < 4:
        return ""
    if (len(valid_words) / len(words)) < 0.60:
        return ""

    # Scanlation banners repeat the group's name and outlive the per-word filters.
    # Only content words count - "the"/"to"/"of" repeat freely in real sentences.
    stopwords = {"the", "to", "of", "a", "i", "and", "in", "is", "it", "you",
                 "that", "this", "for", "be", "my", "me", "at", "on", "as", "if"}
    content = [w.lower().strip(".,!?'") for w in valid_words]
    content = [w for w in content if w and w not in stopwords]
    if content and (len(content) - len(set(content))) >= 3:
        return ""

    text = " ".join(valid_words).strip()
    if text.isupper():
        sentences = re.split(r'([\.\?\!]\s*)', text)
        res = []
        for s in sentences:
            if s and not re.match(r'^[\.\?\!]\s*$', s):
                s = s.strip().capitalize()
                s = re.sub(r'\bi\b', 'I', s)
                s = re.sub(r'\bi\'', "I'", s)
                res.append(s)
            else:
                res.append(s)
        text = "".join(res)
    return text


# Filler narration for panels with no readable dialogue. Deliberately free of
# character and place names so the same pool works for any series.
DYNAMIC_ACTION_BEATS = [
    "The scene shifts, and the weight of the moment settles in.",
    "A tense silence stretches out before anyone dares to move.",
    "The air grows heavy as the confrontation draws closer.",
    "Every gaze narrows, fixed on what comes next.",
    "The tension builds with nowhere left for it to go.",
    "A sudden surge of force changes the shape of the fight.",
    "The balance tips, and neither side can take it back.",
    "What happens next will decide how this ends.",
    "The moment hangs suspended, refusing to break.",
    "Resolve hardens in the face of impossible odds.",
    "The ground seems to shudder under the sheer pressure.",
    "A single movement is all it takes to shatter the calm.",
    "The stakes climb higher with every passing second.",
    "Something unspoken passes between them.",
    "The fight turns, and the advantage slips away.",
    "A quiet breath comes before the storm breaks.",
    "The path ahead narrows to a single choice.",
    "Strength gathers for one decisive strike.",
    "The outcome balances on a knife's edge.",
    "Nothing about this will be settled easily.",
    "The pressure mounts until something has to give.",
    "A cold realization spreads through the scene.",
    "The clash erupts with nothing held in reserve.",
    "Whatever comes next, there is no turning back.",
]


def are_phrases_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_clean = a.lower().strip()
    b_clean = b.lower().strip()
    if a_clean == b_clean:
        return True
    if len(a_clean) > 10 and (a_clean in b_clean or b_clean in a_clean):
        return True
    return False


def are_images_duplicate(img1_path: Path, img2_path: Path) -> bool:
    try:
        im1 = Image.open(img1_path).convert("L").resize((32, 32))
        im2 = Image.open(img2_path).convert("L").resize((32, 32))
        arr1 = np.array(im1, dtype=np.float32)
        arr2 = np.array(im2, dtype=np.float32)
        return float(np.mean(np.abs(arr1 - arr2))) < 6.0
    except Exception:
        return False


def convert_dialogue_to_recap_sentence(dialogue: str, page_num: int, total_pages: int) -> str:
    """
    Turns a panel's dialogue into a narration line. The dialogue is kept in its own
    voice rather than rewritten into third person: rewriting needs to know who is
    speaking, and OCR gives no speaker, so it produced broken lines like "he DON'T
    WANT" and attributed every line to one hardcoded protagonist.
    """
    d = dialogue.strip()
    if not d:
        return ""
    m = re.search(r'([\.\!\?]+)$', d)
    end = "."
    if m:
        end = "?" if "?" in m.group(1) else "!" if "!" in m.group(1) else "."
    rephrased = re.sub(r'[\.\!\?]+$', '', d).strip()
    if not rephrased:
        return ""

    # Manga lettering is all caps; read it as ordinary prose so the voice doesn't
    # shout and the saved script stays readable.
    letters = [c for c in rephrased if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        rephrased = rephrased.lower()
        rephrased = re.sub(r"\bi\b", "I", rephrased)
        rephrased = re.sub(r"\bi'", "I'", rephrased)
        rephrased = re.sub(
            r'(^|[\.\!\?]\s+)([a-z])', lambda mm: mm.group(1) + mm.group(2).upper(), rephrased
        )

    rephrased = rephrased[0].upper() + rephrased[1:]
    return rephrased + end


def is_promo_cover(img: Image.Image, raw_text: str, height: int) -> bool:
    if height < 2200:
        t_low = raw_text.lower()
        if any(w in t_low for w in ['discord', 'proofreader', 'typesetter', 'cleaner', 'redrawer', 'click on this image', 'patreon']):
            return True
    return False


def slice_into_panels(page_paths: list[Path], output_dir: Path) -> list[dict]:
    """
    Slices manga strips into individual panels in strict chronological sequence,
    extracting OCR text directly for each panel slice.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_slices = []

    for page_path in sorted(page_paths):
        try:
            img = Image.open(page_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size

        # Check for promo covers (small height with credit keywords)
        if h < 2200:
            preview_txt = pytesseract.image_to_string(img).strip()
            if is_promo_cover(img, preview_txt, h):
                continue

        arr = np.array(img)

        # Variance across horizontal rows to find natural gutters between panels
        row_std = np.std(arr, axis=1).mean(axis=1)
        row_mean = np.mean(arr, axis=(1, 2))
        is_cut = (row_std < 4.0) | ((row_mean < 15) & (row_std < 8)) | ((row_mean > 240) & (row_std < 8))

        gutters = []
        start = None
        for y in range(h):
            if is_cut[y]:
                if start is None:
                    start = y
            else:
                if start is not None:
                    if (y - start) >= 8:
                        gutters.append((start + y) // 2)
                    start = None
        if start is not None:
            gutters.append((start + h) // 2)

        cuts = [0]
        for g in gutters:
            if g - cuts[-1] >= 450:
                cuts.append(g)
        if h - cuts[-1] < 450 and len(cuts) > 1:
            cuts[-1] = h
        else:
            cuts.append(h)

        page_prefix = page_path.stem.replace("page-", "p")
        for idx in range(len(cuts) - 1):
            top = cuts[idx]
            bottom = cuts[idx + 1]
            if bottom - top < 250:
                continue

            crop = img.crop((0, top, w, bottom))
            name = f"{page_prefix}_panel_{idx+1:02d}.jpg"
            out_path = output_dir / name
            crop.save(out_path, quality=95)

            # Character presence and OCR extraction for this specific panel
            score, is_text_heavy = score_character_presence(out_path)
            raw_text = extract_panel_text(out_path)
            clean_text = ""
            if raw_text and not is_translator_note(raw_text):
                clean_text = sanitize_narration(raw_text)

            all_slices.append({
                "path": out_path,
                "score": score,
                "is_text_heavy": is_text_heavy,
                "text": clean_text,
                "page": page_prefix,
                "top": top,
                "bottom": bottom,
            })

    return all_slices


MAX_PANELS_PER_BEAT = 3
MAX_WORDS_PER_BEAT = 30
MIN_SECONDS_PER_PANEL = 1.8
_SECONDS_PER_WORD = 0.4


def _estimate_speech_seconds(text: str) -> float:
    return 0.6 + _SECONDS_PER_WORD * len(text.split())


def _narration_chunks(raw_text: str) -> list[str]:
    """
    Turns an OCR'd dialogue run into narration lines: each clause becomes a clean
    sentence (keeping its ? or ! so the voice gets the intonation right), and
    consecutive sentences are packed together up to a comfortable length so one
    still image isn't held for a whole paragraph.
    """
    parts = re.split(r'([\.\!\?]+)\s*', raw_text)
    sentences: list[str] = []
    for i in range(0, len(parts), 2):
        clause = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if len(clause.split()) < 2:
            continue
        end = "?" if "?" in punct else "!" if "!" in punct else ""
        sent = convert_dialogue_to_recap_sentence(clause + end, 1, 1)
        if sent:
            sentences.append(sent)

    chunks: list[str] = []
    current: list[str] = []
    for sent in sentences:
        if current and len(" ".join(current + [sent]).split()) > MAX_WORDS_PER_BEAT:
            chunks.append(" ".join(current))
            current = []
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def generate_story_beats(
    panel_slices: list[dict],
    chapter_title: str,
    aspect_ratio: str = "16:9"
) -> list[dict]:
    """
    Turns the ordered panel slices into narration beats. Each beat is one line of
    narration plus the panel(s) shown while it plays:

    - Dialogue is spoken over the panel it was read from when that panel has artwork.
    - Dialogue from a text-only bubble is held and spoken over the next artwork panel.
    - Wordless artwork panels ride along with a neighbouring line - shown before the
      dialogue panel they lead into, or after the line they follow - so the camera
      keeps moving while the narration continues. A filler line is used only when a
      stretch of wordless art has no dialogue nearby to ride on.
    """
    beats: list[dict] = []
    pending_text: list[str] = []
    orphan_panels: list[Path] = []
    last_narration = ""
    last_panel: Optional[Path] = None
    last_seen_panel: Optional[Path] = None
    action_idx = 0

    def add_beat(narration: str, panels: list[Path], primary: Path, page) -> dict:
        beat = {
            "id": f"beat_{len(beats) + 1:03d}",
            "panel": primary,
            "panels": list(panels),
            "narration": narration,
            "page": page,
        }
        beats.append(beat)
        return beat

    def panels_that_fit(narration: str, extra: int) -> int:
        """How many of `extra` wordless panels a line can carry without rushing."""
        fit = 0
        for n in range(1, extra + 1):
            if n + 1 > MAX_PANELS_PER_BEAT:
                break
            if _estimate_speech_seconds(narration) / (n + 1) < MIN_SECONDS_PER_PANEL:
                break
            fit = n
        return fit

    def flush_orphans_as_filler(page) -> None:
        nonlocal orphan_panels, last_narration, action_idx
        if not orphan_panels:
            return
        filler = DYNAMIC_ACTION_BEATS[action_idx % len(DYNAMIC_ACTION_BEATS)]
        action_idx += 1
        add_beat(filler, orphan_panels, orphan_panels[0], page)
        last_narration = filler
        orphan_panels = []

    for s in panel_slices:
        panel: Path = s["path"]
        is_bubble_only = s["is_text_heavy"] and s["score"] < 12.0
        last_seen_panel = panel

        if s["text"]:
            pending_text.append(s["text"])
        if is_bubble_only:
            continue
        if last_panel is not None and are_images_duplicate(panel, last_panel):
            continue
        last_panel = panel

        if pending_text:
            raw = " ".join(pending_text)
            pending_text = []
            chunks = [
                c for c in _narration_chunks(raw)
                if not (last_narration and are_phrases_similar(c, last_narration))
            ]
            if chunks:
                # Lead into the dialogue with the wordless panels that preceded it.
                lead_in = panels_that_fit(chunks[0], len(orphan_panels))
                if lead_in < len(orphan_panels):
                    kept = orphan_panels[len(orphan_panels) - lead_in:] if lead_in else []
                    orphan_panels = orphan_panels[:len(orphan_panels) - lead_in]
                    flush_orphans_as_filler(s["page"])
                    orphan_panels = kept
                add_beat(chunks[0], orphan_panels + [panel], panel, s["page"])
                orphan_panels = []
                for chunk in chunks[1:]:
                    add_beat(chunk, [panel], panel, s["page"])
                last_narration = chunks[-1]
                continue

        # Wordless artwork: ride along with the previous line if it has room.
        if beats and not orphan_panels:
            prev = beats[-1]
            if panels_that_fit(prev["narration"], len(prev["panels"])) >= len(prev["panels"]):
                prev["panels"].append(panel)
                continue

        orphan_panels.append(panel)
        if len(orphan_panels) >= MAX_PANELS_PER_BEAT:
            flush_orphans_as_filler(s["page"])

    if orphan_panels:
        if beats:
            beats[-1]["panels"].extend(orphan_panels[: MAX_PANELS_PER_BEAT - 1])
            orphan_panels = orphan_panels[MAX_PANELS_PER_BEAT - 1:]
        flush_orphans_as_filler(panel_slices[-1]["page"] if panel_slices else "")

    if pending_text:
        # Falls back to the last bubble-only panel seen when a whole chapter never
        # produced an art panel to hang narration on (e.g. an all-dialogue page) -
        # without this, that dialogue was silently dropped and the chapter failed
        # with "no narration beats could be generated".
        tail_panel = beats[-1]["panel"] if beats else last_seen_panel
        tail_page = beats[-1]["page"] if beats else (panel_slices[-1]["page"] if panel_slices else "")
        if tail_panel is not None:
            for chunk in _narration_chunks(" ".join(pending_text)):
                if last_narration and are_phrases_similar(chunk, last_narration):
                    continue
                add_beat(chunk, [tail_panel], tail_panel, tail_page)
                last_narration = chunk

    return beats


def _llm_json(client, prompt: str, max_tokens: int, temperature: float = 0.5,
              model: Optional[str] = None):
    """Calls the model and parses a JSON reply, tolerating ```json fences."""
    response = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    import json
    return json.loads(raw)


def _comprehend_chapter(client, chapter_title: str, panel_manifest: str,
                        model: Optional[str] = None) -> Optional[dict]:
    """
    Pass 1: read the whole chapter and work out what actually happens in it.

    OCR gives unattributed, often garbled speech fragments. Narrating those line by
    line produces disjointed text that doesn't track the story, so first the model
    reconstructs the chapter - who is in it, what happens, in what order - and that
    understanding is what gets narrated in pass 2.
    """
    prompt = (
        f'Below is OCR text extracted from the panels of a manga/manhwa chapter titled '
        f'"{chapter_title}", in reading order. The OCR is noisy: words may be misspelled, '
        f'speech may be split across panels or merged out of order, and speakers are never '
        f'labelled. ART panels have no text.\n\n'
        f'{panel_manifest}\n\n'
        f'Read the whole chapter and work out what is actually happening. Infer who is '
        f'speaking from context, correct obvious OCR corruption, and ignore watermarks, '
        f'credits and scanlation notes.\n\n'
        f'Respond with ONLY JSON:\n'
        f'{{"characters": [{{"name": "<name as best you can tell>", "role": "<one phrase>"}}],\n'
        f'  "setting": "<one sentence on where/when this takes place>",\n'
        f'  "synopsis": "<5-8 sentences telling what happens in this chapter, in order, '
        f'as a story>",\n'
        f'  "beats": ["<short phrase for each major story beat, in order>"]}}\n'
        f'If a character\'s name is never legible, describe them instead ("the masked swordsman"). '
        f'Never invent plot that the panels do not support.'
    )
    try:
        data = _llm_json(client, prompt, max_tokens=1200, temperature=0.3, model=model)
        if isinstance(data, dict) and data.get("synopsis"):
            return data
    except Exception as exc:
        print(f"[WARN] Chapter comprehension failed: {exc}")
    return None


def generate_llm_story_beats(
    panel_slices: list[dict],
    chapter_title: str,
    aspect_ratio: str = "16:9",
    story_context: Optional[dict] = None,
) -> Optional[tuple[list[dict], Optional[dict]]]:
    """
    Writes the narration by first understanding the chapter, then telling it.

    Pass 1 reconstructs the story from the noisy OCR. Pass 2 writes continuous
    third-person narration of that story and assigns each sentence to the panel
    range it describes, so the voice and the art stay in step.

    Returns (beats, chapter_summary) or None when no API key is set or a call
    fails, in which case the caller falls back to generate_story_beats().

    `story_context` carries what happened in previous chapters so narration across a
    multi-chapter video reads as one continuous story.
    """
    provider = resolve_llm_provider()
    if provider is None:
        return None
    api_key, base_url, model = provider

    candidates = [s for s in panel_slices if s["text"] or not s["is_text_heavy"]]
    if not candidates:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    lines = []
    for i, s in enumerate(candidates):
        kind = "DIALOGUE" if s["text"] else "ART"
        text = s["text"] if s["text"] else "(no text - character/action artwork)"
        lines.append(f"{i}. [{kind}] {text}")
    panel_manifest = "\n".join(lines)

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    except Exception:
        return None

    understanding = _comprehend_chapter(client, chapter_title, panel_manifest, model)
    if not understanding:
        return None

    cast = ", ".join(
        f"{c.get('name', '?')} ({c.get('role', '')})".strip()
        for c in understanding.get("characters", [])[:8]
    )

    previously = ""
    if story_context and story_context.get("running_summary"):
        previously = (
            f"PREVIOUSLY IN THE STORY (do not re-narrate, use only for continuity "
            f"and to keep character names consistent):\n{story_context['running_summary']}\n\n"
        )

    last_index = len(candidates) - 1
    prompt = (
        f'{previously}'
        f'You are the narrator of a manga recap video. This is what happens in the '
        f'current chapter:\n\n'
        f'CAST: {cast}\n'
        f'SETTING: {understanding.get("setting", "")}\n'
        f'WHAT HAPPENS: {understanding.get("synopsis", "")}\n\n'
        f'These are the chapter\'s panels in reading order, with their OCR text:\n\n'
        f'{panel_manifest}\n\n'
        f'Narrate this chapter as one continuous third-person story, the way a good recap '
        f'channel tells it, and split that narration across the panels so each sentence is '
        f'spoken while the art it describes is on screen.\n\n'
        f'Rules:\n'
        f'- Third person, past tense. Never "I" or "you" - convert dialogue into narration '
        f'("Itadori admitted he had never seen a curse before"), do not quote it raw.\n'
        f'- It must read as a flowing story, not panel-by-panel captions. Connect events '
        f'with cause and consequence.\n'
        f'- Name characters instead of saying "he"/"the man" wherever you can.\n'
        f'- Each entry covers a consecutive run of panels: "from" to "to" inclusive. The runs '
        f'must be in ascending order, must not overlap, and together must cover panels 0 to '
        f'{last_index} with no gaps.\n'
        f'- Give a wordless stretch of action panels a single sentence spanning that run '
        f'rather than one sentence per panel.\n'
        f'- 12 to 30 words per sentence - long enough to sound natural read aloud.\n'
        f'- Do not mention panels, pages, art or the reader. Never narrate the OCR noise.\n\n'
        f'Respond with ONLY a JSON array:\n'
        f'[{{"from": <int>, "to": <int>, "narration": "<sentence>"}}, ...]'
    )

    try:
        parsed = _llm_json(client, prompt, max_tokens=4000, temperature=0.6, model=model)
    except Exception as exc:
        print(f"[WARN] LLM narration failed: {exc}")
        return None
    if not isinstance(parsed, list):
        return None

    story_beats: list[dict] = []
    covered = 0
    for item in parsed:
        try:
            start = int(item["from"])
            end = int(item["to"])
            narration = str(item["narration"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not narration:
            continue

        # Clamp to the real panel range and keep runs moving forward, so a model
        # slip can't drop panels or show them out of order.
        start = max(start, covered)
        end = max(min(end, last_index), start)
        if start > last_index:
            continue

        panels: list[Path] = []
        for s in candidates[start:end + 1]:
            if panels and are_images_duplicate(s["path"], panels[-1]):
                continue
            panels.append(s["path"])
        if not panels:
            continue

        story_beats.append({
            "id": f"beat_{len(story_beats)+1:03d}",
            "panel": panels[0],
            "panels": panels,
            "narration": narration,
            "page": candidates[start]["page"],
        })
        covered = end + 1

    if not story_beats:
        return None

    # Any panels the model left off the end still get shown, under the last line.
    if covered <= last_index:
        for s in candidates[covered:]:
            if not are_images_duplicate(s["path"], story_beats[-1]["panels"][-1]):
                story_beats[-1]["panels"].append(s["path"])

    return story_beats, understanding


def get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(float(result.stdout.strip()), 0.8)


# ---------------------------------------------------------------------------
# Text-to-speech
#
# Two engines. Kokoro (open-source, runs fully offline on this machine) is the
# preferred one - it is markedly more natural than the network voices. edge-tts
# (Microsoft neural voices) is the fallback when Kokoro isn't installed. Every
# clip then goes through the same finishing pass so loudness, silence trimming
# and sample rate are identical whichever engine produced it.
# ---------------------------------------------------------------------------

# Kokoro voice id -> closest edge-tts voice, used when Kokoro is unavailable.
KOKORO_VOICES = {
    "am_michael": "en-US-AndrewMultilingualNeural",
    "am_fenrir": "en-US-GuyNeural",
    "am_adam": "en-US-BrianMultilingualNeural",
    "am_onyx": "en-US-ChristopherNeural",
    "af_heart": "en-US-AvaMultilingualNeural",
    "af_bella": "en-US-AriaNeural",
    "bm_george": "en-GB-RyanNeural",
    "bm_fable": "en-GB-RyanNeural",
    "bf_emma": "en-GB-SoniaNeural",
}

_kokoro_pipelines: dict = {}
_kokoro_fallback_warned = False


def kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
        import soundfile  # noqa: F401
        return True
    except Exception:
        return False


def _speed_rate_to_factor(speed_rate: str) -> float:
    m = re.match(r'^\s*([+-]?\d+)\s*%\s*$', speed_rate)
    return 1.0 + int(m.group(1)) / 100.0 if m else 1.0


def _normalise_edge_rate(speed_rate: str) -> str:
    # edge-tts rejects an unsigned rate like "0%".
    s = speed_rate.strip()
    return s if s.startswith(("+", "-")) else f"+{s}"


def _kokoro_pipeline(lang_code: str):
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline
        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]


def _synthesize_kokoro(text: str, wav_path: Path, voice: str, speed: float) -> None:
    import soundfile as sf

    pipeline = _kokoro_pipeline(voice[0])  # 'a' = American English, 'b' = British English
    chunks = []
    for item in pipeline(text, voice=voice, speed=speed):
        audio = getattr(item, "audio", None)
        if audio is None and isinstance(item, (tuple, list)):
            audio = item[2]
        if audio is None:
            continue
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size:
            chunks.append(audio)
    if not chunks:
        raise RuntimeError("Kokoro produced no audio for this line")
    sf.write(str(wav_path), np.concatenate(chunks), 24000)


def _finalize_voice_clip(raw_path: Path, output_path: Path) -> None:
    """
    Trims dead air from both ends (keeping a short breath so words aren't clipped),
    normalises to broadcast loudness, adds a brief pause after the line, and writes
    48 kHz mono PCM so the render step never re-decodes lossy audio.
    """
    af = (
        "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,"
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.12,areverse,"
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        "apad=pad_dur=0.35,"
        "aresample=48000"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path), "-af", af, "-ac", "1", "-ar", "48000",
         "-c:a", "pcm_s16le", str(output_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


async def synthesize_voiceover(
    text: str,
    output_path: Path,
    voice: str = "am_michael",
    speed_rate: str = "+0%",
    retries: int = 3,
) -> None:
    """
    Both engines occasionally fail transiently on a single line (dropped connection,
    empty audio); without the retry that exception used to propagate up and silently
    drop the whole chapter from the merged video.
    """
    global _kokoro_fallback_warned
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_kokoro = voice in KOKORO_VOICES or ("_" in voice and "-" not in voice)
    if use_kokoro and not kokoro_available():
        fallback = KOKORO_VOICES.get(voice, "en-US-AndrewMultilingualNeural")
        if not _kokoro_fallback_warned:
            print(f"[WARN] Kokoro TTS is not installed - using edge-tts voice {fallback} instead "
                  f"(pip install kokoro soundfile to enable the local voice).")
            _kokoro_fallback_warned = True
        voice, use_kokoro = fallback, False

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            if use_kokoro:
                raw_path = output_path.with_suffix(".raw.wav")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, _synthesize_kokoro, text, raw_path, voice, _speed_rate_to_factor(speed_rate)
                )
            else:
                raw_path = output_path.with_suffix(".raw.mp3")
                communicate = edge_tts.Communicate(text, voice, rate=_normalise_edge_rate(speed_rate))
                await communicate.save(str(raw_path))
            _finalize_voice_clip(raw_path, output_path)
            raw_path.unlink(missing_ok=True)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
    raise last_exc


VIDEO_FPS = 30
# The push-in zooms from 1.0 to this; frames are composed slightly larger than the
# output so the zoomed crop never has to upscale past the panel's native pixels.
KEN_BURNS_ZOOM = 1.06


def compose_panel_frame(panel_path: Path, width: int, height: int, frames_dir: Path) -> Path:
    """
    Builds the full video frame for a panel - blurred, darkened cover-fit background
    with the panel contain-fit on top - as a cached image. Doing this once per panel
    in PIL (instead of inside every ffmpeg call) keeps the render filter graph a single
    uniform image stream, which is what makes the multi-panel zoom reliable.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    canvas_w = int(width * KEN_BURNS_ZOOM * 1.02) // 2 * 2
    canvas_h = int(height * KEN_BURNS_ZOOM * 1.02) // 2 * 2
    out = frames_dir / f"{panel_path.stem}_{canvas_w}x{canvas_h}.jpg"
    if out.exists():
        return out

    img = Image.open(panel_path).convert("RGB")
    bg = ImageOps.fit(img, (canvas_w, canvas_h), method=Image.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=max(14, canvas_w // 60)))
    bg = ImageEnhance.Brightness(bg).enhance(0.5)
    fg = ImageOps.contain(img, (canvas_w, canvas_h), method=Image.LANCZOS)
    bg.paste(fg, ((canvas_w - fg.width) // 2, (canvas_h - fg.height) // 2))
    bg.save(out, quality=93)
    return out


def render_frame_video(
    frame_paths,
    audio_path: Path,
    output_path: Path,
    duration: float,
    width: int = 1080,
    height: int = 1920,
) -> None:
    """
    Renders one narration beat: the line plays once while its frame(s) are shown in
    sequence, each with a slow push-in.

    The output is frame-exact. The video track is rounded up to a whole number of
    frames and the audio is padded to precisely that length - a mismatch of even a
    single frame per segment compounds across the hundred-odd segments in a chapter
    into clearly audible drift between the voice and the panels.
    """
    if isinstance(frame_paths, (str, Path)):
        frame_paths = [frame_paths]
    frames = [Path(p) for p in frame_paths]
    if not frames:
        raise ValueError("render_frame_video needs at least one frame")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames_per_panel = max(VIDEO_FPS // 2, math.ceil(duration * VIDEO_FPS / len(frames)))
    total_frames = frames_per_panel * len(frames)
    total_duration = total_frames / VIDEO_FPS

    list_path = output_path.with_suffix(".frames.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in frames:
            f.write(f"file '{p.resolve()}'\n")

    zoom_step = (KEN_BURNS_ZOOM - 1.0) / max(frames_per_panel - 1, 1)
    vf = (
        f"[0:v]setsar=1,"
        f"zoompan=z='1+{zoom_step:.6f}*on':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames_per_panel}:s={width}x{height}:fps={VIDEO_FPS},format=yuv420p[outv];"
        f"[1:a]apad=whole_dur={total_duration:.4f},aresample=48000[outa]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-i", str(audio_path),
        "-filter_complex", vf,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(VIDEO_FPS), "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{total_duration:.4f}",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    list_path.unlink(missing_ok=True)


def _probe_video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    value = result.stdout.strip()
    if not value or value == "N/A":
        return get_audio_duration(path)
    return float(value)


def concat_video_segments(segment_paths: list[Path], final_output: Path) -> None:
    """
    Joins segments into one video. The video track is concatenated and re-encoded
    (so differing resolutions/frame rates can't silently break it), but the audio is
    rebuilt from each segment's decoded audio padded to that segment's exact video
    length and joined as PCM. Letting ffmpeg concatenate the AAC streams directly
    adds roughly 10 ms of encoder priming at every join, which over the hundred-plus
    segments in a chapter drifted the voice a full second out of sync by the end.
    """
    if not segment_paths:
        raise RuntimeError(
            f"No video segments to assemble for '{final_output.name}' - "
            f"nothing was rendered upstream (check scraping/panel-slicing/narration steps)."
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = final_output.parent / f".concat_{final_output.stem}"
    work_dir.mkdir(exist_ok=True)

    video_list = work_dir / "video.txt"
    audio_list = work_dir / "audio.txt"
    with open(video_list, "w", encoding="utf-8") as vf, open(audio_list, "w", encoding="utf-8") as af:
        for i, p in enumerate(segment_paths):
            video_seconds = _probe_video_duration(p)
            wav = work_dir / f"{i:04d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(p), "-vn",
                 "-af", f"apad=whole_dur={video_seconds:.4f}", "-t", f"{video_seconds:.4f}",
                 "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            vf.write(f"file '{p.resolve()}'\n")
            af.write(f"file '{wav.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(video_list),
        "-f", "concat", "-safe", "0", "-i", str(audio_list),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(VIDEO_FPS), "-fps_mode", "cfr",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(final_output),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    shutil.rmtree(work_dir, ignore_errors=True)


def discover_series_chapter_videos(output_dir: Path) -> list[Path]:
    """
    Finds every rendered chapter_XXX_video.mp4 already sitting in a series/aspect-ratio
    output folder (including ones from earlier runs), sorted by chapter number, so a
    later run adding new chapters re-merges the whole series instead of just the new part.
    """
    videos = list(output_dir.glob("chapter_*_video.mp4"))

    def chapter_num(p: Path) -> float:
        m = re.search(r"chapter_(\d+(?:\.\d+)?)_video\.mp4$", p.name)
        return float(m.group(1)) if m else float("inf")

    return sorted(videos, key=chapter_num)


async def process_single_chapter(
    chapter_info: dict,
    workspace_dir: Path,
    output_dir: Path,
    voice: str = "en-US-ChristopherNeural",
    speed_rate: str = "+25%",
    aspect_ratio: str = "16:9",
    use_llm: bool = True,
    enable_ai_context: bool = False,
    story_context: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> Path:
    ch_num = chapter_info["number"]
    ch_url = chapter_info["url"]
    ch_title = chapter_info["title"]
    
    ch_str = f"{int(ch_num):03d}" if isinstance(ch_num, (int, float)) and float(ch_num).is_integer() else f"{ch_num}"
    
    ch_work_dir = workspace_dir / f"chapter_{ch_str}"
    input_dir = ch_work_dir / "input"
    panels_dir = ch_work_dir / "panels"
    audio_dir = ch_work_dir / "audio"
    segments_dir = ch_work_dir / "segments"
    
    ch_work_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(f"Scraping Chapter {ch_num} images...", 15)
        
    pages = scrape_chapter_images(ch_url, input_dir)

    if not pages:
        raise RuntimeError(
            f"Chapter {ch_num}: no images could be scraped from {ch_url} - "
            f"the page may not have loaded, or its markup doesn't match what the scraper expects."
        )

    if progress_callback:
        progress_callback(f"Detecting and slicing character panels for Chapter {ch_num}...", 35)

    panel_entries = slice_into_panels(pages, panels_dir)

    if not panel_entries:
        raise RuntimeError(
            f"Chapter {ch_num}: {len(pages)} page image(s) were scraped but no panels could be "
            f"sliced from them - the images may be corrupt, too small, or an unexpected layout."
        )

    if progress_callback:
        progress_callback(f"Generating YouTube recap voiceover script for Chapter {ch_num}...", 55)

    beats = None
    if use_llm:
        if progress_callback:
            progress_callback(f"Reading Chapter {ch_num} and writing the story...", 55)
        result = generate_llm_story_beats(
            panel_entries, ch_title, aspect_ratio=aspect_ratio, story_context=story_context
        )
        if result is not None:
            beats, understanding = result
            if story_context is not None and understanding:
                # Carry this chapter forward so later chapters narrate as one story.
                story_context.setdefault("chapters", []).append(
                    f"Chapter {ch_num}: {understanding.get('synopsis', '')}"
                )
                story_context["running_summary"] = "\n".join(story_context["chapters"][-6:])
                for c in understanding.get("characters", []):
                    name = (c.get("name") or "").strip()
                    if name:
                        story_context.setdefault("cast", {})[name] = c.get("role", "")
    if beats is None:
        beats = generate_story_beats(panel_entries, ch_title, aspect_ratio=aspect_ratio)

    if not beats:
        raise RuntimeError(
            f"Chapter {ch_num}: {len(panel_entries)} panel(s) were sliced but no narration beats "
            f"could be generated from them (no usable text/art detected)."
        )

    width = 1920 if aspect_ratio == "16:9" else 1080
    height = 1080 if aspect_ratio == "16:9" else 1920

    ai_context_map = {}
    if enable_ai_context:
        try:
            from ai_context_generator import identify_and_generate_ai_context
            if progress_callback:
                progress_callback(f"Generating bespoke AI character & scene context for Chapter {ch_num}...", 65)
            ai_context_map = identify_and_generate_ai_context(
                beats=beats,
                chapter_title=ch_title,
                ch_work_dir=ch_work_dir,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            print(f"[WARN] AI context generation skipped: {exc}")

    frames_dir = ch_work_dir / "frames"
    script_path = output_dir / f"chapter_{ch_str}_script.txt"
    script_path.write_text(
        f"{ch_title}\n\n" + "\n".join(b["narration"] for b in beats) + "\n", encoding="utf-8"
    )

    segment_paths = []
    total_beats = len(beats)
    
    for i, beat in enumerate(beats, start=1):
        beat_id = beat["id"]
        panel_path = beat["panel"]
        narration = beat["narration"]

        audio_path = audio_dir / f"{beat_id}.wav"
        await synthesize_voiceover(narration, audio_path, voice=voice, speed_rate=speed_rate)
        duration = get_audio_duration(audio_path)

        segment_path = segments_dir / f"{beat_id}.mp4"
        if beat_id in ai_context_map:
            from ai_context_generator import render_enhanced_segment
            render_enhanced_segment(
                ai_context_map[beat_id],
                audio_path,
                segment_path,
                duration,
                width=width,
                height=height,
                has_zoom=True,
            )
        else:
            frame_paths = [
                compose_panel_frame(Path(p), width, height, frames_dir)
                for p in (beat.get("panels") or [panel_path])
            ]
            render_frame_video(frame_paths, audio_path, segment_path, duration, width=width, height=height)
        segment_paths.append(segment_path)

        if progress_callback and (i % 5 == 0 or i == total_beats):
            pct = 55 + int((i / total_beats) * 35)
            progress_callback(f"Rendered recap beat {i}/{total_beats} for Chapter {ch_num}...", pct)

    if progress_callback:
        progress_callback(f"Assembling Chapter {ch_num} video...", 95)

    chapter_video_path = output_dir / f"chapter_{ch_str}_video.mp4"
    concat_video_segments(segment_paths, chapter_video_path)

    return chapter_video_path


async def process_multi_chapters(
    start_url: str,
    max_chapters: int = 1,
    voice: str = "en-US-ChristopherNeural",
    speed_rate: str = "+25%",
    aspect_ratio: str = "16:9",
    use_llm: bool = True,
    enable_ai_context: bool = False,
    output_dir: Path = Path("output"),
    workspace_dir: Path = Path("workspace"),
    merged_only: bool = False,
    chapter_start: Optional[float] = None,
    chapter_end: Optional[float] = None,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> tuple[list[Path], Path]:
    all_chapters = discover_all_chapters(start_url)

    if chapter_start is not None or chapter_end is not None:
        # Explicit chapter range (e.g. 10-20): filter by the real chapter number
        # rather than a positional slice, so it's correct even with numbering gaps
        # (a missing chapter, a "chapter 18.5" side story, etc).
        lo = chapter_start if chapter_start is not None else float("-inf")
        hi = chapter_end if chapter_end is not None else float("inf")
        to_process = [c for c in all_chapters if lo <= c["number"] <= hi]
    else:
        # Locate starting chapter index
        start_idx = 0
        clean_start = start_url.rstrip("/").lower()
        for idx, c in enumerate(all_chapters):
            if c["url"].rstrip("/").lower() == clean_start:
                start_idx = idx
                break

        to_process = all_chapters[start_idx : start_idx + max_chapters] if max_chapters > 0 else all_chapters[start_idx:]

    if not to_process:
        available = ", ".join(str(c["number"]) for c in all_chapters[:10])
        raise RuntimeError(
            f"No chapters found in the requested range ({len(all_chapters)} chapters discovered total"
            f"{f'; first few: {available}...' if available else ''})."
        )

    chapter_videos: list[Path] = []
    total = len(to_process)
    story_context: dict = {}

    for idx, ch in enumerate(to_process, start=1):
        if progress_callback:
            progress_callback(f"Starting Chapter {ch['number']} ({idx}/{total})...", int(((idx - 1) / total) * 100))

        video_path = await process_single_chapter(
            chapter_info=ch,
            workspace_dir=workspace_dir,
            output_dir=output_dir,
            voice=voice,
            speed_rate=speed_rate,
            aspect_ratio=aspect_ratio,
            use_llm=use_llm,
            enable_ai_context=enable_ai_context,
            story_context=story_context,
            progress_callback=progress_callback,
        )
        chapter_videos.append(video_path)

    # Merge every chapter video in this series/aspect-ratio folder (including ones
    # rendered in earlier runs) into a master combined video, so later runs that add
    # new chapters extend the same merged video instead of replacing it.
    merged_output_path = output_dir / "all_chapters_merged.mp4"
    if progress_callback:
        progress_callback("Merging all chapter videos into final master story video...", 98)

    all_series_videos = discover_series_chapter_videos(output_dir)
    if len(all_series_videos) == 1:
        shutil.copy(all_series_videos[0], merged_output_path)
    else:
        concat_video_segments(all_series_videos, merged_output_path)
    _write_series_script(output_dir)

    if merged_only:
        _remove_per_chapter_videos(all_series_videos, merged_output_path)

    if progress_callback:
        progress_callback("All chapters completed successfully!", 100)

    return chapter_videos, merged_output_path


def _write_series_script(output_dir: Path) -> Path:
    """Joins every chapter's narration script in this series folder into one file to post with the video."""
    def chapter_num(p: Path) -> float:
        m = re.search(r"chapter_(\d+(?:\.\d+)?)_script\.txt$", p.name)
        return float(m.group(1)) if m else float("inf")

    scripts = sorted(output_dir.glob("chapter_*_script.txt"), key=chapter_num)
    combined = output_dir / "all_chapters_script.txt"
    combined.write_text(
        "\n\n".join(p.read_text(encoding="utf-8").strip() for p in scripts) + "\n", encoding="utf-8"
    )
    return combined


def _remove_per_chapter_videos(chapter_videos: list[Path], merged_output_path: Path) -> None:
    """Deletes individual chapter_XXX_video.mp4 files once they're folded into the
    merged video, for users who only want the final combined output on disk.
    Refuses to delete anything if the merge didn't actually produce a real file,
    so a failed/silent merge can never leave the user with nothing in output/."""
    if not merged_output_path.exists() or merged_output_path.stat().st_size == 0:
        return
    for p in chapter_videos:
        if p.resolve() != merged_output_path.resolve() and p.exists():
            p.unlink()


async def process_chapter_urls(
    chapter_urls: list[str],
    voice: str = "en-US-ChristopherNeural",
    speed_rate: str = "+25%",
    aspect_ratio: str = "16:9",
    use_llm: bool = True,
    enable_ai_context: bool = False,
    output_dir: Path = Path("output"),
    workspace_dir: Path = Path("workspace"),
    merged_only: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> tuple[list[Path], Path]:
    """
    Like process_multi_chapters(), but takes an explicit list of chapter URLs
    instead of discovering a run of N chapters forward from a single start URL.
    Each URL gets its own rendered video; all of them are also concatenated
    into one merged master video, in the given order.
    """
    if not chapter_urls:
        raise ValueError("chapter_urls must contain at least one URL")

    chapter_videos: list[Path] = []
    failures: list[tuple[str, str]] = []
    total = len(chapter_urls)
    story_context: dict = {}

    existing_videos = discover_series_chapter_videos(output_dir)
    fallback_num = 0
    if existing_videos:
        m = re.search(r"chapter_(\d+)_video\.mp4$", existing_videos[-1].name)
        fallback_num = int(m.group(1)) if m else len(existing_videos)

    for offset, url in enumerate(chapter_urls, start=1):
        # Use the chapter number encoded in the URL itself (e.g. "chapter-11") so a
        # pasted list saves under its real chapter, not a positional 1, 2, 3, ...
        parsed_num = parse_chapter_number(url)
        if parsed_num is not None:
            idx = int(parsed_num) if parsed_num.is_integer() else parsed_num
        else:
            fallback_num += 1
            idx = fallback_num

        if progress_callback:
            progress_callback(f"Starting chapter {idx} ({offset}/{total})...", int(((offset - 1) / total) * 100))

        chapter_info = {"number": idx, "url": url, "title": f"Chapter {idx}"}
        try:
            video_path = await process_single_chapter(
                chapter_info=chapter_info,
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                voice=voice,
                speed_rate=speed_rate,
                aspect_ratio=aspect_ratio,
                use_llm=use_llm,
                enable_ai_context=enable_ai_context,
                story_context=story_context,
                progress_callback=progress_callback,
            )
            chapter_videos.append(video_path)
        except Exception as exc:
            failures.append((url, str(exc)))
            if progress_callback:
                progress_callback(f"Chapter {idx} failed: {exc}", int((offset / total) * 100))

    if not chapter_videos:
        raise RuntimeError(
            f"All {total} chapter URL(s) failed - no videos were produced. "
            f"First failure: {failures[0][1] if failures else 'unknown'}"
        )

    merged_output_path = output_dir / "all_chapters_merged.mp4"
    if progress_callback:
        progress_callback("Merging all chapter videos into final master story video...", 98)

    all_series_videos = discover_series_chapter_videos(output_dir)
    if len(all_series_videos) == 1:
        shutil.copy(all_series_videos[0], merged_output_path)
    else:
        concat_video_segments(all_series_videos, merged_output_path)
    _write_series_script(output_dir)

    if merged_only:
        _remove_per_chapter_videos(all_series_videos, merged_output_path)

    if progress_callback:
        msg = "All chapters completed successfully!"
        if failures:
            msg = f"Completed with {len(failures)} failure(s) out of {total} chapter(s)."
        progress_callback(msg, 100)

    return chapter_videos, merged_output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manga Story Video Generator CLI")
    parser.add_argument("--url", help="Manga chapter or series URL (discovers --chapters forward from here)")
    parser.add_argument(
        "--urls",
        nargs="+",
        help="One or more explicit chapter URLs to process (space-separated). "
             "Each gets its own video, plus one merged video of all of them, in the given order. "
             "Mutually exclusive with --url/--chapters.",
    )
    parser.add_argument("--chapters", type=int, default=1, help="Number of chapters to process (used with --url)")
    parser.add_argument("--voice", default="am_michael", help="TTS voice name (Kokoro voice like am_michael, or an edge-tts voice name)")
    parser.add_argument("--speed", default="+0%", help="TTS speed rate (e.g. +0% natural, +25% faster)")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"], help="Video aspect ratio")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (OpenAI, requires API_KEY in .env) for narration script generation instead of the regex-based generator",
    )
    parser.add_argument(
        "--ai-context",
        action="store_true",
        help="Generate bespoke cinematic AI character portraits & scene visuals with stylish context badges",
    )

    args = parser.parse_args()

    if not args.url and not args.urls:
        parser.error("one of --url or --urls is required")
    if args.url and args.urls:
        parser.error("--url and --urls are mutually exclusive")

    progress = lambda msg, pct: print(f"[{pct}%] {msg}")

    if args.urls:
        manga_slug = extract_manga_name(args.urls[0])
        out_dir = Path(args.output) / manga_slug
        work_dir = Path(args.workspace) / manga_slug

        asyncio.run(
            process_chapter_urls(
                chapter_urls=args.urls,
                voice=args.voice,
                speed_rate=args.speed,
                aspect_ratio=args.aspect_ratio,
                use_llm=args.use_llm,
                enable_ai_context=args.ai_context,
                output_dir=out_dir,
                workspace_dir=work_dir,
                progress_callback=progress,
            )
        )
    else:
        manga_slug = extract_manga_name(args.url)
        out_dir = Path(args.output) / manga_slug
        work_dir = Path(args.workspace) / manga_slug

        asyncio.run(
            process_multi_chapters(
                start_url=args.url,
                max_chapters=args.chapters,
                voice=args.voice,
                speed_rate=args.speed,
                aspect_ratio=args.aspect_ratio,
                use_llm=args.use_llm,
                enable_ai_context=args.ai_context,
                output_dir=out_dir,
                workspace_dir=work_dir,
                progress_callback=progress,
            )
        )


