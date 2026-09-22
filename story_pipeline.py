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
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import cv2
import edge_tts
import numpy as np
from PIL import Image, ImageStat
import pytesseract

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")


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


def fetch_html(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": origin + "/",
        "Origin": origin,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


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
    chapters = []
    seen_urls = set()

    # Always ensure the starting URL itself is in the chapter list
    start_num = parse_chapter_number(start_url) or 1.0
    start_title = f"Chapter {int(start_num) if start_num.is_integer() else start_num}"
    clean_start = start_url.rstrip("/").lower()

    chapters.append({
        "url": start_url,
        "title": start_title,
        "number": start_num,
    })
    seen_urls.add(clean_start)
    seen_urls.add(clean_start + "/")

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

    # 2. Search <a href="..."> chapter links
    links = re.findall(r'<a\s+[^>]*href=[\"\']([^\"\']+)[\"\'][^>]*>([^<]*)</a>', html, flags=re.IGNORECASE)
    for href, anchor_text in links:
        href_clean = href.strip()
        if not href_clean or href_clean.startswith("#") or href_clean.startswith("javascript:"):
            continue
        full_url = urljoin(start_url, href_clean)
        clean_full = full_url.rstrip("/").lower()
        if clean_full not in seen_urls:
            ch_num = parse_chapter_number(full_url) or parse_chapter_number(anchor_text)
            if ch_num is not None:
                seen_urls.add(clean_full)
                seen_urls.add(clean_full + "/")
                display_title = anchor_text.strip() or f"Chapter {int(ch_num) if ch_num.is_integer() else ch_num}"
                chapters.append({
                    "url": full_url,
                    "title": display_title,
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
    rephrased = re.sub(r'[\.\!\?]+$', '', d)
    if not rephrased:
        return ""

    rephrased = rephrased[0].upper() + rephrased[1:]
    if not rephrased.endswith(('.', '!', '?')):
        rephrased += '.'
    return rephrased


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


def generate_story_beats(
    panel_slices: list[dict],
    chapter_title: str,
    aspect_ratio: str = "16:9"
) -> list[dict]:
    """
    Builds a continuous, cinematic 3rd-person YouTube-style story recap script
    strictly synchronized 1:1 with what is shown on screen.
    """
    story_beats = []
    action_idx = 0
    last_narration = ""
    last_panel_path: Optional[Path] = None
    text_buffer = []

    for s in panel_slices:
        panel_path: Path = s["path"]
        score: float = s["score"]
        is_text: bool = s["is_text_heavy"]
        clean_text: str = s["text"]

        if clean_text:
            text_buffer.append(clean_text)

        # If this is a speech bubble box only (no character/art), buffer its text to speak over the next character panel
        if is_text and score < 12.0:
            continue

        # This is a visual/character artwork panel
        if text_buffer:
            combined_text = " ".join(text_buffer)
            # Split into natural recap thoughts if long
            clauses = re.split(r'[\.\!\?]+\s*', combined_text)
            valid_clauses = [c.strip() for c in clauses if len(c.strip().split()) >= 2]

            if valid_clauses:
                for clause in valid_clauses:
                    recap_sent = convert_dialogue_to_recap_sentence(clause, 1, 1)
                    if not recap_sent or (last_narration and are_phrases_similar(recap_sent, last_narration)):
                        continue

                    chosen_panel = panel_path
                    if last_panel_path and are_images_duplicate(chosen_panel, last_panel_path):
                        chosen_panel = panel_path

                    story_beats.append({
                        "id": f"beat_{len(story_beats)+1:03d}",
                        "panel": chosen_panel,
                        "narration": recap_sent,
                        "page": s["page"],
                    })
                    last_narration = recap_sent
                    last_panel_path = chosen_panel
            text_buffer = []
        else:
            # Give every non-duplicate art panel its own action beat, cycling through
            # filler lines so consecutive beats don't repeat the same sentence verbatim.
            chosen_panel = panel_path
            if last_panel_path and are_images_duplicate(chosen_panel, last_panel_path):
                continue

            act_sent = DYNAMIC_ACTION_BEATS[action_idx % len(DYNAMIC_ACTION_BEATS)]
            action_idx += 1

            story_beats.append({
                "id": f"beat_{len(story_beats)+1:03d}",
                "panel": chosen_panel,
                "narration": act_sent,
                "page": s["page"],
            })
            last_narration = act_sent
            last_panel_path = chosen_panel

    # Flush any remaining dialogue at the end
    if text_buffer and story_beats:
        rem_text = " ".join(text_buffer)
        recap_sent = convert_dialogue_to_recap_sentence(rem_text, 1, 1)
        if recap_sent:
            story_beats.append({
                "id": f"beat_{len(story_beats)+1:03d}",
                "panel": story_beats[-1]["panel"],
                "narration": recap_sent,
                "page": story_beats[-1]["page"],
            })

    return story_beats


def generate_llm_story_beats(
    panel_slices: list[dict],
    chapter_title: str,
    aspect_ratio: str = "16:9",
) -> Optional[list[dict]]:
    """
    LLM-based alternative to generate_story_beats(). Sends the OCR'd panel text
    for the whole chapter to a cheap model in a single call and asks for a
    cinematic third-person recap narration line per beat. Falls back to None
    (caller should use the regex-based generate_story_beats) if no API key is
    configured or the call fails, so this is always an opt-in enhancement.

    Output contract matches generate_story_beats(): list of
    {id, panel, narration, page} dicts, so downstream TTS/render code is
    untouched.
    """
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    # Only panels that actually have usable OCR text or are visual/character
    # panels worth narrating are sent - keeps the prompt (and cost) small.
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
        text = s["text"] if s["text"] else "(no text, character/action panel)"
        lines.append(f"{i}. [{kind}] {text}")
    panel_manifest = "\n".join(lines)

    prompt = (
        f"You are writing a cinematic third-person YouTube recap narration for a manga/manhwa "
        f"chapter titled \"{chapter_title}\".\n\n"
        f"Below is the ordered sequence of panels in this chapter. Each has an index, a kind "
        f"(DIALOGUE = has OCR'd speech/text, ART = a visual/action panel with no text), and the "
        f"raw OCR text if any.\n\n"
        f"{panel_manifest}\n\n"
        f"Write a short third-person recap sentence for each panel that should be narrated. "
        f"Skip panels that add nothing (empty bubbles, redundant lines). Merge consecutive "
        f"DIALOGUE panels into one sentence when they form a single thought. For ART panels, "
        f"write a brief cinematic action/atmosphere line. Keep sentences short (under 20 words), "
        f"natural to read aloud, and consistent in character names.\n\n"
        f"Respond with ONLY a JSON array, no prose, in this exact shape:\n"
        f'[{{"panel_index": <int>, "narration": "<sentence>"}}, ...]\n'
        f"panel_index must reference the index numbers above, in ascending order."
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
    except Exception:
        return None

    import json
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        parsed = json.loads(raw)
    except Exception:
        return None

    story_beats = []
    last_panel_path: Optional[Path] = None
    for item in parsed:
        try:
            idx = int(item["panel_index"])
            narration = str(item["narration"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not narration or idx < 0 or idx >= len(candidates):
            continue

        panel_path = candidates[idx]["path"]
        if last_panel_path and are_images_duplicate(panel_path, last_panel_path):
            panel_path = last_panel_path if last_panel_path else panel_path

        story_beats.append({
            "id": f"beat_{len(story_beats)+1:03d}",
            "panel": panel_path,
            "narration": narration,
            "page": candidates[idx]["page"],
        })
        last_panel_path = panel_path

    return story_beats if story_beats else None


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


async def synthesize_voiceover(
    text: str,
    output_path: Path,
    voice: str = "en-US-ChristopherNeural",
    speed_rate: str = "+25%",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(text, voice, rate=speed_rate)
    await communicate.save(str(output_path))


def render_frame_video(
    panel_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    width: int = 1080,
    height: int = 1920,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},boxblur=25:5[bg];"
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(panel_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        vf,
        "-map",
        "[outv]",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-r",
        "30",
        "-t",
        f"{duration:.3f}",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def concat_video_segments(segment_paths: list[Path], final_output: Path) -> None:
    if not segment_paths:
        raise RuntimeError(
            f"No video segments to assemble for '{final_output.name}' - "
            f"nothing was rendered upstream (check scraping/panel-slicing/narration steps)."
        )
    final_output.parent.mkdir(parents=True, exist_ok=True)
    concat_list = final_output.parent / f"concat_{final_output.stem}.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(final_output),
    ]
    subprocess.run(cmd, check=True)
    if concat_list.exists():
        concat_list.unlink()


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
    use_llm: bool = False,
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
        beats = generate_llm_story_beats(panel_entries, ch_title, aspect_ratio=aspect_ratio)
    if beats is None:
        beats = generate_story_beats(panel_entries, ch_title, aspect_ratio=aspect_ratio)

    if not beats:
        raise RuntimeError(
            f"Chapter {ch_num}: {len(panel_entries)} panel(s) were sliced but no narration beats "
            f"could be generated from them (no usable text/art detected)."
        )

    width = 1920 if aspect_ratio == "16:9" else 1080
    height = 1080 if aspect_ratio == "16:9" else 1920

    segment_paths = []
    total_beats = len(beats)
    
    for i, beat in enumerate(beats, start=1):
        beat_id = beat["id"]
        panel_path = beat["panel"]
        narration = beat["narration"]

        audio_path = audio_dir / f"{beat_id}.mp3"
        await synthesize_voiceover(narration, audio_path, voice=voice, speed_rate=speed_rate)
        duration = get_audio_duration(audio_path)

        segment_path = segments_dir / f"{beat_id}.mp4"
        render_frame_video(panel_path, audio_path, segment_path, duration, width=width, height=height)
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
    use_llm: bool = False,
    output_dir: Path = Path("output"),
    workspace_dir: Path = Path("workspace"),
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> tuple[list[Path], Path]:
    all_chapters = discover_all_chapters(start_url)
    
    # Locate starting chapter index
    start_idx = 0
    clean_start = start_url.rstrip("/").lower()
    for idx, c in enumerate(all_chapters):
        if c["url"].rstrip("/").lower() == clean_start:
            start_idx = idx
            break
            
    to_process = all_chapters[start_idx : start_idx + max_chapters] if max_chapters > 0 else all_chapters[start_idx:]

    chapter_videos: list[Path] = []
    total = len(to_process)

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

    if progress_callback:
        progress_callback("All chapters completed successfully!", 100)

    return chapter_videos, merged_output_path


async def process_chapter_urls(
    chapter_urls: list[str],
    voice: str = "en-US-ChristopherNeural",
    speed_rate: str = "+25%",
    aspect_ratio: str = "16:9",
    use_llm: bool = False,
    output_dir: Path = Path("output"),
    workspace_dir: Path = Path("workspace"),
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
    parser.add_argument("--voice", default="en-US-ChristopherNeural", help="TTS voice name")
    parser.add_argument("--speed", default="+25%", help="TTS speed rate (e.g. +25%)")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"], help="Video aspect ratio")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--workspace", default="workspace", help="Workspace directory")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (OpenAI, requires API_KEY in .env) for narration script generation instead of the regex-based generator",
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
                output_dir=out_dir,
                workspace_dir=work_dir,
                progress_callback=progress,
            )
        )


