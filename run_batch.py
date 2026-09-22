#!/usr/bin/env python3
"""
Batch driver for the manga story video pipeline.
Reads a JSON manifest of jobs and runs each one through the existing
process_multi_chapters() pipeline sequentially, logging progress and results.

Usage:
    python run_batch.py jobs.json
    python run_batch.py jobs.json --max-chapters-per-job 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from story_pipeline import extract_manga_name, process_multi_chapters

DEFAULT_MAX_CHAPTERS_PER_JOB = 5


def load_jobs(manifest_path: Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list):
        raise ValueError("Job manifest must be a JSON array of job objects")
    return jobs


def validate_job(job: dict, index: int, hard_cap: int) -> dict:
    if "url" not in job or not job["url"]:
        raise ValueError(f"Job {index}: missing required 'url'")

    chapters = int(job.get("chapters", 1))
    if chapters < 1:
        raise ValueError(f"Job {index}: 'chapters' must be >= 1")
    if chapters > hard_cap:
        raise ValueError(
            f"Job {index}: 'chapters'={chapters} exceeds max-chapters-per-job cap "
            f"({hard_cap}). Lower it or raise --max-chapters-per-job explicitly."
        )

    speed = float(job.get("speed", 1.25))
    speed_rate = f"+{int((speed - 1.0) * 100)}%" if speed > 1.0 else "0%"

    return {
        "url": job["url"],
        "max_chapters": chapters,
        "voice": job.get("voice", "en-US-ChristopherNeural"),
        "speed_rate": speed_rate,
        "aspect_ratio": job.get("aspect_ratio", "9:16"),
        "use_llm": bool(job.get("use_llm", False)),
    }


async def run_job(job_index: int, raw_job: dict, hard_cap: int, results: list[dict]) -> None:
    label = raw_job.get("url", f"job_{job_index}")
    print(f"\n=== Job {job_index}: {label} ===")

    try:
        job = validate_job(raw_job, job_index, hard_cap)
    except ValueError as exc:
        print(f"[SKIPPED] {exc}")
        results.append({"index": job_index, "url": label, "status": "skipped", "error": str(exc)})
        return

    manga_slug = extract_manga_name(job["url"])
    aspect_slug = job["aspect_ratio"].replace(":", "x")
    manga_work_dir = Path("workspace") / manga_slug / aspect_slug
    manga_out_dir = Path("output") / manga_slug / aspect_slug

    def log(msg: str, pct: int):
        print(f"  [{pct:3d}%] {msg}")

    start = time.monotonic()
    try:
        chapter_videos, merged_video = await process_multi_chapters(
            start_url=job["url"],
            max_chapters=job["max_chapters"],
            voice=job["voice"],
            speed_rate=job["speed_rate"],
            aspect_ratio=job["aspect_ratio"],
            use_llm=job["use_llm"],
            output_dir=manga_out_dir,
            workspace_dir=manga_work_dir,
            progress_callback=log,
        )
        elapsed = time.monotonic() - start
        print(f"[OK] {label} -> {merged_video} ({elapsed:.1f}s, {len(chapter_videos)} chapter(s))")
        results.append({
            "index": job_index,
            "url": label,
            "status": "success",
            "chapter_videos": [str(p) for p in chapter_videos],
            "merged_video": str(merged_video),
            "elapsed_seconds": round(elapsed, 1),
        })
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"[FAILED] {label}: {exc}")
        results.append({
            "index": job_index,
            "url": label,
            "status": "failed",
            "error": str(exc),
            "elapsed_seconds": round(elapsed, 1),
        })


async def main_async(manifest_path: Path, hard_cap: int, log_path: Path) -> int:
    jobs = load_jobs(manifest_path)
    print(f"Loaded {len(jobs)} job(s) from {manifest_path}")

    results: list[dict] = []
    for idx, raw_job in enumerate(jobs, start=1):
        await run_job(idx, raw_job, hard_cap, results)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\n=== Batch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped ===")
    print(f"Results written to {log_path}")

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description="Run a batch of manga-to-video jobs from a manifest file")
    parser.add_argument("manifest", type=Path, help="Path to a JSON job manifest (see jobs.example.json)")
    parser.add_argument(
        "--max-chapters-per-job",
        type=int,
        default=DEFAULT_MAX_CHAPTERS_PER_JOB,
        help=f"Hard cap on 'chapters' per job to prevent runaway batches (default: {DEFAULT_MAX_CHAPTERS_PER_JOB})",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("batch_results.json"),
        help="Where to write the JSON results log (default: batch_results.json)",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Manifest file not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)

    exit_code = asyncio.run(main_async(args.manifest, args.max_chapters_per_job, args.log))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
