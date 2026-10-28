"""Collect YouTube Shorts data for the Longevity Hook Lab.

Step 1 (count check):
    python codes/collect.py count

For each channel in data/channels.csv this:
  1. looks up the channel by handle (channels.list, 1 unit)
  2. lists every upload (playlistItems.list, 1 unit per 50 videos)
  3. gets durations + stats (videos.list, 1 unit per 50 videos)
  4. runs the /shorts check on the newest <=180s uploads (plain web request, no quota)
and prints a summary table. Raw API responses are cached in data/raw/ so the
full collection step can reuse them without spending quota again.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CHANNELS_CSV = ROOT / "data" / "channels.csv"

API_URL = "https://www.googleapis.com/youtube/v3"
MAX_SHORT_SECONDS = 180
MIN_AGE_DAYS = 14
MIN_SHORTS = 75
# Only the newest N <=180s uploads get the /shorts check (enough to judge the 75 minimum / 300 cap)
SHORTS_CHECK_LIMIT = 400
SHORTS_CHECK_DELAY = 0.2

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("YOUTUBE_API_KEY")


def api_get(endpoint, params, retries=3):
    """Call the YouTube Data API. The key goes in a header so it never appears in URLs or errors."""
    for attempt in range(retries):
        try:
            response = requests.get(
                f"{API_URL}/{endpoint}",
                params=params,
                headers={"X-Goog-Api-Key": API_KEY},
                timeout=30,
            )
            break
        except (requests.Timeout, requests.ConnectionError):
            if attempt == retries - 1:
                raise
            print(f"  {endpoint} timed out, retrying ({attempt + 1}/{retries - 1}) ...", flush=True)
            time.sleep(5 * (attempt + 1))
    if response.status_code != 200:
        message = response.json().get("error", {}).get("message", response.text[:200])
        raise RuntimeError(f"{endpoint} failed ({response.status_code}): {message}")
    return response.json()


def get_channel(handle):
    """Return the channel's id, title, upload count and uploads playlist id."""
    data = api_get("channels", {"part": "snippet,contentDetails,statistics", "forHandle": handle})
    if not data.get("items"):
        return None
    item = data["items"][0]
    return {
        "channel_id": item["id"],
        "channel_title": item["snippet"]["title"],
        "video_count": int(item["statistics"].get("videoCount", 0)),
        "uploads_playlist": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def get_upload_ids(playlist_id):
    """Page through the uploads playlist and return every video id (newest first)."""
    video_ids, page_token = [], None
    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        data = api_get("playlistItems", params)
        video_ids += [item["contentDetails"]["videoId"] for item in data["items"]]
        page_token = data.get("nextPageToken")
        if not page_token:
            return video_ids


def get_video_details(video_ids):
    """Fetch snippet, duration and statistics in batches of 50."""
    videos = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        data = api_get("videos", {"part": "snippet,contentDetails,statistics", "id": ",".join(batch)})
        videos += data["items"]
    return videos


def parse_duration(iso_duration):
    """Convert an ISO 8601 duration like 'PT1M5S' to seconds."""
    match = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_duration or "")
    if not match:
        return None
    days, hours, minutes, seconds = (int(x) if x else 0 for x in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def is_short(video_id, session):
    """A Short's /shorts/ URL returns 200; a regular video redirects to /watch."""
    try:
        response = session.head(
            f"https://www.youtube.com/shorts/{video_id}", allow_redirects=False, timeout=15
        )
        return response.status_code == 200
    except requests.RequestException:
        return None


def raw_path(handle, name):
    return RAW_DIR / f"{handle.lstrip('@').lower()}_{name}.json"


def count_check():
    if not API_KEY:
        sys.exit("YOUTUBE_API_KEY not found in .env")

    channels = pd.read_csv(CHANNELS_CSV)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    cutoff = datetime.now(timezone.utc) - timedelta(days=MIN_AGE_DAYS)
    rows = []

    for _, ch in channels.iterrows():
        handle = ch["handle"]
        print(f"Checking {handle} ...", flush=True)

        # Reuse cached API data from an earlier run so quota isn't spent twice
        if raw_path(handle, "videos").exists():
            cached = json.loads(raw_path(handle, "videos").read_text())
            info, videos = cached["channel"], cached["videos"]
            print("  using cached video data", flush=True)
        else:
            info = get_channel(handle)
            if info is None:
                rows.append({"handle": handle, "creator": ch["creator"], "note": "handle not found"})
                continue
            video_ids = get_upload_ids(info["uploads_playlist"])
            videos = get_video_details(video_ids)
            raw_path(handle, "videos").write_text(json.dumps({"channel": info, "videos": videos}))

        # Newest first, keep uploads of 3 minutes or less
        short_length = [
            v for v in videos
            if (parse_duration(v["contentDetails"].get("duration")) or 0) <= MAX_SHORT_SECONDS
        ]
        short_length.sort(key=lambda v: v["snippet"]["publishedAt"], reverse=True)
        to_check = short_length[:SHORTS_CHECK_LIMIT]

        check_file = raw_path(handle, "shorts_check")
        shorts_check = json.loads(check_file.read_text()) if check_file.exists() else {}
        for v in to_check:
            if shorts_check.get(v["id"]) is None:
                shorts_check[v["id"]] = is_short(v["id"], session)
                time.sleep(SHORTS_CHECK_DELAY)
        check_file.write_text(json.dumps(shorts_check))

        passed = [v for v in to_check if shorts_check[v["id"]]]
        passed_old_enough = [
            v for v in passed
            if datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00")) <= cutoff
        ]
        rows.append({
            "handle": handle,
            "creator": ch["creator"],
            "channel_type": ch["channel_type"],
            "total_uploads": len(videos),
            "le_180s": len(short_length),
            "checked": len(to_check),
            "passed_shorts_check": len(passed),
            "check_errors": sum(r is None for r in shorts_check.values()),
            "passed_and_14d_old": len(passed_old_enough),
            "note": "",
        })

    summary = pd.DataFrame(rows)
    summary["meets_75"] = summary["passed_and_14d_old"].fillna(0) >= MIN_SHORTS
    summary.to_csv(RAW_DIR / "count_check.csv", index=False)

    print("\n=== Count check ===")
    print(f"(/shorts check run on the newest {SHORTS_CHECK_LIMIT} uploads <= {MAX_SHORT_SECONDS}s per channel)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "count"
    if step == "count":
        count_check()
    else:
        sys.exit(f"Unknown step '{step}'. Available: count")
