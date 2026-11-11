"""Fetch an ISS TLE nearest to a requested UTC epoch from Orbital Data API.

Keep credentials local. Edit API_KEY below, or set ORBITAL_DATA_API_KEY.
Do not commit a real API key.
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from sgp4.api import Satrec
from astropy.time import Time

API_KEY = "PASTE_YOUR_API_KEY_HERE"
BASE_URL = "https://orbital-data-api.davidhsu.cc/api/v1/gp"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", required=True, help="UTC date, e.g. 2026-11-22T12:00:00Z")
    p.add_argument("--output", default="data/iss/celestrak/iss_test_nearest.tle")
    p.add_argument("--norad", type=int, default=25544)
    p.add_argument("--max-age-days", type=float, default=2.0)
    args = p.parse_args()
    key = os.getenv("ORBITAL_DATA_API_KEY") or API_KEY
    if not key or key == "PASTE_YOUR_API_KEY_HERE":
        p.error("Set API_KEY in this file or ORBITAL_DATA_API_KEY in your environment")
    target = datetime.fromisoformat(args.epoch.replace("Z", "+00:00"))
    if target.tzinfo is None:
        p.error("--epoch must include UTC timezone (Z)")
    target = target.astimezone(timezone.utc)
    if target > datetime.now(timezone.utc):
        p.error("Requested epoch is in the future; wait for historical TLE data")
    if args.max_age_days <= 0:
        p.error("--max-age-days must be positive")
    url = f"{BASE_URL}/{args.norad}/nearest?" + urlencode({
        "epoch": target.strftime("%Y-%m-%dT%H:%M:%SZ"), "format": "tle"})
    request = Request(url, headers={"X-Api-Key": key, "Accept": "text/plain"})
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8-sig")
    except HTTPError as exc:
        p.error(f"API returned HTTP {exc.code}; verify credentials and endpoint")
    except URLError as exc:
        p.error(f"Network error: {exc.reason}")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    line1 = next((line for line in lines if line.startswith("1 ")), None)
    line2 = next((line for line in lines if line.startswith("2 ")), None)
    if not line1 or not line2:
        p.error("Response did not contain TLE lines; check API response format")
    if not line1.startswith(f"1 {args.norad:05d}") or not line2.startswith(f"2 {args.norad:05d}"):
        p.error("Returned TLE is for a different satellite")
    sat = Satrec.twoline2rv(line1, line2)
    tle_epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format="jd", scale="utc").to_datetime(timezone=timezone.utc)
    gap = abs((tle_epoch - target).total_seconds()) / 86400
    print(f"Target: {target.isoformat()} | TLE epoch: {tle_epoch.isoformat()} | gap: {gap:.3f} days")
    if gap > args.max_age_days:
        p.error("Nearest TLE is outside --max-age-days; no file written")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(line1 + "\n" + line2 + "\n")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
