"""Prepare a Starlink Space-Track OMM JSON archive for reproducible experiments.

Retains raw TLEs; does not manufacture independent ground truth.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from sgp4.api import Satrec
from astropy.time import Time


def utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None
            else dt.astimezone(timezone.utc))


def load_records(path, norad):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("records", raw))
    if not isinstance(raw, list):
        raise ValueError("Expected a JSON array of Space-Track OMM records")
    parsed = []
    invalid = []
    for index, rec in enumerate(raw):
        try:
            if int(rec["NORAD_CAT_ID"]) != norad:
                raise ValueError("NORAD ID mismatch")
            if rec.get("REF_FRAME") != "TEME" or rec.get("TIME_SYSTEM") != "UTC":
                raise ValueError("Expected TEME/UTC")
            if rec.get("MEAN_ELEMENT_THEORY") != "SGP4":
                raise ValueError("Expected SGP4 element theory")
            epoch = utc(rec["EPOCH"])
            line1, line2 = rec["TLE_LINE1"], rec["TLE_LINE2"]
            sat = Satrec.twoline2rv(line1, line2)
            if sat.satnum != norad or sat.error != 0:
                raise ValueError("TLE parse error or NORAD mismatch")
            tle_epoch = Time(sat.jdsatepoch, sat.jdsatepochF,
                             format="jd", scale="utc").to_datetime(timezone=timezone.utc)
            if abs((tle_epoch - epoch).total_seconds()) > 2:
                raise ValueError("OMM epoch and embedded TLE epoch differ by >2 seconds")
            creation = utc(rec["CREATION_DATE"]) if rec.get("CREATION_DATE") else epoch
            parsed.append(dict(epoch=epoch, creation=creation, line1=line1,
                               line2=line2, original_index=index,
                               gp_id=str(rec.get("GP_ID", "")),
                               object_name=str(rec.get("OBJECT_NAME", ""))))
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append(dict(index=index, reason=str(exc)))
    return len(raw), parsed, invalid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Space-Track JSON array of OMM records")
    p.add_argument("--norad", type=int, default=45532)
    p.add_argument("--output", default="data/starlink/processed/45532")
    a = p.parse_args()
    count, records, invalid = load_records(a.input, a.norad)
    if not records:
        raise ValueError("No valid records")
    # Deterministic duplicate policy: retain the most recently created record
    # at each epoch, breaking ties by original record order.
    by_epoch = {}
    for rec in records:
        key = rec["epoch"]
        if key not in by_epoch or (rec["creation"], rec["original_index"]) > (
                by_epoch[key]["creation"], by_epoch[key]["original_index"]):
            by_epoch[key] = rec
    ordered = sorted(by_epoch.values(), key=lambda r: r["epoch"])
    gaps_h = [(b["epoch"] - a["epoch"]).total_seconds() / 3600
              for a, b in zip(ordered, ordered[1:])]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tle_catalog.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["epoch_utc", "creation_utc", "norad_cat_id", "object_name",
                  "gp_id", "tle_line1", "tle_line2"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ordered:
            writer.writerow(dict(epoch_utc=r["epoch"].isoformat(),
                                 creation_utc=r["creation"].isoformat(),
                                 norad_cat_id=a.norad, object_name=r["object_name"],
                                 gp_id=r["gp_id"], tle_line1=r["line1"],
                                 tle_line2=r["line2"]))
    summary = dict(source=a.input, norad_cat_id=a.norad, total_records=count,
                   valid_records=len(records), invalid_records=len(invalid),
                   invalid_examples=invalid[:20],
                   duplicate_epochs=len(records)-len(ordered),
                   unique_epochs=len(ordered),
                   first_epoch_utc=ordered[0]["epoch"].isoformat(),
                   last_epoch_utc=ordered[-1]["epoch"].isoformat(),
                   min_gap_hours=min(gaps_h) if gaps_h else None,
                   median_gap_hours=sorted(gaps_h)[len(gaps_h)//2] if gaps_h else None,
                   max_gap_hours=max(gaps_h) if gaps_h else None,
                   gap_count_over_24h=sum(g > 24 for g in gaps_h),
                   reference_frame="TEME", time_system="UTC",
                   note="Later OMM/TLE sets are updated orbital estimates, not independent truth. "
                        "Do not use future records as training inputs for historical forecasts.")
    (out / "catalog_quality.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved", out / "tle_catalog.csv")


if __name__ == "__main__":
    main()
