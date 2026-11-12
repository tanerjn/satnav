"""Corrected Starlink Phase 1: exact-horizon, creation-time-aware SGP4 evaluation.

Historical TLEs are not independent truth. Output is per-forecast and summary.
Forecast origin is the *availability* time of each source TLE, never its earlier
orbital epoch. Reference TLE is selected by closest orbital epoch to the exact
target, subject to creation after the target and a reference-age tolerance.
"""
import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday


def parse_utc(s):
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("UTC timestamp must include offset")
    return d.astimezone(timezone.utc)


def sgp4_state(sat, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    error, r, v = sat.sgp4(jd, fr)
    if error:
        raise ValueError(f"SGP4 error code {error}")
    return np.asarray(r + v, dtype=float)  # km, km/s, TEME


def read_catalog(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError(f"Not enough TLEs in {path}")
    records = []
    ids = set()
    for row in rows:
        sat = Satrec.twoline2rv(row["tle_line1"], row["tle_line2"])
        norad = int(row["norad_cat_id"])
        if sat.error or sat.satnum != norad:
            raise ValueError(f"Invalid TLE for {norad}")
        ids.add(norad)
        records.append(dict(epoch=parse_utc(row["epoch_utc"]),
                            created=parse_utc(row["creation_utc"]),
                            gp_id=row.get("gp_id", ""), sat=sat))
    if len(ids) != 1:
        raise ValueError("Catalog mixes NORAD IDs")
    return ids.pop(), records


def evaluate(catalog, horizons, tolerance_hours):
    norad, records = read_catalog(catalog)
    detail = []
    summary = []
    for hours in horizons:
        errors = []
        skipped = {"no_reference": 0, "propagation_error": 0}
        for source in records:
            # A source TLE cannot be used until it has been published.
            origin = max(source["epoch"], source["created"])
            target = origin + timedelta(hours=hours)
            candidates = [
                ref for ref in records
                if ref is not source and ref["created"] >= target
                and abs((ref["epoch"] - target).total_seconds()) <= tolerance_hours * 3600
            ]
            if not candidates:
                skipped["no_reference"] += 1
                continue
            ref = min(candidates, key=lambda x: (
                abs((x["epoch"] - target).total_seconds()), x["created"], x["gp_id"]))
            try:
                forecast = sgp4_state(source["sat"], target)
                reference = sgp4_state(ref["sat"], target)
            except ValueError:
                skipped["propagation_error"] += 1
                continue
            pos_m = float(np.linalg.norm(forecast[:3] - reference[:3]) * 1000)
            vel_mps = float(np.linalg.norm(forecast[3:] - reference[3:]) * 1000)
            entry = dict(norad_cat_id=norad, horizon_hours=hours,
                         origin_utc=origin.isoformat(), target_utc=target.isoformat(),
                         source_epoch_utc=source["epoch"].isoformat(),
                         source_creation_utc=source["created"].isoformat(),
                         source_gp_id=source["gp_id"],
                         reference_epoch_utc=ref["epoch"].isoformat(),
                         reference_creation_utc=ref["created"].isoformat(),
                         reference_gp_id=ref["gp_id"],
                         reference_epoch_offset_hours=(
                             ref["epoch"] - target).total_seconds() / 3600,
                         position_error_m=pos_m, velocity_error_mps=vel_mps)
            detail.append(entry)
            errors.append(entry)
        summary.append(dict(horizon_hours=hours, samples=len(errors),
                            position_rmse_m=float(np.sqrt(np.mean([
                                e["position_error_m"]**2 for e in errors])))
                            if errors else None,
                            velocity_rmse_mps=float(np.sqrt(np.mean([
                                e["velocity_error_mps"]**2 for e in errors])))
                            if errors else None, **skipped))
    return dict(norad_cat_id=norad, catalog=str(catalog),
                method="SGP4 forward from available TLE versus later TLE estimate at exact target",
                reference_frame="TEME", horizons=summary,
                reference_epoch_tolerance_hours=tolerance_hours,
                forecast_origin_policy="max(source orbital epoch, source creation UTC)",
                reference_policy="closest reference orbital epoch within tolerance, created at or after exact target",
                caveat="Reference states are SGP4-generated from later TLEs, not independent measured truth. "
                       "Forecast origins may overlap; differences can reflect maneuvers and TLE fitting."), detail


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--norads", type=int, nargs="+", default=[45532, 45695, 45755, 46563])
    p.add_argument("--catalog-root", default="data/starlink/processed")
    p.add_argument("--results-root", default="results/starlink")
    p.add_argument("--horizons-hours", type=float, nargs="+", default=[1, 6, 12, 24, 72, 168])
    p.add_argument("--reference-tolerance-hours", type=float, default=3)
    args = p.parse_args()
    if any(h <= 0 for h in args.horizons_hours) or args.reference_tolerance_hours < 0:
        p.error("Horizons must be positive and tolerance nonnegative")
    for norad in args.norads:
        catalog = Path(args.catalog_root) / str(norad) / "tle_catalog.csv"
        report, detail = evaluate(catalog, args.horizons_hours, args.reference_tolerance_hours)
        out = Path(args.results_root) / str(norad) / "phase1_corrected"
        out.mkdir(parents=True, exist_ok=True)
        (out / "sgp4_exact_horizon.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_csv(out / "sgp4_exact_horizon.csv", report["horizons"],
                  ["horizon_hours", "samples", "position_rmse_m", "velocity_rmse_mps",
                   "no_reference", "propagation_error"])
        write_csv(out / "sgp4_per_forecast.csv", detail,
                  ["norad_cat_id", "horizon_hours", "origin_utc", "target_utc",
                   "source_epoch_utc", "source_creation_utc", "source_gp_id",
                   "reference_epoch_utc", "reference_creation_utc", "reference_gp_id",
                   "reference_epoch_offset_hours", "position_error_m", "velocity_error_mps"])
        print(f"{norad}: " + ", ".join(
            f"{x['horizon_hours']:g}h={x['samples']} samples" for x in report["horizons"]))


if __name__ == "__main__":
    main()
