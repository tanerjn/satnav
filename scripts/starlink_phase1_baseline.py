"""Phase 1 Starlink baseline: forward-only SGP4 versus later OMM-derived states.

The reference is a *later TLE propagated at its own epoch*, not measured truth.
Each earlier TLE is used only for forward forecasts. No future data leakage.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sgp4.api import Satrec


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def state(sat, epoch):
    jd, fr = __import__("sgp4.api", fromlist=["jday"]).jday(
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute,
        epoch.second + epoch.microsecond / 1e6)
    err, r, v = sat.sgp4(jd, fr)
    if err:
        raise ValueError(f"SGP4 propagation error {err}")
    return np.array([*r, *v], dtype=float)  # km, km/s, TEME


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", default="data/starlink/processed/45532/tle_catalog.csv")
    p.add_argument("--output", default="results/starlink/45532/phase1")
    p.add_argument("--horizons-hours", type=float, nargs="+",
                   default=[1, 6, 12, 24, 72, 168])
    p.add_argument("--tolerance-hours", type=float, default=2)
    a = p.parse_args()
    if any(h <= 0 for h in a.horizons_hours) or a.tolerance_hours < 0:
        p.error("Horizons must be positive; tolerance must be nonnegative")
    with open(a.catalog, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if len(records) < 2:
        raise ValueError("At least two unique OMM records required")
    epochs = [utc(r["epoch_utc"]) for r in records]
    if any(b <= x for x, b in zip(epochs, epochs[1:])):
        raise ValueError("Catalog must be sorted with unique epochs")
    sats = [Satrec.twoline2rv(r["tle_line1"], r["tle_line2"]) for r in records]
    if any(s.error or s.satnum != int(records[0]["norad_cat_id"]) for s in sats):
        raise ValueError("Invalid or inconsistent satellite IDs")
    # For each start epoch, pick the closest later independent update to
    # the target horizon. Do not interpolate or backfill from the future.
    results = []
    for horizon in a.horizons_hours:
        errors = []
        for i, t0 in enumerate(epochs[:-1]):
            options = [(abs((t - t0).total_seconds() / 3600 - horizon), j)
                       for j, t in enumerate(epochs[i+1:], i+1)]
            if not options:
                continue
            distance, j = min(options)
            if distance > a.tolerance_hours:
                continue
            target = epochs[j]
            try:
                predicted = state(sats[i], target)
                reference = state(sats[j], target)
            except ValueError:
                continue
            errors.append(dict(position_km=float(np.linalg.norm(
                                   predicted[:3] - reference[:3])),
                               velocity_kmps=float(np.linalg.norm(
                                   predicted[3:] - reference[3:])),
                               actual_horizon_hours=(target-t0).total_seconds()/3600))
        result = dict(requested_horizon_hours=horizon, samples=len(errors),
                      position_rmse_m=float(np.sqrt(np.mean(
                          [e["position_km"]**2 for e in errors]))*1000)
                      if errors else None,
                      velocity_rmse_mps=float(np.sqrt(np.mean(
                          [e["velocity_kmps"]**2 for e in errors]))*1000)
                      if errors else None,
                      actual_horizon_min_hours=min(
                          (e["actual_horizon_hours"] for e in errors), default=None),
                      actual_horizon_max_hours=max(
                          (e["actual_horizon_hours"] for e in errors), default=None))
        results.append(result)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(catalog=a.catalog, norad_cat_id=int(records[0]["norad_cat_id"]),
                   method="Earlier TLE SGP4 forecast vs later TLE SGP4 state at later epoch",
                   reference_frame="TEME", position_unit="m", velocity_unit="m/s",
                   tolerance_hours=a.tolerance_hours, horizons=results,
                   caveat="Later TLE states are independently updated model estimates, "
                          "not ground-truth observations. Some forecast/reference "
                          "estimates may share underlying tracking observations. "
                          "Comparisons are descriptive and overlap across horizons.")
    (out/"sgp4_forward_baseline.json").write_text(
        json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    with (out/"sgp4_forward_baseline.csv").open("w", newline="", encoding="utf-8") as f:
        fields=list(results[0])
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
