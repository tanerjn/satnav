"""Compare ISS SGP4 states in GCRS and mean-J2000 (EME2000) axes.

Frame convention: IAU 2000 SOFA/ERFA frame bias (bp00 rb), GCRS -> mean
equator/equinox of J2000. This is an explicitly documented approximation
to OEM EME2000; it does not establish the OEM producer's frame realization.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.time import Time
from sgp4.api import Satrec

from satnav.iss_sgp4 import propagate_tle
from satnav.iss_frames import teme_to_gcrs, gcrs_to_eme2000
from satnav.evaluation import state_rmse

FIELDS = ("x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps")


def load_oem(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [datetime.fromisoformat(row["epoch_utc"].replace("Z", "+00:00")) for row in rows]
    epochs = [dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc) for dt in epochs]
    states = np.array([[float(row[k]) for k in FIELDS] for row in rows])
    if not epochs or states.shape != (len(epochs), 6) or not np.isfinite(states).all():
        raise ValueError("OEM CSV must contain finite Cartesian states")
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("OEM epochs must be strictly increasing")
    return epochs, states


def load_tle(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    line1 = next((line for line in lines if line.startswith("1 ")), None)
    line2 = next((line for line in lines if line.startswith("2 ")), None)
    if line1 is None or line2 is None:
        raise ValueError("TLE file requires line 1 and line 2")
    sat = Satrec.twoline2rv(line1, line2)
    if sat.satnum != 25544:
        raise ValueError("Expected ISS NORAD catalog number 25544")
    epoch = Time(sat.jdsatepoch, sat.jdsatepochF, format="jd", scale="utc").to_datetime(timezone=timezone.utc)
    return line1, line2, epoch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oem", default="data/iss/processed/nasa_iss_states.csv")
    p.add_argument("--tle", default="data/iss/spacetrack/iss.tle")
    p.add_argument("--max-age-days", type=float, default=2.0)
    p.add_argument("--output", default="results/iss/phase1/frame_verification.json")
    args = p.parse_args()
    if args.max_age_days <= 0:
        p.error("--max-age-days must be positive")
    epochs, reference = load_oem(args.oem)
    l1, l2, tle_epoch = load_tle(args.tle)
    matched = [i for i, epoch in enumerate(epochs)
               if abs((epoch - tle_epoch).total_seconds()) <= args.max_age_days * 86400]
    if not matched:
        raise ValueError("No OEM timestamps near TLE epoch; no result saved")
    times = [epochs[i] for i in matched]
    ref = reference[matched]
    teme = propagate_tle(l1, l2, times)
    gcrs = teme_to_gcrs(teme, times)
    eme = gcrs_to_eme2000(gcrs)
    delta = eme - gcrs
    pos_diff = np.linalg.norm(delta[:, :3], axis=1)
    vel_diff = np.linalg.norm(delta[:, 3:], axis=1)
    result = {
        "tle_file": args.tle,
        "oem_file": args.oem,
        "tle_epoch_utc": tle_epoch.isoformat(),
        "matched_oem_epochs": len(matched),
        "first_matched_epoch_utc": times[0].isoformat(),
        "last_matched_epoch_utc": times[-1].isoformat(),
        "max_tle_age_days": args.max_age_days,
        "frame_method": "TEME -> Astropy GCRS -> ERFA/SOFA IAU 2000 bp00 rb mean-J2000 frame bias",
        "frame_difference": {
            "position_rms_m": float(np.sqrt(np.mean(pos_diff ** 2))),
            "position_max_m": float(np.max(pos_diff)),
            "velocity_rms_mps": float(np.sqrt(np.mean(vel_diff ** 2))),
            "velocity_max_mps": float(np.max(vel_diff)),
        },
        "sgp4_gcrs_vs_oem": state_rmse(gcrs, ref),
        "sgp4_eme2000_vs_oem": state_rmse(eme, ref),
        "caveat": (
            "OEM declares EME2000, but its precise frame realization is not independently "
            "validated. The IAU 2000 frame-bias rotation gives a defined mean-J2000 "
            "convention, not a proof of NASA OEM frame equivalence. Differences in RMSE "
            "cannot be uniquely attributed to frame mismatch versus propagation error."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
