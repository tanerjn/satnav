#!/usr/bin/env python3
"""Phase 2: leakage-controlled Starlink ELM experiments on corrected Phase 1 cases.

Train per-satellite, per-horizon direct and SGP4-residual ELM models. Training
labels are allowed only if their reference TLE was CREATED before the split.
Test origins are at/after the split. Both use the exact same Phase 1 cases.
Reference states are later-TLE SGP4 estimates, NOT measured ground truth.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday


def utc(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def state(sat, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    err, r, v = sat.sgp4(jd, fr)
    if err:
        raise ValueError(f"SGP4 error {err}")
    return np.array([*r, *v], dtype=float)


def load(norad, catalog_root, results_root):
    path = Path(catalog_root) / str(norad) / "tle_catalog.csv"
    with path.open(newline="", encoding="utf-8") as f:
        catalog = list(csv.DictReader(f))
    by_id = {}
    for row in catalog:
        key = (row["gp_id"], row["epoch_utc"])
        by_id[key] = Satrec.twoline2rv(row["tle_line1"], row["tle_line2"])
    path = Path(results_root) / str(norad) / "phase1_corrected" / "sgp4_per_forecast.csv"
    with path.open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    rows = []
    for row in cases:
        source = by_id[(row["source_gp_id"], row["source_epoch_utc"])]
        ref = by_id[(row["reference_gp_id"], row["reference_epoch_utc"])]
        origin, target = utc(row["origin_utc"]), utc(row["target_utc"])
        if utc(row["source_creation_utc"]) > origin or utc(row["reference_creation_utc"]) < target:
            raise ValueError("Phase 1 case violates causality")
        x0 = state(source, origin)
        sgp4 = state(source, target)
        reference = state(ref, target)
        rows.append(dict(case=row, origin=origin, reference_created=utc(row["reference_creation_utc"]),
                         x0=x0, sgp4=sgp4, reference=reference))
    return rows


def feature(row, horizon):
    # TEME state at forecast origin. Normalize position/velocity by typical LEO scales.
    # Horizon-specific models do not need horizon as an input feature.
    return np.concatenate([row["x0"][:3] / 7000, row["x0"][3:] / 8])


def fit_elm(x, y, hidden, ridge, solver, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(x.shape[1], hidden))
    b = rng.uniform(-1, 1, size=hidden)
    h = np.column_stack([np.tanh(x @ w + b), np.ones(len(x))])
    penalty = np.eye(h.shape[1]) * ridge
    penalty[-1, -1] = 0
    if solver == "svd":
        # Ridge via augmented least squares, with unpenalized intercept.
        a = np.vstack([h, np.sqrt(penalty)])
        target = np.vstack([y, np.zeros((h.shape[1], y.shape[1]))])
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        beta = vt.T @ ((u.T @ target) / np.maximum(s[:, None], 1e-12))
    else:
        a = np.vstack([h, np.sqrt(penalty)])
        target = np.vstack([y, np.zeros((h.shape[1], y.shape[1]))])
        beta = np.linalg.lstsq(a, target, rcond=None)[0]
    return w, b, beta


def predict(model, x):
    w, b, beta = model
    return np.column_stack([np.tanh(x @ w + b), np.ones(len(x))]) @ beta


def rmse_km(pred, actual):
    return float(np.sqrt(np.mean(np.sum((pred[:, :3] - actual[:, :3]) ** 2, axis=1))))


def evaluate(norad, rows, fraction, hidden, ridge, solver, seed, min_train, min_test):
    if not rows:
        return [], []
    unique_origins = sorted({r["origin"] for r in rows})
    split = unique_origins[min(len(unique_origins)-1, int(len(unique_origins) * fraction))]
    horizons = sorted({float(r["case"]["horizon_hours"]) for r in rows})
    metrics, predictions = [], []
    for horizon in horizons:
        subset = [r for r in rows if float(r["case"]["horizon_hours"]) == horizon]
        train = [r for r in subset if r["origin"] < split and r["reference_created"] <= split]
        test = [r for r in subset if r["origin"] >= split]
        entry = dict(norad_cat_id=norad, horizon_hours=horizon, split_utc=split.isoformat(),
                     train_samples=len(train), test_samples=len(test), hidden=hidden,
                     ridge=ridge, solver=solver)
        if len(train) < min_train or len(test) < min_test:
            entry["status"] = "insufficient_samples"
            metrics.append(entry)
            continue
        x_train = np.vstack([feature(r, horizon) for r in train])
        x_test = np.vstack([feature(r, horizon) for r in test])
        y_ref_train = np.vstack([r["reference"][:3] for r in train])
        y_sgp_train = np.vstack([r["sgp4"][:3] for r in train])
        y_ref_test = np.vstack([r["reference"][:3] for r in test])
        y_sgp_test = np.vstack([r["sgp4"][:3] for r in test])
        # Both targets use km. Residual correction learns (reference - SGP4).
        t0 = time.perf_counter()
        direct = fit_elm(x_train, y_ref_train / 7000, hidden, ridge, solver, seed)
        residual = fit_elm(x_train, (y_ref_train-y_sgp_train) / 7000,
                           hidden, ridge, solver, seed)
        train_seconds = time.perf_counter()-t0
        direct_pred = predict(direct, x_test)*7000
        residual_pred = y_sgp_test + predict(residual, x_test)*7000
        sgp4_rmse = rmse_km(y_sgp_test, y_ref_test)
        entry.update(status="ok", sgp4_rmse_km=sgp4_rmse,
                     direct_elm_rmse_km=rmse_km(direct_pred, y_ref_test),
                     residual_elm_rmse_km=rmse_km(residual_pred, y_ref_test),
                     fit_seconds=train_seconds)
        metrics.append(entry)
        for i, r in enumerate(test):
            predictions.append(dict(norad_cat_id=norad, horizon_hours=horizon,
                                    source_gp_id=r["case"]["source_gp_id"],
                                    origin_utc=r["origin"].isoformat(),
                                    target_utc=r["case"]["target_utc"],
                                    reference_gp_id=r["case"]["reference_gp_id"],
                                    sgp4_error_km=float(np.linalg.norm(y_sgp_test[i]-y_ref_test[i])),
                                    direct_elm_error_km=float(np.linalg.norm(direct_pred[i]-y_ref_test[i])),
                                    residual_elm_error_km=float(np.linalg.norm(residual_pred[i]-y_ref_test[i]))))
    return metrics, predictions


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--norads", type=int, nargs="+", default=[45532, 45695, 45755, 46563])
    p.add_argument("--catalog-root", default="data/starlink/processed")
    p.add_argument("--results-root", default="results/starlink")
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--solver", choices=["qr", "svd"], default="qr")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train", type=int, default=8)
    p.add_argument("--min-test", type=int, default=3)
    a = p.parse_args()
    if not 0 < a.train_fraction < 1 or a.hidden < 1 or a.ridge <= 0:
        p.error("Require 0 < train-fraction < 1, hidden >= 1, ridge > 0")
    for norad in a.norads:
        rows = load(norad, a.catalog_root, a.results_root)
        metrics, predictions = evaluate(norad, rows, a.train_fraction, a.hidden,
                                        a.ridge, a.solver, a.seed, a.min_train, a.min_test)
        out = Path(a.results_root) / str(norad) / "phase2"
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "elm_metrics.csv", metrics,
                  ["norad_cat_id", "horizon_hours", "split_utc", "train_samples",
                   "test_samples", "hidden", "ridge", "solver", "status",
                   "sgp4_rmse_km", "direct_elm_rmse_km",
                   "residual_elm_rmse_km", "fit_seconds"])
        write_csv(out / "elm_per_forecast.csv", predictions,
                  ["norad_cat_id", "horizon_hours", "source_gp_id", "origin_utc",
                   "target_utc", "reference_gp_id", "sgp4_error_km",
                   "direct_elm_error_km", "residual_elm_error_km"])
        (out / "experiment.json").write_text(json.dumps(
            dict(norad_cat_id=norad, train_fraction=a.train_fraction, hidden=a.hidden,
                 ridge=a.ridge, solver=a.solver, seed=a.seed,
                 reference="later TLE-derived SGP4, not ground truth",
                 leakage_policy="training reference creation <= split; test origin >= split",
                 caveat="Small, irregular TLE samples. Models are per-horizon and per-satellite. "
                        "No RK4 in Phase 2; test metrics are exploratory, not final benchmark."),
            indent=2)+"\n", encoding="utf-8")
        print(f"{norad}: " + ", ".join(
            f"{m['horizon_hours']:g}h {m['status']} (train={m['train_samples']}, test={m['test_samples']})"
            for m in metrics))


if __name__ == "__main__":
    main()
