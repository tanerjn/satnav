#!/usr/bin/env python3
"""Phase 2 orbital-feature ELM: chronological 70/30 train-validation experiment.

Reference positions are retrospective later-TLE SGP4 estimates, NOT measurements.
The direct model predicts target position in an origin-fixed RTN basis (no target
SGP4 input). The residual model learns target-SGP4 corrections in target RTN.
The final 30% is used BOTH for hyperparameter selection and reporting; these
are validation (not independent test) metrics. The original script is unchanged.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, jday


def utc(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)


def state(sat, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    error, r, v = sat.sgp4(jd, fr)
    if error:
        raise ValueError(f'SGP4 returned {error}')
    return np.array([*r, *v], dtype=float)


def rtn(s):
    r, v = s[:3], s[3:]
    radial = r / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    return np.stack([radial, along, normal], axis=1)


def load_cases(norad, catalog_root, results_root):
    path = Path(catalog_root) / str(norad) / 'tle_catalog.csv'
    with path.open(newline='', encoding='utf-8') as f:
        catalog = list(csv.DictReader(f))
    sats = {(r['gp_id'], r['epoch_utc']): Satrec.twoline2rv(r['tle_line1'], r['tle_line2'])
            for r in catalog}
    path = Path(results_root) / str(norad) / 'phase1_corrected' / 'sgp4_per_forecast.csv'
    with path.open(newline='', encoding='utf-8') as f:
        cases = list(csv.DictReader(f))
    result = []
    for c in cases:
        origin, target = utc(c['origin_utc']), utc(c['target_utc'])
        ref_created = utc(c['reference_creation_utc'])
        if utc(c['source_creation_utc']) > origin or ref_created < target:
            raise ValueError('Invalid Phase 1 availability constraint')
        source = sats[(c['source_gp_id'], c['source_epoch_utc'])]
        reference = sats[(c['reference_gp_id'], c['reference_epoch_utc'])]
        x0, sgp, ref = state(source, origin), state(source, target), state(reference, target)
        origin_basis, target_basis = rtn(x0), rtn(sgp)
        # Origin-fixed coordinates make direct prediction independent of future SGP4.
        direct_target = origin_basis.T @ ref[:3] / 7000.0
        # Small corrections, expressed in physically meaningful target RTN axes.
        residual_target = target_basis.T @ (ref[:3] - sgp[:3]) / 100.0
        result.append(dict(case=c, origin=origin, ref_created=ref_created,
                           source_sat=source, source_epoch=utc(c['source_epoch_utc']),
                           source_created=utc(c['source_creation_utc']),
                           x0=x0, sgp=sgp, ref=ref, origin_basis=origin_basis,
                           target_basis=target_basis, direct_target=direct_target,
                           residual_target=residual_target))
    return result


def features(rows, feature_set='full'):
    """All features are derived from the source TLE known at forecast origin.

    Angles are encoded continuously. e*cos(omega), e*sin(omega) avoid
    unstable perigee direction for near-circular orbits.
    """
    vectors = []
    for row in rows:
        x0, sat = row['x0'], row['source_sat']
        base = np.r_[x0[:3] / 7000, x0[3:] / 8,
                     np.linalg.norm(x0[:3]) / 7000,
                     np.linalg.norm(x0[3:]) / 8]
        if feature_set == 'baseline':
            vectors.append(base)
            continue
        e, inc, node, argp, anomaly = (
            sat.ecco, sat.inclo, sat.nodeo, sat.argpo, sat.mo)
        # Longitude of mean argument of latitude is better behaved when e~0.
        phase = (argp + anomaly) % (2 * np.pi)
        orbital = np.r_[e, sat.no_kozai * 1440 / (2 * np.pi),
                        np.sin(inc), np.cos(inc),
                        np.sin(node), np.cos(node),
                        e * np.sin(argp), e * np.cos(argp),
                        np.sin(phase), np.cos(phase)]
        if feature_set == 'orbital':
            vectors.append(np.r_[base, orbital])
            continue
        age_hours = (row['origin'] - row['source_epoch']).total_seconds() / 3600
        publication_lag_hours = (
            row['source_created'] - row['source_epoch']).total_seconds() / 3600
        drag = np.r_[sat.bstar, age_hours, publication_lag_hours]
        if feature_set == 'drag':
            vectors.append(np.r_[base, orbital, drag])
            continue
        # SGP4 state is a valid as-of-origin prediction from the source TLE.
        # Express it in target RTN to avoid raw TEME orientation dependence.
        target = row['sgp']
        target_rtn_velocity = row['target_basis'].T @ target[3:]
        horizon_hours = float(row['case']['horizon_hours'])
        physics = np.r_[np.linalg.norm(target[:3]) / 7000,
                        target_rtn_velocity / 8,
                        horizon_hours / 168]
        vectors.append(np.r_[base, orbital, drag, physics])
    return np.vstack(vectors)


def fit_scaler(x):
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd < 1e-8, 1, sd)
    return mu, sd


def hidden_matrix(x, w, b):
    return np.column_stack([np.tanh(x @ w + b), np.ones(len(x))])


def solve(h, y, ridge, solver):
    # Augmented ridge system; intercept is not penalized.
    penalty = np.eye(h.shape[1])
    penalty[-1, -1] = 0
    a = np.vstack([h, np.sqrt(ridge) * penalty])
    rhs = np.vstack([y, np.zeros((h.shape[1], y.shape[1]))])
    if solver == 'qr':
        q, r = np.linalg.qr(a, mode='reduced')
        return np.linalg.solve(r, q.T @ rhs)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return vt.T @ ((u.T @ rhs) / s[:, None])


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def decode_direct(rows, y):
    return np.vstack([r['origin_basis'] @ (p * 7000) for r, p in zip(rows, y)])


def decode_residual(rows, y, alpha):
    return np.vstack([r['sgp'][:3] + alpha * (r['target_basis'] @ (p * 100))
                      for r, p in zip(rows, y)])


def run_group(norad, horizon, rows, args, rng):
    # Chronological 70/30 split by distinct forecast origins.
    origins = sorted({r['origin'] for r in rows})
    if len(origins) < 2:
        return dict(status='insufficient_origins'), []
    i = max(1, min(len(origins) - 1, int(len(origins) * args.train_fraction)))
    train_cut = origins[i]
    # Avoid look-ahead: training labels must have been published by cutoff.
    train = [r for r in rows if r['origin'] < train_cut and r['ref_created'] <= train_cut]
    val = [r for r in rows if r['origin'] >= train_cut]
    result = dict(norad_cat_id=norad, horizon_hours=horizon,
                  train_cut_utc=train_cut.isoformat(),
                  train_samples=len(train), val_samples=len(val),
                  hidden=args.hidden, solver=args.solver, seed=args.seed)
    if len(train) < args.min_train or len(val) < args.min_val:
        return dict(result, status='insufficient_samples'), []
    xtrain, xval = features(train, args.feature_set), features(val, args.feature_set)
    mu, sd = fit_scaler(xtrain)
    xtrain, xval = (xtrain - mu) / sd, (xval - mu) / sd
    w = rng.normal(0, 1 / np.sqrt(xtrain.shape[1]), (xtrain.shape[1], args.hidden))
    b = rng.uniform(-1, 1, args.hidden)
    ht, hv = hidden_matrix(xtrain, w, b), hidden_matrix(xval, w, b)
    ydirect = np.vstack([r['direct_target'] for r in train])
    yresidual = np.vstack([r['residual_target'] for r in train])
    val_ref = np.vstack([r['ref'][:3] for r in val])
    sgp_val = np.vstack([r['sgp'][:3] for r in val])
    t0 = time.perf_counter()
    best_direct, best_res = None, None
    for ridge in args.ridges:
        beta_d = solve(ht, ydirect, ridge, args.solver)
        beta_r = solve(ht, yresidual, ridge, args.solver)
        dscore = rmse(decode_direct(val, hv @ beta_d), val_ref)
        if best_direct is None or dscore < best_direct[0]:
            best_direct = (dscore, ridge, beta_d)
        raw_res = hv @ beta_r
        for alpha in args.alphas:
            score = rmse(decode_residual(val, raw_res, alpha), val_ref)
            if best_res is None or score < best_res[0]:
                best_res = (score, ridge, alpha, beta_r)
    elapsed = time.perf_counter() - t0
    direct_val = decode_direct(val, hv @ best_direct[2])
    residual_val = decode_residual(val, hv @ best_res[3], best_res[2])
    result.update(status='ok', direct_ridge=best_direct[1],
                  residual_ridge=best_res[1], residual_alpha=best_res[2],
                  sgp4_val_rmse_km=rmse(sgp_val, val_ref),
                  direct_elm_val_rmse_km=rmse(direct_val, val_ref),
                  residual_elm_val_rmse_km=rmse(residual_val, val_ref),
                  fit_and_selection_seconds=elapsed)
    predictions = []
    for k, r in enumerate(val):
        predictions.append(dict(norad_cat_id=norad, horizon_hours=horizon,
                                origin_utc=r['case']['origin_utc'],
                                target_utc=r['case']['target_utc'],
                                source_gp_id=r['case']['source_gp_id'],
                                reference_gp_id=r['case']['reference_gp_id'],
                                sgp4_error_km=float(np.linalg.norm(sgp_val[k] - val_ref[k])),
                                direct_elm_error_km=float(np.linalg.norm(direct_val[k] - val_ref[k])),
                                residual_elm_error_km=float(np.linalg.norm(residual_val[k] - val_ref[k]))))
    return result, predictions


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--norads', type=int, nargs='+', default=[45532, 45695, 45755, 46563])
    p.add_argument('--catalog-root', default='data/starlink/processed')
    p.add_argument('--results-root', default='results/starlink')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--feature-set', choices=['baseline', 'orbital', 'drag', 'full'], default='full')
    p.add_argument('--include-mean-motion-derivatives', action='store_true',
                   help='Reserved for future ablation; currently unsupported')
    p.add_argument('--solver', choices=['qr', 'svd'], default='qr')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-fraction', type=float, default=0.70)
    p.add_argument('--ridges', type=float, nargs='+', default=[0.01, 0.1, 1, 10, 100])
    p.add_argument('--alphas', type=float, nargs='+', default=[0, 0.25, 0.5, 1])
    p.add_argument('--min-train', type=int, default=8)
    p.add_argument('--min-val', type=int, default=3)
    a = p.parse_args()
    if a.include_mean_motion_derivatives:
        p.error('Mean-motion derivatives are not implemented; use the supported feature sets')
    if (not 0 < a.train_fraction < 1 or a.hidden < 1 or
            not a.ridges or min(a.ridges) <= 0 or not a.alphas or
            min(a.alphas) < 0 or max(a.alphas) > 1):
        p.error('Invalid split, hidden size, ridge, or alpha configuration')
    for norad in a.norads:
        cases = load_cases(norad, a.catalog_root, a.results_root)
        metrics, forecasts = [], []
        for horizon in sorted({float(r['case']['horizon_hours']) for r in cases}):
            group = [r for r in cases if float(r['case']['horizon_hours']) == horizon]
            # Stable random layer per satellite/horizon, independent of execution order.
            rng = np.random.default_rng(np.random.SeedSequence([a.seed, norad, int(horizon)]))
            m, predictions = run_group(norad, horizon, group, a, rng)
            metrics.append(dict(norad_cat_id=norad, horizon_hours=horizon, **{
                k: v for k, v in m.items() if k not in ('norad_cat_id', 'horizon_hours')}))
            forecasts.extend(predictions)
        out = Path(a.results_root) / str(norad) / f'phase2_orbital_70_30_{a.feature_set}_{a.solver}_{a.hidden}n'
        out.mkdir(parents=True, exist_ok=True)
        metric_cols = ['norad_cat_id', 'horizon_hours', 'status', 'train_cut_utc',
                       'train_samples', 'val_samples', 'hidden', 'solver', 'seed',
                       'direct_ridge', 'residual_ridge', 'residual_alpha',
                       'sgp4_val_rmse_km', 'direct_elm_val_rmse_km',
                       'residual_elm_val_rmse_km', 'fit_and_selection_seconds']
        write_csv(out / 'elm_metrics.csv', metrics, metric_cols)
        write_csv(out / 'elm_per_forecast.csv', forecasts,
                  ['norad_cat_id', 'horizon_hours', 'origin_utc', 'target_utc',
                   'source_gp_id', 'reference_gp_id', 'sgp4_error_km',
                   'direct_elm_error_km', 'residual_elm_error_km'])
        (out / 'experiment.json').write_text(json.dumps(dict(vars(a),
            norad_cat_id=norad, split='chronological 70/30 by distinct forecast origins',
            feature_set=a.feature_set, feature_dimensions=int(features(cases[:1], a.feature_set).shape[1]),
            reference='retrospective later-TLE SGP4 estimate, not truth',
            protocol='training reference creation <= train cutoff; validation uses later origins',
            warning='The 30% validation set selects ridge/alpha AND reports RMSE; '
                    'these are not unbiased held-out test metrics. '
                    'No independent test split exists.'), indent=2)+'\n')
        print(norad, [(m['horizon_hours'], m['status'],
                      m.get('sgp4_val_rmse_km'), m.get('direct_elm_val_rmse_km'),
                      m.get('residual_elm_val_rmse_km')) for m in metrics])


if __name__ == '__main__':
    main()
