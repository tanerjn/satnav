#!/usr/bin/env python3
"""Plot SGP4 forecast versus historical TLE-derived reference around Earth.

Inputs: catalog from starlink_prepare.py and per-forecast CSV from
starlink_phase1_corrected.py. All positions are TEME km, shown in a common
inertial-style Cartesian frame. TLE-derived references are NOT measured truth.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sgp4.api import Satrec, jday

EARTH_RADIUS_KM = 6378.137
HORIZONS = (1, 6, 12, 24, 72, 168)


def parse_time(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError(f'Timestamp lacks UTC offset: {value}')
    return t.astimezone(timezone.utc)


def sgp4_position(sat, t):
    jd, fraction = jday(t.year, t.month, t.day, t.hour, t.minute,
                        t.second + t.microsecond / 1e6)
    error, position, _ = sat.sgp4(jd, fraction)
    if error:
        return np.full(3, np.nan)
    return np.asarray(position, dtype=float)


def read_catalog(path):
    with open(path, newline='', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError(f'Empty catalog: {path}')
    result = []
    for record in records:
        sat = Satrec.twoline2rv(record['tle_line1'], record['tle_line2'])
        if sat.error:
            raise ValueError('Invalid TLE in catalog')
        result.append({
            'epoch': parse_time(record['epoch_utc']),
            'created': parse_time(record['creation_utc']),
            'gp_id': record.get('gp_id', ''),
            'sat': sat,
        })
    return result


def read_cases(path, horizon):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if abs(float(r['horizon_hours']) - horizon) < 1e-8]


def find_record(records, gp_id, epoch):
    matches = [r for r in records if r['gp_id'] == gp_id and r['epoch'] == epoch]
    if len(matches) != 1:
        raise ValueError(f'Cannot uniquely match TLE GP_ID={gp_id}, epoch={epoch}')
    return matches[0]


def reference_at_time(records, t, origin, tolerance_hours):
    """Use closest TLE epoch among records created no later than time t.

    This is a time-varying, as-of-time TLE-derived reference, not a single
    physical ground-truth trajectory. Before the first post-origin TLE becomes
    available, the reference is unavailable (NaN).
    """
    eligible = [r for r in records
                if origin < r['created'] <= t
                and abs((r['epoch'] - t).total_seconds()) <= tolerance_hours * 3600]
    if not eligible:
        return np.full(3, np.nan)
    best = min(eligible, key=lambda r: (abs((r['epoch'] - t).total_seconds()),
                                        -r['created'].timestamp()))
    return sgp4_position(best['sat'], t)


def plot_earth(ax):
    u = np.linspace(0, 2*np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='steelblue', alpha=0.23,
                    linewidth=0, shade=True, zorder=0)


def set_equal_axes(ax, positions):
    valid = positions[np.isfinite(positions).all(axis=1)]
    if len(valid) == 0:
        valid = np.zeros((1, 3))
    span = max(EARTH_RADIUS_KM * 1.15, np.max(np.abs(valid)))
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-span, span)
    ax.set_box_aspect((1, 1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--norad', type=int, default=45532)
    parser.add_argument('--horizon-hours', type=float, choices=HORIZONS, default=24)
    parser.add_argument('--case-index', type=int, default=0,
                        help='Zero-based row among matching horizon cases in per-forecast CSV')
    parser.add_argument('--samples', type=int, default=400)
    parser.add_argument('--reference-tolerance-hours', type=float, default=3,
                        help='Maximum distance between reference TLE epoch and sampled time')
    parser.add_argument('--catalog-root', default='data/starlink/processed')
    parser.add_argument('--results-root', default='results/starlink')
    args = parser.parse_args()
    if args.samples < 2 or args.case_index < 0 or args.reference_tolerance_hours < 0:
        parser.error('samples >= 2, case-index >= 0, tolerance >= 0 required')

    catalog_path = Path(args.catalog_root) / str(args.norad) / 'tle_catalog.csv'
    output_dir = Path(args.results_root) / str(args.norad) / 'phase1_corrected'
    cases_path = output_dir / 'sgp4_per_forecast.csv'
    records = read_catalog(catalog_path)
    cases = read_cases(cases_path, args.horizon_hours)
    if not cases:
        raise ValueError('No eligible forecast cases for this horizon. Check corrected Phase 1 results.')
    if args.case_index >= len(cases):
        raise ValueError(f'case-index must be 0..{len(cases)-1}')

    case = cases[args.case_index]
    origin = parse_time(case['origin_utc'])
    target = parse_time(case['target_utc'])
    source = find_record(records, case['source_gp_id'], parse_time(case['source_epoch_utc']))
    endpoint_ref = find_record(records, case['reference_gp_id'],
                               parse_time(case['reference_epoch_utc']))
    if source['created'] > origin or endpoint_ref['created'] < target:
        raise ValueError('Case violates creation-time availability rules')

    times = [origin + (target-origin) * (i/(args.samples-1))
             for i in range(args.samples)]
    prediction = np.vstack([sgp4_position(source['sat'], t) for t in times])
    reference = np.vstack([reference_at_time(records, t, origin,
                          args.reference_tolerance_hours) for t in times])
    # The selected reference TLE is known to be available at the exact target.
    reference[-1] = sgp4_position(endpoint_ref['sat'], target)
    valid = np.isfinite(prediction).all(axis=1) & np.isfinite(reference).all(axis=1)
    errors = np.full(len(times), np.nan)
    errors[valid] = np.linalg.norm(prediction[valid] - reference[valid], axis=1)
    endpoint_error = np.linalg.norm(prediction[-1] - reference[-1])

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(121, projection='3d')
    plot_earth(ax)
    ax.plot(*prediction.T, color='darkorange', lw=1.6,
            label='Earlier TLE: SGP4 forecast')
    # NaNs break the line where there is no valid as-of-time reference.
    ax.plot(*reference.T, color='royalblue', lw=1.5,
            label='Updated TLE reference (available at each time)')
    ax.scatter(*prediction[0], c='green', s=40, label='Forecast origin')
    ax.scatter(*prediction[-1], c='darkorange', marker='x', s=60)
    ax.scatter(*reference[-1], c='royalblue', marker='o', s=35)
    ax.plot([prediction[-1, 0], reference[-1, 0]],
            [prediction[-1, 1], reference[-1, 1]],
            [prediction[-1, 2], reference[-1, 2]],
            'r--', lw=2, label=f'Endpoint separation: {endpoint_error:.1f} km')
    set_equal_axes(ax, np.vstack([prediction, reference]))
    ax.set(xlabel='TEME X (km)', ylabel='TEME Y (km)', zlabel='TEME Z (km)')
    ax.set_title('3D Earth and orbital trajectories')
    ax.legend(loc='upper center', fontsize=8)

    bx = fig.add_subplot(122)
    hours = np.array([(t-origin).total_seconds()/3600 for t in times])
    bx.plot(hours, errors, color='crimson', lw=1.8)
    bx.scatter([args.horizon_hours], [endpoint_error], color='crimson', zorder=3)
    bx.set(xlabel='Time since forecast origin (hours)',
           ylabel='Position difference (km)',
           title='SGP4 forecast vs updated TLE estimates')
    bx.grid(True, alpha=0.3)
    bx.text(0.02, 0.98,
            f'Endpoint error: {endpoint_error:.2f} km\n'
            f'Reference available at {int(valid.sum())}/{len(valid)} samples',
            transform=bx.transAxes, va='top', fontsize=10)
    fig.suptitle(f'STARLINK NORAD {args.norad} | {args.horizon_hours:g} h | '
                 f'case {args.case_index} | origin {origin:%Y-%m-%d %H:%M} UTC',
                 fontsize=12)
    fig.text(0.5, 0.015,
             'Reference is SGP4-derived from later TLEs, not independently measured truth. '
             'Reference TLE may change along the curve.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.horizon_hours:g}h'.replace('.', 'p')
    output = output_dir / f'trajectory_{tag}_case{args.case_index:03d}.png'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output}')
    print(f'Endpoint error: {endpoint_error:.3f} km')
    print(f'Available reference samples: {int(valid.sum())}/{len(valid)}')


if __name__ == '__main__':
    main()

