#!/usr/bin/env python3
"""Visualize historical Starlink SGP4 forecasts and as-of-time TLE estimates.

The historical TLE-derived reference is NOT independent truth. The reference
changes as newer TLEs become available; gaps and switches are made explicit.

Usage:
 python scripts/plot_starlink_trajectories_improved.py --norad 45532 --horizon-hours 24 --case-index 0
 python scripts/plot_starlink_trajectories_improved.py --norad 45532 --horizon-hours 72 --source-gp-id 307394565
 python scripts/plot_starlink_trajectories_improved.py --norad 45532 --compare-horizons --source-gp-id 307394565
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

EARTH_KM = 6378.137
DEFAULT_HORIZONS = (1, 6, 12, 24, 72, 168)


def utc(value):
    t = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('Timestamp must contain UTC offset: ' + value)
    return t.astimezone(timezone.utc)


def position(sat, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    code, r, _ = sat.sgp4(jd, fr)
    return np.asarray(r if code == 0 else (np.nan,) * 3, dtype=float)


def load_catalog(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        sat = Satrec.twoline2rv(row['tle_line1'], row['tle_line2'])
        if sat.error:
            raise ValueError('Invalid TLE: ' + str(row.get('gp_id')))
        result.append(dict(gp_id=row['gp_id'], epoch=utc(row['epoch_utc']),
                           created=utc(row['creation_utc']), sat=sat))
    if not result:
        raise ValueError('Empty catalog')
    return result


def load_cases(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def select_case(cases, horizon, index, gp_id, origin):
    candidates = [c for c in cases if abs(float(c['horizon_hours']) - horizon) < 1e-8]
    if gp_id is not None:
        candidates = [c for c in candidates if c['source_gp_id'] == gp_id]
    if origin is not None:
        candidates = [c for c in candidates if utc(c['origin_utc']) == utc(origin)]
    if not candidates:
        return None
    if index >= len(candidates):
        raise ValueError(f'case-index {index} exceeds matching case count {len(candidates)}')
    return candidates[index]


def lookup(records, gp_id, epoch):
    found = [r for r in records if r['gp_id'] == gp_id and r['epoch'] == epoch]
    if len(found) != 1:
        raise ValueError(f'Cannot uniquely find GP_ID={gp_id} at {epoch}')
    return found[0]


def reference_series(records, times, origin, tolerance_h):
    """As-of-time reference; never uses an update before its creation time.

    The reference at t uses a post-origin TLE published by t, with orbital epoch
    no farther than tolerance_h from t. NaNs indicate unavailable references.
    """
    coords = np.full((len(times), 3), np.nan)
    selected = [None] * len(times)
    for i, t in enumerate(times):
        eligible = [r for r in records if origin < r['created'] <= t and
                    abs((r['epoch'] - t).total_seconds()) <= tolerance_h * 3600]
        if not eligible:
            continue
        best = min(eligible, key=lambda r: (abs((r['epoch']-t).total_seconds()),
                                            -r['created'].timestamp(), r['gp_id']))
        coords[i] = position(best['sat'], t)
        selected[i] = best['gp_id']
    return coords, selected


def draw_earth(ax):
    u = np.linspace(0, 2*np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    x = EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='steelblue', alpha=.22, linewidth=0)


def equal_axes(ax, coords):
    valid = coords[np.isfinite(coords).all(axis=1)]
    span = max(EARTH_KM*1.12, np.max(np.abs(valid)) * 1.04) if len(valid) else EARTH_KM*1.12
    ax.set(xlim=(-span, span), ylim=(-span, span), zlim=(-span, span))
    ax.set_box_aspect((1, 1, 1))


def prepare(records, case, samples, tolerance_h):
    origin, target = utc(case['origin_utc']), utc(case['target_utc'])
    source = lookup(records, case['source_gp_id'], utc(case['source_epoch_utc']))
    endpoint = lookup(records, case['reference_gp_id'], utc(case['reference_epoch_utc']))
    if source['created'] > origin or endpoint['created'] < target:
        raise ValueError('Creation-time availability rule violated')
    times = [origin + (target-origin)*i/(samples-1) for i in range(samples)]
    pred = np.vstack([position(source['sat'], t) for t in times])
    ref, ids = reference_series(records, times, origin, tolerance_h)
    # Match the corrected Phase 1 CSV exactly at the endpoint, and mark this
    # explicitly: the selected endpoint TLE may differ from as-of-time curve.
    endpoint_position = position(endpoint['sat'], target)
    endpoint_error = float(np.linalg.norm(pred[-1] - endpoint_position))
    if not np.isclose(endpoint_error * 1000, float(case['position_error_m']), atol=.1):
        raise ValueError('Endpoint mismatch with corrected Phase 1 CSV')
    valid = np.isfinite(ref).all(axis=1) & np.isfinite(pred).all(axis=1)
    errors = np.full(samples, np.nan)
    errors[valid] = np.linalg.norm(pred[valid] - ref[valid], axis=1)
    # Discontinuities from changing TLE should not be drawn as continuous arcs.
    segmented = ref.copy()
    for i in range(1, len(ids)):
        if ids[i] != ids[i-1]:
            segmented[i] = np.nan
    changes = [i for i in range(1, len(ids)) if ids[i] is not None and ids[i] != ids[i-1]]
    updates = sorted({r['gp_id']: r for r in records
                      if origin < r['created'] <= target}.values(), key=lambda r: r['created'])
    return dict(origin=origin, target=target, source=source, endpoint=endpoint,
                times=times, pred=pred, ref=ref, segmented=segmented, ids=ids,
                errors=errors, valid=valid, changes=changes, updates=updates,
                endpoint_position=endpoint_position, endpoint_error=endpoint_error)


def plot_single(data, norad, horizon, case, output):
    hours = np.array([(t-data['origin']).total_seconds()/3600 for t in data['times']])
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection='3d')
    draw_earth(ax)
    ax.plot(*data['pred'].T, color='darkorange', lw=1.4, label='SGP4 forecast')
    ax.plot(*data['segmented'].T, color='royalblue', lw=1.4,
            label='As-of-time updated TLE estimates (gaps/switches)')
    ax.scatter(*data['pred'][0], color='green', s=36, label='Forecast origin')
    ax.scatter(*data['pred'][-1], color='darkorange', marker='x', s=60)
    ax.scatter(*data['endpoint_position'], color='royalblue', s=38,
               label='Phase 1 endpoint reference')
    ax.plot([data['pred'][-1, 0], data['endpoint_position'][0]],
            [data['pred'][-1, 1], data['endpoint_position'][1]],
            [data['pred'][-1, 2], data['endpoint_position'][2]],
            'r--', lw=1.8, label=f'Endpoint error: {data["endpoint_error"]:.2f} km')
    equal_axes(ax, np.vstack([data['pred'], data['ref'], data['endpoint_position'][None]]))
    ax.set(xlabel='TEME X (km)', ylabel='TEME Y (km)', zlabel='TEME Z (km)',
           title='Earth and SGP4 trajectories')
    ax.legend(fontsize=7, loc='upper center')

    bx = fig.add_subplot(122)
    # Draw each continuous reference regime separately to avoid false jumps.
    regime = data['ids']
    for i in range(len(hours)-1):
        if regime[i] is not None and regime[i] == regime[i+1] and np.isfinite(data['errors'][i:i+2]).all():
            bx.plot(hours[i:i+2], data['errors'][i:i+2], color='crimson', lw=1.3)
    for i in data['changes']:
        bx.axvline(hours[i], color='grey', lw=.5, alpha=.25)
    bx.scatter([horizon], [data['endpoint_error']], c='black', marker='x', s=55,
               label='Corrected Phase 1 endpoint error')
    bx.set(xlabel='Hours since forecast origin', ylabel='Position difference (km)',
           title='Error against as-of-time updated TLE estimates')
    bx.grid(alpha=.25)
    bx.legend(fontsize=8)
    used = len(set(x for x in data['ids'] if x is not None))
    bx.text(.02, .98, f'Endpoint error: {data["endpoint_error"]:.2f} km\n'
            f'Available plotted samples: {int(data["valid"].sum())}/{len(hours)}\n'
            f'Actual new TLE releases: {len(data["updates"])}\n'
            f'Distinct reference TLEs used: {used}\n'
            f'Reference regime starts/switches: {len(data["changes"])}',
            transform=bx.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    fig.suptitle(f'STARLINK {norad} | {horizon:g} h | GP_ID {case["source_gp_id"]} | '
                 f'origin {data["origin"]:%Y-%m-%d %H:%M} UTC', fontsize=12)
    fig.text(.5, .012, 'TLE-derived reference is NOT measured ground truth. Endpoint reference '
             'may differ from intermediate as-of-time TLE selections.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(rows, records, norad, samples, tolerance, output):
    fig, ax = plt.subplots(figsize=(11, 6))
    lines = []
    for case in rows:
        horizon = float(case['horizon_hours'])
        d = prepare(records, case, samples, tolerance)
        hrs = np.array([(t-d['origin']).total_seconds()/3600 for t in d['times']])
        # Same source, and all plotted time series have the same as-of-time
        # reference rule. Only the last endpoint is a separately selected TLE.
        ax.plot(hrs, d['errors'], lw=1.4, label=f'{horizon:g}h (endpoint {d["endpoint_error"]:.1f} km)')
        ax.scatter([horizon], [d['endpoint_error']], marker='x', s=40)
        lines.append((horizon, d['endpoint_error']))
    ax.set(xlabel='Hours since common forecast origin', ylabel='Position difference (km)',
           title=f'STARLINK {norad}: same source TLE, multiple horizons')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.text(.5, .015, 'Reference is updated TLE-derived, not measured truth; x markers use '
             'Phase 1 selected endpoint TLEs.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--norad', type=int, default=45532)
    p.add_argument('--horizon-hours', type=float, choices=DEFAULT_HORIZONS, default=24)
    p.add_argument('--case-index', type=int, default=0)
    p.add_argument('--source-gp-id', help='Select fixed starting TLE (recommended for comparisons)')
    p.add_argument('--origin-utc', help='Exact forecast origin, e.g. 2026-01-01T03:14:26+00:00')
    p.add_argument('--compare-horizons', action='store_true', help='Compare horizons with one common source')
    p.add_argument('--samples', type=int, default=400)
    p.add_argument('--reference-tolerance-hours', type=float, default=3)
    p.add_argument('--catalog-root', default='data/starlink/processed')
    p.add_argument('--results-root', default='results/starlink')
    args = p.parse_args()
    if args.samples < 2 or args.case_index < 0 or args.reference_tolerance_hours < 0:
        p.error('samples >= 2, case-index >= 0, tolerance >= 0 required')
    outdir = Path(args.results_root) / str(args.norad) / 'phase1_corrected'
    records = load_catalog(Path(args.catalog_root) / str(args.norad) / 'tle_catalog.csv')
    cases = load_cases(outdir / 'sgp4_per_forecast.csv')
    if args.compare_horizons:
        if not args.source_gp_id and not args.origin_utc:
            p.error('--compare-horizons requires --source-gp-id or --origin-utc to fix the origin')
        chosen = []
        for h in DEFAULT_HORIZONS:
            c = select_case(cases, h, 0, args.source_gp_id, args.origin_utc)
            if c is not None:
                chosen.append(c)
        if not chosen:
            p.error('No matching corrected Phase 1 forecast cases')
        origins = {(c['source_gp_id'], c['origin_utc']) for c in chosen}
        if len(origins) != 1:
            p.error('Selected horizons do not share the same source GP_ID and origin')
        source_id = chosen[0]['source_gp_id']
        output = outdir / f'trajectory_comparison_source{source_id}.png'
        report = plot_comparison(chosen, records, args.norad, args.samples,
                                 args.reference_tolerance_hours, output)
        print('Horizon endpoint errors (km):', report)
        missing = sorted(set(DEFAULT_HORIZONS) - {float(c['horizon_hours']) for c in chosen})
        if missing:
            print('No eligible endpoint reference for these horizons:', missing)
    else:
        case = select_case(cases, args.horizon_hours, args.case_index,
                           args.source_gp_id, args.origin_utc)
        if case is None:
            p.error('No matching case. Choose another horizon/source/origin')
        d = prepare(records, case, args.samples, args.reference_tolerance_hours)
        source_id = case['source_gp_id']
        output = outdir / (f'trajectory_{args.horizon_hours:g}h_'
                           f'source{source_id}.png')
        plot_single(d, args.norad, args.horizon_hours, case, output)
        print(f'Endpoint error: {d["endpoint_error"]:.3f} km')
        print(f'Actual new TLE releases: {len(d["updates"])}; '
              f'available reference samples: {int(d["valid"].sum())}/{args.samples}')
        print('New TLE release timestamps (UTC):')
        for rec in d['updates']:
            print(f'  {rec["created"].isoformat()} GP_ID={rec["gp_id"]}')
    print('Saved:', output)


if __name__ == '__main__':
    main()
