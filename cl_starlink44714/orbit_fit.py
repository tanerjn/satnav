"""
Satellite trajectory estimation by least squares (QR / SVD)
Data: Space-Track OMM JSON (SGP4 mean elements, TEME frame, UTC)

Pipeline
  1. Build a "reference" trajectory by propagating the element sets with SGP4
     (nearest-epoch stitching) -> positions r(t) in TEME [km].
  2. Fit a model on a FIT window (QR and SVD solvers, Gauss-Newton).
  3. Predict over a VALIDATION window and compare with the reference -> errors (3D, R/T/N).
"""
import json, sys
import numpy as np, pandas as pd
from sgp4.api import Satrec, WGS72
from sgp4.api import jday
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MU, RE, J2 = 398600.4418, 6378.137, 1.08262668e-3

# ------------------------------------------------------------------ data
def load(path):
    d = pd.DataFrame(json.load(open(path)))
    d["EPOCH"] = pd.to_datetime(d["EPOCH"])
    d = d.sort_values("EPOCH").drop_duplicates("EPOCH").reset_index(drop=True)
    return d

def make_sat(row):
    s = Satrec()
    ep = row.EPOCH
    jd, fr = jday(ep.year, ep.month, ep.day, ep.hour, ep.minute, ep.second + ep.microsecond*1e-6)
    ep_days = (jd + fr) - 2433281.5                      # days since 1949-12-31 00:00
    s.sgp4init(WGS72, 'i', int(row.NORAD_CAT_ID), ep_days, float(row.BSTAR),
               float(row.MEAN_MOTION_DOT)*2*np.pi/1440.0**2,
               0.0, float(row.ECCENTRICITY), np.radians(float(row.ARG_OF_PERICENTER)),
               np.radians(float(row.INCLINATION)), np.radians(float(row.MEAN_ANOMALY)),
               float(row.MEAN_MOTION)*2*np.pi/1440.0, np.radians(float(row.RA_OF_ASC_NODE)))
    return s, jd, fr

def reference_track(df, t_start, t_end, step=120.0, max_age_min=30.0):
    """Reference positions: SGP4 evaluated only within +-max_age_min of an element-set epoch
    (nearest epoch). Avoids extrapolating a TLE over many hours / stitching jumps."""
    sats = [make_sat(r) for r in df.itertuples()]
    ep = df.EPOCH.values.astype("datetime64[us]")
    grid = pd.date_range(t_start, t_end, freq=f"{int(step)}s")
    keep, pos, vel = [], [], []
    for t in grid:
        age = np.abs(ep - np.datetime64(t)).astype("timedelta64[s]").astype(float)
        k = int(np.argmin(age))
        if age[k] > max_age_min*60: continue
        s_, jd0, fr0 = sats[k]
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        e, r, v = s_.sgp4(jd, fr)
        if e != 0: continue
        keep.append(t); pos.append(r); vel.append(v)
    keep = pd.DatetimeIndex(keep)
    return keep, np.array((keep - t_start).total_seconds()), np.array(pos), np.array(vel)

# ------------------------------------------------------------------ model
def kepler(M, e):
    E = M.copy()
    for _ in range(8):
        E = E - (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
    return E

# parameter vector p:
# [a, ex, ey, i, Om0, u0, n, ndot, Omdot, | harmonics (optional): R,T,N x (cos 2u, sin 2u)]
NAMES_BASE = ["a","ex","ey","i","Om0","u0","n","ndot","Omdot"]
def model(p, t, harmonics=True):
    a, ex, ey, inc, Om0, u0, n, ndot, Omd = p[:9]
    e = np.hypot(ex, ey); w0 = np.arctan2(ey, ex)
    pp = a*(1-e**2)
    wdot = 1.5*n*J2*(RE/pp)**2*(2 - 2.5*np.sin(inc)**2)          # J2 secular perigee drift (fixed)
    w  = w0 + wdot*t
    M  = (u0 - w0) + n*t + 0.5*ndot*t**2      # u0 = w0 + M0 : well-conditioned for e ~ 0
    Om = Om0 + Omd*t
    E  = kepler(np.mod(M, 2*np.pi), e)
    xo, yo = a*(np.cos(E) - e), a*np.sqrt(1-e**2)*np.sin(E)
    out = np.empty((len(t), 3))
    u = w + M                                                       # mean argument of latitude
    # RTN correction (like the Crc/Crs/Cuc/Cus/Cic/Cis terms of GNSS broadcast ephemeris)
    dR = dT = dN = 0.0
    if harmonics and len(p) > 9:
        h = p[9:].reshape(3, 2)
        B = np.stack([np.cos(2*u), np.sin(2*u)], axis=1)      # 2/rev only: 1/rev is degenerate with e, i, Om
        dR, dT, dN = B @ h[0], B @ h[1], B @ h[2]
    # rotate each point (vectorised)
    cO, sO, cw, sw, ci, si = np.cos(Om), np.sin(Om), np.cos(w), np.sin(w), np.cos(inc), np.sin(inc)
    X = (cO*cw - sO*sw*ci)*xo + (-cO*sw - sO*cw*ci)*yo
    Y = (sO*cw + cO*sw*ci)*xo + (-sO*sw + cO*cw*ci)*yo
    Z = (sw*si)*xo + (cw*si)*yo
    r = np.stack([X, Y, Z], axis=1)
    if harmonics and len(p) > 9:
        rh = r/np.linalg.norm(r, axis=1)[:, None]
        nh = np.stack([si*np.sin(Om)*np.ones_like(u), -si*np.cos(Om)*np.ones_like(u), ci*np.ones_like(u)], axis=1)
        th = np.cross(nh, rh)
        r = r + dR[:, None]*rh + dT[:, None]*th + dN[:, None]*nh
    return r

def initial_guess(row, t_fit0, harmonics):
    e = float(row.ECCENTRICITY); w = np.radians(float(row.ARG_OF_PERICENTER))
    n = float(row.MEAN_MOTION)*2*np.pi/86400
    a = (MU/n**2)**(1/3); inc = np.radians(float(row.INCLINATION))
    pp = a*(1-e**2)
    Omd = -1.5*n*J2*(RE/pp)**2*np.cos(inc)
    dt = (t_fit0 - row.EPOCH).total_seconds()
    M0 = np.radians(float(row.MEAN_ANOMALY)) + n*dt
    Om0 = np.radians(float(row.RA_OF_ASC_NODE)) + Omd*dt
    p = [a, e*np.cos(w), e*np.sin(w), inc, Om0, M0 + w, n, 0.0, Omd]
    if harmonics: p += [0.0]*6
    return np.array(p)

# ------------------------------------------------------------------ least squares
def solve_ls(J, r, method):
    """min ||J dx - r||_2 ; column-equilibrated; returns dx, cond"""
    sc = np.linalg.norm(J, axis=0); sc[sc == 0] = 1
    Js = J/sc
    if method == "qr":
        Q, R = np.linalg.qr(Js)
        dx = np.linalg.solve(R, Q.T@r); cond = np.linalg.cond(R)
    else:
        U, s, Vt = np.linalg.svd(Js, full_matrices=False)
        keep = s > 1e-12*s[0]
        sk = s[keep] if r.ndim == 1 else s[keep][:, None]
        dx = Vt[keep].T@((U[:, keep].T@r)/sk); cond = s[0]/s[-1]
    return (dx/sc if dx.ndim == 1 else dx/sc[:, None]), cond

def fit(t, obs, p0, method="qr", harmonics=True, iters=15, verbose=False, fix_ndot=False):
    p = p0.copy()
    # per-parameter finite-difference steps
    step = np.array([1e-3, 1e-6, 1e-6, 1e-7, 1e-7, 1e-6, 1e-10, 1e-14, 1e-12] + [1e-3]*(len(p)-9))
    for it in range(iters):
        f0 = model(p, t, harmonics)
        res = (obs - f0).ravel()                       # stacked x,y,z residuals
        J = np.empty((res.size, len(p)))
        for j in range(len(p)):
            q = p.copy(); q[j] += step[j]
            J[:, j] = ((model(q, t, harmonics) - f0)/step[j]).ravel()
        free = np.ones(len(p), bool); free[7] = not fix_ndot       # optionally freeze ndot (index 7)
        dx = np.zeros(len(p)); dx[free], cond = solve_ls(J[:, free], res, method)
        p = p + dx
        rms = np.sqrt(np.mean(res**2))
        if verbose: print(f"  it{it}: rms={rms*1e3:9.2f} m  |dx|={np.linalg.norm(dx):.2e} cond={cond:.2e}")
        if it > 0 and abs(prev - rms) < 1e-6*rms: break
        prev = rms
    return p, cond

# baseline: Chebyshev polynomial per axis (linear LS)
def cheb_fit(t, obs, deg, method):
    t0, T = t.mean(), (t[-1]-t[0])/2
    A = np.polynomial.chebyshev.chebvander((t-t0)/T, deg)
    sc = np.linalg.norm(A, axis=0)
    c, _ = solve_ls(A, obs, method)
    f = lambda tt: np.polynomial.chebyshev.chebvander((tt-t0)/T, deg) @ c
    return f, np.linalg.cond(A)

# ------------------------------------------------------------------ errors
def rtn_errors(r_ref, v_ref, r_pred):
    Rh = r_ref/np.linalg.norm(r_ref, axis=1)[:, None]
    Nh = np.cross(r_ref, v_ref); Nh /= np.linalg.norm(Nh, axis=1)[:, None]
    Th = np.cross(Nh, Rh)
    d = r_pred - r_ref
    return np.stack([(d*Rh).sum(1), (d*Th).sum(1), (d*Nh).sum(1)], axis=1)

# ------------------------------------------------------------------ main
def run_case(df, t0, fit_h, val_h, harm=True, method="qr", fix_ndot=False):
    grid, ts, pos, vel = reference_track(df, t0, t0 + pd.Timedelta(hours=fit_h+val_h))
    fm = ts <= fit_h*3600
    if fm.sum() < 20 or (~fm).sum() < 5: return None
    nearest = df[df.EPOCH <= t0].iloc[-1]
    p0 = initial_guess(nearest, t0, harm)
    p, cond = fit(ts[fm], pos[fm], p0, method, harm, fix_ndot=fix_ndot)
    fit_rms = np.sqrt(np.mean(np.sum((model(p, ts[fm], harm) - pos[fm])**2, 1)))
    err = rtn_errors(pos[~fm], vel[~fm], model(p, ts[~fm], harm))
    # operational baseline: latest element set available at the end of the fit window, SGP4-propagated
    t_end = t0 + pd.Timedelta(hours=fit_h)
    last = df[df.EPOCH <= t_end].iloc[-1]; sat = make_sat(last)[0]
    base = []
    for tt in grid[~fm]:
        jd, fr = jday(tt.year, tt.month, tt.day, tt.hour, tt.minute, tt.second)
        base.append(sat.sgp4(jd, fr)[1])
    err_b = np.linalg.norm(np.array(base) - pos[~fm], axis=1)
    return dict(p=p, cond=cond, fit_rms=fit_rms, t_val=ts[~fm] - fit_h*3600, err=err, err3_sgp4=err_b,
                err3=np.linalg.norm(err, axis=1), ts=ts, pos=pos, vel=vel, fm=fm)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/starlink_44714.json"
    df = load(path)
    # quiet arc after the January orbit-lowering manoeuvre (stable ~472 km)
    QUIET0, QUIET1 = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-10")

    # ---------- 1) one detailed case: 48 h fit -> 72 h prediction
    t0 = pd.Timestamp("2026-02-05 00:00:00"); FIT_H, VAL_H = 72, 72
    case = {}
    for harm in (False, True):
        for m in ("qr", "svd"):
            case[(harm, m)] = run_case(df, t0, FIT_H, VAL_H, harm, m, fix_ndot=True)
    case[('free', 'qr')] = run_case(df, t0, FIT_H, VAL_H, True, 'qr', fix_ndot=False)
    # polynomial baseline
    grid, ts, pos, vel = reference_track(df, t0, t0 + pd.Timedelta(hours=FIT_H+VAL_H))
    fm = ts <= FIT_H*3600
    poly = {}
    for m in ("qr", "svd"):
        f, c = cheb_fit(ts[fm], pos[fm], 8, m)
        e = rtn_errors(pos[~fm], vel[~fm], f(ts[~fm]))
        poly[m] = dict(cond=c, fit_rms=np.sqrt(np.mean(np.sum((f(ts[fm])-pos[fm])**2, 1))), err=e, err3=np.linalg.norm(e, axis=1), t_val=ts[~fm]-FIT_H*3600)

    print(f"Case: fit {FIT_H} h from {t0}, predict next {VAL_H} h  ({fm.sum()} fit pts, {(~fm).sum()} validation pts)")
    print(f"{'model':36s}{'cond':>9s}{'fit RMS m':>11s}{'val RMS km':>12s}{'max km':>9s}{'err@24h':>9s}{'err@72h':>9s}")
    def row(name, r):
        tv, e3 = r["t_val"]/3600, r["err3"]
        at = lambda h: e3[np.argmin(np.abs(tv-h))]
        print(f"{name:36s}{r['cond']:9.1e}{r['fit_rms']*1e3:11.0f}{np.sqrt(np.mean(e3**2)):12.3g}{e3.max():10.3g}{at(24):10.3g}{at(72):10.3g}")
    for (h, m), r in case.items(): row({True: "Kepler+J2+2u harm, ndot=0", False: "Kepler+J2, ndot=0", "free": "Kepler+J2+2u harm, ndot free"}[h] + f" [{m}]", r)
    row('SGP4 baseline (last element set)', dict(cond=np.nan, fit_rms=np.nan, t_val=case[(True,'qr')]['t_val'], err3=case[(True,'qr')]['err3_sgp4']))
    for m, r in poly.items(): row(f"Chebyshev deg 8 [{m}]", r)

    # ---------- 2) rolling experiment: error vs. prediction horizon
    horizons = np.array([6, 12, 24, 48, 72])
    summary = {}
    for fit_h, fixn in ((24, False), (48, False), (72, False), (48, True), (72, True)):
        key = fit_h if not fixn else f'{fit_h}fix'
        errs = {h: [] for h in horizons}; errb = {h: [] for h in horizons}
        for start in pd.date_range(QUIET0, QUIET1, freq="12h"):
            r = run_case(df, start, fit_h, 72, True, "qr", fixn)
            if r is None: continue
            for h in horizons:
                i = np.where(np.abs(r["t_val"]/3600 - h) < 0.6)[0]
                if len(i): j = i[np.argmin(r['err3'][i])]; errs[h].append(r['err3'][j]); errb[h].append(r['err3_sgp4'][j])
        summary[key] = {h: (np.median(v), np.percentile(v, 90), len(v)) for h, v in errs.items() if v}
        if fixn: continue
        summary[f'sgp4_{fit_h}'] = {h: (np.median(v), np.percentile(v, 90), len(v)) for h, v in errb.items() if v}
    print("\nRolling experiment (starts every 12 h, %s .. %s); 3D error in km, median / 90th pct" % (QUIET0.date(), QUIET1.date()))
    print("fit window  " + "".join(f"{'+%dh'%h:>16s}" for h in horizons))
    for fit_h, d in summary.items():
        if str(fit_h).startswith('sgp4_') and fit_h != 'sgp4_48': continue
        print(f"{(str(fit_h)+' h') if isinstance(fit_h,int) else fit_h:>10s}  " + "".join(f"{d[h][0]:8.2f} /{d[h][1]:6.2f}" if h in d else " "*16 for h in horizons))

    # ---------- plots
    best, kep = case[(True, "qr")], case[(False, "qr")]
    th = best["t_val"]/3600
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    pass  # Chebyshev error (~1e9 km) omitted from plot: it would flatten the other curves
    ax[0,0].semilogy(kep["t_val"]/3600, kep["err3"], ".", ms=3, label="Kepler+J2 (ndot=0)")
    ax[0,0].semilogy(case[("free","qr")]["t_val"]/3600, case[("free","qr")]["err3"], ".", ms=3, label="Kepler+J2+harm (ndot free)")
    ax[0,0].semilogy(best["t_val"]/3600, best["err3_sgp4"], ".", ms=3, c="gray", label="SGP4, last element set")
    ax[0,0].semilogy(th, best["err3"], ".", ms=3, label="Kepler+J2+2u harm (ndot=0)")
    ax[0,0].set(title=f"3D error after {FIT_H} h fit window", xlabel="hours after end of fit window", ylabel="km"); ax[0,0].legend(); ax[0,0].grid(alpha=.3)
    for j, nm in enumerate(["Radial", "Along-track", "Cross-track"]): ax[0,1].plot(th, best["err"][:, j], ".", ms=3, label=nm)
    ax[0,1].set(title="Best model: RTN errors", xlabel="hours after end of fit window", ylabel="km"); ax[0,1].legend(); ax[0,1].grid(alpha=.3)
    q, sv = case[(True, "qr")], case[(True, "svd")]
    ax[1,0].plot(th, q["err3"], "o", ms=5, label="QR"); ax[1,0].plot(th, sv["err3"], "x", ms=5, label="SVD")
    ax[1,0].set(title=f"QR vs SVD solution (max |dp| = {np.max(np.abs(q['p']-sv['p'])/(np.abs(q['p'])+1e-30)):.1e} relative)", xlabel="hours", ylabel="km"); ax[1,0].legend(); ax[1,0].grid(alpha=.3)
    for fit_h, d in summary.items():
        if str(fit_h).startswith('sgp4_') and fit_h != 'sgp4_48': continue
        hs = sorted(d); ax[1,1].semilogy(hs, [d[h][0] for h in hs], marker="o", label=(f"fit {fit_h} h" if isinstance(fit_h,int) else ("fit "+fit_h.replace("fix"," h, ndot=0") if "fix" in fit_h else "SGP4 baseline (last set)")), ls=("--" if not isinstance(fit_h,int) else "-"))
    ax[1,1].set(title="Rolling experiment: error vs prediction horizon", xlabel="hours after fit window", ylabel="km"); ax[1,1].legend(); ax[1,1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("/mnt/user-data/outputs/orbit_fit_results.png", dpi=110)
