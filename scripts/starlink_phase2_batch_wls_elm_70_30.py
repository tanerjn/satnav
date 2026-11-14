#!/usr/bin/env python3
"""Compare SGP4, batch J2/J3/J4 WLS, and residual ELM (<=72h).

All state vectors are TLE/SGP4-derived TEME estimates, NOT independent truth.
Uses only TLEs created by each forecast origin. The final 30% is validation
and is also used to select ELM hyperparameters: NOT an independent test.
Requires scipy, numpy, sgp4. Run Phase 1 corrected first.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sgp4.api import Satrec, jday

MU, RE = 398600.4418, 6378.137
J2, J3, J4 = 1.08262668e-3, -2.53265649e-6, -1.61962159e-6

def utc(x):
    return datetime.fromisoformat(x.replace('Z', '+00:00')).astimezone(timezone.utc)

def sgp(s, t):
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                  t.second + t.microsecond / 1e6)
    err, r, v = s.sgp4(jd, fr)
    if err:
        raise ValueError(f'SGP4 error {err}')
    return np.array([*r, *v])

def acceleration(r):
    x, y, z = r
    rn = np.linalg.norm(r)
    zz = (z / rn)**2
    a0 = -MU * r / rn**3
    k2 = -1.5 * J2 * MU * RE**2 / rn**5
    a2 = k2 * np.array([x*(1-5*zz), y*(1-5*zz), z*(3-5*zz)])
    k3 = -2.5 * J3 * MU * RE**3 / rn**7
    f3 = 3*z - 7*z**3/rn**2
    a3 = k3 * np.array([x*f3, y*f3, 6*z*z-7*z**4/rn**2-0.6*rn**2])
    k4 = 15/8 * J4 * MU * RE**4 / rn**7
    f4 = 1-14*zz+21*zz**2
    a4 = k4 * np.array([x*f4, y*f4, z*(5-70/3*zz+21*zz**2)])
    return a0+a2+a3+a4

def propagate(p, seconds):
    def rhs(t, x):
        v = x[3:]
        return np.r_[v, acceleration(x[:3])+p[6]*v/np.linalg.norm(v)]
    if abs(seconds) < 1e-10:
        return p[:6].copy()
    sol = solve_ivp(rhs, (0, seconds), p[:6], method='DOP853',
                    rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise ValueError(sol.message)
    return sol.y[:, -1]

def fit_batch(records, origin, window_days, min_obs, accel, max_nfev):
    eligible = [r for r in records if r['created'] <= origin
                and origin.timestamp()-window_days*86400 <= r['epoch'].timestamp()
                and r['epoch'] <= origin]
    # One observation per epoch; prefer latest available publication.
    by_epoch = {}
    for r in eligible:
        key = r['epoch']
        if key not in by_epoch or r['created'] > by_epoch[key]['created']:
            by_epoch[key] = r
    selected = sorted(by_epoch.values(), key=lambda x: x['epoch'])
    if len(selected) < min_obs:
        raise ValueError('insufficient historical observations')
    # Use the most recent historical epoch as the fit reference.
    t0 = selected[-1]['epoch']
    ts = np.array([(r['epoch']-t0).total_seconds() for r in selected])
    obs = np.array([sgp(r['sat'], r['epoch']) for r in selected])
    p0 = np.r_[obs[-1], 0.]
    # Fit parameters in km, km/s, and km/s^2 with scaled optimization.
    scale = np.array([1., 1., 1., .001, .001, .001, 1e-9])
    def residual(q):
        p = p0 + scale*q
        if not accel:
            p[6] = 0.
        predicted = np.array([propagate(p, t) for t in ts])
        return ((predicted-obs)/np.array([.3,.3,.3,.0003,.0003,.0003])).ravel()
    dim = 7 if accel else 6
    solution = least_squares(lambda q: residual(np.r_[q, 0.] if not accel else q),
                             np.zeros(dim), method='trf', max_nfev=max_nfev,
                             xtol=1e-7, ftol=1e-7, gtol=1e-7)
    p = p0 + scale*(solution.x if accel else np.r_[solution.x, 0.])
    return p, t0, len(selected), float(np.sqrt(np.mean(solution.fun**2)))

def rtn(s):
    radial = s[:3]/np.linalg.norm(s[:3])
    normal = np.cross(s[:3],s[3:]); normal /= np.linalg.norm(normal)
    along = np.cross(normal, radial)
    return np.stack([radial, along, normal], axis=1)

def features(r):
    s = r['source']
    x = r['sgp_origin']
    age = (r['origin']-r['source_epoch']).total_seconds()/3600
    return np.r_[x[:3]/7000, x[3:]/8, s.ecco,
                 s.no_kozai*1440/(2*np.pi), s.bstar,
                 np.sin(s.inclo),np.cos(s.inclo),
                 np.sin(s.nodeo),np.cos(s.nodeo),
                 s.ecco*np.sin(s.argpo),s.ecco*np.cos(s.argpo),
                 np.sin(s.argpo+s.mo),np.cos(s.argpo+s.mo),
                 age/72, r['fit_rms'], r['fit_count']/30,
                 r['batch_origin'][:3]/7000,
                 r['batch_origin'][3:]/8]

def elm_fit_predict(train, val, hidden, ridge, seed):
    xtr=np.vstack([features(r) for r in train])
    xva=np.vstack([features(r) for r in val])
    mu=xtr.mean(0); sd=xtr.std(0); sd[sd<1e-8]=1
    xtr=(xtr-mu)/sd; xva=(xva-mu)/sd
    rng=np.random.default_rng(seed)
    w=rng.normal(0,1/np.sqrt(xtr.shape[1]),(xtr.shape[1],hidden))
    b=rng.uniform(-1,1,hidden)
    ht=np.c_[np.tanh(xtr@w+b),np.ones(len(train))]
    hv=np.c_[np.tanh(xva@w+b),np.ones(len(val))]
    y=np.vstack([r['basis'].T@(r['reference'][:3]-r['batch'][:3]) for r in train])
    # Residual target km, scaled for numerical stability.
    reg=np.sqrt(ridge)*np.diag(np.r_[np.ones(hidden),0.])
    A=np.vstack([ht,reg])
    B=np.vstack([y/100,np.zeros((hidden+1,3))])
    beta=np.linalg.lstsq(A,B,rcond=None)[0]
    return np.vstack([r['batch'][:3]+r['basis']@(p*100)
                      for r,p in zip(val,hv@beta)])

def rmse(pred, truth):
    return float(np.sqrt(np.mean(np.sum((pred-truth)**2,axis=1))))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--norads',nargs='+',type=int,default=[44714])
    ap.add_argument('--catalog-root',default='data/starlink/processed')
    ap.add_argument('--results-root',default='results/starlink')
    ap.add_argument('--horizons-hours',nargs='+',type=float,default=[1,6,12,24,72])
    ap.add_argument('--fit-window-days',type=float,default=7)
    ap.add_argument('--min-observations',type=int,default=4)
    ap.add_argument('--estimate-acceleration',action='store_true')
    ap.add_argument('--max-nfev',type=int,default=20)
    ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--ridges',nargs='+',type=float,default=[.01,.1,1,10,100])
    ap.add_argument('--alphas',nargs='+',type=float,default=[0,.25,.5,1])
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train-fraction',type=float,default=.7)
    args=ap.parse_args()
    if any(h<=0 or h>72 for h in args.horizons_hours):
        ap.error('Only positive horizons up to 72 hours are supported')
    if not 0<args.train_fraction<1 or args.fit_window_days<=0:
        ap.error('Invalid split or fit window')
    for norad in args.norads:
        root=Path(args.results_root)/str(norad)
        with (Path(args.catalog_root)/str(norad)/'tle_catalog.csv').open() as f:
            raw=list(csv.DictReader(f))
        records=[dict(epoch=utc(r['epoch_utc']),created=utc(r['creation_utc']),
                      gp_id=r['gp_id'],sat=Satrec.twoline2rv(r['tle_line1'],r['tle_line2']))
                 for r in raw]
        lookup={(r['gp_id'],r['epoch']):r for r in records}
        with (root/'phase1_corrected'/'sgp4_per_forecast.csv').open() as f:
            cases=list(csv.DictReader(f))
        rows=[]; skipped={}
        cache={}
        for c in cases:
            h=float(c['horizon_hours'])
            if h not in args.horizons_hours: continue
            origin,target=utc(c['origin_utc']),utc(c['target_utc'])
            source=lookup[(c['source_gp_id'],utc(c['source_epoch_utc']))]
            ref=lookup[(c['reference_gp_id'],utc(c['reference_epoch_utc']))]
            if source['created']>origin or ref['created']<target:
                raise ValueError('Phase 1 publication-time violation')
            key=origin
            try:
                if key not in cache:
                    cache[key]=fit_batch(records,origin,args.fit_window_days,
                                         args.min_observations,
                                         args.estimate_acceleration,args.max_nfev)
                p,t0,count,fit_rms=cache[key]
                batch=propagate(p,(target-t0).total_seconds())
                batch_origin=propagate(p,(origin-t0).total_seconds())
                sgp_target=sgp(source['sat'],target)
                reference=sgp(ref['sat'],target)
                sgp_origin=sgp(source['sat'],origin)
                rows.append(dict(h=h,origin=origin,target=target,case=c,
                                 source=source['sat'],source_epoch=source['epoch'],
                                 sgp=sgp_target,reference=reference,batch=batch,
                                 basis=rtn(batch),sgp_origin=sgp_origin,
                                 batch_origin=batch_origin,fit_count=count,fit_rms=fit_rms))
            except (ValueError,OverflowError,KeyError) as exc:
                skipped[str(exc)]=skipped.get(str(exc),0)+1
        summaries=[]; details=[]
        for h in sorted(set(args.horizons_hours)):
            group=sorted([r for r in rows if r['h']==h],key=lambda r:r['origin'])
            origins=sorted({r['origin'] for r in group})
            if len(origins)<2: continue
            cut=origins[max(1,min(len(origins)-1,int(len(origins)*args.train_fraction)))]
            # The batch model uses only information available at each origin.
            # ELM labels must additionally have been published by training cutoff.
            train=[r for r in group if r['origin']<cut
                   and utc(r['case']['reference_creation_utc'])<=cut]
            val=[r for r in group if r['origin']>=cut]
            if len(train)<8 or len(val)<3: continue
            truth=np.vstack([r['reference'][:3] for r in val])
            sgp_pred=np.vstack([r['sgp'][:3] for r in val])
            batch_pred=np.vstack([r['batch'][:3] for r in val])
            best=None
            for ridge in args.ridges:
                correction=elm_fit_predict(train,val,args.hidden,ridge,
                                            args.seed+norad+int(h))
                for alpha in args.alphas:
                    pred=batch_pred+alpha*(correction-batch_pred)
                    score=rmse(pred,truth)
                    if best is None or score<best[0]:
                        best=(score,ridge,alpha,pred)
            summaries.append(dict(horizon_hours=h,train_samples=len(train),
                                  validation_samples=len(val),
                                  train_cut_utc=cut.isoformat(),
                                  sgp4_rmse_km=rmse(sgp_pred,truth),
                                  batch_wls_rmse_km=rmse(batch_pred,truth),
                                  residual_elm_rmse_km=best[0],
                                  ridge=best[1],alpha=best[2]))
            for i,r in enumerate(val):
                details.append(dict(horizon_hours=h,origin_utc=r['origin'].isoformat(),
                                    target_utc=r['target'].isoformat(),
                                    sgp4_error_km=float(np.linalg.norm(sgp_pred[i]-truth[i])),
                                    batch_wls_error_km=float(np.linalg.norm(batch_pred[i]-truth[i])),
                                    residual_elm_error_km=float(np.linalg.norm(best[3][i]-truth[i]))))
        dest=root/('phase2_batch_wls_elm_70_30_'+('accel' if args.estimate_acceleration else 'noaccel'))
        dest.mkdir(parents=True,exist_ok=True)
        for name,data in [('metrics.csv',summaries),('per_forecast.csv',details)]:
            if data:
                with (dest/name).open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(data[0]))
                    writer.writeheader();writer.writerows(data)
        (dest/'experiment.json').write_text(json.dumps(dict(vars(args),norad=norad,
            skipped=skipped,frame='Approximate TEME-aligned fixed-z zonal gravity',
            caveat='TLE-derived references, not ground truth; validation tunes ridge/alpha; no test set'),indent=2)+'\n')
        print(norad,[(r['horizon_hours'],round(r['sgp4_rmse_km'],2),
                      round(r['batch_wls_rmse_km'],2),round(r['residual_elm_rmse_km'],2))
                     for r in summaries], 'skipped:',sum(skipped.values()))

if __name__=='__main__':
    main()
