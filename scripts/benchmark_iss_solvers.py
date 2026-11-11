"""Repeatable QR/SVD output-layer benchmarks on identical ISS ELM hidden matrices.

Reports warm-up-separated timing, least-squares residuals, coefficient norms,
hidden matrix condition/rank, and QR/SVD coefficient agreement. No test-set
selection; training portion only. Timings cover solver ONLY, not feature generation.
"""
import argparse,csv,json
from pathlib import Path
from time import perf_counter
import numpy as np
from satnav.elm_model import ELM
from satnav.least_squares import solve_least_squares
from satnav.dynamics import rk4_step
from iss_experiments import load,scale

def benchmark(A,B,solver,ridge,warmups,repeats):
    for _ in range(warmups):
        solve_least_squares(A,B,method=solver,ridge=ridge)
    times=[]
    beta=None
    for _ in range(repeats):
        t0=perf_counter()
        beta=solve_least_squares(A,B,method=solver,ridge=ridge)
        times.append((perf_counter()-t0)*1000)
    residual=np.linalg.norm(A@beta-B,ord="fro")
    coeff=np.linalg.norm(beta,ord="fro")
    objective=residual**2+ridge*coeff**2
    return dict(time_median_ms=float(np.median(times)),
        time_min_ms=float(np.min(times)),time_max_ms=float(np.max(times)),
        time_std_ms=float(np.std(times)),residual_fro=float(residual),
        coefficient_fro=float(coeff),ridge_objective=float(objective)),beta

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/iss/processed/nasa_iss_states.csv")
    p.add_argument("--output",default="results/iss/solver_benchmark")
    p.add_argument("--dt",type=float,default=240.)
    p.add_argument("--hidden",type=int,nargs="+",default=[16,32,64,128,256])
    p.add_argument("--ridge",type=float,nargs="+",default=[0.,1e-6,1e-4,1e-2,1.])
    p.add_argument("--warmups",type=int,default=2)
    p.add_argument("--repeats",type=int,default=7)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.repeats<2 or args.warmups<0 or any(h<1 for h in args.hidden) or any(v<0 for v in args.ridge):
        p.error("Require repeats>=2, warmups>=0, hidden>=1, ridge>=0")
    t,x=load(args.csv)
    boundary=int(.6*len(x))
    train=np.where((np.abs(np.diff(t)-args.dt)<=.01)&(np.arange(len(t)-1)+1<boundary))[0]
    if len(train)<20:raise ValueError("Insufficient uniform training pairs")
    X=x[train]
    xm,xs=scale(X)
    Xn=(X-xm)/xs
    rows=[]
    for family in ("direct_elm","physics_residual_elm"):
        base=np.array([rk4_step(0.,s,args.dt) for s in X]) if family=="physics_residual_elm" else X
        Y=x[train+1]-base
        ym,ys=scale(Y)
        Yn=(Y-ym)/ys
        for hidden in args.hidden:
            model=ELM(hidden=hidden,seed=args.seed)
            rng=np.random.default_rng(args.seed)
            model.W=rng.normal(size=(Xn.shape[1],hidden))/np.sqrt(Xn.shape[1])
            model.bias=rng.uniform(-1,1,size=hidden)
            H=model._hidden(Xn)
            singular=np.linalg.svd(H,compute_uv=False)
            rank=int(np.linalg.matrix_rank(H))
            condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf")
            for ridge in args.ridge:
                results={}
                for solver in ("svd","qr"):
                    try:
                        metrics,beta=benchmark(H,Yn,solver,ridge,args.warmups,args.repeats)
                    except ValueError as exc:
                        print("Skipped",family,hidden,ridge,solver,str(exc))
                        continue
                    results[solver]=beta
                    rows.append(dict(model=family,hidden=hidden,ridge=ridge,solver=solver,
                        samples=len(train),rank=rank,condition_number=condition,**metrics))
                if len(results)==2:
                    diff=float(np.linalg.norm(results["svd"]-results["qr"],ord="fro"))
                    for row in rows[-2:]:row["qr_svd_coefficient_difference_fro"]=diff
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if not rows:raise RuntimeError("No benchmark results")
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/"solver_benchmark.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (out/"benchmark_metadata.json").write_text(json.dumps(dict(
        source=args.csv,train_pairs=len(train),warmups=args.warmups,repeats=args.repeats,
        seed=args.seed,hidden=args.hidden,ridge=args.ridge,
        timing="solver only, repeated on fixed H and normalized Y; system BLAS threading affects results",
        warning="SVD/QR numerical agreement may degrade for highly ill-conditioned unregularized systems"),indent=2))
    print("Saved",out/"solver_benchmark.csv","rows",len(rows))
if __name__=="__main__":main()
