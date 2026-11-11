"""Chronological ISS ELM hyperparameter selection with untouched test segment.

Runs a 60/20/20 chronological split. Validation selects hidden size, ridge,
solver and model family. Test is evaluated once for the selected configuration.
No random shuffling or test-based selection.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from satnav.elm_model import ELM
from satnav.dynamics import rk4_step
from satnav.evaluation import state_rmse
from iss_experiments import load,scale

def pairs(t,dt):
    return np.where(np.abs(np.diff(t)-dt)<=0.01)[0]

def fit_predict(x,train,eval_idx,dt,hidden,ridge,solver,residual):
    X=x[train]
    base_train=np.array([rk4_step(0.,s,dt) for s in X]) if residual else X
    Y=x[train+1]-base_train
    xm,xs=scale(X);ym,ys=scale(Y)
    model=ELM(hidden=hidden,seed=42,solver=solver,ridge=ridge)
    model.fit((X-xm)/xs,(Y-ym)/ys)
    Z=x[eval_idx]
    base=np.array([rk4_step(0.,s,dt) for s in Z]) if residual else Z
    return base+model.predict((Z-xm)/xs)*ys+ym,model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/iss/processed/nasa_iss_states.csv")
    p.add_argument("--output",default="results/iss/validated")
    p.add_argument("--dt",type=float,default=240.)
    p.add_argument("--hidden",type=int,nargs="+",default=[16,32,64,128,256])
    p.add_argument("--ridge",type=float,nargs="+",default=[0.,1e-6,1e-4,1e-2,1.,100.])
    p.add_argument("--solver",nargs="+",choices=["svd","qr"],default=["svd","qr"])
    args=p.parse_args()
    t,x=load(args.csv);good=pairs(t,args.dt)
    a,b=int(len(x)*.6),int(len(x)*.8)
    train=good[good+1<a]
    val=good[(good>=a)&(good+1<b)]
    test=good[good>=b]
    if min(len(train),len(val),len(test))<10:raise ValueError("Insufficient split samples")
    records=[];best=None
    for residual in [False,True]:
        for hidden in args.hidden:
            for ridge in args.ridge:
                for solver in args.solver:
                    try:
                        pred,model=fit_predict(x,train,val,args.dt,hidden,ridge,solver,residual)
                    except ValueError as exc:
                        print("Skip",hidden,ridge,solver,exc);continue
                    metrics=state_rmse(pred,x[val+1])
                    record=dict(model="physics_residual_elm" if residual else "direct_elm",hidden=hidden,ridge=ridge,solver=solver,**metrics)
                    records.append(record)
                    if best is None or metrics["position_rmse_m"]<best["position_rmse_m"]:
                        best=record
    if best is None:raise RuntimeError("No valid candidate")
    residual=best["model"]=="physics_residual_elm"
    # Refit using train+validation, then evaluate once on untouched test.
    fit_idx=good[good+1<b]
    pred,model=fit_predict(x,fit_idx,test,args.dt,best["hidden"],best["ridge"],best["solver"],residual)
    test_metrics=state_rmse(pred,x[test+1])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with (output/"validation_grid.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(selected_on_validation=best,test_metrics=test_metrics,
        split=dict(train_states=a,validation_states=b-a,test_states=len(x)-b,
            validation_pairs=len(val),test_pairs=len(test)),
        note="One-step teacher-forced comparison to NASA OEM operational ephemeris; not independent ground truth. Selection uses validation only.")
    (output/"selected_model_test.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
