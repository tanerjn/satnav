"""ISS OEM sensitivity experiment: chronological splits, uniform 240s pairs, QR/SVD, physics baselines, recursive rollout."""
import argparse,csv,json
from datetime import datetime
from pathlib import Path
from time import perf_counter
import numpy as np
from satnav.elm_model import ELM
from satnav.dynamics import rk4_step
from satnav.evaluation import state_rmse

def load(path):
    with open(path,newline="") as f: rows=list(csv.DictReader(f))
    keys=["x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"]
    t=np.array([datetime.fromisoformat(r["epoch_utc"].replace("Z","+00:00")).timestamp() for r in rows])
    x=np.array([[float(r[k]) for k in keys] for r in rows])
    if len(x)<30 or not np.all(np.diff(t)>0) or not np.all(np.isfinite(x)):raise ValueError("Invalid epochs/states")
    return t,x

def physics(x,dt):
    return rk4_step(0.,x,dt)

def scale(x):
    m=x.mean(0);s=x.std(0);s[s<1e-12]=1
    return m,s

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--csv",default="data/iss/processed/nasa_iss_states.csv")
    a.add_argument("--output",default="results/iss")
    a.add_argument("--dt",type=float,default=240)
    a.add_argument("--hidden",type=int,nargs="+",default=[32,64,128,256])
    a.add_argument("--horizon",type=int,default=5)
    args=a.parse_args()
    t,x=load(args.csv);d=np.diff(t)
    intervals,counts=np.unique(d,return_counts=True)
    print("Intervals (seconds,count):",list(zip(intervals.tolist(),counts.tolist())))
    good=np.where(np.abs(d-args.dt)<=0.01)[0]
    boundary=int(.7*len(x))
    train=good[good+1<boundary];test=good[good>=boundary]
    if len(train)<20 or len(test)<10:raise ValueError("Insufficient uniform train/test pairs")
    print("Train/test pairs:",len(train),len(test),"Excluded:",len(d)-len(good))
    truth=x[test+1]
    output=[]
    baselines={
        "persistence":x[test],
        "constant_velocity":np.column_stack([x[test,:3]+args.dt*x[test,3:],x[test,3:]]),
        "two_body_rk4":np.array([physics(s,args.dt) for s in x[test]])
    }
    for name,pred in baselines.items():
        output.append(dict(model=name,**state_rmse(pred,truth)))
    for residual in [False,True]:
        for hidden in args.hidden:
            for solver in ["svd","qr"]:
                name="physics_residual_elm" if residual else "direct_elm"
                X=x[train];Y=x[train+1]-X
                if residual:Y=Y-np.array([physics(s,args.dt)-s for s in X])
                xm,xs=scale(X);ym,ys=scale(Y)
                try:
                    tic=perf_counter()
                    model=ELM(hidden=hidden,seed=42,solver=solver).fit((X-xm)/xs,(Y-ym)/ys)
                    ms=(perf_counter()-tic)*1000
                except ValueError as e:
                    print("Skipped",name,hidden,solver,e);continue
                correction=model.predict((x[test]-xm)/xs)*ys+ym
                base=np.array([physics(s,args.dt) for s in x[test]]) if residual else x[test]
                pred=base+correction
                H=model._hidden((X-xm)/xs)
                singular=np.linalg.svd(H,compute_uv=False)
                rank=int(np.sum(singular>singular[0]*max(H.shape)*np.finfo(float).eps))
                cond=float(singular[0]/singular[-1]) if singular[-1] else float("inf")
                record=dict(model=name,hidden=hidden,solver=solver,training_ms=ms,rank=rank,condition_number=cond,**state_rmse(pred,truth))
                if hidden==64:
                    forecasts=[];references=[]
                    for start in range(boundary,len(x)-args.horizon,args.horizon):
                        if not np.all(np.abs(np.diff(t[start:start+args.horizon+1])-args.dt)<=0.01):continue
                        s=x[start].copy()
                        for _ in range(args.horizon):
                            c=model.predict(((s-xm)/xs)[None,:])[0]*ys+ym
                            s=(physics(s,args.dt) if residual else s)+c
                        forecasts.append(s);references.append(x[start+args.horizon])
                    if forecasts:
                        m=state_rmse(np.array(forecasts),np.array(references))
                        record.update(recursive_horizon_s=args.horizon*args.dt,recursive_samples=len(forecasts),recursive_position_rmse_m=m["position_rmse_m"],recursive_velocity_rmse_mps=m["velocity_rmse_mps"])
                output.append(record)
                print(name,hidden,solver,"pos RMSE m",round(record["position_rmse_m"],3),"vel RMSE m/s",round(record["velocity_rmse_mps"],3),"train ms",round(ms,2),"cond",round(cond))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fields=["model","hidden","solver","position_rmse_m","velocity_rmse_mps","training_ms","rank","condition_number","recursive_horizon_s","recursive_samples","recursive_position_rmse_m","recursive_velocity_rmse_mps"]
    with (out/"iss_sensitivity.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(output)
    (out/"experiment_metadata.json").write_text(json.dumps(dict(source=args.csv,states=len(x),dt_s=args.dt,train_pairs=len(train),test_pairs=len(test),excluded_pairs=len(d)-len(good),note="OEM is operational ephemeris, not raw ground truth. One-step is teacher-forced; recursive is free-run."),indent=2))
    print("Saved",out/"iss_sensitivity.csv")

if __name__=="__main__":main()
