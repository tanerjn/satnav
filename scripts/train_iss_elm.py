"""Train one-step ISS ELM on processed NASA OEM states (chronological split)."""
import argparse
import csv
from pathlib import Path
import numpy as np
from satnav.elm_model import ELM
from satnav.evaluation import state_rmse

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/iss/processed/nasa_iss_states.csv")
    p.add_argument("--hidden",type=int,default=64)
    p.add_argument("--solver",choices=["svd","qr"],default="svd")
    a=p.parse_args()
    with Path(a.csv).open(newline="") as f:
        rows=list(csv.DictReader(f))
    if len(rows)<20:
        raise ValueError("Need at least 20 chronologically ordered OEM states")
    epochs=[r["epoch_utc"] for r in rows]
    if epochs != sorted(epochs) or len(set(epochs))!=len(epochs):
        raise ValueError("Epochs must be strictly increasing and unique")
    keys=["x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"]
    states=np.array([[float(row[k]) for k in keys] for row in rows])
    X,Y=states[:-1],states[1:]-states[:-1]
    cut=int(len(X)*0.7)
    mu,sd=X[:cut].mean(0),X[:cut].std(0)
    sd[sd<1e-12]=1
    ym,ys=Y[:cut].mean(0),Y[:cut].std(0)
    ys[ys<1e-12]=1
    model=ELM(hidden=a.hidden,solver=a.solver).fit((X[:cut]-mu)/sd,(Y[:cut]-ym)/ys)
    pred=X[cut:]+model.predict((X[cut:]-mu)/sd)*ys+ym
    print("One-step held-out OEM agreement:",state_rmse(pred,states[cut+1:]))
    print("This is NOT independent raw-measurement validation or open-loop forecasting.")

if __name__=="__main__":
    main()
