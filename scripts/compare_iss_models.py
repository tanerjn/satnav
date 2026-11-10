"""Compare aligned (N,6) state CSVs in SI units and the SAME reference frame."""
import argparse
import csv
import numpy as np
from satnav.evaluation import state_rmse

def load(path):
    with open(path,newline="") as f:
        rows=list(csv.DictReader(f))
    keys=["x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"]
    return [r["epoch_utc"] for r in rows],np.array([[float(r[k]) for k in keys] for r in rows])

def main():
    p=argparse.ArgumentParser(description="Both files MUST be in the same frame; TEME vs J2000 is invalid")
    p.add_argument("--reference",required=True)
    p.add_argument("--prediction",required=True)
    args=p.parse_args()
    rt,rs=load(args.reference)
    pt,ps=load(args.prediction)
    if rt!=pt:
        raise ValueError("Epoch mismatch; align timestamps before comparison")
    print(state_rmse(ps,rs))

if __name__=="__main__":
    main()
