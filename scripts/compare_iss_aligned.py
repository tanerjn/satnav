"""Evaluate SGP4 and a validation-selected ELM on exactly the same test epochs.

Single TLE; reports limited epoch overlap and GCRS/OEM frame caveat.
"""
import argparse,json,csv
from datetime import datetime,timezone
import numpy as np
from astropy.time import Time
from sgp4.api import Satrec
from satnav.iss_sgp4 import propagate_tle
from satnav.iss_frames import teme_to_gcrs
from satnav.evaluation import state_rmse
from iss_experiments import load
from iss_validate import fit_predict,pairs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default="data/iss/processed/nasa_iss_states.csv")
    p.add_argument("--selected",default="results/iss/validated/selected_model_test.json")
    p.add_argument("--tle",required=True)
    p.add_argument("--max-age-days",type=float,default=2.)
    p.add_argument("--dt",type=float,default=240.)
    args=p.parse_args()
    if args.max_age_days<=0:p.error("--max-age-days must be positive")
    t,x=load(args.csv)
    with open(args.selected) as f: selection=json.load(f)["selected_on_validation"]
    good=pairs(t,args.dt)
    split=int(.8*len(x))
    fit_idx=good[good+1<split]
    test=good[good>=split]
    lines=[line.strip() for line in open(args.tle) if line.strip()]
    line1=next((line for line in lines if line.startswith("1 ")),None)
    line2=next((line for line in lines if line.startswith("2 ")),None)
    if not line1 or not line2:raise ValueError("TLE requires lines 1 and 2")
    sat=Satrec.twoline2rv(line1,line2)
    tle_epoch=Time(sat.jdsatepoch+sat.jdsatepochF,format="jd",scale="utc").unix
    mask=np.abs(t[test+1]-tle_epoch)<=args.max_age_days*86400
    matched=test[mask]
    if len(matched)==0:
        raise ValueError("No overlap with FINAL TEST epochs. Obtain historical TLE near test dates; do not compare mismatched subsets.")
    epochs=[datetime.fromtimestamp(float(v),tz=timezone.utc) for v in t[matched+1]]
    sgp4_pred=teme_to_gcrs(propagate_tle(line1,line2,epochs),epochs)
    elm_pred,_=fit_predict(x,fit_idx,matched,args.dt,int(selection["hidden"]),
        float(selection["ridge"]),selection["solver"],selection["model"]=="physics_residual_elm")
    ref=x[matched+1]
    result=dict(tle_epoch_utc=datetime.fromtimestamp(float(tle_epoch),tz=timezone.utc).isoformat(),
        matched_test_pairs=len(matched),total_test_pairs=len(test),
        first_epoch_utc=epochs[0].isoformat(),last_epoch_utc=epochs[-1].isoformat(),
        sgp4_gcrs_vs_oem=state_rmse(sgp4_pred,ref),
        selected_elm_vs_oem=state_rmse(elm_pred,ref),
        warning="Same epochs, but SGP4 GCRS vs OEM J2000/EME2000 is not fully frame-validated; ELM is one-step teacher-forced. Single-TLE propagation.")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
