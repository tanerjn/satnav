"""SGP4 versus NASA OEM comparison, with explicit frame caution.

This exploratory comparison transforms TEME to GCRS, which is not
identical to NASA OEM EME2000. Do not label errors as precise J2000
comparisons until frame definitions are verified.
"""
import argparse,csv
from datetime import datetime,timezone
import numpy as np
from satnav.iss_sgp4 import propagate_tle
from satnav.iss_frames import teme_to_gcrs
from satnav.evaluation import state_rmse

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oem",default="data/iss/processed/nasa_iss_states.csv")
    p.add_argument("--tle",required=True,help="File containing ISS TLE lines")
    p.add_argument("--max-age-days",type=float,default=2.)
    args=p.parse_args()
    with open(args.oem,newline="") as f:rows=list(csv.DictReader(f))
    epochs=[datetime.fromisoformat(r["epoch_utc"].replace("Z","+00:00")).replace(tzinfo=timezone.utc) if datetime.fromisoformat(r["epoch_utc"].replace("Z","+00:00")).tzinfo is None else datetime.fromisoformat(r["epoch_utc"].replace("Z","+00:00")).astimezone(timezone.utc) for r in rows]
    keys=["x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"]
    ref=np.array([[float(r[k]) for k in keys] for r in rows])
    lines=[line.strip() for line in open(args.tle) if line.strip()]
    one=next((s for s in lines if s.startswith("1 ")),None)
    two=next((s for s in lines if s.startswith("2 ")),None)
    if not one or not two:raise ValueError("TLE file must contain line 1 and line 2")
    from sgp4.api import Satrec
    from astropy.time import Time
    sat=Satrec.twoline2rv(one,two)
    tle_epoch=Time(sat.jdsatepoch+sat.jdsatepochF,format="jd",scale="utc").to_datetime(timezone=timezone.utc)
    mask=np.array([abs((dt-tle_epoch).total_seconds())<=args.max_age_days*86400 for dt in epochs])
    if not np.any(mask):raise ValueError("No OEM epochs near TLE epoch; obtain historical matching TLE")
    selected=[dt for dt,keep in zip(epochs,mask) if keep]
    teme=propagate_tle(one,two,selected)
    gcrs=teme_to_gcrs(teme,selected)
    print("TLE epoch UTC:",tle_epoch,"Matched OEM epochs:",len(selected))
    print("Exploratory TEME->GCRS vs OEM:",state_rmse(gcrs,ref[mask]))
    print("WARNING: GCRS and OEM J2000/EME2000 are not guaranteed identical.")
if __name__=="__main__":main()
