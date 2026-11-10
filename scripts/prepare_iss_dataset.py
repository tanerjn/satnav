"""Parse a downloaded NASA OEM file to normalized-format CSV (SI units)."""
import argparse
import csv
from pathlib import Path
from satnav.iss_oem_parser import parse_oem_text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oem", required=True, help="Path to NASA ISS OEM TXT file")
    p.add_argument("--output", default="data/iss/processed/nasa_iss_states.csv")
    args = p.parse_args()
    epochs, states = parse_oem_text(args.oem)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["epoch_utc","x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"])
        w.writerows(([epoch,*s] for epoch,s in zip(epochs,states)))
    print(f"Wrote {len(states)} OEM states to {output}; verify OEM frame/metadata before comparison.")

if __name__ == "__main__":
    main()
