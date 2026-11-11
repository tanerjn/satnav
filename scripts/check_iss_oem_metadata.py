"""Verify declared OEM frame and time system; does not prove frame identity."""
import argparse,json
from satnav.oem_metadata import inspect_oem_metadata
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oem",default="data/iss/nasa_oem/ISS.OEM_J2K_EPH.txt")
    args=p.parse_args()
    meta=inspect_oem_metadata(args.oem)
    print(json.dumps(meta,indent=2))
    print("Metadata checks passed; frame declaration alone does NOT establish GCRS/EME2000 equivalence.")
if __name__=="__main__":main()
