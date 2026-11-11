"""Read and check CCSDS OEM header metadata without assuming frame equivalence."""
from pathlib import Path

def inspect_oem_metadata(path):
    meta={}
    for raw in Path(path).read_text().splitlines():
        line=raw.strip()
        if line.startswith("META_START"):continue
        if line.startswith("META_STOP"):break
        if "=" in line:
            key,value=line.split("=",1)
            key=key.strip().upper()
            if key in ("REF_FRAME","TIME_SYSTEM","CENTER_NAME","OBJECT_ID","OBJECT_NAME","START_TIME","STOP_TIME"):
                meta[key]=value.strip()
    required=("REF_FRAME","TIME_SYSTEM","CENTER_NAME")
    missing=[key for key in required if not meta.get(key)]
    if missing:raise ValueError("Missing OEM metadata: "+", ".join(missing))
    if meta["TIME_SYSTEM"].upper()!="UTC":raise ValueError("Expected UTC OEM times, got "+meta["TIME_SYSTEM"])
    if meta["CENTER_NAME"].upper() not in ("EARTH","399"):raise ValueError("Expected Earth-centered OEM")
    frame=meta["REF_FRAME"].upper()
    if frame not in ("EME2000","J2000","GCRF","GCRS"):
        raise ValueError("Unsupported/unverified OEM reference frame: "+frame)
    return meta
