"""NASA ISS OEM ingestion.

OEM metadata and reference frame must be validated before model comparison.
"""
from pathlib import Path
import numpy as np

def parse_oem_text(path):
    """Read NASA ISS OEM text records; return ISO epochs and SI (N,6) states.

    Expects OEM records as: YYYY-MM-DDTHH:MM:SS[.fff] x y z vx vy vz.
    OEM units are km and km/s; output is meters and meters/second.
    """
    epochs, states = [], []
    for line in Path(path).read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) != 7 or "T" not in tokens[0]:
            continue
        try:
            values = [float(v) for v in tokens[1:]]
        except ValueError:
            continue
        epochs.append(tokens[0])
        states.append(values)
    if not states:
        raise ValueError("No OEM state records found; inspect format and metadata")
    return epochs, np.asarray(states, dtype=float)*1000.0
