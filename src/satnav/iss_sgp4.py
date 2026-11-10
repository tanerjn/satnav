"""ISS SGP4 propagation in TEME frame; NOT J2000.

Requires optional dependency 'sgp4'. Convert frames before comparing to NASA OEM.
"""
from datetime import datetime, timezone
import numpy as np

def propagate_tle(line1, line2, utc_datetimes):
    try:
        from sgp4.api import Satrec, jday
    except ImportError as exc:
        raise ImportError("Install sgp4 to propagate ISS TLEs") from exc
    sat = Satrec.twoline2rv(line1, line2)
    states = []
    for dt in utc_datetimes:
        if dt.tzinfo is None:
            raise ValueError("UTC datetime must be timezone-aware")
        d = dt.astimezone(timezone.utc)
        sec = d.second + d.microsecond/1e6
        jd, fr = jday(d.year,d.month,d.day,d.hour,d.minute,sec)
        err, r, v = sat.sgp4(jd,fr)
        if err:
            raise RuntimeError(f"SGP4 error code {err} at {dt.isoformat()}")
        states.append([*r,*v])
    return np.asarray(states,float)*1000.0
