"""TEME to GCRS (geocentric celestial frame) using Astropy.

GCRS is not exactly EME2000/J2000; frame differences must be documented
and assessed before claiming high-precision OEM comparison.
"""
import numpy as np

def teme_to_gcrs(states_m, utc_epochs):
    from astropy import units as u
    from astropy.coordinates import TEME, GCRS, CartesianRepresentation, CartesianDifferential
    from astropy.time import Time
    states=np.asarray(states_m,float)
    if states.ndim!=2 or states.shape[1]!=6 or len(states)!=len(utc_epochs):
        raise ValueError("Expected (N,6) states and matching UTC epochs")
    t=Time(list(utc_epochs),scale="utc")
    rep=CartesianRepresentation(states[:,:3].T*u.m,
        differentials=CartesianDifferential(states[:,3:].T*u.m/u.s))
    frame=TEME(rep,obstime=t).transform_to(GCRS(obstime=t))
    pos=frame.cartesian.xyz.to_value(u.m).T
    vel=frame.cartesian.differentials["s"].d_xyz.to_value(u.m/u.s).T
    return np.column_stack((pos,vel))


def gcrs_to_eme2000(states_m):
    """Rotate GCRS Cartesian states to mean equator/equinox J2000 (EME2000).

    Uses the IAU 2000 frame-bias matrix rb from ERFA/SOFA bp00:
    GCRS -> mean J2000. This is a *constant* geocentric-axis rotation;
    the same matrix applies to position and velocity. It is not the
    precession-nutation matrix rbp (which targets mean equator of date).
    OEM provider-specific frame realizations must still be verified.
    """
    import erfa
    states = np.asarray(states_m, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError("Expected (N,6) Cartesian states")
    rb, _, _ = erfa.bp00(2451545.0, 0.0)
    result = np.empty_like(states)
    result[:, :3] = states[:, :3] @ rb.T
    result[:, 3:] = states[:, 3:] @ rb.T
    return result
