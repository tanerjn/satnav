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
