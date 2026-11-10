"""Two-body dynamics in inertial Cartesian coordinates; SI units."""
import numpy as np
MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0

def circular_state(altitude_m=500_000.0):
    r = R_EARTH + altitude_m
    return np.array([r, 0, 0, 0, np.sqrt(MU_EARTH/r), 0], dtype=float)

def rhs(t, state):
    s = np.asarray(state, dtype=float)
    r, v = s[:3], s[3:]
    radius = np.linalg.norm(r)
    if radius <= R_EARTH * 0.5:
        raise ValueError("Invalid orbital radius")
    return np.concatenate([v, -MU_EARTH * r / radius**3])

def rk4_step(t, state, dt):
    k1 = rhs(t, state)
    k2 = rhs(t + dt/2, state + dt*k1/2)
    k3 = rhs(t + dt/2, state + dt*k2/2)
    k4 = rhs(t + dt, state + dt*k3)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def propagate(initial, duration_s=5400, dt_s=10):
    if dt_s <= 0 or duration_s <= 0:
        raise ValueError("Positive time step and duration required")
    n = round(duration_s/dt_s)
    if not np.isclose(n*dt_s, duration_s):
        raise ValueError("duration_s must be divisible by dt_s")
    times = np.arange(n+1, dtype=float)*dt_s
    states = np.empty((n+1,6), dtype=float)
    states[0] = np.asarray(initial, dtype=float)
    for k in range(n):
        states[k+1] = rk4_step(times[k], states[k], dt_s)
    return times, states

def invariants(states):
    r, v = states[:,:3], states[:,3:]
    energy = 0.5*np.sum(v*v, axis=1) - MU_EARTH/np.linalg.norm(r,axis=1)
    momentum = np.linalg.norm(np.cross(r,v),axis=1)
    return energy, momentum
