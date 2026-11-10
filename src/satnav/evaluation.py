"""Position and velocity errors for time-aligned states in a common frame."""
import numpy as np

def state_rmse(predicted, reference):
    p, r = np.asarray(predicted, float), np.asarray(reference, float)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 6 or not len(p):
        raise ValueError("Expected matching nonempty (N,6) state arrays")
    diff = p-r
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(diff[:, :3]**2, axis=1)))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(np.sum(diff[:, 3:]**2, axis=1))))
    }
