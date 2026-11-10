"""Phase 1 smoke run: synthetic two-body trajectory and initial ELM fit."""
from pathlib import Path
import csv
import numpy as np
from satnav.dynamics import circular_state, propagate, invariants
from satnav.elm_model import ELM

times, states = propagate(circular_state(), duration_s=5400, dt_s=10)
energy, h = invariants(states)
out = Path("results/synthetic")
out.mkdir(parents=True, exist_ok=True)
with (out/"two_body_trajectory.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_s","x_m","y_m","z_m","vx_mps","vy_mps","vz_mps"])
    writer.writerows(np.column_stack([times,states]))

# Demonstration only: standardized inputs and next-step increments.
# Training and testing are chronological, with no overlapping snapshot pairs.
X, Y = states[:-1], states[1:]-states[:-1]
cut = int(0.7*len(X))
mean, std = X[:cut].mean(axis=0), X[:cut].std(axis=0)
std[std < 1e-12] = 1.0
dy_mean, dy_std = Y[:cut].mean(axis=0), Y[:cut].std(axis=0)
dy_std[dy_std < 1e-12] = 1.0
model = ELM(hidden=64,seed=42).fit((X[:cut]-mean)/std,(Y[:cut]-dy_mean)/dy_std)
delta_pred = model.predict((X[cut:]-mean)/std)*dy_std+dy_mean
pred = X[cut:]+delta_pred
pos_rmse = np.sqrt(np.mean(np.sum((pred[:,:3]-states[cut+1:,:3])**2,axis=1)))
print(f"Generated {len(states)} synthetic states -> {out/'two_body_trajectory.csv'}")
print(f"Relative energy drift: {max(abs(energy/energy[0]-1)):.3e}")
print(f"Relative angular-momentum drift: {max(abs(h/h[0]-1)):.3e}")
print(f"ELM one-step held-out position RMSE: {pos_rmse:.3f} m (synthetic only)")
print("Not an ISS measurement validation; not an open-loop forecast.")
