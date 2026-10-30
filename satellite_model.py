import numpy as np

# 1. Load Constants from Parameters File
mu = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
r0 = np.array([6878137.0, 0.0, 0.0])  # Initial Position (m)
v0 = np.array([0.0, 7612.6, 0.0])  # Initial Velocity (m/s)

# Environmental parameters for perturbation calculation
Cd = 2.2
Area_mass_ratio = 0.02
rho_500km = 6.048e-14
P_srp = 4.56e-6
Cr = 1.4

# 2. Simulate/Construct Observation Matrix A and Measurement Vector b
# Let's assume m measurements from ground stations at known positions
num_measurements = 10
A = np.zeros((num_measurements, 6))  # 6 state variables (x, y, z, vx, vy, vz)
b = np.zeros((num_measurements, 1))

ground_station = np.array([6378137.0, 0.0, 0.0])  # Example ground station

for i in range(num_measurements):
  # Propagated satellite position at time step i (using r0, v0, drag, SRP)
  sat_pos = r0 + v0 * (i * 10)  # Simplified linear propagation for demo

  # Vector from ground station to satellite
  diff = sat_pos - ground_station
  R = np.linalg.norm(diff)

  # Fill row of A matrix (Line-of-sight unit vector components for position)
  A[i, 0:3] = diff / R
  A[i, 3:6] = 0.0  # Velocity partials for pure range

  # Simulated noisy measurement (True range + white noise)
  b[i] = R + np.random.normal(0, 5.0)  # 5 meters standard deviation noise

# 3. Solve the Least Squares Problem: Ax ≈ b using QR Factorization or SVD
# (Tied to MA 532 course modules: Chapter 7 & 8)
Q, R_mat = np.linalg.qr(A)
x_estimated = np.linalg.solve(R_mat[:6, :6], Q.T[:6, :] @ b)

print('Estimated State Corrections x:', x_estimated.flatten())
