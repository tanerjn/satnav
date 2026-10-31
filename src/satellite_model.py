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
num_measurements = 10
A = np.zeros((num_measurements, 6))  # 6 state variables (x, y, z, vx, vy, vz)
b = np.zeros((num_measurements, 1))

ground_station = np.array([6378137.0, 0.0, 0.0])  # Example ground station

for i in range(num_measurements):
  t_i = i * 10.0  # Time step
  
  # Propagated satellite position at time step i
  sat_pos = r0 + v0 * t_i  

  # Vector from ground station to satellite
  diff = sat_pos - ground_station
  R = np.linalg.norm(diff)

  # Line-of-sight unit vector
  los = diff / R

  # Fill row of A matrix
  A[i, 0:3] = los         # Position partials (dr / dr0)
  A[i, 3:6] = los * t_i   # Velocity partials (dr / dv0 scaled by time t_i)

  # Simulated noisy measurement (True range + white noise)
  b[i] = R + np.random.normal(0, 5.0)  # 5 meters standard deviation noise

# 3. Solve the Least Squares Problem: Ax ≈ b using QR Factorization
Q, R_mat = np.linalg.qr(A)

# Safeguard solve using lstsq in case of any ill-conditioning
x_estimated, residuals, rank, s = np.linalg.lstsq(R_mat[:6, :6], Q.T[:6, :] @ b, rcond=None)

print('Estimated State Corrections x:', x_estimated.flatten())
