from datetime import datetime
import numpy as np
from sgp4.api import Satrec, jday


def run_sgp4_orbit_determination():
  print("--- LEO Satellite Orbit Determination using Real SGP4 TLE Data ---")

  # 1. Define Two-Line Element (TLE) for a real LEO satellite (e.g., ISS / ZARYA)
  line1 = (
      "1 25544U 98067A   26050.51234567  .00016717  00000-0  10270-3 0  9993"
  )
  line2 = (
      "2 25544  51.6416 247.4627 0006791 130.5360 329.7405 15.49575355412346"
  )

  # Initialize the SGP4 satellite record
  satellite = Satrec.twoline2rv(line1, line2)

  # Ground station position (e.g., a tracking station on Earth's surface in meters)
  ground_station = np.array([6378137.0, 0.0, 0.0])

  # 2. Define baseline epoch time for simulation (Year, Month, Day, Hour, Minute, Second)
  # Let's use a baseline date aligned near the TLE epoch
  year, month, day, hour, minute, second = 2026, 2, 19, 12, 0, 0
  jd_base, fr_base = jday(year, month, day, hour, minute, second)

  # 3. Simulate Measurements over a Time Series
  num_measurements = 20
  dt_seconds = 10.0  # 10-second intervals between observations

  A = np.zeros((num_measurements, 6))  # Design Matrix (m x 6)
  b = np.zeros((num_measurements, 1))  # Residual Vector (m x 1)

  print(
      f"Propagating TLE data and generating {num_measurements} synthetic range"
      " measurements..."
  )

  # Get nominal state at t = 0 from SGP4 (returns km and km/s, convert to meters & m/s)
  e_nom, r_nom_km, v_nom_km = satellite.sgp4(jd_base, fr_base)
  if e_nom != 0:
    raise ValueError(f"SGP4 Propagation Error Code: {e_nom}")

  r_nom = np.array(r_nom_km) * 1000.0
  v_nom = np.array(v_nom_km) * 1000.0

  # Inject a true state offset (simulating real error: +100m pos, -0.1 m/s vel)
  true_state_offset = np.array([100.0, -80.0, 40.0, -0.1, 0.2, 0.05])

  for i in range(num_measurements):
    tsince_minutes = (i * dt_seconds) / 60.0  # SGP4 takes time-since in minutes

    # Compute Nominal Position via SGP4 time offset method
    # Alternatively, step Julian date fractions:
    fractional_days_offset = (i * dt_seconds) / 86400.0
    e_n, r_n_km, v_n_km = satellite.sgp4(jd_base, fr_base + fractional_days_offset)

    r_nominal = np.array(r_n_km) * 1000.0

    # Compute True Position (using perturbed initial state linearly propagated for simplicity)
    r_true = (r_nom + true_state_offset[0:3]) + (
        (v_nom + true_state_offset[3:6]) * (i * dt_seconds)
    )

    # Line-of-sight geometry vector from ground station to nominal satellite position
    diff_nom = r_nominal - ground_station
    R_nom = np.linalg.norm(diff_nom)
    los = diff_nom / R_nom

    # True range with added Gaussian white noise (e.g., 3 meters standard deviation)
    R_true = np.linalg.norm(r_true - ground_station)
    R_measured = R_true + np.random.normal(0, 3.0)

    # Populate Design Matrix A (Line-of-sight partial derivatives)
    A[i, 0:3] = los
    A[i, 3:6] = los * (i * dt_seconds)

    # Populate Residual Vector b (Measured Range - Nominal Range)
    b[i] = R_measured - R_nom

  # 4. Solve Least Squares via QR Factorization
  Q, R_mat = np.linalg.qr(A)
  Q1_b = Q.T @ b
  delta_x = np.linalg.solve(R_mat[:6, :6], Q1_b[:6])

  # 5. Compute Matrix Condition Number for Numerical Stability Analysis
  cond_A = np.linalg.cond(A)

  print("\n--- SGP4 Orbit Determination Results ---")
  print(f"Injected True State Offset: {true_state_offset}")
  print(
      "Estimated State Corrections (delta_x):\n",
      np.round(delta_x.flatten(), 4),
  )
  print(f"Matrix Condition Number kappa(A): {cond_A:.2f}")


if __name__ == "__main__":
  run_sgp4_orbit_determination()
