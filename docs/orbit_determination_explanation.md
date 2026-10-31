This script performs a simplified **Orbit Determination (OD)** calculation—a fundamental task in satellite navigation and aerospace engineering.

In short, it simulates tracking a Low Earth Orbit (LEO) satellite from a ground station, adds realistic sensor noise, and uses linear least squares to estimate the satellite's position and velocity state vector.

---

## Step-by-Step Breakdown of What the Script Does

### 1. Defining Constants and Initial States

* It sets up fundamental physics constants like Earth’s gravitational parameter ($\mu$) and defines a hypothetical initial position (`r0`) and velocity (`v0`) for the satellite.

* It also defines environmental variables (like atmospheric drag coefficient $C_d$ and solar radiation pressure parameters), which are included as placeholders for a full perturbation-aware orbit propagator.

### 2. Simulating Measurements & Building Matrix $A$

* The script loops through 10 time steps (spaced 10 seconds apart: $t_i = i \times 10$).

* At each step, it linearly propagates where the satellite should be (`sat_pos = r0 + v0 * t_i`).

* It calculates the true straight-line distance (range $R$) from a ground station on Earth to the satellite.

* It adds random Gaussian white noise (`np.random.normal(0, 5.0)`) to simulate real-world radar or GPS measurement errors (with a standard deviation of 5 meters).

* It constructs the Design Matrix $A$ (size $10 \times 6$). Each row contains the partial derivatives showing how a tiny change in position ($x, y, z$) and velocity ($v_x, v_y, v_z$) would affect the measured range over time.

### 3. Solving the System ($Ax \approx b$)

* It uses QR Factorization and Least Squares (`np.linalg.lstsq`) to solve the overdetermined system of equations.

* Because you have more measurements (10 time steps) than unknowns (6 state variables), least squares finds the statistically best-fit state vector that minimizes the measurement errors.

---

## Explaining the Output Results

When running the script, the solver outputs an estimated 6-element state vector:

```text
Estimated State Corrections x: 
[ 4.99996520e+05   1.51244392e-03   0.00000000e+00   9.93381973e-02   7.61258289e+03   0.00000000e+00 ]
```

Because of how the linear approximation and the absolute range vector $b$ were set up in this demo, the resulting vector $x = [\Delta x, \Delta y, \Delta z, \Delta v_x, \Delta v_y, \Delta v_z]^T$ directly recovers the key kinematic components of your initial setup:

1. **Position along X-Axis ($4.99996520 \times 10^5 \approx 500,000\text{ m}$):**
   This represents the initial separation along the X-axis between your ground station ($6,378,137\text{ m}$) and the satellite's starting position ($6,878,137\text{ m}$):
   $$\Delta x = 6,878,137 - 6,378,137 = 500,000\text{ m}$$

2. **Velocity along Y-Axis ($7.61258289 \times 10^3 \approx 7,612.6\text{ m/s}$):**
   This value directly matches the initial velocity component $v_y$ defined in your vector `v0` ($7,612.6\text{ m/s}$), demonstrating that the solver correctly reconstructed the satellite's orbital speed from noisy range data over time.

3. **Zeros and Minor Residuals:**
   * **Z-axis components ($0.00000000e+00$):** Since there is zero initial position displacement or velocity along the Z-axis, the state remains exactly zero.
   * **Minor noise fluctuations ($1.51 \times 10^{-3}\text{ m}$ and $9.93 \times 10^{-2}\text{ m/s}$):** These tiny non-zero values on the Y-position and X-velocity represent the small fitting adjustments made by the least-squares solver to absorb the 5-meter Gaussian measurement noise added during simulation.
