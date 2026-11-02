# LEO Satellite Orbit Determination via QR Factorization

A data-driven, deterministic state estimation pipeline for Low Earth Orbit (LEO) satellite navigation. This project uses real-world NORAD Two-Line Element (TLE) data propagated via the SGP4 model, linearizes the observation equations, and stably solves for 6-DOF orbital state corrections using **Householder QR Factorization** and matrix equilibration.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tanerjn/satnav.git
   cd satnav
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

To execute the main orbit determination script and run the least-squares solver:

```bash
python src/satellite_model.py
```

---

## {4} Mathematical Calculations Performed

The python pipeline performs the following computational steps:

1. **Ephemeris Propagation (SGP4):**
   * Uses two-line element sets to calculate nominal satellite positions $\mathbf{r}_{\text{nominal}}(t)$ across observation time steps.

2. **Observation Matrix Assembly ($\mathbf{A}$):**
   * Constructs the $m \times 6$ design matrix $\mathbf{A} \in \mathbb{R}^{m \times 6}$ using line-of-sight range partial derivatives for position and time-scaled partials for velocity:
     $$\mathbf{A}_{i,:} = \begin{bmatrix} \frac{\mathbf{r}_i - \mathbf{R}_g}{\|\mathbf{r}_i - \mathbf{R}_g\|}, & t_i \cdot \frac{\mathbf{r}_i - \mathbf{R}_g}{\|\mathbf{r}_i - \mathbf{R}_g\|} \end{bmatrix}$$

3. **Differential Residual Vector Formulation ($\mathbf{b}$):**
   * Computes measurement range residuals:
     $$\Delta b_i = R_{\text{measured}, i} - R_{\text{nominal}, i}$$

4. **Matrix Conditioning & Equilibration:**
   * Applies column scaling to prevent severe ill-conditioning ($\kappa(\mathbf{A}) \approx 10^{10}$) caused by scale discrepancies between position and velocity partials.

5. **Stable Numerical Least-Squares Solution:**
   * Solves $\mathbf{A} \Delta \mathbf{x} \approx \mathbf{b}$ via **Householder QR Factorization** ($\mathbf{A} = \mathbf{Q}\mathbf{R}$):
     $$\mathbf{R}_1 \Delta \mathbf{x} = \mathbf{Q}_1^T \mathbf{b}$$
   * Computes the matrix condition number $\kappa(\mathbf{A}_{\text{scaled}})$ using Singular Value Decomposition (SVD).

---

## 📈 Expected Outcome & Results

Upon successful execution, the script prints:
* **Injected True State Offset:** The simulated ground-truth position/velocity error ($\Delta \mathbf{x}_{\text{true}}$).
* **Estimated State Corrections ($\Delta \mathbf{x}$):** The least-squares estimate computed via QR factorization.
* **Matrix Condition Number $\kappa(\mathbf{A})$:** Quantitative evaluation of matrix stability.

### Sample Console Output:
```text
--- LEO Satellite Orbit Determination using Real SGP4 TLE Data ---
Propagating TLE data and generating 50 measurements over extended arc...

--- Improved SGP4 Orbit Determination Results ---
Injected True State Offset: [ 100.  -80.   40.   -0.1   0.2   0.05]
Estimated State Corrections (delta_x):
 [ 99.8432 -80.1205  39.9512  -0.0998   0.2003   0.0501]
Matrix Condition Number kappa(A_scaled): 142.35
```

---

## 📁 Repository Structure

```text
satnav/
│
├── data/              # Orbital parameters and TLE records
├── src/               # Python source files (satellite_model.py)
├── results/           # Output residual logs and condition number analysis
├── docs/              # LaTeX source code and figure assets
├── README.md          # Project overview & usage
└── requirements.txt   # Python dependency list
```
