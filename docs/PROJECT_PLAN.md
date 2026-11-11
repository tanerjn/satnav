# ISS-first satellite state prediction project

## Layout
- `data/iss/nasa_oem/`: downloaded NASA OEM source files
- `data/iss/celestrak/`: ISS CelesTrak GP/TLE snapshots
- `data/iss/processed/`: parsed SI-unit state CSV files
- `src/satnav/iss_oem_parser.py`: NASA OEM text state parser
- `src/satnav/iss_sgp4.py`: SGP4 propagation in TEME (requires `sgp4`)
- `src/satnav/elm_model.py`: ELM regression
- `src/satnav/least_squares.py`: QR and SVD least-squares solvers
- `src/satnav/evaluation.py`: position and velocity RMSE
- `src/satnav/dynamics.py`: existing two-body dynamics, retained
- `scripts/prepare_iss_dataset.py`: convert OEM to CSV
- `scripts/train_iss_elm.py`: one-step ELM demonstration
- `scripts/compare_iss_models.py`: compare aligned state CSVs
- `results/iss/`: ISS experiment outputs

## Commands
```bash
pip install -r requirements-project.txt
pip install sgp4  # only needed for SGP4 propagation
PYTHONPATH=src python scripts/prepare_iss_dataset.py --oem data/iss/nasa_oem/ISS.OEM_J2K_EPH.txt
PYTHONPATH=src python scripts/train_iss_elm.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

**Validation caveat:** NASA OEM is an operational ephemeris rather than raw measurements. SGP4 produces TEME states, whereas NASA OEM is J2000; perform a validated frame conversion and epoch alignment before comparing. The simple OEM parser reads state rows but does not yet validate all OEM metadata, sampling gaps or maneuvers. Do not interpret one-step held-out results as open-loop orbit forecast accuracy.

Legacy `src/satellite_model*.py` scripts and original datasets are retained to avoid losing previous work.


## Phase 1 verification — NASA OEM, RK4 and ISS SGP4

Run from the repository root after pulling the latest changes:

```bash
git pull origin main
python -m pip install -r requirements-project.txt
PYTHONPATH=src python -m unittest discover -s tests -v
```

### 1A. Prepare and inspect the NASA OEM states

Download the NASA ISS OEM text file into `data/iss/nasa_oem/ISS.OEM_J2K_EPH.txt` if it is not already present. Prepare the SI-unit CSV:

```bash
PYTHONPATH=src python scripts/prepare_iss_dataset.py --oem data/iss/nasa_oem/ISS.OEM_J2K_EPH.txt
```

Inspect `data/iss/processed/nasa_iss_states.csv` for chronological `epoch_utc` and finite `x_m,y_m,z_m,vx_mps,vy_mps,vz_mps`. The existing experiment scripts print sampling-interval counts and exclude non-240-second pairs. Verify the OEM header metadata manually: `REF_FRAME`, `TIME_SYSTEM`, `CENTER_NAME`, units and any maneuver/quality flags. The parser does not yet fully validate this metadata.

### 1B. Reproduce the two-body RK4 baseline

```bash
PYTHONPATH=src python scripts/iss_experiments.py --hidden 64 --output results/iss/phase1_check
```

Open `results/iss/phase1_check/iss_sensitivity.csv` and find `two_body_rk4`. On the earlier 5,403-state dataset and the original 70/30 test split, the observed one-step RMSE was about **377.616 m position** and **3.050 m/s velocity**. Differences can result from dataset revisions or different splits. This is a one-step calculation starting from the observed state, not a free-running forecast.

### 1C. Check ISS TLE/SGP4 propagation and coordinate conversion

Save an ISS TLE (NORAD 25544) as `data/iss/celestrak/iss.tle`. CelesTrak's current TLE endpoint is:

`https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE`

The TLE should have an epoch near the OEM data; for historical comparisons use archived TLEs from the same period. Then run:

```bash
PYTHONPATH=src python scripts/compare_iss_sgp4.py \
  --oem data/iss/processed/nasa_iss_states.csv \
  --tle data/iss/celestrak/iss.tle \
  --max-age-days 2
```

Expected output includes the TLE epoch, number of matched OEM epochs, position and velocity RMSE, and a frame warning. An error saying `No OEM epochs near TLE epoch` means the chosen TLE does not overlap the requested OEM window; obtain an appropriate historical TLE rather than arbitrarily relaxing the age limit. An SGP4 error may indicate an invalid TLE or a propagation failure.

**User-reported exploratory run (2026-11-11):**

| Quantity | Reported value |
| --- | ---: |
| TLE epoch (UTC) | 2026-11-11 08:03:57.286073 |
| Matched OEM epochs (within ±2 days) | 1,381 |
| Position RMSE | 901.186008 m |
| Velocity RMSE | 0.990020 m/s |

**Scientific limitations:** SGP4 produces TEME states. `iss_frames.py` converts them to **GCRS**, which is not guaranteed identical to the NASA OEM's J2000/EME2000 definition. Confirm the OEM reference frame and validate the transformation before interpreting meter-level differences as a rigorous comparison. This script uses **one TLE**, not an epoch-matched historical TLE series; its ±2-day matching window does not mean every prediction is equally reliable. The SGP4 and ELM numbers below are on different epoch subsets and **must not be compared as a controlled head-to-head test**.

## Phase 2 verification — ELM, ridge regularization and QR/SVD

### 2A. Check the least-squares implementations

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from satnav.least_squares import solve_least_squares

rng = np.random.default_rng(42)
A = rng.normal(size=(100, 20))
B = rng.normal(size=(100, 6))
for ridge in (0, 1e-4, 0.01, 1):
    qr = solve_least_squares(A, B, method="qr", ridge=ridge)
    svd = solve_least_squares(A, B, method="svd", ridge=ridge)
    delta = np.linalg.norm(qr - svd)
    print(f"ridge={ridge:g} QR/SVD coefficient difference={delta:.3e}")
    assert np.allclose(qr, svd, rtol=1e-8, atol=1e-10)
print("PASS: QR and SVD agree for the well-conditioned test matrix")
PY
```

For the ridge objective, the output weights minimize
`||H beta - Y||_F^2 + lambda ||beta||_F^2`.
QR uses an augmented system; SVD uses regularized singular-value factors. Agreement on this test is a **numerical consistency check**, not proof that both solvers are equally stable on ill-conditioned ISS features.

### 2B. Select hyperparameters using chronological validation

```bash
PYTHONPATH=src python scripts/iss_validate.py \
  --hidden 16 32 64 128 256 \
  --ridge 0 0.000001 0.0001 0.01 1 100
```

The experiment evaluates 2 architectures × 5 hidden sizes × 6 ridge values × 2 solvers = **120 candidate fits**. It uses chronological 60% training, 20% validation and 20% test state segments (no shuffling). Training-only scaling is applied for each fit. The best model is selected using **validation position RMSE**; the selected configuration is refitted on the first 80% and evaluated once on the held-out final 20%. Irregular time-step pairs are excluded.

Inspect:
- `results/iss/validated/validation_grid.csv` — all candidate validation results.
- `results/iss/validated/selected_model_test.json` — selected configuration, final test metrics, and split counts.

**User-reported run (2026-11-11):**

| Quantity | Reported value |
| --- | ---: |
| Selected model | physics_residual_elm |
| Hidden neurons | 128 |
| Ridge λ | 0.0001 |
| Solver | QR |
| Validation position RMSE | 6.994822 m |
| Validation velocity RMSE | 0.057624 m/s |
| Held-out test position RMSE | 7.351744 m |
| Held-out test velocity RMSE | 0.060672 m/s |
| Train / validation / test states | 3,241 / 1,081 / 1,081 |
| Validation / test pairs | 1,080 / 1,077 |

These are **one-step, teacher-forced** predictions against NASA OEM operational ephemeris, not independent measured ground truth or multi-step free-running forecasts. The earlier 43.177 m result used a different split and is not directly comparable. The reported 7.352 m is not a guaranteed accuracy on future ISS ephemerides.

### 2B.1. Historical pre-ridge baseline and interpretation

Before ridge regularization was implemented, we ran `scripts/iss_experiments.py` on the NASA ISS OEM dataset with a chronological **70% train / 30% test** split and a 240-second one-step prediction horizon. The best reported physics-residual ELM among the originally tested 32/64/128/256-neuron configurations used **64 hidden neurons**, **no ridge penalty (λ = 0)**, and either QR or SVD (the two solvers agreed numerically):

#### Comparing our two experiments

| Configuration | Earlier experiment | New experiment |
| --- | --- | --- |
| Architecture | Physics-residual ELM | Physics-residual ELM |
| Hidden neurons | 64 | 128 |
| Ridge \(\lambda\) | 0 | 0.0001 |
| Solver | QR or SVD | QR |
| Position RMSE | **43.177 m** | **7.352 m** |
| Velocity RMSE | **0.354 m/s** | **0.0607 m/s** |
| Evaluation | 70/30 split | 60/20/20 split |

The earlier **70/30 split** used the first 70% of chronologically ordered ISS states for training and the remaining 30% for testing, without a separate validation set. The newer **60/20/20 split** used 60% for training, 20% for selecting the hidden-neuron count, ridge parameter, and solver, and the final 20% for testing after refitting the selected model on the first 80%. Both results are **one-step (240-second), teacher-forced predictions** from observed ISS states, rather than free-running predictions of an entire future orbit.

Both RMSE columns report one-step, 240-second held-out test errors against NASA OEM operational ephemeris. The new experiment selected its configuration using validation data; its validation position RMSE was **6.994822 m**. The earlier experiment's physics-only RK4 baseline had **377.616274 m** position RMSE on its original test split.

**Do not interpret 43.177 m versus 7.352 m as a controlled measure of ridge's improvement.** The neuron count, train/test boundaries, training amount, and evaluated timestamps differ. To isolate ridge's effect, compare λ = 0 with λ = 0.0001 at the *same* hidden-neuron count and solver using the *same validation split*, as recorded in `results/iss/validated/validation_grid.csv`. For a clean final test, select hyperparameters using validation only.

#### Meaning of ridge λ = 0.0001

ELM fits the output-weight matrix β from hidden-layer features H and normalized training targets Y. Without regularization, it minimizes `||Hβ − Y||_F²`. With ridge (Tikhonov) regularization, it minimizes:

```text
||Hβ − Y||_F² + λ ||β||_F²
```

For **λ = 0.0001 = 10⁻⁴**, a small positive penalty discourages excessively large output coefficients and can reduce sensitivity to nearly dependent hidden features. λ is a *weight penalty*, not a position-error tolerance, percentage improvement, or measurement unit. Its effective strength depends on feature/target scaling and the scale of H; in our script, normalization is fitted on the training subset. Ridge is implemented using regularized singular-value factors for SVD and an augmented least-squares system for QR, avoiding an explicit normal-equation inverse.

### 2C. Additional checks before marking Phase 2 complete

1. Inspect `validation_grid.csv` for finite RMSE and compare QR/SVD accuracy at identical architecture, neuron count and λ.
2. Verify that the chosen configuration is selected **only** from validation results, not the test set.
3. Benchmark QR and SVD with warm-up and multiple timed repeats on the **same fixed hidden matrix**; report median and dispersion rather than drawing conclusions from a single run.
4. Report hidden-matrix rank, condition number, residual norm and coefficient norm; assess how ridge λ changes stability and validation accuracy.
5. Run multi-horizon **free-running** evaluation against an RK4-only baseline on the same held-out windows. The one-step results above do not establish long-horizon accuracy.
6. For final research claims, use a genuinely new temporal holdout or independent ephemeris period, especially because earlier test-set results were inspected while choosing model families and neuron sizes.

## Phase completion checklist

- **Phase 1:** OEM parsing and RK4 baseline implemented; exploratory SGP4 test executed. Still requires authoritative OEM frame verification, robust frame alignment, and preferably historical epoch-matched TLE comparisons.
- **Phase 2:** ELM architectures, QR/SVD, ridge and chronological validation implemented; first validation/test run completed. Still requires solver-consistency check execution, repeated runtime/stability benchmarks, and final independent evaluation.
- **Phase 3:** Full aligned head-to-head model comparison and multi-horizon experiments remain future work.
- **Phase 4:** Final report, tables, figures and reproducibility discussion remain future work.
