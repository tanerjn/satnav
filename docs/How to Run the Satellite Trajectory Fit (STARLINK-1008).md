# How to Run the Satellite Trajectory Fit (STARLINK-1008)

A step-by-step guide to running `orbit_fit.py`, reading its output, and changing its settings.

## 1. What the program does

The script reads your Space-Track element-set file (`starlink_44714.json`, NORAD 44714) and does the following:

1. Builds reference positions by propagating each element set with SGP4, only within ±30 minutes of that set's own epoch (TEME frame, km).
2. Fits an orbit model to a **fit window** using least squares, solved with both QR and SVD.
3. Predicts the satellite's position over the following **validation window** and compares it with the reference positions.
4. Prints error tables and saves a four-panel figure.

The model is a Keplerian orbit with J2 secular drift. It is fitted by Gauss-Newton, with the drag term ṅ either fixed at zero or estimated.

## 2. Requirements

- Python 3.9 or newer
- Python packages: `numpy`, `pandas`, `matplotlib`, `sgp4`
- The data file `starlink_44714.json`
- The script `orbit_fit.py`

Install the packages with:

```
pip install numpy pandas matplotlib sgp4
```

## 3. Folder setup

Put both files anywhere you like, for example:

```
my_project/
    orbit_fit.py
    starlink_44714.json
```

Two paths inside the script are hard-coded for the environment where it was written, and you will need to adapt them:

| What | Where in the script | Default value |
| --- | --- | --- |
| Input data | `path = sys.argv[1] if ... else "..."` in the main block | `/mnt/user-data/uploads/starlink_44714.json` |
| Output figure | `plt.savefig("...")` on the last line | `/mnt/user-data/outputs/orbit_fit_results.png` |

The input path can be passed on the command line, so you don't need to edit it. The figure path has to be edited. Change it to something like `orbit_fit_results.png` so the image is saved next to the script.

## 4. Running it

From the folder that holds the script:

```
python orbit_fit.py starlink_44714.json
```

To keep the text output in a file as well as seeing it on screen:

```
python orbit_fit.py starlink_44714.json | tee run_log.txt
```

Do not pipe the output into `head`. When `head` closes the pipe, Python is stopped before it reaches the plotting step and the figure is not saved or is left stale.

The rolling experiment fits the model about 70 start times for several window lengths, so allow some time. A modern laptop should finish in a few minutes at most.

## 5. Reading the output

### 5.1 Detailed case

The first block of output is one worked example: a fit window starting 2026-02-05 00:00 UTC, followed by a validation window.

```
model                  cond  fit RMS m  val RMS km  max km  err@24h  err@72h
```

| Column | Meaning |
| --- | --- |
| `cond` | Condition number of the scaled least-squares problem. Values around 10 to 40 mean well-conditioned. |
| `fit RMS m` | 3D RMS residual inside the fit window, in metres. |
| `val RMS km` | 3D RMS error over the validation window, in km. |
| `max km` | Largest 3D error in the validation window. |
| `err@24h`, `err@72h` | 3D error at 24 h and 72 h after the end of the fit window. |

Rows you will see:

- **Kepler+J2, ndot=0** and **Kepler+J2+2u harm, ndot=0**, each solved with `[qr]` and `[svd]`. The QR and SVD rows should be identical to the digits shown.
- **Kepler+J2+2u harm, ndot free**: shows what happens when the drag term is estimated. It extrapolates badly.
- **SGP4 baseline (last element set)**: the operational comparison, which propagates the newest element set available at the end of the fit window.
- **Chebyshev deg 8**: the naive polynomial baseline. Its enormous values (around 1e9 km) are expected, because polynomials diverge outside the fit window.

### 5.2 Rolling experiment

The second block repeats the fit from many start times between 2026-02-01 and 2026-03-10, every 12 hours. It reports the **median / 90th-percentile** 3D error in km at +6, +12, +24, +48 and +72 hours after the fit window. The row labels are:

- `24 h`, `48 h`, `72 h`: fit-window length, with ṅ estimated.
- `48fix`, `72fix`: fit-window length, with ṅ fixed at zero (the best-performing setup).
- `sgp4_48`: the SGP4 baseline.

### 5.3 Figure

The saved image has four panels:

1. 3D error versus time after the fit window for each model.
2. Radial, along-track and cross-track errors for the best model. Along-track dominates.
3. QR and SVD predictions overlaid, which should coincide.
4. Error versus prediction horizon from the rolling experiment.

The gaps in the plotted curves are expected. Reference positions exist only near element-set epochs.

## 6. Changing the settings

All of these are in the `if __name__ == "__main__":` block unless noted.

| To change | Edit |
| --- | --- |
| Detailed-case start time | `t0 = pd.Timestamp("2026-02-05 00:00:00")` |
| Fit and prediction lengths of the detailed case | `FIT_H, VAL_H = 72, 72` (hours) |
| Period used for the rolling experiment | `QUIET0`, `QUIET1` |
| Fit-window lengths and ṅ options in the rolling experiment | the tuple `((24, False), (48, False), (72, False), (48, True), (72, True))`, where `True` means ṅ fixed at zero |
| Prediction horizons that are reported | `horizons = np.array([6, 12, 24, 48, 72])` |
| Start-time spacing of the rolling experiment | `freq="12h"` |
| Reference sampling interval and maximum element-set age | `reference_track(..., step=120.0, max_age_min=30.0)` |
| Polynomial baseline degree | the `12` or `8` passed to `cheb_fit(...)` |
| Physical constants | `MU`, `RE`, `J2` near the top of the file |

### Choosing the quiet period

The satellite changed its orbit in January (about 546 km down to 472 km). Windows that include that maneuver will fit poorly, because the model assumes smooth, unpowered motion. Keep the fit and validation windows inside a stable stretch. You can check this by printing `MEAN_MOTION` against `EPOCH` from the JSON file and looking for a flat region.

## 7. Using a different satellite or file

The script works with any Space-Track OMM/GP JSON that holds a list of records with these fields: `EPOCH`, `MEAN_MOTION`, `ECCENTRICITY`, `INCLINATION`, `RA_OF_ASC_NODE`, `ARG_OF_PERICENTER`, `MEAN_ANOMALY`, `BSTAR`, `MEAN_MOTION_DOT`, `NORAD_CAT_ID`. Then:

1. Pass the new file as the command-line argument.
2. Update `QUIET0`, `QUIET1` and `t0` to a stable period for that satellite.
3. For a satellite at a very different altitude or eccentricity, review the model. It assumes a near-circular orbit and only J2 perturbations.

## 8. Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `ModuleNotFoundError: sgp4` (or numpy, pandas, matplotlib) | Install the missing package with pip. |
| `FileNotFoundError` for the JSON file | Pass the correct path as the first argument. |
| Error when saving the figure (directory not found) | Change the `plt.savefig` path to a folder that exists. |
| Figure missing or out of date | The script was interrupted before the last line, for example by piping into `head`. Run it again without piping. |
| A run returns nothing or `None` for a start time | Too few reference points fall in the fit or validation window. Use longer windows or a period with more element sets. |
| Very large errors for a window | The window probably includes a maneuver or an inconsistent element set. Move it to a quieter period, or fix ṅ at zero. |
| Fit does not converge or gives NaN | Check that the start time lies inside your data range. The initial guess comes from the last element set before `t0`. |

## 9. Known limitations

- The "real" positions are SGP4 outputs from the element sets, not independent measurements. They carry about 1 km of their own error, so absolute error numbers are relative to that reference.
- Starlink satellites thrust continuously, so the along-track acceleration changes sign over time. This limits any smooth model and explains why the 90th-percentile errors are several times the medians.
- For true ground truth, use precise ephemerides (SP3 files) or GNSS data.

## 10. Suggested next steps

- Estimate a drag or thrust acceleration in piecewise-constant intervals (for example one per day).
- Use weighted least squares to down-weight element sets that disagree with their neighbours.
- Replace the analytic J2 model with numerical propagation (J2 to J4 gravity, estimated drag coefficient).
