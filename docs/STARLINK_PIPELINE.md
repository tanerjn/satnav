# Starlink pipeline (separate from ISS)

Initial target: **NORAD 45532 / STARLINK-1338**. Other candidates: 45695, 45755, 46563. The raw inputs are Space-Track CCSDS OMM JSON records with embedded TLEs, not independent position/velocity measurements. All SGP4 comparisons in this pipeline use **TEME**, avoiding the ISS OEM EME2000 issue.

## Phases

1. **Phase 1 — Data integrity and SGP4 baseline:** validate JSON, NORAD identity, TEME/UTC, TLE epochs, duplicates, update gaps; compare *earlier* TLE forecasts to *later* TLE-derived states at their own epochs.
2. **Phase 2 — ELM:** construct carefully time-split training states and compare direct ELM vs physics-residual ELM with QR/SVD and ridge. Clearly label any SGP4-generated training targets as synthetic/model-derived.
3. **Phase 3 — Open-loop forecasting:** frozen historical model and initial state, forecast 1h, 6h, 12h, 1d, 3d, 7d; evaluate only against later available estimates. No future TLEs used in forecasting.
4. **Phase 4 — Reporting and generalization:** accuracy versus horizon, per-satellite metrics, solver runtime, and holdout-satellite experiments.

## Phase 1 commands

From the repository root:

```bash
python -m pip install -r requirements-project.txt

PYTHONPATH=src python scripts/starlink_prepare.py \
  --input data/starlink/STARLINK-45532_2026-01-01_2026-02-01.json \
  --norad 45532 \
  --output data/starlink/processed/45532

PYTHONPATH=src python scripts/starlink_phase1_baseline.py \
  --catalog data/starlink/processed/45532/tle_catalog.csv \
  --output results/starlink/45532/phase1
```

**Check the actual input filename** in your `data/starlink` directory and adjust `--input` accordingly. The uploaded file uses the example filename above; GitHub's path has not been verified.

Outputs:

- `data/starlink/processed/45532/tle_catalog.csv` — sorted, deduplicated OMM/TLE catalog
- `data/starlink/processed/45532/catalog_quality.json` — record counts, gaps, validation issues
- `results/starlink/45532/phase1/sgp4_forward_baseline.csv` — error per requested forecast horizon
- `results/starlink/45532/phase1/sgp4_forward_baseline.json` — results and provenance

**Interpretation:** the baseline compares two different TLE-based orbit estimates; it is **not** a validation against measured ground truth. Requested horizons are matched to the closest available later update within the configured tolerance; actual horizons and sample counts are reported. For rigorous evaluation, inspect update gaps, maneuvers, and selection bias before drawing performance conclusions.

ISS scripts and results remain in their original locations.

## Historical single-satellite workflow: STARLINK-1008 (NORAD 44714)

The original 45532 Phase 1 workflow above is retained for reproducibility. The following commands use the historical Space-Track OMM/GP JSON file for NORAD 44714 (January–March 2026). Run from the repository root after installing `requirements-project.txt`. The raw file currently referenced by the committed quality report is `data/starlink/starlink_44714.json`; adjust `--input` if yours has a different name.

### 1. Prepare and validate the historical catalog

```bash
PYTHONPATH=src python scripts/starlink_prepare.py \
  --input data/starlink/starlink_44714.json \
  --norad 44714 \
  --output data/starlink/processed/44714
```

Inspect `data/starlink/processed/44714/catalog_quality.json` and `tle_catalog.csv`. The preparation script checks NORAD identity, TEME/UTC, embedded TLE validity, and TLE/OMM epoch agreement. It keeps the latest-created record at each *exactly matching* epoch; very close but nonidentical epochs can remain. Inspect gaps and near-duplicates rather than assuming every row is an independent orbit estimate.

The committed 44714 quality report lists 266 valid records, 234 unique epochs, 32 duplicate epochs, no invalid records, and a maximum epoch gap of about 71.21 hours.

### 2. Phase 1 corrected: exact-horizon SGP4 evaluation

```bash
PYTHONPATH=src python scripts/starlink_phase1_corrected.py \
  --norads 44714 \
  --horizons-hours 1 6 12 24 72 168 \
  --reference-tolerance-hours 3
```

**Argument spelling:** Phase 1 corrected uses `--horizons-hours` (plural). Do not use `--horizon-hours` with this script.

Outputs under `results/starlink/44714/phase1_corrected/`:

- `sgp4_exact_horizon.json`: method, assumptions, counts and RMSE by horizon.
- `sgp4_exact_horizon.csv`: summary metrics and skipped-case counts.
- `sgp4_per_forecast.csv`: individual origins, targets, source/reference GP IDs, creation dates and errors.

The forecast origin is `max(source EPOCH, source CREATION_DATE)`; the target is exactly origin plus the requested horizon. Reference states are produced by propagating a later-published TLE to that target, selecting the closest reference epoch within the tolerance. They are **retrospective TLE-derived estimates, not independent measured truth**. A later-published reference is used only for retrospective scoring, never as a forecast input.

The currently committed 44714 Phase 1 results are:

| Horizon | Eligible cases | Position RMSE (km) | No reference |
| --- | ---: | ---: | ---: |
| 1 h | 121 | 21.69 | 113 |
| 6 h | 121 | 28.72 | 113 |
| 12 h | 112 | 34.28 | 122 |
| 24 h | 114 | 112.54 | 120 |
| 72 h | 91 | 1439.45 | 143 |
| 168 h | 86 | 6157.42 | 148 |

All six horizons have zero reported SGP4 propagation errors in this run. These aggregate Phase 1 errors include all eligible cases and are **not directly comparable** to Phase 2 test-only RMSE. Long-horizon discrepancies may include maneuvers, element fitting differences, and propagation error.

### 3. Plot improved SGP4 trajectories

Run **Phase 1 corrected first**, since the improved plotting script reads the processed `tle_catalog.csv` and `phase1_corrected/sgp4_per_forecast.csv`.

For one 24-hour forecast case:

```bash
python scripts/plot_starlink_trajectories_improved.py \
  --norad 44714 \
  --horizon-hours 24 \
  --case-index 0
```

For other horizons, change `--horizon-hours` to one of `1 6 12 24 72 168`; this plotting script uses the **singular** `--horizon-hours` flag, unlike Phase 1 corrected. `--case-index` selects the zero-based matching row from the Phase 1 forecast cases for that horizon. If there are no eligible cases, choose another horizon or source.

For a fixed starting TLE, first inspect `source_gp_id` in `results/starlink/44714/phase1_corrected/sgp4_per_forecast.csv`. Then run (replace the placeholder with an actual GP ID present in the file):

```bash
python scripts/plot_starlink_trajectories_improved.py \
  --norad 44714 \
  --horizon-hours 72 \
  --source-gp-id YOUR_SOURCE_GP_ID
```

To compare horizons from the **same source GP ID and forecast origin**:

```bash
python scripts/plot_starlink_trajectories_improved.py \
  --norad 44714 \
  --compare-horizons \
  --source-gp-id YOUR_SOURCE_GP_ID
```

Only horizons with an eligible Phase 1 reference for that source are included. The script rejects comparisons whose selected cases do not share the same source and origin. It is also possible to fix the case with `--origin-utc`.

To change plot density or reference-epoch tolerance:

```bash
python scripts/plot_starlink_trajectories_improved.py \
  --norad 44714 \
  --horizon-hours 24 \
  --case-index 0 \
  --samples 600 \
  --reference-tolerance-hours 3
```

PNG outputs are saved to `results/starlink/44714/phase1_corrected/` as `trajectory_24h_source<GP_ID>.png` for individual cases and `trajectory_comparison_source<GP_ID>.png` for comparisons. The single-case image shows a 3D Earth/SGP4 forecast, the updated as-of-time TLE estimate with gaps and switches, and position differences. The endpoint marker uses the exact reference selected by corrected Phase 1; intermediate estimates may use different TLEs. **These plots visualize SGP4 versus updated TLE-derived estimates, not direct/residual ELM trajectories.**

### 4. Phase 2 v2: direct ELM and SGP4-residual ELM

Run Phase 1 corrected before Phase 2 v2; the ELM script consumes its per-forecast CSV and the same processed catalog. For a first QR run:

```bash
PYTHONPATH=src python scripts/starlink_phase2_elm_v2.py \
  --norads 44714 \
  --hidden 64 \
  --solver qr
```

For a small hidden-layer and solver sweep:

```bash
for solver in qr svd; do
  for hidden in 32 64 128 256; do
    PYTHONPATH=src python scripts/starlink_phase2_elm_v2.py \
      --norads 44714 --hidden "$hidden" --solver "$solver"
  done
done
```

Each run writes to `results/starlink/44714/phase2_v2_<solver>_<hidden>n/`:

- `elm_metrics.csv`: chronological train/validation/test counts, chosen ridge and residual alpha, validation RMSE, test SGP4/direct/residual RMSE and fit/selection time.
- `elm_per_forecast.csv`: held-out case errors for SGP4, direct ELM and residual ELM.
- `experiment.json`: parameters, reference definition and protocol.

The script uses chronological forecast-origin splits (defaults: 55% training, 20% validation, 25% testing), with reference-creation cutoffs for training and validation. Check the sample counts for every horizon; long-horizon cases can be sparse, and overlapping forecasts are not independent. The residual correction factor `alpha` is validation-selected from `0, 0.25, 0.5, 1`; **alpha=0 is exactly SGP4**, not an ELM improvement. Compare ELM and SGP4 on the *same Phase 2 test cases*, not against the Phase 1 all-case summary.

The committed 44714 **QR, 64-neuron** run reports:

| Horizon | Train / val / test | Test SGP4 RMSE (km) | Direct ELM RMSE (km) | Residual ELM RMSE (km) | Residual alpha |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 h | 66 / 24 / 31 | 10.04 | 34.17 | 10.04 | 0 |
| 6 h | 66 / 24 / 31 | 15.58 | 194.49 | 15.58 | 0 |
| 12 h | 61 / 22 / 28 | 19.32 | 202.58 | 19.32 | 0 |
| 24 h | 60 / 22 / 29 | 34.98 | 1059.76 | 34.98 | 0 |
| 72 h | 50 / 14 / 23 | 148.10 | 6282.27 | 148.10 | 0 |
| 168 h | 45 / 9 / 22 | 766.62 | 4595.90 | 766.62 | 0 |

**Interpretation:** the tested direct ELM is worse than SGP4 at every horizon; the residual ELM falls back to SGP4 at every horizon, so this run does not demonstrate learned improvement. Additional model sizes and satellites are needed before drawing general conclusions.

### 5. Review and reproducibility checklist

1. Inspect `catalog_quality.json` for duplicates, missing dates and long update gaps.
2. Inspect `sgp4_exact_horizon.csv` for eligible-case and skipped-reference counts.
3. Use `sgp4_per_forecast.csv` to select plot cases and investigate outliers or possible maneuvers.
4. Run QR and SVD Phase 2 v2 sweeps, then inspect `elm_metrics.csv` and `elm_per_forecast.csv`.
5. Report the test subset and sample counts alongside every RMSE; do not claim alpha=0 is an ELM gain.
6. Preserve the earlier 45532/45695/45755/46563 experiments and ISS outputs; keep 44714 results in their own directories.
