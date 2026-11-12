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
