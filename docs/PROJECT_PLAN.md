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
