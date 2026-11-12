"""Generate reproducible ISS experiment figures from saved CSV/JSON outputs.

No model retraining. Only plots files that already exist.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("Saved", path)


def validation_plot(path, out):
    rows = read_csv(path)
    if not rows:
        return
    # For each family and hidden size, show its best validation score across
    # ridge values and solvers. This is NOT a held-out test comparison.
    best = {}
    for r in rows:
        k = (r["model"], int(r["hidden"]))
        v = float(r["position_rmse_m"])
        best[k] = min(best.get(k, float("inf")), v)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for family in sorted({k[0] for k in best}):
        hidden = sorted(h for f, h in best if f == family)
        ax.plot(hidden, [best[(family, h)] for h in hidden], marker="o", label=family)
    ax.set(xlabel="ELM hidden neurons", ylabel="Best validation position RMSE (m)",
           title="ISS ELM validation (60/20/20 chronological split)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, out / "validation_position_rmse.png")


def solver_plot(path, out, ridge):
    rows = [r for r in read_csv(path)
            if r["model"] == "physics_residual_elm"
            and abs(float(r["ridge"]) - ridge) <= 1e-12]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for solver in ("qr", "svd"):
        subset = sorted((int(r["hidden"]), float(r["time_median_ms"]))
                        for r in rows if r["solver"] == solver)
        if subset:
            ax.plot([x for x, _ in subset], [y for _, y in subset],
                    marker="o", label=solver.upper())
    ax.set(xlabel="ELM hidden neurons", ylabel="Median solver-only runtime (ms)",
           title=f"Physics-residual ELM: QR vs SVD (ridge={ridge:g})")
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, out / "solver_runtime.png")


def frame_plot(path, out):
    data = json.loads(path.read_text(encoding="utf-8"))
    gcrs = data["sgp4_gcrs_vs_oem"]
    eme = data["sgp4_eme2000_vs_oem"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, ylabel in zip(
            axes, ("position_rmse_m", "velocity_rmse_mps"),
            ("Position RMSE (m)", "Velocity RMSE (m/s)")):
        vals = [float(gcrs[metric]), float(eme[metric])]
        ax.bar(["GCRS", "Mean J2000"], vals)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.18)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.suptitle("SGP4 vs NASA OEM: frame-convention comparison")
    save(fig, out / "frame_comparison.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/iss/figures")
    p.add_argument("--validated", default="results/iss/validated/validation_grid.csv")
    p.add_argument("--benchmark", default="results/iss/solver_benchmark/solver_benchmark.csv")
    p.add_argument("--frames", default="results/iss/phase1/frame_verification.json")
    p.add_argument("--ridge", type=float, default=1e-4)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(Path(args.validated), validation_plot),
             (Path(args.benchmark), lambda a, b: solver_plot(a, b, args.ridge)),
             (Path(args.frames), frame_plot)]
    for path, plot in tasks:
        if path.exists():
            plot(path, out)
        else:
            print("Skipped missing input:", path)


if __name__ == "__main__":
    main()
