#!/usr/bin/env python3
"""Create one comparison figure/table for the five retrained MPPI models.

No command-line arguments are required.  Edit the constants below only when
the result directory names change.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "model_tuning/results"
OUTPUT_DIR = RESULT_ROOT / "retrained_five_model_comparison"

MODELS = {
    "Slip kinematic (regressed)": "slip_classic",
    "Dynamic (regressed)": "dynamic_classic",
    "Slip kinematic + MLP": "slip_mlp",
    "Dynamic + MLP": "dynamic_mlp",
    "E2E MLP": "e2e",
}
DATASETS = {"0807": "suite_0807_", "Collision pre-impact": "suite_collision_"}


def read_metrics(directory: Path):
    for name in ("metrics.json", "replay_metrics.json"):
        path = directory / name
        if path.exists():
            data = json.loads(path.read_text())
            # evaluate_mppi_bag_replay.py stores several state-clamp policies
            # and horizons; use the runtime-equivalent 1 s result.
            if "fixed_preserve_measured_overspeed" in data:
                data = data["fixed_preserve_measured_overspeed"]["1.00s"]
            return data
    raise FileNotFoundError(f"metrics JSON not found in {directory}")


def pick(d, *keys):
    for key in keys:
        if key in d:
            return float(d[key])
    raise KeyError(f"none of {keys} in metrics")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset, prefix in DATASETS.items():
        for label, suffix in MODELS.items():
            m = read_metrics(RESULT_ROOT / f"{prefix}{suffix}")
            rows.append({
                "dataset": dataset, "model": label,
                "trajectory_mean_m": pick(m, "trajectory_mean_m", "trajectory_error_mean_m", "mean_distance_error_m"),
                "trajectory_median_m": pick(m, "trajectory_median_m", "trajectory_error_median_m", "median_distance_error_m"),
                "trajectory_p95_m": pick(m, "trajectory_p95_m", "trajectory_error_p95_m", "p95_distance_error_m"),
                "trajectory_worst_m": pick(m, "trajectory_worst_m", "trajectory_error_worst_m", "worst_distance_error_m"),
                "speed_mae_mps": pick(m, "speed_mae_mps", "speed_rmse_mps"),
                "yaw_rate_mae_radps": pick(m, "yaw_rate_mae_radps", "yaw_rate_rmse_radps"),
                "yaw_mae_deg": pick(m, "yaw_mae_deg", "yaw_rmse_deg"),
            })

    with (OUTPUT_DIR / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    metrics = (("trajectory_mean_m", "1 s trajectory mean [m]"),
               ("speed_mae_mps", "speed MAE [m/s]"),
               ("yaw_rate_mae_radps", "yaw-rate MAE [rad/s]"),
               ("yaw_mae_deg", "yaw MAE [deg]"))
    labels = list(MODELS)
    x = np.arange(len(labels)); width = .36
    for ax, (key, title) in zip(axes.ravel(), metrics):
        for j, dataset in enumerate(DATASETS):
            vals = [next(r[key] for r in rows if r["dataset"] == dataset and r["model"] == label)
                    for label in labels]
            ax.bar(x + (j-.5)*width, vals, width, label=dataset)
        ax.set_title(title); ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "five_model_quantitative_comparison.png", dpi=180)
    print(OUTPUT_DIR / "five_model_quantitative_comparison.png")
    print(OUTPUT_DIR / "metrics.csv")


if __name__ == "__main__":
    main()
