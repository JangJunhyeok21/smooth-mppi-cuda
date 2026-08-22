#!/usr/bin/env python3
"""Visualize alternating-identification convergence and representative rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STATE_NAMES = ("x", "y", "yaw", "vx", "vy", "yaw_rate")
PARAM_NAMES = (
    "speed_kp", "speed_accel_tau", "speed_brake_tau",
    "steer_scale", "steer_bias", "steer_tau", "Iz",
    "B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r",
)


def _load_iterations(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("iteration_*/iteration.json")):
        with path.open() as stream:
            record = json.load(stream)
        record["_directory"] = str(path.parent)
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No iteration records under {root}")
    return records


def _plot_parameter_history(records: list[dict], output: Path) -> None:
    iterations = np.asarray([record["iteration"] for record in records])
    fig, axes = plt.subplots(5, 3, figsize=(15, 16), constrained_layout=True)
    for ax, name in zip(axes.flat, PARAM_NAMES):
        values = np.asarray([record["classic_parameters"][name] for record in records])
        ax.plot(iterations, values, "o-", linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel("alternating iteration")
        ax.grid(alpha=0.3)
        for x, value in zip(iterations, values):
            ax.annotate(f"{value:.4g}", (x, value), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Classic parameter history (0817-0820)", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_parameter_csv(records: list[dict], output: Path) -> None:
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("iteration", "validation_score", "parameter_movement", *PARAM_NAMES),
        )
        writer.writeheader()
        for record in records:
            params = record["classic_parameters"]
            writer.writerow({
                "iteration": record["iteration"],
                "validation_score": record["validation_score"],
                "parameter_movement": record["classic_parameter_movement"],
                **{name: params[name] for name in PARAM_NAMES},
            })


def _plot_cases(npz_path: Path, output: Path, metadata_path: Path) -> None:
    data = np.load(npz_path)
    predicted = data["predicted"]
    truth = data["ground_truth"]
    starts = data["starts"]
    final_xy_error = np.linalg.norm(predicted[:, -1, :2] - truth[:, -1, :2], axis=1)
    order = np.argsort(final_xy_error)
    selected = (order[0], order[len(order) // 2], order[-1])
    labels = ("best", "median", "worst")
    dt = 1.2 / max(predicted.shape[1] - 1, 1)
    time = np.arange(predicted.shape[1]) * dt

    fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
    metadata = {"selection_metric": "final 1.2 s XY error [m]", "cases": {}}
    for column, (label, index) in enumerate(zip(labels, selected)):
        ax = axes[0, column]
        ax.plot(truth[index, :, 0], truth[index, :, 1], "k-", label="KF GT")
        ax.plot(predicted[index, :, 0], predicted[index, :, 1], "C1--", label="model")
        ax.scatter(truth[index, 0, 0], truth[index, 0, 1], marker="o", color="C2", label="start")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{label}: final XY error={final_xy_error[index]:.3f} m")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.3); ax.legend()

        ax = axes[1, column]
        for state, color in ((2, "C0"), (3, "C2"), (4, "C3")):
            ax.plot(time, truth[index, :, state], color=color, label=f"GT {STATE_NAMES[state]}")
            ax.plot(time, predicted[index, :, state], "--", color=color,
                    label=f"pred {STATE_NAMES[state]}")
        ax.set_xlabel("horizon [s]"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)

        ax = axes[2, column]
        ax.plot(time, truth[index, :, 5], "k-", label="GT yaw_rate")
        ax.plot(time, predicted[index, :, 5], "C4--", label="pred yaw_rate")
        ax.set_xlabel("horizon [s]"); ax.set_ylabel("rad/s"); ax.grid(alpha=0.3); ax.legend()
        metadata["cases"][label] = {
            "array_index": int(index), "dataset_start_index": int(starts[index]),
            "final_xy_error_m": float(final_xy_error[index]),
        }
    fig.suptitle(f"Representative test rollouts: {npz_path.parent.name}", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("model_tuning/results/alternating"))
    args = parser.parse_args()
    root = args.root.resolve()
    records = _load_iterations(root)
    best = min(records, key=lambda record: record["validation_score"])
    best_dir = Path(best["_directory"])
    _plot_parameter_history(records, root / "parameter_convergence.png")
    _write_parameter_csv(records, root / "parameter_history.csv")
    _plot_cases(best_dir / "evaluation.npz", root / "best_median_worst.png",
                root / "best_median_worst.json")
    print(f"best iteration: {best['iteration']} (validation score={best['validation_score']:.6f})")
    print(root / "parameter_convergence.png")
    print(root / "best_median_worst.png")


if __name__ == "__main__":
    main()
