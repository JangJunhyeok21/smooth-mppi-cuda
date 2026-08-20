#!/usr/bin/env python3
"""Visualize deployed accel=2 versus retrained accel=4 on aggressive holdout."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "model_tuning/results/dynamic_0817_0820_accel4_from_deployed_seed31"
BASE = RESULT / "no_manual_baseline_accel2.npz"
NEW = RESULT / "no_manual_candidate_accel4.npz"
OUT = RESULT / "no_manual_aggressive_best_p95_worst_comparison.png"
REPORT = OUT.with_suffix(".json")
DT = 0.04


def endpoint_errors(prediction, truth):
    return {
        "trajectory_m": np.linalg.norm(prediction[:, -1, :2] - truth[:, -1, :2], axis=1),
        "vx_mps": np.abs(prediction[:, -1, 3] - truth[:, -1, 3]),
        "vy_mps": np.abs(prediction[:, -1, 4] - truth[:, -1, 4]),
        "yaw_rate_rps": np.abs(prediction[:, -1, 5] - truth[:, -1, 5]),
    }


def main():
    baseline = np.load(BASE)
    candidate = np.load(NEW)
    if not np.array_equal(baseline["starts"], candidate["starts"]):
        raise RuntimeError("evaluation windows are not identical")
    truth = candidate["ground_truth"]
    old = baseline["predicted"]
    new = candidate["predicted"]
    old_error = endpoint_errors(old, truth)
    new_error = endpoint_errors(new, truth)
    trajectory_order = np.argsort(new_error["trajectory_m"])
    cases = (
        ("Best trajectory", int(trajectory_order[0])),
        ("Trajectory P95", int(trajectory_order[min(len(trajectory_order)-1, round(.95*(len(trajectory_order)-1)))])),
        ("Worst trajectory", int(trajectory_order[-1])),
        ("Worst $v_y$", int(np.argmax(new_error["vy_mps"]))),
        ("Worst yaw-rate", int(np.argmax(new_error["yaw_rate_rps"]))),
    )
    time = np.arange(old.shape[1]) * DT
    fig, axes = plt.subplots(5, len(cases), figsize=(22, 18), constrained_layout=True)
    colors = {"gt": "black", "old": "tab:blue", "new": "tab:orange"}
    report = {"split": "test_aggressive", "horizon_s": float(time[-1]), "cases": []}
    for column, (title, index) in enumerate(cases):
        axes[0, column].plot(truth[index, :, 0], truth[index, :, 1], color=colors["gt"], lw=2.4, label="GT")
        axes[0, column].plot(old[index, :, 0], old[index, :, 1], "--", color=colors["old"], lw=2, label="deployed ±2")
        axes[0, column].plot(new[index, :, 0], new[index, :, 1], "-.", color=colors["new"], lw=2, label="retrained ±4")
        axes[0, column].axis("equal")
        axes[0, column].set_title(f"{title}\nwindow={int(candidate['starts'][index])}")
        axes[0, column].set_xlabel("relative x [m]")
        axes[0, column].set_ylabel("relative y [m]")
        for row, state_index, label, unit in ((1, 3, "$v_x$", "m/s"), (2, 4, "$v_y$", "m/s"), (3, 5, "yaw-rate", "rad/s")):
            axes[row, column].plot(time, truth[index, :, state_index], color=colors["gt"], lw=2.2, label="GT")
            axes[row, column].plot(time, old[index, :, state_index], "--", color=colors["old"], lw=1.8, label="deployed ±2")
            axes[row, column].plot(time, new[index, :, state_index], "-.", color=colors["new"], lw=1.8, label="retrained ±4")
            axes[row, column].set_ylabel(f"{label} [{unit}]")
            axes[row, column].set_xlabel("time [s]")
            axes[row, column].grid(alpha=.25)
        old_position = np.linalg.norm(old[index, :, :2] - truth[index, :, :2], axis=1)
        new_position = np.linalg.norm(new[index, :, :2] - truth[index, :, :2], axis=1)
        axes[4, column].plot(time, old_position, "--", color=colors["old"], lw=2, label="deployed ±2")
        axes[4, column].plot(time, new_position, "-.", color=colors["new"], lw=2, label="retrained ±4")
        axes[4, column].axhline(.10, color="tab:red", ls=":", lw=1.5, label="10 cm goal")
        axes[4, column].set_ylabel("trajectory error [m]")
        axes[4, column].set_xlabel("time [s]")
        axes[4, column].grid(alpha=.25)
        report["cases"].append({
            "name": title,
            "window_start": int(candidate["starts"][index]),
            "deployed": {name: float(value[index]) for name, value in old_error.items()},
            "retrained": {name: float(value[index]) for name, value in new_error.items()},
        })
    axes[0, 0].legend(loc="best", fontsize=8)
    axes[4, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Aggressive holdout · deployed accel ±2 vs retrained accel ±4", fontsize=17)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"plot": str(OUT), "report": str(REPORT), **report}, indent=2))


if __name__ == "__main__":
    main()
