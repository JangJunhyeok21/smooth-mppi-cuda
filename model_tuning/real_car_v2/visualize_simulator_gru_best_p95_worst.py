#!/usr/bin/env python3
"""F5로 simulator GRU E2E의 best/P95/worst 1.2 s rollout을 시각화한다."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT = ROOT / "model_tuning/results/simulator_gru_0817_0820_seed31"
DT = 0.02


def wrapped_abs(value: np.ndarray) -> np.ndarray:
    return np.abs((value + np.pi) % (2.0 * np.pi) - np.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--show", action="store_true", help="저장 후 그래프 창도 표시")
    args = parser.parse_args()

    replay_path = args.result / "rollout_60step_metrics.npz"
    replay = np.load(replay_path)
    predicted = replay["predicted"]
    truth = replay["ground_truth"]
    starts = replay["starts"]
    if predicted.shape != truth.shape or predicted.shape[2] != 6:
        raise ValueError(f"Unexpected replay shapes: {predicted.shape}, {truth.shape}")

    endpoint_error = np.linalg.norm(predicted[:, -1, :2] - truth[:, -1, :2], axis=1)
    p95_value = float(np.quantile(endpoint_error, 0.95))
    selected = {
        "Best": int(np.argmin(endpoint_error)),
        "P95": int(np.argmin(np.abs(endpoint_error - p95_value))),
        "Worst": int(np.argmax(endpoint_error)),
    }
    colors = {"Best": "tab:green", "P95": "tab:orange", "Worst": "tab:red"}
    time = np.arange(predicted.shape[1]) * DT

    fig, axes = plt.subplots(4, 3, figsize=(16, 14), constrained_layout=True)
    report = {
        "selection_set": "held-out aggressive test",
        "horizon_s": float(time[-1]),
        "windows": int(len(endpoint_error)),
        "trajectory_endpoint_distribution_m": {
            "mean": float(endpoint_error.mean()),
            "p95": p95_value,
            "worst": float(endpoint_error.max()),
        },
        "cases": {},
    }

    for column, (label, index) in enumerate(selected.items()):
        prediction = predicted[index]
        target = truth[index]
        color = colors[label]

        trajectory_axis = axes[0, column]
        trajectory_axis.plot(target[:, 0], target[:, 1], "k-", lw=2.2, label="GT")
        trajectory_axis.plot(prediction[:, 0], prediction[:, 1], "--", color=color,
                             lw=2.2, label="GRU E2E")
        trajectory_axis.scatter(target[0, 0], target[0, 1], color="tab:blue", s=35,
                                zorder=4, label="start")
        trajectory_axis.scatter(target[-1, 0], target[-1, 1], color="black", s=35,
                                zorder=4, label="GT end")
        trajectory_axis.scatter(prediction[-1, 0], prediction[-1, 1], color=color, s=45,
                                zorder=4, label="prediction end")
        trajectory_axis.set_aspect("equal", adjustable="datalim")
        trajectory_axis.set(title=f"{label}: endpoint error {endpoint_error[index]:.3f} m",
                            xlabel="relative x [m]", ylabel="relative y [m]")
        trajectory_axis.grid(alpha=0.3)
        trajectory_axis.legend(fontsize=8)

        for row, state_index, state_label, unit in (
            (1, 3, "$v_x$", "m/s"),
            (2, 4, "$v_y$", "m/s"),
            (3, 5, "yaw rate", "rad/s"),
        ):
            axis = axes[row, column]
            axis.plot(time, target[:, state_index], "k-", lw=1.8, label="GT")
            axis.plot(time, prediction[:, state_index], "--", color=color, lw=1.8,
                      label="GRU E2E")
            axis.set(xlabel="rollout time [s]", ylabel=f"{state_label} [{unit}]")
            axis.grid(alpha=0.3)
            if column == 0:
                axis.legend(fontsize=8)

        state_error = np.abs(prediction[:, 3:6] - target[:, 3:6])
        report["cases"][label.lower()] = {
            "replay_index": index,
            "source_row": int(starts[index]),
            "trajectory_endpoint_error_m": float(endpoint_error[index]),
            "yaw_endpoint_error_rad": float(wrapped_abs(prediction[-1, 2] - target[-1, 2])),
            "state_endpoint_abs_error": {
                "vx_mps": float(state_error[-1, 0]),
                "vy_mps": float(state_error[-1, 1]),
                "yaw_rate_rps": float(state_error[-1, 2]),
            },
            "state_rollout_mae": {
                "vx_mps": float(state_error[:, 0].mean()),
                "vy_mps": float(state_error[:, 1].mean()),
                "yaw_rate_rps": float(state_error[:, 2].mean()),
            },
        }

    fig.suptitle("Simulator GRU E2E · aggressive held-out recursive rollout (1.2 s)", fontsize=16)
    output_png = args.result / "simulator_gru_best_p95_worst.png"
    output_json = args.result / "simulator_gru_best_p95_worst.json"
    fig.savefig(output_png, dpi=180)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(output_png)
    print(output_json)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
