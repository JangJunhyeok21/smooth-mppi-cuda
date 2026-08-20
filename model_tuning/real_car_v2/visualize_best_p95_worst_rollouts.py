#!/usr/bin/env python3
"""F5로 최신 모델의 best/P95/worst trajectory와 state를 시각화한다."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "model_tuning/results/compare_0817_0820_inertial_ekf_bias"
STATE_NAMES = ("x", "y", "yaw", "vx", "vy", "yaw-rate")
STATE_UNITS = ("m", "m", "rad", "m/s", "m/s", "rad/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    new = np.load(result_dir / "new_lateral_only.npz")
    old = np.load(result_dir / "deployed_lateral_only.npz")
    if not np.array_equal(new["starts"], old["starts"]):
        raise RuntimeError("새 모델과 배포 모델의 rollout window 순서가 다릅니다.")

    gt = new["ground_truth"]
    pred = new["predicted"]
    old_pred = old["predicted"]
    point_position_error = np.linalg.norm(pred[:, :, :2] - gt[:, :, :2], axis=2)
    window_error = point_position_error.mean(axis=1)
    p95_value = float(np.percentile(window_error, 95))
    cases = {
        "best": int(np.argmin(window_error)),
        "p95": int(np.argmin(np.abs(window_error - p95_value))),
        "worst": int(np.argmax(window_error)),
    }
    colors = {"best": "tab:green", "p95": "tab:orange", "worst": "tab:red"}
    time = np.arange(gt.shape[1]) * args.dt

    fig_traj, traj_axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig_state, state_axes = plt.subplots(3, 6, figsize=(22, 10), sharex=True)
    report = {"selection_metric": "mean XY error over 30-step rollout", "dt_s": args.dt}

    for row, (case, index) in enumerate(cases.items()):
        g, n, o = gt[index], pred[index], old_pred[index]
        new_xy = np.linalg.norm(n[:, :2] - g[:, :2], axis=1)
        old_xy = np.linalg.norm(o[:, :2] - g[:, :2], axis=1)
        state_mae = np.mean(np.abs(n - g), axis=0)
        state_mae[2] = np.mean(np.abs(angle_error(n[:, 2], g[:, 2])))
        report[case] = {
            "window_index": index,
            "dataset_start_index": int(new["starts"][index]),
            "new_trajectory_mean_m": float(new_xy.mean()),
            "new_trajectory_final_m": float(new_xy[-1]),
            "new_trajectory_max_m": float(new_xy.max()),
            "deployed_trajectory_mean_m": float(old_xy.mean()),
            "state_mae": {name: float(value) for name, value in zip(STATE_NAMES, state_mae)},
        }

        ax = traj_axes[row]
        ax.plot(g[:, 0], g[:, 1], "k-o", ms=2.5, label="GT")
        ax.plot(o[:, 0], o[:, 1], "--", color="tab:blue", label="deployed MLP")
        ax.plot(n[:, 0], n[:, 1], color=colors[case], label="new EKF+MLP")
        ax.scatter(n[-1, 0], n[-1, 1], color=colors[case], s=35, zorder=5)
        ax.set_title(
            f"{case.upper()} | mean={new_xy.mean():.3f} m | "
            f"final={new_xy[-1]:.3f} m"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend()

        for col, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
            sax = state_axes[row, col]
            sax.plot(time, g[:, col], "k", linewidth=1.8, label="GT")
            sax.plot(time, o[:, col], "--", color="tab:blue", linewidth=1.2,
                     label="deployed MLP")
            sax.plot(time, n[:, col], color=colors[case], linewidth=1.4,
                     label="new EKF+MLP")
            sax.set_title(f"{case.upper()} {name} | MAE={state_mae[col]:.3f} {unit}")
            sax.set_ylabel(f"{name} [{unit}]")
            sax.grid(alpha=0.25)
            if row == 2:
                sax.set_xlabel("time [s]")
            if row == 0 and col == 0:
                sax.legend(fontsize=8)

    fig_traj.suptitle("Latest model: best / P95 / worst trajectory cases")
    fig_traj.tight_layout()
    trajectory_path = result_dir / "best_p95_worst_trajectories.png"
    fig_traj.savefig(trajectory_path, dpi=180)

    fig_state.suptitle("Latest model: best / P95 / worst state rollouts", fontsize=15)
    fig_state.tight_layout()
    state_path = result_dir / "best_p95_worst_states.png"
    fig_state.savefig(state_path, dpi=160)

    report_path = result_dir / "best_p95_worst_cases.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(trajectory_path)
    print(state_path)
    print(report_path)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
