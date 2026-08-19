#!/usr/bin/env python3
"""Compare the 0807 slip-MLP vx/yaw loss-weight experiments."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smppi-cache")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "Baseline yaw2": ROOT / "model_tuning/results/ifac0807_slip_with_imu_kf_regressed",
    "Yaw8": ROOT / "model_tuning/results/ifac0807_slip_kf_yaw8_lossckpt",
    "Vx6 + yaw4": ROOT / "model_tuning/results/ifac0807_slip_kf_targeted_vx6_yaw4",
    "Vx6 + yaw8": ROOT / "model_tuning/results/ifac0807_slip_kf_pareto_vx6_yaw8",
    "No vx residual + yaw8": ROOT / "model_tuning/results/ifac0807_slip_kf_no_vx_residual_yaw8",
}
OUTPUT = ROOT / "model_tuning/results/ifac0807_vx_yaw_weight_comparison"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    curves = {}
    for label, directory in EXPERIMENTS.items():
        metadata = json.loads((directory / "metrics.json").read_text())
        data = np.load(directory / "test_predictions.npz")
        prediction, gt_pose, gt_state = (
            data["prediction"], data["gt_pose"], data["gt_state"])
        trajectory = data["position_error"]
        vx = np.abs(prediction[:, :, 3] - gt_state[:, :, 0])
        yaw_rate = np.abs(prediction[:, :, 5] - gt_state[:, :, 2])
        yaw = np.abs(np.arctan2(np.sin(prediction[:, :, 2] - gt_pose[:, :, 2]),
                                np.cos(prediction[:, :, 2] - gt_pose[:, :, 2])))
        curves[label] = (trajectory, vx, yaw_rate, yaw, float(metadata["dt"]))
        summary[label] = {
            "trajectory_mean_m": float(trajectory[:, -1].mean()),
            "trajectory_p95_m": float(np.quantile(trajectory[:, -1], .95)),
            "vx_mae_mps": float(vx[:, -1].mean()),
            "yaw_rate_mae_radps": float(yaw_rate[:, -1].mean()),
            "yaw_angle_mae_rad": float(yaw[:, -1].mean()),
            "speed_mae_mps": float(metadata["final_speed_mae_mps"]),
            "loss_weights": metadata["loss_weights"],
            "velocity_residual_enabled": metadata["velocity_residual_enabled"],
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = ((0, "Trajectory error", "m"), (1, "Vx error", "m/s"),
             (2, "Yaw-rate error", "rad/s"), (3, "Yaw-angle error", "rad"))
    for axis, (index, title, unit) in zip(axes.flat, specs):
        for label, values in curves.items():
            value, dt = values[index], values[-1]
            time = np.arange(value.shape[1]) * dt
            axis.plot(time, value.mean(0), lw=1.8, label=label)
        axis.set(title=title, xlabel="open-loop time [s]",
                 ylabel=f"mean absolute error [{unit}]")
        axis.grid(alpha=.25); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUTPUT / "error_over_horizon_all_experiments.png", dpi=180)
    plt.close(fig)

    labels = list(summary)
    keys = ("trajectory_mean_m", "vx_mae_mps", "yaw_rate_mae_radps",
            "yaw_angle_mae_rad")
    titles = ("Trajectory mean [m]", "Vx MAE [m/s]",
              "Yaw-rate MAE [rad/s]", "Yaw-angle MAE [rad]")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, key, title in zip(axes.flat, keys, titles):
        values = [summary[label][key] for label in labels]
        bars = axis.bar(np.arange(len(labels)), values)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
        axis.set_title(title); axis.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT / "final_error_bars_all_experiments.png", dpi=180)
    plt.close(fig)
    (OUTPUT / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
