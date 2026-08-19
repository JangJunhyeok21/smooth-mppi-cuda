#!/usr/bin/env python3
"""Quantify deployed-model response mismatch on the clean pre-impact interval."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "model_tuning/data/ifac0808_221038_mppi_observation.npz"
REPLAY = ROOT / "model_tuning/results/failed_bag_replay/old_kf_221038_h06/predictions.npz"
OUTPUT = ROOT / "model_tuning/results/failed_bag_collision_analysis"
PREDICTION_KEY = "fixed_preserve_measured_overspeed"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")


def main():
    data = np.load(DATASET)
    replay = np.load(REPLAY)
    raw = data["samples"]
    dt = float(data["dt"])
    gt_pose = replay["gt_pose"]
    prediction = replay[PREDICTION_KEY]

    # Locate each replay initial state in the extracted bag.  This avoids
    # depending on the evaluator's history-offset option.
    initial_indices = np.array([
        np.argmin(np.linalg.norm(raw[:, 1:3] - pose[0, :2], axis=1))
        for pose in gt_pose
    ])
    horizon = prediction.shape[1]
    gt_state = np.stack([
        raw[index:index + horizon, [1, 2, 3, 4, 5, 6]]
        for index in initial_indices
    ])
    # MPPI/body convention: the 0807/0808 IMU is FRD, therefore wz is negated
    # at the subscriber boundary and causally EMA-filtered (alpha=0.25).
    imu_wz = -raw[:, 12].astype(float)
    imu_wz_filtered = imu_wz.copy()
    for index in range(1, len(imu_wz_filtered)):
        imu_wz_filtered[index] = .25 * imu_wz[index] + .75 * imu_wz_filtered[index - 1]
    for window, index in enumerate(initial_indices):
        gt_state[window, :, 5] = imu_wz_filtered[index:index + horizon]

    position_error = np.linalg.norm(prediction[:, :, :2] - gt_state[:, :, :2], axis=2)
    pred_speed = np.hypot(prediction[:, :, 3], prediction[:, :, 4])
    gt_speed = np.hypot(gt_state[:, :, 3], gt_state[:, :, 4])
    speed_error = np.abs(pred_speed - gt_speed)
    yaw_rate_error = np.abs(prediction[:, :, 5] - gt_state[:, :, 5])
    yaw_error = np.abs(np.arctan2(
        np.sin(prediction[:, :, 2] - gt_state[:, :, 2]),
        np.cos(prediction[:, :, 2] - gt_state[:, :, 2])))

    final = -1
    worst = int(np.argmax(position_error[:, final]))
    metrics = {
        "model": "deployed 0807 slip-kinematic residual MLP with original KF",
        "bag_interval": "clean pre-impact portion of 22:10:38 bag",
        "yaw_rate_gt": "-imu_wz with causal EMA alpha=0.25",
        "windows": int(len(prediction)),
        "horizon_s": float((horizon - 1) * dt),
        "endpoint": {
            "trajectory_mean_m": float(position_error[:, final].mean()),
            "trajectory_median_m": float(np.median(position_error[:, final])),
            "trajectory_p95_m": float(np.quantile(position_error[:, final], .95)),
            "trajectory_worst_m": float(position_error[:, final].max()),
            "speed_mae_mps": float(speed_error[:, final].mean()),
            "yaw_rate_mae_radps": float(yaw_rate_error[:, final].mean()),
            "yaw_mae_deg": float(np.degrees(yaw_error[:, final].mean())),
        },
        "worst_window": {
            "start_from_extracted_bag_s": float(raw[initial_indices[worst], 0]),
            "trajectory_error_m": float(position_error[worst, final]),
            "speed_error_mps": float(speed_error[worst, final]),
            "yaw_rate_error_radps": float(yaw_rate_error[worst, final]),
            "yaw_error_deg": float(np.degrees(yaw_error[worst, final])),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "model_response_mismatch.json").write_text(json.dumps(metrics, indent=2) + "\n")

    import matplotlib.pyplot as plt
    time = np.arange(horizon) * dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(gt_state[worst, :, 0], gt_state[worst, :, 1], lw=2.5, label="GT from MCL")
    axes[0, 0].plot(prediction[worst, :, 0], prediction[worst, :, 1], "--", lw=2.5,
                    label="model replay of /drive")
    axes[0, 0].axis("equal"); axes[0, 0].set_title("Worst pre-impact 0.6 s trajectory")
    axes[0, 0].legend(); axes[0, 0].grid(alpha=.25)

    axes[0, 1].plot(time, gt_speed[worst], lw=2, label="GT speed")
    axes[0, 1].plot(time, pred_speed[worst], "--", lw=2, label="predicted speed")
    axes[0, 1].set(title="Worst-window speed response", xlabel="rollout time [s]", ylabel="m/s")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=.25)

    axes[1, 0].plot(time, gt_state[worst, :, 5], lw=2, label="GT yaw rate")
    axes[1, 0].plot(time, prediction[worst, :, 5], "--", lw=2, label="predicted yaw rate")
    axes[1, 0].set(title="Worst-window yaw-rate response", xlabel="rollout time [s]", ylabel="rad/s")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=.25)

    axes[1, 1].plot(time, position_error.mean(0), label="position mean")
    axes[1, 1].plot(time, np.quantile(position_error, .95, axis=0), label="position p95")
    axes[1, 1].set(title=f"All {len(prediction)} pre-impact replay windows",
                   xlabel="rollout time [s]", ylabel="position error [m]")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=.25)
    fig.suptitle("Recorded /drive response: deployed model prediction vs actual vehicle")
    fig.tight_layout()
    fig.savefig(OUTPUT / "model_response_mismatch.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
