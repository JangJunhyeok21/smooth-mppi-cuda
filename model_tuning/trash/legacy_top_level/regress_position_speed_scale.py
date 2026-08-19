#!/usr/bin/env python3
"""Regress odometry-speed -> MCL-displacement scale using train data only."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "model_tuning/data/ifac0807_0808_hardcase_train_test.npz"
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/results/position_speed_scale_regression"
HORIZON_S = 1.0
MAX_ABS_STEER_CMD = 0.08
MAX_ABS_YAW_RATE = 0.30
MAX_POSE_STEP_M = 0.25
MIN_WINDOWS = 100
IMU_WZ_SIGN = -1.0
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")


def main():
    z = np.load(DATASET_PATH)
    samples = z["samples"].astype(np.float64)
    columns = {str(name): i for i, name in enumerate(z["columns"])}
    dt = float(z["dt"])
    horizon = max(1, round(HORIZON_S / dt))
    bag = samples[:, columns["bag_id"]].astype(int)
    split = samples[:, columns["split"]].astype(int)
    yaw_rate = IMU_WZ_SIGN * samples[:, columns["imu_wz"]]

    predicted, measured, start_rows = [], [], []
    for start in range(len(samples) - horizon):
        stop = start + horizon
        if split[start] != 0 or split[stop] != 0 or bag[start] != bag[stop]:
            continue
        block = samples[start:stop]
        if np.any(np.hypot(np.diff(samples[start:stop+1, columns["x"]]),
                           np.diff(samples[start:stop+1, columns["y"]])) > MAX_POSE_STEP_M):
            continue
        yaw = block[:, columns["yaw"]]
        vx = block[:, columns["vx"]]
        vy = block[:, columns["vy"]]
        displacement = np.array([
            np.sum((vx*np.cos(yaw) - vy*np.sin(yaw))*dt),
            np.sum((vx*np.sin(yaw) + vy*np.cos(yaw))*dt),
        ])
        ground_truth = samples[stop, [columns["x"], columns["y"]]] - samples[start, [columns["x"], columns["y"]]]
        if not np.all(np.isfinite(np.r_[displacement, ground_truth])) or np.linalg.norm(displacement) < 1e-3:
            continue
        predicted.append(displacement); measured.append(ground_truth); start_rows.append(start)

    predicted = np.asarray(predicted); measured = np.asarray(measured); start_rows = np.asarray(start_rows)
    straight = np.array([
        np.max(np.abs(samples[i:i+horizon, columns["steer"]])) <= MAX_ABS_STEER_CMD and
        np.max(np.abs(yaw_rate[i:i+horizon])) <= MAX_ABS_YAW_RATE
        for i in start_rows
    ])
    used_straight_filter = int(straight.sum()) >= MIN_WINDOWS
    selected = straight if used_straight_filter else np.ones(len(predicted), dtype=bool)
    p, g = predicted[selected], measured[selected]
    if len(p) < MIN_WINDOWS:
        raise SystemExit(f"only {len(p)} valid train windows; need at least {MIN_WINDOWS}")

    residual = lambda value: (value[0]*p-g).reshape(-1)
    initial = np.array([np.sum(p*g)/np.sum(p*p)])
    result = least_squares(residual, initial, bounds=(0.5, 1.5), loss="soft_l1", f_scale=.05)
    scale = float(result.x[0])
    before = np.linalg.norm(p-g, axis=1)
    after = np.linalg.norm(scale*p-g, axis=1)
    report = {
        "dataset": str(DATASET_PATH), "split_used": "train only (split == 0)",
        "method": "robust least_squares soft_l1", "horizon_s": HORIZON_S,
        "position_speed_scale": scale, "windows": int(len(p)),
        "straight_filter_used": used_straight_filter,
        "straight_filter": {"max_abs_steer_cmd_rad": MAX_ABS_STEER_CMD,
                            "max_abs_yaw_rate_radps": MAX_ABS_YAW_RATE},
        "unscaled_error_mean_m": float(before.mean()),
        "scaled_error_mean_m": float(after.mean()),
        "scaled_error_p95_m": float(np.quantile(after, .95)),
    }
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "position_speed_scale.json").write_text(json.dumps(report, indent=2)+"\n")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    norm_p = np.linalg.norm(p, axis=1); norm_g = np.linalg.norm(g, axis=1)
    axes[0].scatter(norm_p, norm_g, s=7, alpha=.25)
    xx=np.linspace(0,max(norm_p.max(),norm_g.max()),100)
    axes[0].plot(xx,xx,"k--",label="scale=1");axes[0].plot(xx,scale*xx,"r-",label=f"fit={scale:.4f}")
    axes[0].set(xlabel="integrated odom displacement [m]",ylabel="MCL displacement [m]",title="Train-window regression")
    axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].hist(before,bins=50,alpha=.55,label="unscaled");axes[1].hist(after,bins=50,alpha=.55,label="scaled")
    axes[1].set(xlabel="vector displacement error [m]",title=f"{HORIZON_S:.1f} s error");axes[1].legend();axes[1].grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUTPUT_PATH/"position_speed_scale_regression.png",dpi=180);plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
