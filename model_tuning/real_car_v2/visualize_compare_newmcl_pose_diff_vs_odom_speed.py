#!/usr/bin/env python3
"""Visualize /newmcl_pose-derived speed against the published /odom vx."""
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from step_1_extract_data import read_streams

DEFAULT_BAG = Path(
    "/mnt/nas_custom/F1tenth/2026 IFAC/0820/0820/"
    "rosbag2_2026_08_20-04_23_48/rosbag2_2026_08_20-04_23_48_0.db3")
DEFAULT_OUTPUT = HERE.parent / "results/newmcl_pose_speed_comparison.png"


def pose_difference_velocity(pose, half_window):
    """Centered map-position difference, projected into the vehicle x axis."""
    dt = pose[2 * half_window:, 0] - pose[:-2 * half_window, 0]
    time = 0.5 * (pose[2 * half_window:, 0] + pose[:-2 * half_window, 0])
    dx = pose[2 * half_window:, 1] - pose[:-2 * half_window, 1]
    dy = pose[2 * half_window:, 2] - pose[:-2 * half_window, 2]
    yaw = pose[half_window:-half_window, 3]
    vx = (dx * np.cos(yaw) + dy * np.sin(yaw)) / dt
    return time, vx, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pose, odom, _, _ = read_streams(
        args.bag, "/newmcl_pose", "/odom", "/ackermann_cmd", "/imu/data")
    short_t, short_v, short_dt = pose_difference_velocity(pose, 2)
    smooth_t, smooth_v, smooth_dt = pose_difference_velocity(pose, 6)
    odom_short = np.interp(short_t, odom[:, 0], odom[:, 1])
    odom_smooth = np.interp(smooth_t, odom[:, 0], odom[:, 1])
    valid = ((odom_smooth > 0.5) & np.isfinite(smooth_v)
             & (smooth_dt > 0.05) & (smooth_dt < 0.6))
    error = smooth_v[valid] - odom_smooth[valid]
    ratio = smooth_v[valid] / odom_smooth[valid]

    t0 = max(pose[0, 0], odom[0, 0])
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(odom[:, 0] - t0, odom[:, 1], color="black", lw=1.5,
                 label="/odom published vx")
    axes[0].plot(short_t - t0, short_v, color="tab:orange", alpha=.42, lw=.8,
                 label=f"/newmcl_pose x,y diff ({np.median(short_dt):.2f} s)")
    axes[0].plot(smooth_t - t0, smooth_v, color="tab:blue", lw=1.2,
                 label=f"/newmcl_pose x,y diff ({np.median(smooth_dt):.2f} s)")
    axes[0].set_ylabel("longitudinal speed [m/s]")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=.25)

    axes[1].plot(smooth_t - t0, smooth_v - odom_smooth, color="tab:red", lw=1)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_ylabel("MCL diff - odom [m/s]")
    axes[1].grid(alpha=.25)

    axes[2].scatter(odom_smooth[valid], smooth_v[valid], s=5, alpha=.25)
    limit = max(float(np.quantile(odom_smooth[valid], .995)), .5)
    axes[2].plot([0, limit], [0, limit], "k--", lw=1, label="1:1")
    axes[2].set_xlim(0, limit); axes[2].set_ylim(0, limit)
    axes[2].set_xlabel("/odom published vx [m/s]")
    axes[2].set_ylabel("MCL pose-difference vx [m/s]")
    axes[2].grid(alpha=.25); axes[2].legend()

    fig.suptitle(
        f"{args.bag.parent.name}\n"
        f"moving samples={valid.sum()}, bias={error.mean():+.3f} m/s, "
        f"MAE={np.abs(error).mean():.3f} m/s, P95={np.quantile(np.abs(error), .95):.3f} m/s, "
        f"median ratio={np.median(ratio):.3f}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"saved: {args.output}")
    print({"moving_samples": int(valid.sum()), "bias_mps": float(error.mean()),
           "mae_mps": float(np.abs(error).mean()),
           "p95_abs_error_mps": float(np.quantile(np.abs(error), .95)),
           "correlation": float(np.corrcoef(smooth_v[valid], odom_smooth[valid])[0, 1]),
           "median_mcl_to_odom_ratio": float(np.median(ratio))})


if __name__ == "__main__":
    main()
