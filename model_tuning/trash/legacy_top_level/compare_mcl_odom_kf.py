#!/usr/bin/env python3
"""Compare MCL pose-derived body velocity with odom and the runtime IMU/KF.

MCL is treated as offline GT, so its derivative uses a centered Savitzky-Golay
window (no phase-delay-induced fake lateral velocity).  The online KF, steering
conversion, IMU signs, EMA and causal stream alignment match the real-car SMPPI
observation path configured in config/params.yaml.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_tuning.extract_training_data import causal_hold, read_streams
from model_tuning_utils.filter_collision_recovery_episodes import collision_recovery_mask
from model_tuning_utils.lateral_velocity_kf import (
    LateralVelocityKFParams, estimate_dataset)

# USER SETTINGS
ROOT=Path(__file__).resolve().parents[1]
BAG_0807=Path('/mnt/nas_custom/F1tenth/2026 IFAC/0807/rosbag2_2026_08_07-19_13_58/rosbag2_2026_08_07-19_13_58_0.db3')
COLLISION_BAG=Path('/home/a/Downloads/rosbag2_2026_08_08-22_1/rosbag2_2026_08_08-22_10_38/rosbag2_2026_08_08-22_10_38_0.db3')
OUTPUT_PATH=ROOT/'model_tuning/results/observation_source_comparison';PARAMS_PATH=ROOT/'config/params.yaml'
POSE_TOPIC='/newmcl_pose';VELOCITY_TOPIC='/odom';DRIVE_TOPIC_0807='/ackermann_cmd';DRIVE_TOPIC_COLLISION='/drive';IMU_TOPIC='/imu/data'
DT=.02;POSE_VELOCITY_WINDOW=.20;MAX_POSE_AGE=1.;MAX_VELOCITY_AGE=1.;MAX_COMMAND_AGE=1.;MAX_IMU_AGE=.05

SHOW_PLOTS = True  # True: save PNGs and open both figures; False: save only.
INTERACTIVE_BACKEND = "TkAgg"

# Select the backend before importing pyplot. Previously pyplot was imported
# above this setting and each figure was closed before fig.show().
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smppi-cache")
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
import matplotlib
matplotlib.use(INTERACTIVE_BACKEND if SHOW_PLOTS and HAS_DISPLAY else "Agg", force=True)
import matplotlib.pyplot as plt

def centered_slope(value, dt, window_s):
    """Zero-phase offline-GT derivative; not an input available to MPPI."""
    window = max(5, int(round(window_s / dt)) | 1)
    window = min(window, len(value) // 2 * 2 - 1)
    return savgol_filter(value, window, min(3, window - 2), deriv=1, delta=dt)


def aligned_raw(bag, cfg, args):
    pose, velocity, drive, imu = read_streams(
        bag, args.pose_topic, args.velocity_topic, args.drive_topic,
        args.imu_topic)
    start = max(x[0, 0] for x in (pose, velocity, drive, imu))
    end = min(x[-1, 0] for x in (pose, velocity, drive, imu))
    time = np.arange(start, end, args.dt)
    pp, p_ok = causal_hold(pose, time, args.max_pose_age)
    vv, v_ok = causal_hold(velocity, time, args.max_velocity_age)
    dd, d_ok = causal_hold(drive, time, args.max_command_age)
    ii, i_ok = causal_hold(imu, time, args.max_imu_age)
    valid = p_ok & v_ok & d_ok & i_ok
    time, pp, vv, dd, ii = (x[valid] for x in (time, pp, vv, dd, ii))
    time = time - time[0]

    # Keep the full collision bag for diagnosis; bag_id remains constant.
    samples = np.c_[time, pp[:, 1:4], vv[:, 1:4], dd[:, 1:4],
                    np.ones(len(time)), np.zeros(len(time)), ii[:, 1:4]]
    columns = np.array(["t", "x", "y", "yaw", "vx", "vy", "omega",
                        "steer", "accel", "speed_cmd", "split", "bag_id",
                        "imu_wz", "imu_ax", "imu_ay"])
    params = LateralVelocityKFParams(
        cornering_stiffness_front=float(cfg["kf_cornering_stiffness_front"]),
        cornering_stiffness_rear=float(cfg["kf_cornering_stiffness_rear"]),
        mass=float(cfg["mass"]), yaw_inertia=float(cfg["I_z"]),
        l_f=float(cfg["l_f"]), l_r=float(cfg["l_r"]), dt=args.dt,
        min_longitudinal_speed=float(cfg["kf_min_vx"]),
        low_speed_threshold=float(cfg["kf_low_speed_threshold"]),
        max_abs_vy=float(cfg["kf_max_abs_vy"]),
        process_var_vy=float(cfg["kf_q_vy"]),
        process_var_yaw_rate=float(cfg["kf_q_yaw_rate"]),
        measurement_var_lateral_accel=float(cfg["kf_r_lateral_accel"]),
        measurement_var_yaw_rate=float(cfg["kf_r_yaw_rate"]),
        initial_var_vy=float(cfg["kf_initial_p_vy"]),
        initial_var_yaw_rate=float(cfg["kf_initial_p_yaw_rate"]),
        imu_lateral_accel_sign=float(cfg["imu_lateral_accel_sign"]))
    kf_vy, kf_w = estimate_dataset(
        samples, columns, args.dt, params=params,
        steer_scale=float(cfg["kf_steer_scale"]),
        steer_bias=float(cfg["kf_steer_bias"]),
        max_steer=float(cfg["kf_max_steer"]),
        imu_ema_alpha=float(cfg["imu_ema_alpha"]),
        imu_wz_sign=float(cfg["imu_wz_sign"]),
        imu_ay_sign=float(cfg["imu_ay_sign"]))

    # Pose-derived map velocity -> vehicle FLU body velocity.
    yaw_unwrapped = np.unwrap(pp[:, 3])
    world_vx = centered_slope(pp[:, 1], args.dt, args.pose_velocity_window)
    world_vy = centered_slope(pp[:, 2], args.dt, args.pose_velocity_window)
    cy, sy = np.cos(yaw_unwrapped), np.sin(yaw_unwrapped)
    mcl_vx = cy * world_vx + sy * world_vy
    mcl_vy = -sy * world_vx + cy * world_vy

    # Runtime signed causal EMA, shown separately from the KF yaw state.
    imu_w = float(cfg["imu_wz_sign"]) * ii[:, 1]
    alpha = float(cfg["imu_ema_alpha"])
    for i in range(1, len(imu_w)):
        imu_w[i] = alpha * imu_w[i] + (1.0 - alpha) * imu_w[i - 1]
    base = np.c_[time, pp[:, 1:4], vv[:, 1:4], dd[:, 1:4]]
    _, collision_episodes = collision_recovery_mask(base, args.dt)
    pose_w = centered_slope(yaw_unwrapped, args.dt, args.pose_velocity_window)
    return dict(t=time, mcl_vx=mcl_vx, odom_vx=vv[:, 1],
                mcl_vy=mcl_vy, kf_vy=kf_vy, imu_w=imu_w, kf_w=kf_w,
                pose_w=pose_w,
                collision_episodes=collision_episodes)


def finite_metrics(reference, estimate):
    ok = np.isfinite(reference) & np.isfinite(estimate)
    error = estimate[ok] - reference[ok]
    return {"samples": int(ok.sum()), "bias": float(error.mean()),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "p95_abs": float(np.quantile(np.abs(error), .95)),
            "correlation": float(np.corrcoef(reference[ok], estimate[ok])[0, 1])}


def plot(data, title, output, shade_collision):
    comparisons = [
        ("Longitudinal velocity", "MCL pose-derived $v_x$", data["mcl_vx"],
         "Odom $v_x$", data["odom_vx"], "$v_x$ [m/s]"),
        ("Lateral velocity", "MCL pose-derived $v_y$", data["mcl_vy"],
         "2-state KF $v_y$", data["kf_vy"], "$v_y$ [m/s]"),
        ("Yaw rate", "Signed + causal EMA IMU yaw rate", data["imu_w"],
         "2-state KF yaw rate", data["kf_w"], "Yaw rate [rad/s]")]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for ax, (name, ref_name, ref, est_name, est, ylabel) in zip(axes, comparisons):
        ax.plot(data["t"], ref, lw=1.4, color="tab:blue", label=ref_name)
        ax.plot(data["t"], est, lw=1.1, color="tab:orange", label=est_name)
        if shade_collision:
            for episode in data["collision_episodes"]:
                ax.axvspan(episode["start_time_s"], episode["end_time_s"],
                           color="tab:red", alpha=.10, label="collision/recovery" if ax is axes[0] else None)
                ax.axvspan(episode["reverse_start_time_s"], episode["reverse_end_time_s"],
                           color="tab:purple", alpha=.10, label="reverse command" if ax is axes[0] else None)
        metric = finite_metrics(ref, est)
        ax.set_title(f"{name}  |  MAE={metric['mae']:.3f}, RMSE={metric['rmse']:.3f}, "
                     f"bias={metric['bias']:+.3f}, corr={metric['correlation']:.3f}")
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    # Keep GUI figures alive until main() starts one shared event loop.
    if not (SHOW_PLOTS and HAS_DISPLAY):
        plt.close(fig)
    return {"vx_mcl_vs_odom": finite_metrics(data["mcl_vx"], data["odom_vx"]),
            "vy_mcl_vs_kf": finite_metrics(data["mcl_vy"], data["kf_vy"]),
            "yaw_rate_imu_vs_kf": finite_metrics(data["imu_w"], data["kf_w"]),
            "yaw_rate_mcl_pose_vs_imu_validation": finite_metrics(data["pose_w"], data["imu_w"]),
            "collision_episodes": data["collision_episodes"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bag-0807", default=str(BAG_0807))
    p.add_argument("--collision-bag", default=str(COLLISION_BAG))
    p.add_argument("-o", "--output", default=str(OUTPUT_PATH))
    p.add_argument("--params", default=str(PARAMS_PATH))
    p.add_argument("--pose-topic", default=POSE_TOPIC)
    p.add_argument("--velocity-topic", default=VELOCITY_TOPIC)
    p.add_argument("--drive-topic-0807", default=DRIVE_TOPIC_0807)
    p.add_argument("--drive-topic-collision", default=DRIVE_TOPIC_COLLISION)
    p.add_argument("--imu-topic", default=IMU_TOPIC)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--pose-velocity-window", type=float, default=POSE_VELOCITY_WINDOW)
    p.add_argument("--max-pose-age", type=float, default=MAX_POSE_AGE)
    p.add_argument("--max-velocity-age", type=float, default=MAX_VELOCITY_AGE)
    p.add_argument("--max-command-age", type=float, default=MAX_COMMAND_AGE)
    p.add_argument("--max-imu-age", type=float, default=MAX_IMU_AGE)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.params).read_text())["/**"]["ros__parameters"]
    output = Path(args.output)
    results = {}
    for key, bag, drive_topic, title, filename, collision in (
        ("0807", args.bag_0807, args.drive_topic_0807,
         "Figure 1. 0807 bag: MCL/Odom/IMU/KF state comparison",
         "figure1_0807_vx_vy_yaw_rate_comparison.png", False),
        ("collision", args.collision_bag, args.drive_topic_collision,
         "Figure 2. Collision bag: MCL/Odom/IMU/KF state comparison",
         "figure2_collision_vx_vy_yaw_rate_comparison.png", True)):
        args.drive_topic = drive_topic
        data = aligned_raw(bag, cfg, args)
        results[key] = plot(data, title, output / filename, collision)
        results[key]["bag"] = str(Path(bag).resolve())
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if SHOW_PLOTS:
        if HAS_DISPLAY:
            plt.show(block=True)
            plt.close("all")
        else:
            print("SHOW_PLOTS=True, but DISPLAY/WAYLAND_DISPLAY is unavailable; "
                  f"PNG files were saved to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
