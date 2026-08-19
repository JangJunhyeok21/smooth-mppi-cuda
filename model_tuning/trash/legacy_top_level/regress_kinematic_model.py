#!/usr/bin/env python3
"""Regress the classic base of the active MPPI slip-kinematic model.

The residual MLP is deliberately excluded.  The fitted parameters are:
steering command scale/bias, servo time constant/rate limit, speed-loop Kp,
and odom-speed-to-map-position scale.  Every candidate is evaluated with a
recursive 1 s rollout matching update_slip_kinematic_with_imu_direct().
"""
import json
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

# =============================================================================
# USER SETTINGS -- run with: python3 model_tuning/regress_kinematic_model.py
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "model_tuning/data/ifac_0808_0810_train_test.npz"
PARAMS_YAML_PATH = ROOT / "config/params.yaml"
OUTPUT_PATH = ROOT / "model_tuning/results/ifac_0808_0810_slip_regression"

HORIZON_S = 1.0
WINDOW_STRIDE = 5
MAX_WINDOWS = 1200
RANDOM_SEED = 17
MAX_NFEV = 350

# Residual scales: smaller means that signal has more influence.
POSITION_RESIDUAL_SCALE_M = 0.15
YAW_RESIDUAL_SCALE_RAD = 0.15
SPEED_RESIDUAL_SCALE_MPS = 0.25
REGULARIZATION_WEIGHT = 0.02

# [steer_scale, steer_bias, servo_tau, max_steer_rate, speed_kp, position_scale]
LOWER_BOUNDS = np.array([0.10, -0.15, 0.02, 0.20, 0.5, 0.50])
UPPER_BOUNDS = np.array([2.00,  0.15, 0.60, 8.00, 20.0, 1.50])

SHOW_PLOTS = False
INTERACTIVE_BACKEND = "TkAgg"
# Keep False until the regression plot/metrics have been reviewed.
WRITE_PARAMS_YAML = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smppi-cache")
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
import matplotlib
matplotlib.use(INTERACTIVE_BACKEND if SHOW_PLOTS and HAS_DISPLAY else "Agg", force=True)
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(ROOT))
from model_tuning_utils.lateral_velocity_kf import (
    LateralVelocityKFParams, estimate_dataset)


PARAMETER_NAMES = (
    "kinematic_steer_scale", "kinematic_steer_bias",
    "steer_servo_time_constant", "actuator_max_steer_rate",
    "speed_servo_kp", "kinematic_position_speed_scale")


def angle_error(prediction, target):
    return np.arctan2(np.sin(prediction-target), np.cos(prediction-target))


def make_kf_params(cfg, dt):
    return LateralVelocityKFParams(
        cornering_stiffness_front=float(cfg["kf_cornering_stiffness_front"]),
        cornering_stiffness_rear=float(cfg["kf_cornering_stiffness_rear"]),
        mass=float(cfg["mass"]), yaw_inertia=float(cfg["I_z"]),
        l_f=float(cfg["l_f"]), l_r=float(cfg["l_r"]), dt=dt,
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


def actuator_trace(command, bag, dt, scale, bias, tau, rate_limit):
    trace = np.empty(len(command), dtype=np.float64)
    for bid in np.unique(bag):
        rows = np.flatnonzero(bag == bid)
        trace[rows[0]] = np.clip(scale*command[rows[0], 0]+bias, -.55, .55)
        for previous, current in zip(rows[:-1], rows[1:]):
            target = np.clip(scale*command[current, 0]+bias, -.55, .55)
            rate = np.clip((target-trace[previous])/max(tau, 1e-3),
                           -rate_limit, rate_limit)
            trace[current] = np.clip(trace[previous]+rate*dt, -.55, .55)
    return trace


def rollout(theta, starts, horizon, dt, pose, vx, vy, command, bag, cfg):
    steer_scale, steer_bias, tau, rate_limit, kp, position_scale = theta
    delta_trace = actuator_trace(command, bag, dt, steer_scale, steer_bias,
                                 tau, rate_limit)
    rows = starts.copy()
    x, y, yaw = pose[rows, 0].copy(), pose[rows, 1].copy(), pose[rows, 2].copy()
    speed = np.hypot(vx[rows], vy[rows])
    beta = np.arctan2(vy[rows], np.maximum(vx[rows], 1e-4))
    previous_rows = np.maximum(rows-1, 0)
    previous_rows = np.where(bag[previous_rows] == bag[rows], previous_rows, rows)
    # MPPI history[10] is the actuator state before applying u[k].
    delta = delta_trace[previous_rows].copy()
    wb = float(cfg["l_f"])+float(cfg["l_r"])
    min_speed, max_speed = float(cfg["min_speed"]), float(cfg["max_speed"])
    min_accel, max_accel = float(cfg["min_accel"]), float(cfg["max_accel"])
    predicted_pose = np.empty((len(starts), horizon, 3), dtype=np.float64)
    predicted_speed = np.empty((len(starts), horizon), dtype=np.float64)
    for step in range(horizon):
        index = rows+step
        target_delta = np.clip(steer_scale*command[index, 0]+steer_bias, -.55, .55)
        steer_rate = np.clip((target_delta-delta)/max(tau, 1e-3),
                             -rate_limit, rate_limit)
        delta = np.clip(delta+steer_rate*dt, -.55, .55)
        speed_cmd = np.clip(command[index, 1], min_speed, max_speed)
        acceleration = np.clip(kp*(speed_cmd-speed), min_accel, max_accel)
        lower, upper = np.minimum(min_speed, speed), np.maximum(max_speed, speed)
        speed = np.minimum(upper, np.maximum(lower, speed+acceleration*dt))
        next_vx, next_vy = speed*np.cos(beta), speed*np.sin(beta)
        yaw_rate = next_vx*np.tan(delta)/wb
        x = x+position_scale*(next_vx*np.cos(yaw)-next_vy*np.sin(yaw))*dt
        y = y+position_scale*(next_vx*np.sin(yaw)+next_vy*np.cos(yaw))*dt
        yaw = yaw+yaw_rate*dt
        predicted_pose[:, step] = np.column_stack((x, y, yaw))
        predicted_speed[:, step] = speed
    return predicted_pose, predicted_speed


def choose_windows(samples, names, horizon, split_value):
    bag = samples[:, names["bag_id"]].astype(int)
    split = samples[:, names["split"]].astype(int)
    time = samples[:, names["t"]]
    finite = np.all(np.isfinite(samples[:, [names[x] for x in
        ("x", "y", "yaw", "vx", "steer", "speed_cmd")]]), axis=1)
    candidates = []
    for start in range(0, len(samples)-horizon-1, WINDOW_STRIDE):
        stop = start+horizon
        if split[start] != split_value or split[stop] != split_value or bag[start] != bag[stop]:
            continue
        if not np.all(finite[start:stop+1]):
            continue
        if np.any(np.abs(np.diff(time[start:stop+1])-float(np.median(np.diff(time[start:stop+1])))) > 1e-3):
            continue
        if np.any(np.hypot(np.diff(samples[start:stop+1, names["x"]]),
                           np.diff(samples[start:stop+1, names["y"]])) > .25):
            continue
        candidates.append(start)
    candidates = np.asarray(candidates, dtype=int)
    if len(candidates) > MAX_WINDOWS:
        candidates = np.random.default_rng(RANDOM_SEED).choice(
            candidates, MAX_WINDOWS, replace=False)
    return np.sort(candidates)


def summarize(pred_pose, pred_speed, starts, horizon, pose, speed):
    target_rows = starts+horizon
    position_error = np.linalg.norm(pred_pose[:, -1, :2]-pose[target_rows, :2], axis=1)
    yaw_error = np.abs(angle_error(pred_pose[:, -1, 2], pose[target_rows, 2]))
    speed_error = np.abs(pred_speed[:, -1]-speed[target_rows])
    return {
        "trajectory_mean_m": float(position_error.mean()),
        "trajectory_median_m": float(np.median(position_error)),
        "trajectory_p95_m": float(np.quantile(position_error, .95)),
        "yaw_mae_deg": float(np.degrees(yaw_error.mean())),
        "speed_mae_mps": float(speed_error.mean())}


def main():
    archive = np.load(DATASET_PATH)
    samples = archive["samples"].astype(np.float64)
    names = {str(name): i for i, name in enumerate(archive["columns"])}
    dt = float(archive["dt"]); horizon = max(1, round(HORIZON_S/dt))
    cfg = yaml.safe_load(PARAMS_YAML_PATH.read_text())["/**"]["ros__parameters"]
    bag = samples[:, names["bag_id"]].astype(int)
    pose = samples[:, [names["x"], names["y"], names["yaw"]]]
    vx = samples[:, names["vx"]]
    command = samples[:, [names["steer"], names["speed_cmd"]]]
    vy, _ = estimate_dataset(
        samples, archive["columns"], dt, params=make_kf_params(cfg, dt),
        steer_scale=float(cfg["kf_steer_scale"]),
        steer_bias=float(cfg["kf_steer_bias"]),
        max_steer=float(cfg["kf_max_steer"]),
        imu_ema_alpha=float(cfg["imu_ema_alpha"]),
        imu_wz_sign=float(cfg["imu_wz_sign"]),
        imu_ay_sign=float(cfg["imu_ay_sign"]))
    speed = np.hypot(vx, vy)
    starts = choose_windows(samples, names, horizon, split_value=0)
    if len(starts) < 30:
        raise SystemExit(f"only {len(starts)} valid train windows")
    initial = np.array([
        cfg["kinematic_steer_scale"], cfg["kinematic_steer_bias"],
        cfg["steer_servo_time_constant"], cfg["actuator_max_steer_rate"],
        cfg["speed_servo_kp"], cfg["kinematic_position_speed_scale"]], dtype=float)

    target_pose = np.stack([pose[starts+k] for k in range(1, horizon+1)], axis=1)
    target_speed = np.stack([speed[starts+k] for k in range(1, horizon+1)], axis=1)
    parameter_scale = np.maximum(UPPER_BOUNDS-LOWER_BOUNDS, 1e-6)

    def residual(theta):
        pred_pose, pred_speed = rollout(
            theta, starts, horizon, dt, pose, vx, vy, command, bag, cfg)
        # Use several points across the horizon, not only the endpoint.
        sample_steps = np.unique(np.linspace(4, horizon-1, 8).astype(int))
        position = ((pred_pose[:, sample_steps, :2]-target_pose[:, sample_steps, :2])
                    / POSITION_RESIDUAL_SCALE_M).reshape(-1)
        yaw = (angle_error(pred_pose[:, sample_steps, 2],
                           target_pose[:, sample_steps, 2])
               / YAW_RESIDUAL_SCALE_RAD).reshape(-1)
        velocity = ((pred_speed[:, sample_steps]-target_speed[:, sample_steps])
                    / SPEED_RESIDUAL_SCALE_MPS).reshape(-1)
        regularizer = REGULARIZATION_WEIGHT*(theta-initial)/parameter_scale
        return np.r_[position, yaw, velocity, regularizer]

    initial_pose, initial_speed = rollout(
        initial, starts, horizon, dt, pose, vx, vy, command, bag, cfg)
    fit = least_squares(residual, initial, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                        loss="soft_l1", f_scale=1.0, max_nfev=MAX_NFEV,
                        verbose=1)
    fitted = fit.x
    fitted_pose, fitted_speed = rollout(
        fitted, starts, horizon, dt, pose, vx, vy, command, bag, cfg)
    test_starts = choose_windows(samples, names, horizon, split_value=1)
    test_before = test_after = None
    if len(test_starts):
        test_initial_pose, test_initial_speed = rollout(
            initial, test_starts, horizon, dt, pose, vx, vy, command, bag, cfg)
        test_fitted_pose, test_fitted_speed = rollout(
            fitted, test_starts, horizon, dt, pose, vx, vy, command, bag, cfg)
        test_before = summarize(test_initial_pose, test_initial_speed, test_starts,
                                horizon, pose, speed)
        test_after = summarize(test_fitted_pose, test_fitted_speed, test_starts,
                               horizon, pose, speed)
    report = {
        "dataset": str(DATASET_PATH), "split_used": "train only (split == 0)",
        "model": "MPPI slip-kinematic classic base, residual MLP disabled",
        "horizon_s": HORIZON_S, "windows": int(len(starts)),
        "success": bool(fit.success), "message": fit.message,
        "initial": dict(zip(PARAMETER_NAMES, map(float, initial))),
        "fitted": dict(zip(PARAMETER_NAMES, map(float, fitted))),
        "before": summarize(initial_pose, initial_speed, starts, horizon, pose, speed),
        "after": summarize(fitted_pose, fitted_speed, starts, horizon, pose, speed),
        "test_windows": int(len(test_starts)), "test_before": test_before,
        "test_after": test_after}
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH/"kinematic_model_regression.json").write_text(
        json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUTPUT_PATH/"kinematic_model_regression_predictions.npz",
        starts=starts, initial_pose=initial_pose, fitted_pose=fitted_pose,
        target_pose=target_pose, initial_speed=initial_speed,
        fitted_speed=fitted_speed, target_speed=target_speed)

    labels = ["Initial", "Fitted"]
    keys = ("trajectory_mean_m", "trajectory_p95_m", "yaw_mae_deg", "speed_mae_mps")
    titles = ("Trajectory mean [m]", "Trajectory P95 [m]", "Yaw MAE [deg]", "Speed MAE [m/s]")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, key, title in zip(axes.flat, keys, titles):
        values = [report["before"][key], report["after"][key]]
        axis.bar(labels, values, color=("0.55", "tab:blue"));axis.set_title(title);axis.grid(axis="y", alpha=.25)
    fig.suptitle("Train-only 1 s recursive kinematic parameter regression")
    fig.tight_layout();plot_path=OUTPUT_PATH/"kinematic_model_regression.png";fig.savefig(plot_path,dpi=180)
    if WRITE_PARAMS_YAML:
        text = PARAMS_YAML_PATH.read_text()
        for name, value in report["fitted"].items():
            import re
            text = re.sub(rf"(^\s*{name}:\s*)[^#\n]+", rf"\g<1>{value:.9g} ", text, flags=re.MULTILINE)
        PARAMS_YAML_PATH.write_text(text)
    print(json.dumps(report, indent=2));print(f"Saved plot: {plot_path}")
    if SHOW_PLOTS and HAS_DISPLAY: plt.show(block=True)
    elif SHOW_PLOTS: print("SHOW_PLOTS=True, but no display is available; PNG was saved.")
    plt.close(fig)


if __name__ == "__main__":
    main()
