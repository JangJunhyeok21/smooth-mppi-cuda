#!/usr/bin/env python3
"""Robustly regress the 2-state lateral KF cornering stiffnesses.

The regression uses the same FLU signs and steering conversion as SMPPI.  MCL
pose is used only as an offline body-vy target; deployment still uses the
causal IMU/KF.  Cf and Cr are fitted jointly to the linear bicycle lateral
acceleration and yaw-acceleration equations, then validated by replaying the
complete KF with the old and fitted parameters.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# =============================================================================
# USER SETTINGS
# Edit this block only. No command-line arguments are required.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "model_tuning/data/ifac_all_ackermann_bagdisjoint_train_test.npz"
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/results/ifac_all_ackermann_kf_threshold0_regression"
PARAMS_YAML_PATH = PROJECT_ROOT / "config/params.yaml"

POSE_VELOCITY_WINDOW_S = 0.40
# False: yaw-rate regression target is signed + causal EMA IMU.
# True : yaw-rate regression target is /newmcl_pose yaw derivative.
# The recursive KF input remains IMU in both cases, matching SMPPI deployment.
USE_MCL_YAW_RATE_TARGET = True
MIN_SPEED_MPS = 0.0
MAX_SPEED_MPS = 10.0
MAX_ABS_VY_MPS = 5.0
MAX_ABS_YAW_RATE_RADPS = 4.0
MAX_ABS_YAW_ACCEL_RADPS2 = 25.0

# Larger residual scales reduce the corresponding target's influence.
LATERAL_ACCEL_RESIDUAL_SCALE = 2.0
YAW_ACCEL_RESIDUAL_SCALE = 8.0
KF_VY_RESIDUAL_SCALE = 0.35
KF_YAW_RATE_RESIDUAL_SCALE = 0.35

# Cf and Cr bounds [N/rad].
INITIAL_CORNERING_STIFFNESS = (110.0, 199.0)
CORNERING_STIFFNESS_LOWER = (1.0, 1.0)
CORNERING_STIFFNESS_UPPER = (500.0, 500.0)

# True: save the result and open an interactive comparison window.
# False: save files only (for SSH/headless execution).
SHOW_PLOTS = True
INTERACTIVE_BACKEND = "TkAgg"
# True: additionally run prediction-only rollouts for both the current and
# regressed Cf/Cr. False: run only the normal IMU-feedback KF comparison.
RUN_OPEN_LOOP = True

# Do not require MPLCONFIGDIR to be exported in the shell.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smppi-cache")
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if SHOW_PLOTS and HAS_DISPLAY:
    os.environ["MPLBACKEND"] = INTERACTIVE_BACKEND
    import matplotlib
    matplotlib.use(INTERACTIVE_BACKEND, force=True)
else:
    import matplotlib
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

sys.path.insert(0, str(PROJECT_ROOT))
from model_tuning_utils.lateral_velocity_kf import (
    LateralVelocityKFParams, estimate_dataset)


def smooth_by_segment(values, bag_id, dt, window_s, derivative=0):
    result = np.full(len(values), np.nan)
    requested = max(5, int(round(window_s / dt)) | 1)
    for bid in np.unique(bag_id):
        indices = np.flatnonzero(bag_id == bid)
        window = min(requested, len(indices) // 2 * 2 - 1)
        if window >= 5:
            result[indices] = savgol_filter(
                values[indices], window, min(3, window - 2),
                deriv=derivative, delta=dt)
    return result


def ema_by_segment(values, bag_id, alpha):
    output = values.copy()
    for bid in np.unique(bag_id):
        indices = np.flatnonzero(bag_id == bid)
        for j in range(1, len(indices)):
            i, previous = indices[j], indices[j - 1]
            output[i] = alpha * values[i] + (1.0 - alpha) * output[previous]
    return output


def pose_body_kinematics(samples, names, bag_id, dt, window_s):
    yaw = samples[:, names["yaw"]]
    unwrapped = np.full(len(yaw), np.nan)
    for bid in np.unique(bag_id):
        ii = np.flatnonzero(bag_id == bid)
        unwrapped[ii] = np.unwrap(yaw[ii])
    dx = smooth_by_segment(samples[:, names["x"]], bag_id, dt, window_s, 1)
    dy = smooth_by_segment(samples[:, names["y"]], bag_id, dt, window_s, 1)
    smooth_yaw = smooth_by_segment(unwrapped, bag_id, dt, window_s, 0)
    cy, sy = np.cos(smooth_yaw), np.sin(smooth_yaw)
    body_vx = cy * dx + sy * dy
    body_vy = -sy * dx + cy * dy
    yaw_rate = smooth_by_segment(unwrapped, bag_id, dt, window_s, 1)
    return body_vx, body_vy, yaw_rate


def kf_params(cfg, dt, cf, cr):
    return LateralVelocityKFParams(
        cornering_stiffness_front=float(cf),
        cornering_stiffness_rear=float(cr), mass=float(cfg["mass"]),
        yaw_inertia=float(cfg["I_z"]), l_f=float(cfg["l_f"]),
        l_r=float(cfg["l_r"]), dt=dt,
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


def prediction_only_rollout(samples, names, bag_id, dt, cfg, stiffness,
                            initial_vy, initial_yaw_rate, delta):
    """Run x[k+1]=Ad(vx[k])x[k]+Bd*delta[k] without IMU correction.

    Each collision-free segment is initialized once from the offline target.
    Afterwards the previously predicted [vy, yaw_rate] is fed back.  No z_k,
    Kalman gain, MCL state, IMU ay, or IMU yaw-rate is used inside the segment.
    """
    cf, cr = map(float, stiffness)
    m, iz = float(cfg["mass"]), float(cfg["I_z"])
    lf, lr = float(cfg["l_f"]), float(cfg["l_r"])
    min_vx = float(cfg["kf_min_vx"])
    low_speed = float(cfg["kf_low_speed_threshold"])
    vx = samples[:, names["vx"]]
    output_vy = np.full(len(samples), np.nan)
    output_w = np.full(len(samples), np.nan)
    for bid in np.unique(bag_id):
        indices = np.flatnonzero(bag_id == bid)
        if not len(indices):
            continue
        first = indices[0]
        vy_state = float(initial_vy[first]) if np.isfinite(initial_vy[first]) else 0.0
        w_state = (float(initial_yaw_rate[first])
                   if np.isfinite(initial_yaw_rate[first]) else 0.0)
        output_vy[first], output_w[first] = vy_state, w_state
        for previous, current in zip(indices[:-1], indices[1:]):
            # Match the causal convention: command/state at k predicts k+1.
            speed = abs(float(vx[previous])) if np.isfinite(vx[previous]) else 0.0
            if speed < low_speed:
                # Without an IMU measurement the deployed low-speed correction
                # is unavailable. Decay the unobservable lateral state to zero.
                vy_state = 0.0
                w_state = 0.0
            else:
                safe_vx = max(speed, min_vx)
                inv_vx = 1.0 / safe_vx
                a00 = -(cf + cr) * inv_vx / m
                a01 = -(safe_vx + (lf * cf - lr * cr) * inv_vx / m)
                a10 = -(lf * cf - lr * cr) * inv_vx / iz
                a11 = -(lf * lf * cf + lr * lr * cr) * inv_vx / iz
                next_vy = ((1.0 + dt * a00) * vy_state + dt * a01 * w_state
                           + dt * cf / m * delta[previous])
                next_w = (dt * a10 * vy_state + (1.0 + dt * a11) * w_state
                          + dt * lf * cf / iz * delta[previous])
                vy_state, w_state = next_vy, next_w
            output_vy[current], output_w[current] = vy_state, w_state
    return output_vy, output_w


def metrics(reference, estimate, valid):
    error = estimate[valid] - reference[valid]
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "p95_abs": float(np.quantile(np.abs(error), .95)),
            "correlation": float(np.corrcoef(reference[valid], estimate[valid])[0, 1])}


def main():
    args = SimpleNamespace(
        dataset=DATASET_PATH,
        output=OUTPUT_PATH,
        params=PARAMS_YAML_PATH,
        pose_window=POSE_VELOCITY_WINDOW_S,
        min_speed=MIN_SPEED_MPS,
        max_speed=MAX_SPEED_MPS,
        max_abs_vy=MAX_ABS_VY_MPS,
        max_abs_yaw_rate=MAX_ABS_YAW_RATE_RADPS,
        max_abs_yaw_accel=MAX_ABS_YAW_ACCEL_RADPS2,
        ay_scale=LATERAL_ACCEL_RESIDUAL_SCALE,
        yaw_accel_scale=YAW_ACCEL_RESIDUAL_SCALE,
        kf_vy_scale=KF_VY_RESIDUAL_SCALE,
        kf_yaw_rate_scale=KF_YAW_RATE_RESIDUAL_SCALE,
        lower=CORNERING_STIFFNESS_LOWER,
        upper=CORNERING_STIFFNESS_UPPER)

    archive = np.load(args.dataset)
    samples = archive["samples"].astype(np.float64)
    columns = archive["columns"]
    dt = float(archive["dt"])
    names = {str(name): i for i, name in enumerate(columns)}
    required = ("x", "y", "yaw", "vx", "steer", "bag_id",
                "imu_wz", "imu_ay")
    missing = [name for name in required if name not in names]
    if missing:
        raise SystemExit(f"dataset missing columns: {missing}")
    cfg = yaml.safe_load(Path(args.params).read_text())["/**"]["ros__parameters"]
    bag_id = samples[:, names["bag_id"]].astype(int)
    _, mcl_vy, mcl_yaw_rate = pose_body_kinematics(
        samples, names, bag_id, dt, args.pose_window)
    alpha = float(cfg["imu_ema_alpha"])
    imu_yaw_rate = ema_by_segment(
        float(cfg["imu_wz_sign"]) * samples[:, names["imu_wz"]], bag_id, alpha)
    omega = mcl_yaw_rate if USE_MCL_YAW_RATE_TARGET else imu_yaw_rate
    yaw_rate_target_source = (
        "mcl_pose_derivative" if USE_MCL_YAW_RATE_TARGET else "signed_ema_imu")
    ay = ema_by_segment(float(cfg["imu_ay_sign"]) * samples[:, names["imu_ay"]],
                        bag_id, alpha)
    yaw_accel = smooth_by_segment(omega, bag_id, dt, args.pose_window, 1)
    vx = samples[:, names["vx"]]
    delta = np.clip(float(cfg["kf_steer_scale"]) * samples[:, names["steer"]]
                    + float(cfg["kf_steer_bias"]),
                    -float(cfg["kf_max_steer"]), float(cfg["kf_max_steer"]))
    valid = (np.isfinite(mcl_vy) & np.isfinite(yaw_accel) &
             (np.abs(vx) >= args.min_speed) & (np.abs(vx) <= args.max_speed) &
             (np.abs(mcl_vy) <= args.max_abs_vy) &
             (np.abs(omega) <= args.max_abs_yaw_rate) &
             (np.abs(yaw_accel) <= args.max_abs_yaw_accel))
    if valid.sum() < 100:
        raise SystemExit(f"only {valid.sum()} valid regression samples")

    m, iz, lf, lr = (float(cfg["mass"]), float(cfg["I_z"]),
                     float(cfg["l_f"]), float(cfg["l_r"]))
    safe_vx = np.maximum(np.abs(vx[valid]), float(cfg["kf_min_vx"]))
    vyv, wv, dv = mcl_vy[valid], omega[valid], delta[valid]

    def residual(theta):
        cf, cr = theta
        alpha_f = dv - (vyv + lf * wv) / safe_vx
        alpha_r = -(vyv - lr * wv) / safe_vx
        fyf, fyr = cf * alpha_f, cr * alpha_r
        predicted_ay = (fyf + fyr) / m
        predicted_yaw_accel = (lf * fyf - lr * fyr) / iz
        return np.r_[(predicted_ay - ay[valid]) / args.ay_scale,
                     (predicted_yaw_accel - yaw_accel[valid]) /
                     args.yaw_accel_scale]

    old = np.asarray(INITIAL_CORNERING_STIFFNESS, dtype=np.float64)
    equation_fit = least_squares(residual, old, bounds=(args.lower, args.upper),
                                 loss="soft_l1", f_scale=1.0, max_nfev=1000)
    equation_fitted = equation_fit.x

    # The final deployed parameters minimize the output of the actual recursive
    # KF, rather than assuming noisy MCL derivatives satisfy both continuous
    # bicycle equations exactly.
    def replay(theta):
        # estimate_dataset always feeds signed IMU yaw-rate to the KF. The
        # switch changes supervision only, never the deployed KF input.
        return estimate_dataset(
            samples, columns, dt, params=kf_params(cfg, dt, *theta),
            steer_scale=float(cfg["kf_steer_scale"]),
            steer_bias=float(cfg["kf_steer_bias"]),
            max_steer=float(cfg["kf_max_steer"]), imu_ema_alpha=alpha,
            imu_wz_sign=float(cfg["imu_wz_sign"]),
            imu_ay_sign=float(cfg["imu_ay_sign"]))

    def kf_residual(theta):
        predicted_vy, predicted_yaw_rate = replay(theta)
        return np.r_[(predicted_vy[valid] - mcl_vy[valid]) / args.kf_vy_scale,
                     (predicted_yaw_rate[valid] - omega[valid]) /
                     args.kf_yaw_rate_scale]

    kf_fit = least_squares(kf_residual, equation_fitted,
                           bounds=(args.lower, args.upper), loss="soft_l1",
                           f_scale=1.0, max_nfev=150, diff_step=.02)
    fitted = kf_fit.x

    estimates = {}
    for label, stiffness in (("current", old), ("regressed", fitted)):
        vy, yaw_rate = replay(stiffness)
        estimates[label] = (vy, yaw_rate)

    open_loop_estimates = {}
    if RUN_OPEN_LOOP:
        for label, stiffness in (("current", old), ("regressed", fitted)):
            open_loop_estimates[label] = prediction_only_rollout(
                samples, names, bag_id, dt, cfg, stiffness,
                mcl_vy, omega, delta)

    evaluation = valid & (np.abs(vx) >= args.min_speed)
    result = {
        "dataset": str(Path(args.dataset).resolve()),
        "yaw_rate_target_source": yaw_rate_target_source,
        "samples_total": int(len(samples)),
        "samples_regression": int(valid.sum()),
        "old": {"Cf_N_per_rad": float(old[0]), "Cr_N_per_rad": float(old[1])},
        "equation_regression": {"Cf_N_per_rad": float(equation_fitted[0]),
                                "Cr_N_per_rad": float(equation_fitted[1]),
                                "success": bool(equation_fit.success),
                                "cost": float(equation_fit.cost)},
        "fitted": {"Cf_N_per_rad": float(fitted[0]), "Cr_N_per_rad": float(fitted[1])},
        "optimizer": {"success": bool(kf_fit.success), "cost": float(kf_fit.cost),
                      "message": kf_fit.message,
                      "objective": "recursive KF vy + yaw-rate robust residual"},
        "yaw_rate_sensor_comparison": {
            "mcl_pose_vs_signed_ema_imu": metrics(
                mcl_yaw_rate, imu_yaw_rate,
                evaluation & np.isfinite(mcl_yaw_rate) & np.isfinite(imu_yaw_rate))},
        "validation": {}}
    for label, (vy, yaw_rate) in estimates.items():
        result["validation"][label] = {
            "vy_mcl_vs_kf": metrics(mcl_vy, vy, evaluation),
            "yaw_rate_target_vs_kf": metrics(omega, yaw_rate, evaluation)}
    result["open_loop_enabled"] = RUN_OPEN_LOOP
    if RUN_OPEN_LOOP:
        result["open_loop_validation"] = {
            "definition": ("prediction-only linear bicycle rollout; initialized once "
                           "per segment; no IMU/MCL measurement correction")}
        for label, (vy, yaw_rate) in open_loop_estimates.items():
            finite = evaluation & np.isfinite(vy) & np.isfinite(yaw_rate)
            result["open_loop_validation"][label] = {
                "vy_mcl_vs_prediction": metrics(mcl_vy, vy, finite),
                "yaw_rate_target_vs_prediction": metrics(omega, yaw_rate, finite)}

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "kf_cornering_stiffness.json").write_text(
        json.dumps(result, indent=2) + "\n")
    prediction_arrays = dict(
        t=samples[:, names["t"]], mcl_vy=mcl_vy,
        yaw_rate_target=omega, imu_yaw_rate=imu_yaw_rate,
        mcl_yaw_rate=mcl_yaw_rate,
        current_kf_vy=estimates["current"][0],
        fitted_kf_vy=estimates["regressed"][0],
        current_kf_yaw_rate=estimates["current"][1],
        fitted_kf_yaw_rate=estimates["regressed"][1], valid=evaluation)
    if RUN_OPEN_LOOP:
        prediction_arrays.update(
            current_open_loop_vy=open_loop_estimates["current"][0],
            fitted_open_loop_vy=open_loop_estimates["regressed"][0],
            current_open_loop_yaw_rate=open_loop_estimates["current"][1],
            fitted_open_loop_yaw_rate=open_loop_estimates["regressed"][1])
    np.savez_compressed(output / "kf_cornering_stiffness_predictions.npz",
                        **prediction_arrays)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    time = samples[:, names["t"]].copy()
    # Extracted collision-free episodes restart at t=0.  Use a monotonic display
    # clock so matplotlib never draws misleading cross-episode diagonals.
    offset = 0.0
    for bid in np.unique(bag_id):
        ii = np.flatnonzero(bag_id == bid)
        time[ii] = samples[ii, names["t"]] + offset
        offset = time[ii[-1]] + 2.0 * dt
    axes[0].plot(time, mcl_vy, color="black", lw=1, label="MCL pose-derived vy")
    axes[0].plot(time, estimates["current"][0], lw=.9, label="KF current Cf/Cr")
    axes[0].plot(time, estimates["regressed"][0], lw=.9, label="KF regressed Cf/Cr")
    if RUN_OPEN_LOOP:
        axes[0].plot(time, open_loop_estimates["current"][0], color="tab:purple",
                     lw=.9, ls=":", label="Current prediction-only open loop")
        axes[0].plot(time, open_loop_estimates["regressed"][0], color="tab:red",
                     lw=1, ls="--", label="Regressed prediction-only open loop")
    axes[0].set_ylabel("vy [m/s]"); axes[0].legend(); axes[0].grid(alpha=.25)
    mcl_label = "MCL pose-derived yaw rate"
    imu_label = "signed EMA IMU yaw rate"
    if USE_MCL_YAW_RATE_TARGET:
        mcl_label += " [target]"
    else:
        imu_label += " [target]"
    axes[1].plot(time, mcl_yaw_rate, color="black",
                 lw=1.4 if USE_MCL_YAW_RATE_TARGET else .9,
                 ls="-" if USE_MCL_YAW_RATE_TARGET else "--", label=mcl_label)
    axes[1].plot(time, imu_yaw_rate, color="tab:green",
                 lw=1.4 if not USE_MCL_YAW_RATE_TARGET else .9,
                 ls="-" if not USE_MCL_YAW_RATE_TARGET else ":", label=imu_label)
    axes[1].plot(time, estimates["current"][1], lw=.9, label="KF current Cf/Cr")
    axes[1].plot(time, estimates["regressed"][1], lw=.9, label="KF regressed Cf/Cr")
    if RUN_OPEN_LOOP:
        axes[1].plot(time, open_loop_estimates["current"][1], color="tab:purple",
                     lw=.9, ls=":", label="Current prediction-only open loop")
        axes[1].plot(time, open_loop_estimates["regressed"][1], color="tab:red",
                     lw=1, ls="--", label="Regressed prediction-only open loop")
    axes[1].set_ylabel("yaw rate [rad/s]"); axes[1].set_xlabel("time [s]")
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle(
        f"0807 KF stiffness regression ({yaw_rate_target_source}): "
        f"Cf={fitted[0]:.2f}, Cr={fitted[1]:.2f} N/rad")
    plot_path = output / "kf_cornering_stiffness_comparison.png"
    fig.tight_layout(); fig.savefig(plot_path, dpi=180)
    print(json.dumps(result, indent=2))
    print(f"Saved plot: {plot_path.resolve()}")
    if SHOW_PLOTS and HAS_DISPLAY:
        print("Plot window is open. Close it to finish the program.")
        plt.show()
    elif SHOW_PLOTS:
        print("SHOW_PLOTS=True, but no display is available; PNG was saved.")
    plt.close(fig)


if __name__ == "__main__":
    main()
