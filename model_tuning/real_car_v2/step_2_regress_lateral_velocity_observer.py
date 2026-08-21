#!/usr/bin/env python3
"""Step 2 optional: tune current Pacejka-EKF noise, bias, and MCL-vy update."""
from pathlib import Path
import json
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from helper_lateral_velocity_kf import LateralVelocityKFParams, estimate_dataset
from offline_lateral_velocity_smoother import smooth_segment_vy

DATA = ROOT/"model_tuning/data/ifac0810_0819_autonomous_physics_clean"
OUT = ROOT/"model_tuning/results/pacejka_vy_observer_regression"
CFG = ROOT/"config/params.yaml"
SEED = 43


def ema(values, alpha):
    result = values.copy()
    for i in range(1, len(result)):
        result[i] = alpha*values[i]+(1-alpha)*result[i-1]
    return result


def load_records(cfg):
    records = []
    for path in sorted(DATA.glob("*.npz")):
        archive = np.load(path); samples = archive["samples"].astype(float)
        columns = archive["columns"]; names = {str(v): i for i, v in enumerate(columns)}
        dt = float(archive["dt"]); signs = archive["imu_axis_signs"].astype(float)
        alpha = float(archive["imu_ema_alpha"])
        for segment in np.unique(samples[:, names["bag_id"]].astype(int)):
            part = samples[samples[:, names["bag_id"]].astype(int) == segment]
            if len(part) < 80:
                continue
            yaw_rate = ema(signs[0]*part[:, names["imu_wz"]], alpha)
            lateral_accel = ema(signs[2]*part[:, names["imu_ay"]], alpha)
            teacher, diagnostic = smooth_segment_vy(
                part[:, names["x"]], part[:, names["y"]], part[:, names["yaw"]],
                part[:, names["vx"]], yaw_rate, lateral_accel, dt)
            if not diagnostic.get("usable"):
                continue
            valid = np.isfinite(teacher)
            valid[:8] = False; valid[-8:] = False
            if valid.sum() >= 40:
                records.append((part, columns, dt, signs, alpha, teacher, valid))
    return records


def make_params(cfg, theta, dt):
    q_vy, r_ay, r_pose, q_bias, p_bias, max_bias, pose_gate = theta
    return LateralVelocityKFParams(
        mass=float(cfg["mass"]), yaw_inertia=float(cfg["I_z"]),
        l_f=float(cfg["l_f"]), l_r=float(cfg["l_r"]), dt=dt,
        pacejka_b_front=float(cfg["dynamic_mlp_B_f"]),
        pacejka_c_front=float(cfg["dynamic_mlp_C_f"]),
        pacejka_d_front=float(cfg["dynamic_mlp_D_f"]),
        pacejka_e_front=float(cfg["dynamic_mlp_E_f"]),
        pacejka_b_rear=float(cfg["dynamic_mlp_B_r"]),
        pacejka_c_rear=float(cfg["dynamic_mlp_C_r"]),
        pacejka_d_rear=float(cfg["dynamic_mlp_D_r"]),
        pacejka_e_rear=float(cfg["dynamic_mlp_E_r"]),
        min_longitudinal_speed=float(cfg["kf_min_vx"]),
        low_speed_threshold=0.0, max_abs_vy=float("inf"), process_var_vy=q_vy,
        process_var_yaw_rate=float(cfg["kf_q_yaw_rate"]),
        measurement_var_lateral_accel=r_ay,
        measurement_var_yaw_rate=float(cfg["kf_r_yaw_rate"]),
        initial_var_vy=float(cfg["kf_initial_p_vy"]),
        initial_var_yaw_rate=float(cfg["kf_initial_p_yaw_rate"]),
        imu_lateral_accel_sign=float(cfg["imu_lateral_accel_sign"]),
        process_var_ay_bias=q_bias, initial_var_ay_bias=p_bias,
        max_abs_ay_bias=max_bias, measurement_var_pose_vy=r_pose,
        pose_vy_gate=pose_gate)


def errors(records, cfg, theta):
    result = []
    for samples, columns, dt, signs, alpha, teacher, valid in records:
        estimate, _ = estimate_dataset(
            samples, columns, dt, make_params(cfg, theta, dt),
            steer_scale=float(cfg["kf_steer_scale"]),
            steer_bias=float(cfg["kf_steer_bias"]), max_steer=float(cfg["kf_max_steer"]),
            imu_ema_alpha=alpha, imu_wz_sign=float(signs[0]), imu_ay_sign=float(signs[2]),
            use_pose_vy=bool(cfg["kf_pose_vy_enabled"]),
            pose_window_s=float(cfg["kf_pose_vy_window_s"]))
        result.extend(estimate[valid]-teacher[valid])
    return np.asarray(result)


def metrics(error):
    absolute = np.abs(error)
    return {"samples": int(len(error)), "mae_mps": float(absolute.mean()),
            "rmse_mps": float(np.sqrt(np.mean(error*error))),
            "p95_abs_mps": float(np.quantile(absolute, .95)),
            "max_abs_mps": float(absolute.max()), "bias_mps": float(error.mean())}


def score(records, cfg, theta):
    error = errors(records, cfg, theta)
    return float(np.mean(np.abs(error)) + np.quantile(np.abs(error), .95) + .3*abs(np.mean(error)))


def main():
    cfg = yaml.safe_load(CFG.read_text())["/**"]["ros__parameters"]
    records = load_records(cfg); train = [r for i, r in enumerate(records) if i % 5]
    heldout = [r for i, r in enumerate(records) if not i % 5]
    baseline = np.array([cfg["kf_q_vy"], cfg["kf_r_lateral_accel"], cfg["kf_r_pose_vy"],
                         cfg["kf_q_ay_bias"], cfg["kf_initial_p_ay_bias"],
                         cfg["kf_max_abs_ay_bias"], cfg["kf_pose_vy_gate"]], float)
    rng = np.random.default_rng(SEED); candidates = [baseline]
    for _ in range(160):
        candidates.append(np.array((10**rng.uniform(-3., -.4), 10**rng.uniform(-1.3, .9),
            10**rng.uniform(-2.2, -.1), 10**rng.uniform(-8., -3.5),
            10**rng.uniform(-3., -.3), rng.uniform(.1, 1.2), rng.uniform(.2, 1.8))))
    best = min(candidates, key=lambda theta: score(train, cfg, theta))
    names = ("kf_q_vy", "kf_r_lateral_accel", "kf_r_pose_vy", "kf_q_ay_bias",
             "kf_initial_p_ay_bias", "kf_max_abs_ay_bias", "kf_pose_vy_gate")
    report = {"method": "Pacejka EKF Q/R, ay-bias, MCL-vy gate regression",
              "selection": "train records only; heldout used only for final report",
              "records": len(records), "train_records": len(train), "heldout_records": len(heldout),
              "baseline": metrics(errors(heldout, cfg, baseline)),
              "fitted": metrics(errors(heldout, cfg, best)),
              "fitted_parameters": dict(zip(names, map(float, best)))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"params.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
