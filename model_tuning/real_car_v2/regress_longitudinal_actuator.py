#!/usr/bin/env python3
"""Identify the deployed speed-reference actuator parameters from bag data.

No CLI arguments are required: edit the constants below, then run this file.
The objective is recursive vx rollout error, not residual-MLP training loss.
Complete source sessions are kept in train/validation/test partitions.
"""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_DIR = ROOT / "model_tuning/data/real_car_v2_drive"
OUTPUT_DIR = ROOT / "model_tuning/results/longitudinal_actuator_regression"
CONFIG_PATH = ROOT / "config/params.yaml"
ROLLOUT_STEPS = 50                 # 1.0 s at 50 Hz
START_STRIDE = 5                   # overlapping causal rollouts
MAX_ROLLOUTS_PER_SESSION = 800
RANDOM_SEED = 31
# Keep False until the fitted candidate also passes full residual-rollout
# validation. Actuator-only vx improvement does not guarantee safer yaw tails.
UPDATE_CONFIG = False
SHOW_PLOTS = False

# These are source-session splits, identical to build_dataset.py.
VALIDATION_FILES = {"rosbag2_2026_08_08-16_54_33.npz"}
TEST_FILES = {
    "rosbag2_2026_08_10-21_45_57.npz",
    "rosbag2_2026_08_10-21_52_23.npz",
}

# Bounds: accel tau [s], brake tau [s], reference slew rate [m/s^2].
BOUNDS = ((0.002, 0.8), (0.002, 0.8), (0.25, 100.0))


def load_sessions():
    sessions = []
    for path in sorted(SOURCE_DIR.glob("*.npz")):
        z = np.load(path)
        names = {str(v): i for i, v in enumerate(z["columns"])}
        samples = np.asarray(z["samples"], float)
        for segment in np.unique(samples[:, names["bag_id"]].astype(int)):
            a = samples[samples[:, names["bag_id"]].astype(int) == segment]
            if len(a) < ROLLOUT_STEPS + 6:
                continue
            split = "test" if path.name in TEST_FILES else "validation" if path.name in VALIDATION_FILES else "train"
            sessions.append({
                "name": f"{path.stem}:segment{segment}", "source": path.name,
                "split": split, "vx": a[:, names["vx"]].astype(float),
                "cmd": a[:, names["speed_cmd"]].astype(float),
                "dt": float(z["dt"]),
            })
    return sessions


def choose_starts(session):
    n = len(session["vx"])
    starts = np.arange(0, n - ROLLOUT_STEPS, START_STRIDE)
    # Exclude nearly stationary windows: they mostly identify the acceleration
    # clamp and provide little information about tau/rate.
    informative = np.array([
        np.ptp(session["cmd"][i:i + ROLLOUT_STEPS]) > 0.08 or
        np.mean(np.abs(session["cmd"][i:i + ROLLOUT_STEPS] - session["vx"][i:i + ROLLOUT_STEPS])) > 0.12
        for i in starts
    ])
    starts = starts[informative]
    if len(starts) > MAX_ROLLOUTS_PER_SESSION:
        starts = starts[np.linspace(0, len(starts) - 1, MAX_ROLLOUTS_PER_SESSION).astype(int)]
    return starts


def predict_window(vx_gt, cmd, dt, params, kp, min_accel, max_accel):
    tau_accel, tau_brake, max_rate = params
    predicted = np.empty(len(cmd), float)
    predicted[0] = vx_gt[0]
    speed_reference = vx_gt[0]
    for k in range(len(cmd) - 1):
        tau = tau_accel if cmd[k] >= speed_reference else tau_brake
        reference_rate = np.clip((cmd[k] - speed_reference) / tau, -max_rate, max_rate)
        speed_reference += reference_rate * dt
        acceleration = np.clip(kp * (speed_reference - predicted[k]), min_accel, max_accel)
        predicted[k + 1] = max(0.0, predicted[k] + acceleration * dt)
    return predicted


def residual_vector(params, sessions, split, cfg):
    errors = []
    for session in sessions:
        if session["split"] != split:
            continue
        starts = session["starts"]
        if not len(starts):
            continue
        indices = starts[:, None] + np.arange(ROLLOUT_STEPS + 1)[None, :]
        gt = session["vx"][indices]
        cmd = session["cmd"][indices]
        pred = np.empty_like(gt)
        pred[:, 0] = gt[:, 0]
        speed_reference = gt[:, 0].copy()
        tau_accel, tau_brake, max_rate = params
        for k in range(ROLLOUT_STEPS):
            tau = np.where(cmd[:, k] >= speed_reference, tau_accel, tau_brake)
            reference_rate = np.clip((cmd[:, k] - speed_reference) / tau, -max_rate, max_rate)
            speed_reference += reference_rate * session["dt"]
            acceleration = np.clip(cfg["speed_servo_kp"] * (speed_reference - pred[:, k]),
                                   cfg["min_accel"], cfg["max_accel"])
            pred[:, k + 1] = np.maximum(0.0, pred[:, k] + acceleration * session["dt"])
        # Weight later samples more: the parameter values must remain good
        # recursively, rather than only matching the first transition.
        weights = np.linspace(0.35, 1.0, gt.shape[1])[None, :]
        errors.append((weights * (pred - gt)).ravel())
    return np.concatenate(errors) if errors else np.empty(0)


def robust_objective(params, sessions, cfg):
    e = residual_vector(params, sessions, "train", cfg)
    delta = 0.35
    ae = np.abs(e)
    huber = np.where(ae <= delta, 0.5 * e * e, delta * (ae - 0.5 * delta))
    return float(np.mean(huber))


def metrics(params, sessions, split, cfg):
    e = residual_vector(params, sessions, split, cfg)
    ae = np.abs(e)
    return {"samples": int(len(e)), "mae_mps": float(np.mean(ae)),
            "rmse_mps": float(np.sqrt(np.mean(e * e))),
            "p95_abs_mps": float(np.percentile(ae, 95)),
            "max_abs_mps": float(np.max(ae))}


def update_yaml(params):
    lines = CONFIG_PATH.read_text().splitlines()
    values = dict(zip(("speed_reference_accel_time_constant", "speed_reference_brake_time_constant",
                       "actuator_max_speed_reference_rate"), params))
    found = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in values.items():
            if stripped.startswith(key + ":"):
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}{key}: {value:.9g}  # bag/session-disjoint vx rollout regression"
                found.add(key)
    if found != set(values):
        raise RuntimeError(f"could not update YAML keys: {set(values) - found}")
    CONFIG_PATH.write_text("\n".join(lines) + "\n")


def plot_examples(sessions, old, fitted, cfg):
    selected = []
    for split in ("validation", "test"):
        candidates = [s for s in sessions if s["split"] == split and len(s["starts"])]
        if candidates:
            selected.append(candidates[0])
    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 4.2 * len(selected)), squeeze=False)
    for ax, session in zip(axes[:, 0], selected):
        start = int(session["starts"][len(session["starts"]) // 2])
        sl = slice(start, start + ROLLOUT_STEPS + 1)
        gt, cmd = session["vx"][sl], session["cmd"][sl]
        t = np.arange(len(gt)) * session["dt"]
        ax.plot(t, gt, color="black", lw=2, label="GT odom vx")
        ax.plot(t, cmd, color="tab:gray", ls=":", label="/drive speed command")
        ax.plot(t, predict_window(gt, cmd, session["dt"], old, cfg["speed_servo_kp"], cfg["min_accel"], cfg["max_accel"]), ls="--", label="previous parameters")
        ax.plot(t, predict_window(gt, cmd, session["dt"], fitted, cfg["speed_servo_kp"], cfg["min_accel"], cfg["max_accel"]), color="tab:red", label="fitted parameters")
        ax.set(title=f"{session['split']}: {session['name']}", xlabel="rollout time [s]", ylabel="speed [m/s]")
        ax.grid(alpha=.3); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "rollout_comparison.png", dpi=180)
    if SHOW_PLOTS: plt.show()
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["/**"]["ros__parameters"]
    sessions = load_sessions()
    for session in sessions:
        session["starts"] = choose_starts(session)
    old = np.array([cfg["speed_reference_accel_time_constant"],
                    cfg["speed_reference_brake_time_constant"],
                    cfg["actuator_max_speed_reference_rate"]], float)
    result = differential_evolution(lambda p: robust_objective(p, sessions, cfg), BOUNDS,
                                    seed=RANDOM_SEED, popsize=12, maxiter=80, tol=2e-5,
                                    polish=False, workers=1, updating="immediate")
    refined = minimize(lambda p: robust_objective(p, sessions, cfg), result.x,
                       method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-6, "fatol": 1e-9})
    fitted = np.clip(refined.x, np.array(BOUNDS)[:, 0], np.array(BOUNDS)[:, 1])
    report = {
        "parameter_order": ["speed_reference_accel_time_constant", "speed_reference_brake_time_constant", "actuator_max_speed_reference_rate"],
        "previous": old.tolist(), "fitted": fitted.tolist(),
        "fixed_speed_servo_kp": float(cfg["speed_servo_kp"]),
        "objective": "1.0 s recursive odom-vx rollout, train Huber; source-session-disjoint validation",
        "metrics_previous": {s: metrics(old, sessions, s, cfg) for s in ("train", "validation", "test")},
        "metrics_fitted": {s: metrics(fitted, sessions, s, cfg) for s in ("train", "validation", "test")},
        "sessions": [{"name": s["name"], "split": s["split"], "rollouts": int(len(s["starts"]))} for s in sessions],
    }
    (OUTPUT_DIR / "regression.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_examples(sessions, old, fitted, cfg)
    if UPDATE_CONFIG:
        update_yaml(fitted)
    print(json.dumps(report, indent=2))
    print(f"plot: {OUTPUT_DIR / 'rollout_comparison.png'}")
    print(f"config updated: {UPDATE_CONFIG}")


if __name__ == "__main__":
    main()
