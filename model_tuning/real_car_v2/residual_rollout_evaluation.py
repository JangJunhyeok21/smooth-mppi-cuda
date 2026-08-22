#!/usr/bin/env python3
"""Evaluate Step 6 directly from Step-1 callback archives."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from callback_training_data import load_callback_archives
from contract import (ClassicModelParameters, Contract, actuator_step,
                      longitudinal_actuator_step, low_speed_gate)

DATA = ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
PARAMS = ROOT / "model_tuning/results/dynamic_40ms_regression/params.json"
RESULT_PATH = ROOT / "model_tuning/results/step_6_residual/recursive"
OUTPUT_PATH = RESULT_PATH / "rollout_metrics.json"
DT = 0.04


def load_network(path):
    values = np.fromfile(path, dtype="<f4")
    input_dim = {3563: 20, 3695: 22}.get(len(values))
    if input_dim is None:
        raise ValueError(f"unexpected residual binary size: {len(values)}")
    offset = 0
    def take(count):
        nonlocal offset
        result = values[offset:offset + count]
        offset += count
        return result
    return (take(64 * input_dim).reshape(64, input_dim), take(64),
            take(2048).reshape(32, 64), take(32),
            take(96).reshape(3, 32), take(3),
            take(input_dim), take(input_dim))


def forward(features, weights):
    w1, b1, w2, b2, w3, b3, mean, std = weights
    hidden = np.maximum(((features - mean) / std) @ w1.T + b1, 0.0)
    hidden = np.maximum(hidden @ w2.T + b2, 0.0)
    return hidden @ w3.T + b3


def statistics(values):
    values = np.asarray(values, float)
    return {"rmse": float(np.sqrt(np.mean(values ** 2))),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)),
            "max": float(np.max(values))}


def configured_parameters(classic_path):
    parameters = ClassicModelParameters.from_yaml(ROOT / "config/params.yaml")
    path = Path(classic_path)
    if path.exists():
        fitted = json.loads(path.read_text()).get("expanded_fitted", {})
        updates = {
            key: float(value) for key, value in fitted.items()
            if hasattr(parameters, key)}
        if "I_z" in fitted:
            updates["Iz"] = float(fitted["I_z"])
        parameters = replace(parameters, **updates)
    return parameters


def rollout(data, indices, weights, parameters, disable_mlp, residual_limit):
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    contract = Contract.from_parameters(parameters, dt=DT)
    state = data["initial_state"][indices].astype(float).copy()
    pose = data["initial_pose"][indices].astype(float).copy()
    applied = data["actuator"][indices, 0].astype(float).copy()
    speed_reference = data["actuator"][indices, 1].astype(float).copy()
    history = data["history"][indices].astype(float).reshape(-1, 5, 2).copy()
    acceleration = data["imu"][indices].astype(float).copy()
    commands = data["commands"][indices]
    horizon = commands.shape[1]
    state_trace = [state.copy()]
    pose_trace = [pose.copy()]
    acceleration_trace = [acceleration.copy()]
    lf, lr, mass, inertia = (float(cfg["l_f"]), float(cfg["l_r"]),
                             float(cfg["mass"]), parameters.Iz)
    front_load = mass * 9.81 * lr / (lf + lr)
    rear_load = mass * 9.81 * lf / (lf + lr)
    for step in range(horizon):
        command = commands[:, step]
        if step:
            history = np.concatenate((history[:, 1:], command[:, None]), axis=1)
        previous_steer = history[:, -2, 0]
        current_state = state.copy()
        for row in range(len(indices)):
            applied[row], _ = actuator_step(
                applied[row], command[row, 0], command[row, 1], state[row, 0], contract)
            speed_reference[row], base_ax = longitudinal_actuator_step(
                speed_reference[row], command[row, 1], state[row, 0], contract)
            vx, vy, yaw_rate = state[row]
            safe_vx = max(abs(vx), .5)
            alpha_front = applied[row] - np.arctan2(vy + lf * yaw_rate, safe_vx)
            alpha_rear = -np.arctan2(vy - lr * yaw_rate, safe_vx)
            front_term = parameters.B_f * alpha_front
            rear_term = parameters.B_r * alpha_rear
            front_force = front_load * parameters.D_f * np.sin(parameters.C_f * np.arctan(
                front_term - parameters.E_f * (front_term - np.arctan(front_term))))
            rear_force = rear_load * parameters.D_r * np.sin(parameters.C_r * np.arctan(
                rear_term - parameters.E_r * (rear_term - np.arctan(rear_term))))
            blend=low_speed_gate(vx,contract)
            dynamic_ay = (front_force * np.cos(applied[row]) + rear_force) / mass
            dynamic_yaw_accel = (lf * front_force * np.cos(applied[row]) - lr * rear_force) / inertia
            kinematic_yaw_rate=vx*np.tan(applied[row])/max(lf+lr,1e-6)
            base_ay=blend*dynamic_ay+(1.-blend)*(vx*yaw_rate-vy/.1)
            yaw_accel=blend*dynamic_yaw_accel+(1.-blend)*(kinematic_yaw_rate-yaw_rate)/.1
            state[row] = (vx + (base_ax + vy * yaw_rate) * DT,
                          vy + (base_ay - vx * yaw_rate) * DT,
                          yaw_rate + yaw_accel * DT)
            acceleration[row] = (base_ax, base_ay)
        features = np.concatenate((current_state, command, applied[:, None],
            (command[:, 0] - previous_steer)[:, None], state,
            history.reshape(len(indices), -1), acceleration_trace[-1]), axis=1)
        residual = np.zeros_like(state) if disable_mlp else np.clip(
            forward(features, weights), -residual_limit, residual_limit)
        residual *= low_speed_gate(current_state[:,0],contract)[:,None]
        state += residual * DT
        acceleration += residual[:, :2]
        yaw = pose[:, 2]
        pose = np.column_stack((
            pose[:, 0] + (state[:, 0] * np.cos(yaw) - state[:, 1] * np.sin(yaw)) * DT,
            pose[:, 1] + (state[:, 0] * np.sin(yaw) + state[:, 1] * np.cos(yaw)) * DT,
            pose[:, 2] + state[:, 2] * DT))
        state_trace.append(state.copy()); pose_trace.append(pose.copy())
        acceleration_trace.append(acceleration.copy())
    return (np.stack(pose_trace, 1), np.stack(state_trace, 1),
            np.stack(acceleration_trace, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", default=str(RESULT_PATH))
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--classic-params", default=str(PARAMS))
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--disable-mlp", action="store_true")
    parser.add_argument("--bag-id", type=int)
    args = parser.parse_args()
    if args.horizon_steps < 1:
        parser.error("--horizon-steps must be positive")
    data = load_callback_archives(args.data, model_dt=DT, horizon=args.horizon_steps)
    bag_names = np.unique(data["bag_name"])
    bag_id_by_name = {name: index for index, name in enumerate(bag_names)}
    bag_ids = np.asarray([bag_id_by_name[name] for name in data["bag_name"]], int)
    weights = load_network(Path(args.result) / "dynamic_40ms_residual.bin")
    parameters = configured_parameters(args.classic_params)
    mlp_cfg = yaml.safe_load((ROOT / "config/MLP_params.yaml").read_text())["/**"]["ros__parameters"]
    residual_limit = np.asarray((mlp_cfg["mlp_max_residual_ax"],
                                 mlp_cfg["mlp_max_residual_ay"],
                                 mlp_cfg["mlp_max_residual_yaw_accel"]), float)
    selected = np.flatnonzero(bag_ids == args.bag_id) if args.bag_id is not None else np.arange(len(bag_ids))[::5]
    if not len(selected):
        raise RuntimeError(f"no callback windows for bag_id={args.bag_id}")
    predicted_pose, predicted_state, predicted_accel = rollout(
        data, selected, weights, parameters, args.disable_mlp, residual_limit)
    gt_pose = np.concatenate((data["initial_pose"][selected, None],
                              data["target_pose"][selected]), axis=1)
    gt_state = np.concatenate((data["initial_state"][selected, None],
                               data["target_state"][selected]), axis=1)
    # Future raw IMU is not stored in the callback grid. Derive body acceleration
    # consistently from the actual future KF state for rollout diagnostics.
    dv = np.diff(gt_state[:, :, :2], axis=1) / DT
    future = gt_state[:, 1:]
    gt_future_accel = np.stack((dv[:, :, 0] - future[:, :, 1] * future[:, :, 2],
                                dv[:, :, 1] + future[:, :, 0] * future[:, :, 2]), axis=2)
    gt_accel = np.concatenate((data["imu"][selected, None], gt_future_accel), axis=1)
    xy = np.linalg.norm(predicted_pose[:, -1, :2] - gt_pose[:, -1, :2], axis=1)
    yaw = np.abs(np.arctan2(np.sin(predicted_pose[:, -1, 2] - gt_pose[:, -1, 2]),
                            np.cos(predicted_pose[:, -1, 2] - gt_pose[:, -1, 2])))
    state_error = np.abs(predicted_state[:, -1] - gt_state[:, -1])
    final = {"trajectory_m": statistics(xy), "yaw_rad": statistics(yaw),
             "vx_mps": statistics(state_error[:, 0]),
             "vy_mps": statistics(state_error[:, 1]),
             "yaw_rate_rps": statistics(state_error[:, 2])}
    report = {"evaluation_contract": {"source": "direct Step-1 callback archives",
        "horizon_steps": args.horizon_steps, "horizon_s": args.horizon_steps * DT,
        "state_target": "future actual MPPI-model KF", "pose_target": "future MCL pose"},
        "selected": {"windows": len(selected), "final_horizon": final}}
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    start_time = data["anchor_time"][selected] - np.asarray([
        np.min(data["anchor_time"][bag_ids == bag]) for bag in bag_ids[selected]])
    np.savez_compressed(output.with_suffix(".npz"), starts=selected,
        bag_ids=bag_ids[selected], start_time_s=start_time,
        predicted=np.concatenate((predicted_pose, predicted_state), axis=2),
        ground_truth=np.concatenate((gt_pose, gt_state), axis=2),
        predicted_acceleration=predicted_accel,
        ground_truth_acceleration=gt_accel)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
