#!/usr/bin/env python3
"""Fair callback/KF-GT comparison of E2E and deployed classic+residual models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import train_e2e_kf_predictor as e2e
from contract import ClassicModelParameters, Contract

DATA = ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
E2E_RESULT = ROOT / "model_tuning/results/e2e_kf_predictor"
RESIDUAL_BINARY = ROOT / "config/dynamic_40ms_residual_servo_lag.bin"
OUTPUT = ROOT / "model_tuning/results/e2e_kf_vs_residual"
MAX_EVALUATION_ANCHORS = 0
USE_PLOT = True


def load_residual(path):
    raw = np.fromfile(path, dtype="<f4")
    input_dim = {3563: 20, 3695: 22}.get(len(raw))
    if input_dim is None:
        raise RuntimeError(f"unsupported residual binary length {len(raw)}: {path}")
    offset = 0
    def take(count):
        nonlocal offset
        value = raw[offset:offset + count]; offset += count
        return value
    return {"w1": take(64*input_dim).reshape(64, input_dim), "b1": take(64),
            "w2": take(2048).reshape(32, 64), "b2": take(32),
            "w3": take(96).reshape(3, 32), "b3": take(3),
            "mean": take(input_dim), "std": take(input_dim), "input_dim": input_dim}


def residual_net(features, weights):
    value = (features - weights["mean"]) / weights["std"]
    value = np.maximum(value @ weights["w1"].T + weights["b1"], 0.)
    value = np.maximum(value @ weights["w2"].T + weights["b2"], 0.)
    return value @ weights["w3"].T + weights["b3"]


def rollout_residual(data, weights, cfg, mlp_cfg):
    p = ClassicModelParameters.from_mapping(cfg)
    c = Contract.from_parameters(p, dt=e2e.MODEL_DT_S)
    lf, lr, mass = (float(cfg[name]) for name in ("l_f", "l_r", "mass"))
    wheelbase = lf + lr
    front_load = mass * 9.81 * lr / wheelbase
    rear_load = mass * 9.81 * lf / wheelbase
    limit = np.array((mlp_cfg["mlp_max_residual_ax"], mlp_cfg["mlp_max_residual_ay"],
                      mlp_cfg["mlp_max_residual_yaw_accel"]), float)
    pose = data["initial_pose"].copy(); state = data["initial_state"].copy()
    applied = data["actuator"][:, 0].copy(); speed_reference = data["actuator"][:, 1].copy()
    history = data["history"].reshape(-1, 5, 2).copy()
    acceleration = data["imu"][:, [1, 2]].copy()
    poses, states, accelerations = [], [], []
    dt = e2e.MODEL_DT_S
    for k in range(data["commands"].shape[1]):
        command = data["commands"][:, k]
        if k:
            history = np.concatenate((history[:, 1:], command[:, None]), axis=1)
        current = state.copy(); previous_steer = history[:, -2, 0]
        steer_target = np.clip(
            command[:, 0], -p.max_steer, p.max_steer)
        applied = np.clip(applied + np.clip((steer_target-applied)/max(p.steer_tau, 1e-3),
                                            -p.max_steer_rate, p.max_steer_rate)*dt,
                          -p.max_steer, p.max_steer)
        tau = np.where(command[:, 1] >= speed_reference, p.speed_accel_tau, p.speed_brake_tau)
        speed_reference += np.clip((command[:, 1]-speed_reference)/np.maximum(tau, 1e-3),
                                   -p.v_ref_slew_rate_max, p.v_ref_slew_rate_max)*dt
        vx, vy, yaw_rate = state.T
        ax = np.clip(p.speed_kp*(speed_reference-vx), p.ax_min, p.ax_max)
        safe_vx = np.maximum(np.abs(vx), .5)
        alpha_f = applied - np.arctan2(vy + lf*yaw_rate, safe_vx)
        alpha_r = -np.arctan2(vy - lr*yaw_rate, safe_vx)
        bf, br = p.B_f*alpha_f, p.B_r*alpha_r
        front_inner = bf - p.E_f*(bf-np.arctan(bf))
        rear_inner = br - p.E_r*(br-np.arctan(br))
        fyf = front_load*p.D_f*np.sin(p.C_f*np.arctan(front_inner))
        fyr = rear_load*p.D_r*np.sin(p.C_r*np.arctan(rear_inner))
        ay = (fyf*np.cos(applied)+fyr)/mass
        yaw_accel = (lf*fyf*np.cos(applied)-lr*fyr)/p.Iz
        base_next = np.c_[vx+(ax+vy*yaw_rate)*dt,
                          vy+(ay-vx*yaw_rate)*dt, yaw_rate+yaw_accel*dt]
        feature = np.c_[current, command, applied, command[:, 0]-previous_steer,
                        base_next, history.reshape(len(state), -1)]
        if weights["input_dim"] == 22:
            feature = np.c_[feature, acceleration]
        residual = np.clip(residual_net(feature, weights), -limit, limit)
        state = base_next + residual*dt
        acceleration = np.c_[ax+residual[:, 0], ay+residual[:, 1]]
        yaw = pose[:, 2]
        pose = np.c_[pose[:, 0]+(state[:, 0]*np.cos(yaw)-state[:, 1]*np.sin(yaw))*dt,
                     pose[:, 1]+(state[:, 0]*np.sin(yaw)+state[:, 1]*np.cos(yaw))*dt,
                     pose[:, 2]+state[:, 2]*dt]
        poses.append(pose.copy()); states.append(state.copy()); accelerations.append(acceleration.copy())
    return np.stack(poses, 1), np.stack(states, 1), np.stack(accelerations, 1)


def acceleration_from_state(initial_state, states):
    previous = np.concatenate((initial_state[:, None], states[:, :-1]), axis=1)
    derivative = (states - previous) / e2e.MODEL_DT_S
    return np.stack((derivative[:, :, 0]-previous[:, :, 1]*previous[:, :, 2],
                     derivative[:, :, 1]+previous[:, :, 0]*previous[:, :, 2]), axis=2)


def metrics(pose, state, target, initial_state):
    pose_error = pose-target[:, :, :3]
    pose_error[:, :, 2] = np.arctan2(np.sin(pose_error[:, :, 2]), np.cos(pose_error[:, :, 2]))
    state_error = state-target[:, :, 3:]
    distance = np.linalg.norm(pose_error[:, :, :2], axis=2)
    final_distance = distance[:, -1]
    gt_accel = acceleration_from_state(initial_state, target[:, :, 3:])
    pred_accel = acceleration_from_state(initial_state, state)
    return {
        "one_step_state_rmse": dict(zip(e2e.STATE_NAMES, np.sqrt(np.mean(state_error[:, 0]**2, 0)).tolist())),
        "rollout_state_rmse": dict(zip(e2e.STATE_NAMES, np.sqrt(np.mean(state_error**2, (0, 1))).tolist())),
        "rollout_pose_rmse": dict(zip(e2e.POSE_NAMES, np.sqrt(np.mean(pose_error**2, (0, 1))).tolist())),
        "rollout_acceleration_rmse": {"ax": float(np.sqrt(np.mean((pred_accel[:, :, 0]-gt_accel[:, :, 0])**2))),
                                      "ay": float(np.sqrt(np.mean((pred_accel[:, :, 1]-gt_accel[:, :, 1])**2)))},
        "trajectory_error_mean_m": float(distance.mean()),
        "trajectory_error_p95_m": float(np.quantile(distance, .95)),
        "final_error_mean_m": float(final_distance.mean()),
        "final_error_p95_m": float(np.quantile(final_distance, .95)),
    }, final_distance, pred_accel, gt_accel


def plot(data, predictions, target, final_score, output):
    order = np.argsort(np.minimum(final_score["e2e"], final_score["residual"]))
    selected = (order[0], order[int(.95*(len(order)-1))], order[-1])
    names = ("best", "p95", "worst"); time = e2e.MODEL_DT_S*np.arange(1, target.shape[1]+1)
    fig, axes = plt.subplots(3, 7, figsize=(28, 13), constrained_layout=True)
    colors = {"e2e": "C3", "residual": "C0"}
    for row, (index, case) in enumerate(zip(selected, names)):
        axes[row, 0].plot(target[index, :, 0], target[index, :, 1], "k-", label="KF GT")
        for model in ("residual", "e2e"):
            axes[row, 0].plot(predictions[model][0][index, :, 0], predictions[model][0][index, :, 1],
                              "--", color=colors[model], label=model)
        axes[row, 0].set_title(f"{case} trajectory"); axes[row, 0].axis("equal")
        for column, state_name in enumerate(e2e.STATE_NAMES, 1):
            j = column-1; axes[row, column].plot(time, target[index, :, 3+j], "k-")
            for model in ("residual", "e2e"):
                axes[row, column].plot(time, predictions[model][1][index, :, j], "--", color=colors[model])
            axes[row, column].set_title(state_name)
        axes[row, 4].plot(time, target[index, :, 2]-data["initial_pose"][index, 2], "k-")
        for model in ("residual", "e2e"):
            axes[row, 4].plot(time, predictions[model][0][index, :, 2]-data["initial_pose"][index, 2],
                              "--", color=colors[model])
        axes[row, 4].set_title("yaw change")
        for column, accel_name in ((5, "ax"), (6, "ay")):
            j = column-5; axes[row, column].plot(time, predictions["gt_acceleration"][index, :, j], "k-")
            for model in ("residual", "e2e"):
                axes[row, column].plot(time, predictions[model][2][index, :, j], "--", color=colors[model])
            axes[row, column].set_title(accel_name)
        for ax in axes[row]: ax.grid(True, alpha=.25)
    axes[0, 0].legend(); fig.suptitle("Same callback anchors / same future KF GT")
    fig.savefig(output, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--e2e-result", type=Path, default=E2E_RESULT)
    parser.add_argument("--residual", type=Path, default=RESIDUAL_BINARY)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--max-samples", type=int, default=MAX_EVALUATION_ANCHORS)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.data.glob("*.npz")) if args.data.is_dir() else [args.data]
    horizon = int(round(e2e.ROLLOUT_HORIZON_S/e2e.MODEL_DT_S))
    bags = [e2e.load_bag(path, horizon) for path in paths]
    bags = [bag for bag in bags if len(bag["initial_state"])]
    _, _, test_ids = e2e.split_bags(len(bags)); data = e2e.concatenate(bags, test_ids)
    if args.max_samples and len(data["initial_state"]) > args.max_samples:
        rng = np.random.default_rng(e2e.SEED); choice = rng.choice(len(data["initial_state"]), args.max_samples, False)
        for key in data: data[key] = data[key][choice]

    normalization = np.load(args.e2e_result/"normalization.npz")
    model = e2e.E2EKFTransition(normalization["input_mean"], normalization["input_std"],
                                normalization["derivative_mean"], normalization["derivative_std"])
    checkpoint = torch.load(args.e2e_result/"model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.no_grad():
        e2e_pose, e2e_state = e2e.rollout(model, *e2e.tensors(data)[:2],
                                          e2e.tensors(data)[3], e2e.tensors(data)[4])
    e2e_pose, e2e_state = e2e_pose.numpy(), e2e_state.numpy()
    config = yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    mlp_config = yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    residual_pose, residual_state, residual_accel = rollout_residual(
        data, load_residual(args.residual), config, mlp_config)
    target = data["target"]
    e2e_metrics, e2e_final, e2e_accel, gt_accel = metrics(
        e2e_pose, e2e_state, target, data["initial_state"])
    residual_metrics, residual_final, _, _ = metrics(
        residual_pose, residual_state, target, data["initial_state"])
    report = {"contract": {"anchors": int(len(target)), "bags": [paths[k].name for k in test_ids],
                            "dt_s": e2e.MODEL_DT_S, "horizon_s": e2e.ROLLOUT_HORIZON_S,
                            "gt": "Step-1 causal KF fields interpolated at identical callback-relative knots",
                            "residual_binary": str(args.residual)},
              "e2e": e2e_metrics, "classic_plus_residual": residual_metrics}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    predictions = {"e2e": (e2e_pose, e2e_state, e2e_accel),
                   "residual": (residual_pose, residual_state, residual_accel),
                   "gt_acceleration": gt_accel}
    fig = plot(data, predictions, target, {"e2e": e2e_final, "residual": residual_final},
               args.out/"comparison.png")
    print(json.dumps(report, indent=2)); print(f"saved: {args.out}")
    if USE_PLOT and not args.no_show: plt.show()
    else: plt.close(fig)


if __name__ == "__main__":
    main()
