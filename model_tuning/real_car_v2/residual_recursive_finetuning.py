#!/usr/bin/env python3
"""Recursive residual fine-tuning directly on Step-1 callback archives."""

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from callback_training_data import load_callback_archives
from contract import ClassicModelParameters
from residual_mlp_training import Net

DATA = ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
PARAMS = Path(os.environ.get("DYNAMIC_CLASSIC_PARAMS",
    ROOT / "model_tuning/results/dynamic_40ms_regression/params.json"))
INITIAL_MODEL_PATH = ROOT / "model_tuning/results/step_6_residual/one_step"
OUTPUT_PATH = ROOT / "model_tuning/results/step_6_residual/recursive"
EPOCHS = 100
SEED = 31
DT = 0.04
STATE_WEIGHTS = tuple(float(value) for value in
    os.environ.get("RECURSIVE_STATE_WEIGHTS", "2,5,6").split(","))
if len(STATE_WEIGHTS) != 3:
    raise ValueError("RECURSIVE_STATE_WEIGHTS must contain vx,vy,yaw_rate")
POSITION_WEIGHT = float(os.environ.get("RECURSIVE_POSITION_WEIGHT", "5"))
YAW_WEIGHT = float(os.environ.get("RECURSIVE_YAW_WEIGHT", "3"))
TAIL_POSITION_WEIGHT = float(os.environ.get("RECURSIVE_TAIL_POSITION_WEIGHT", "10"))
TAIL_YAW_WEIGHT = float(os.environ.get("RECURSIVE_TAIL_YAW_WEIGHT", "6"))
RESIDUAL_WEIGHT = float(os.environ.get("RECURSIVE_RESIDUAL_WEIGHT", "1e-4"))
TAIL_FRACTION = float(os.environ.get("RECURSIVE_TAIL_FRACTION", ".10"))
CHECKPOINT_P95_WEIGHT = float(os.environ.get(
    "RECURSIVE_CHECKPOINT_P95_WEIGHT", "2"))
LEARNING_RATE = float(os.environ.get("RECURSIVE_LEARNING_RATE", "2e-5"))
BATCHES_PER_EPOCH = max(1, int(os.environ.get("RECURSIVE_BATCHES_PER_EPOCH", "32")))


def load_normalization(path):
    values = np.fromfile(path, dtype="<f4")
    return values[-44:-22].copy(), values[-22:].copy()


def configured_parameters():
    parameters = ClassicModelParameters.from_yaml(ROOT / "config/params.yaml")
    if PARAMS.exists():
        fitted = json.loads(PARAMS.read_text()).get("expanded_fitted", {})
        updates = {
            key: float(value) for key, value in fitted.items()
            if hasattr(parameters, key)}
        if "I_z" in fitted:
            updates["Iz"] = float(fitted["I_z"])
        parameters = replace(parameters, **updates)
    return parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial", nargs="?", default=str(INITIAL_MODEL_PATH))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--horizon-steps", type=int, default=30)
    args = parser.parse_args()
    if args.horizon_steps < 1:
        parser.error("--horizon-steps must be positive")
    if not 0.0 < TAIL_FRACTION <= 1.0:
        parser.error("RECURSIVE_TAIL_FRACTION must lie in (0, 1]")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_callback_archives(args.data, model_dt=DT,
                                  horizon=args.horizon_steps)
    train = np.flatnonzero(data["split"] == 0)
    validation = np.flatnonzero(data["split"] == 1)
    if not len(train):
        raise RuntimeError("Step 1 callback data has no training windows")
    validation_source = "held-out validation split"
    if not len(validation):
        validation = train.copy()
        validation_source = "training split fallback (no validation bags matched)"
        print("WARNING: recursive validation split is empty; using training "
              "windows for checkpoint selection.", flush=True)
    print(f"recursive windows: train={len(train)}, validation={len(validation)} "
          f"({validation_source})", flush=True)
    validation = validation[::max(1, len(validation) // 256)][:256]
    parameters = configured_parameters()
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    mlp_cfg = cfg
    mean, std = load_normalization(Path(args.initial) / "dynamic_40ms_residual.bin")
    network = Net()
    network.load_state_dict(torch.load(Path(args.initial) / "model.pt",
                                       map_location="cpu", weights_only=True))
    network.to(device)
    initial_state = torch.as_tensor(data["initial_state"], device=device,
                                    dtype=torch.float32)
    initial_pose = torch.as_tensor(data["initial_pose"], device=device,
                                   dtype=torch.float32)
    target_state = torch.as_tensor(data["target_state"], device=device,
                                   dtype=torch.float32)
    target_pose = torch.as_tensor(data["target_pose"], device=device,
                                  dtype=torch.float32)
    commands = torch.as_tensor(data["commands"], device=device,
                               dtype=torch.float32)
    history_all = torch.as_tensor(data["history"], device=device,
                                  dtype=torch.float32)
    imu = torch.as_tensor(data["imu"], device=device, dtype=torch.float32)
    actuator = torch.as_tensor(data["actuator"], device=device,
                               dtype=torch.float32)
    mean = torch.as_tensor(mean, device=device); std = torch.as_tensor(std, device=device)
    residual_limit = torch.tensor((mlp_cfg["mlp_max_residual_ax"],
        mlp_cfg["mlp_max_residual_ay"], mlp_cfg["mlp_max_residual_yaw_accel"]),
        device=device)
    lf, lr, mass, inertia = (float(cfg["l_f"]), float(cfg["l_r"]),
                             float(cfg["mass"]), parameters.Iz)
    front_load = mass * 9.81 * lr / (lf + lr)
    rear_load = mass * 9.81 * lf / (lf + lr)

    def rollout(indices):
        ids = torch.as_tensor(indices, device=device, dtype=torch.long)
        state = initial_state[ids].clone()  # actual runtime KF input, never GT-adjusted
        pose = initial_pose[ids].clone()
        applied = actuator[ids, 0].clone()
        speed_reference = actuator[ids, 1].clone()
        history = history_all[ids].reshape(-1, 5, 2).clone()
        acceleration = imu[ids].clone()
        state_trace, pose_trace, residual_trace = [], [], []
        for step in range(args.horizon_steps):
            command = commands[ids, step]
            if step:
                history = torch.cat((history[:, 1:], command[:, None]), 1)
            previous_steer = history[:, -2, 0]
            current_state = state
            steer_target = torch.clamp(
                                       parameters.steer_scale * command[:, 0]
                                       + parameters.steer_bias,
                                       -parameters.max_steer,
                                       parameters.max_steer)
            steer_rate = torch.clamp((steer_target - applied) /
                                     max(parameters.steer_tau, 1e-3),
                                     -parameters.max_steer_rate,
                                     parameters.max_steer_rate)
            applied = torch.clamp(applied + steer_rate * DT,
                                  -parameters.max_steer,
                                  parameters.max_steer)
            tau = torch.where(command[:, 1] >= speed_reference,
                              torch.full_like(speed_reference, parameters.speed_accel_tau),
                              torch.full_like(speed_reference, parameters.speed_brake_tau))
            speed_reference = speed_reference + torch.clamp(
                (command[:, 1] - speed_reference) / tau,
                -parameters.v_ref_slew_rate_max,
                parameters.v_ref_slew_rate_max) * DT
            vx, vy, yaw_rate = state.unbind(1)
            base_ax = torch.clamp(parameters.speed_kp * (speed_reference - vx),
                                  parameters.ax_min, parameters.ax_max)
            safe_vx = torch.clamp(torch.abs(vx), min=.5)
            alpha_front = applied - torch.atan2(vy + lf * yaw_rate, safe_vx)
            alpha_rear = -torch.atan2(vy - lr * yaw_rate, safe_vx)
            front_term = parameters.B_f * alpha_front
            rear_term = parameters.B_r * alpha_rear
            front_force = front_load * parameters.D_f * torch.sin(parameters.C_f * torch.atan(
                front_term - parameters.E_f * (front_term - torch.atan(front_term))))
            rear_force = rear_load * parameters.D_r * torch.sin(parameters.C_r * torch.atan(
                rear_term - parameters.E_r * (rear_term - torch.atan(rear_term))))
            blend_input=torch.clamp((torch.abs(vx)-.2)/.3,0.,1.)
            dynamic_blend=blend_input*blend_input*(3.-2.*blend_input)
            dynamic_ay = (front_force * torch.cos(applied) + rear_force) / mass
            dynamic_yaw_accel = (lf * front_force * torch.cos(applied) - lr * rear_force) / inertia
            kinematic_yaw_rate=vx*torch.tan(applied)/max(lf+lr,1e-6)
            base_ay=dynamic_blend*dynamic_ay+(1.-dynamic_blend)*(vx*yaw_rate-vy/.1)
            yaw_accel=dynamic_blend*dynamic_yaw_accel+(1.-dynamic_blend)*(kinematic_yaw_rate-yaw_rate)/.1
            classic_next = torch.stack((vx + (base_ax + vy * yaw_rate) * DT,
                vy + (base_ay - vx * yaw_rate) * DT,
                yaw_rate + yaw_accel * DT), 1)
            features = torch.cat((current_state, command, applied[:, None],
                (command[:, 0] - previous_steer)[:, None], classic_next,
                history.reshape(len(ids), -1), acceleration), 1)
            residual = torch.clamp(network((features - mean) / std),
                                   -residual_limit, residual_limit)
            residual=residual*dynamic_blend[:,None]
            state = classic_next + residual * DT
            acceleration = torch.stack((base_ax + residual[:, 0],
                                        base_ay + residual[:, 1]), 1)
            yaw = pose[:, 2]
            pose = torch.stack((pose[:, 0] + (state[:, 0] * torch.cos(yaw)
                    - state[:, 1] * torch.sin(yaw)) * DT,
                pose[:, 1] + (state[:, 0] * torch.sin(yaw)
                    + state[:, 1] * torch.cos(yaw)) * DT,
                pose[:, 2] + state[:, 2] * DT), 1)
            state_trace.append(state); pose_trace.append(pose); residual_trace.append(residual)
        return (torch.stack(state_trace, 1), torch.stack(pose_trace, 1),
                torch.stack(residual_trace, 1), ids)

    state_weights = torch.tensor(STATE_WEIGHTS, device=device)
    def loss(indices, per_window=False):
        predicted_state, predicted_pose, residual, ids = rollout(indices)
        state_error = torch.nn.functional.smooth_l1_loss(
            predicted_state, target_state[ids], reduction="none")
        state_loss = (state_error * state_weights).mean()
        position_error = torch.linalg.vector_norm(
            predicted_pose[:, :, :2] - target_pose[ids, :, :2], dim=2)
        yaw_delta = predicted_pose[:, :, 2] - target_pose[ids, :, 2]
        yaw_error = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
        count = max(1, int(np.ceil(len(indices) * TAIL_FRACTION)))
        endpoint_position = position_error[:, -1]
        endpoint_yaw = yaw_error[:, -1]
        value = (state_loss + POSITION_WEIGHT * position_error.mean()
                 + YAW_WEIGHT * yaw_error.mean()
                 + TAIL_POSITION_WEIGHT * torch.topk(endpoint_position, count).values.mean()
                 + TAIL_YAW_WEIGHT * torch.topk(endpoint_yaw, count).values.mean()
                 + RESIDUAL_WEIGHT * residual.square().mean())
        score = position_error.mean(1) + .5 * endpoint_position + .5 * yaw_error.mean(1)
        return (value, score) if per_window else value

    optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE,
                                  weight_decay=1e-5)
    best = (float("inf"), {key: value.detach().cpu().clone()
                           for key, value in network.state_dict().items()}, 0)
    stale = 0
    for epoch in range(args.epochs):
        network.train(); train_values = []
        for _ in range(BATCHES_PER_EPOCH):
            batch = rng.choice(train, min(64, len(train)), replace=True)
            value = loss(batch)
            if not torch.isfinite(value):
                raise RuntimeError(
                    f"non-finite recursive training loss at epoch {epoch + 1}")
            optimizer.zero_grad(); value.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step(); train_values.append(float(value.detach()))
        network.eval(); validation_scores = []
        with torch.no_grad():
            for batch in np.array_split(validation, max(1, len(validation) // 64)):
                _, score = loss(batch, True); validation_scores.extend(score.cpu().numpy())
        validation_scores = np.asarray(validation_scores)
        score = float(validation_scores.mean()
                      + CHECKPOINT_P95_WEIGHT * np.quantile(validation_scores, .95))
        if not np.isfinite(score):
            raise RuntimeError(
                f"non-finite recursive validation score at epoch {epoch + 1}")
        print(f"epoch={epoch + 1} train={np.mean(train_values):.6f} val={score:.6f}",
              flush=True)
        if score < best[0] - 1e-5:
            best = (score, {key: value.detach().cpu().clone()
                            for key, value in network.state_dict().items()}, epoch + 1)
            stale = 0
        else:
            stale += 1
        if stale >= 18:
            break
    network.load_state_dict(best[1]); network.cpu()
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), output / "model.pt")
    layers = (network.net[0], network.net[2], network.net[4])
    blob = np.concatenate([value.detach().numpy().ravel() for layer in layers
        for value in (layer.weight, layer.bias)] + [mean.cpu().numpy(), std.cpu().numpy()]).astype("<f4")
    if len(blob) != 3695:
        raise RuntimeError(f"unexpected exported residual size {len(blob)}")
    blob.tofile(output / "dynamic_40ms_residual.bin")
    metrics = {"source": "direct Step-1 callback archives", "seed": args.seed,
        "horizon_steps": args.horizon_steps, "horizon_s": args.horizon_steps * DT,
        "input_state": "actual callback MPPI-model KF",
        "state_target": "future actual MPPI-model KF",
        "pose_target": "future MCL pose", "best_epoch": best[2],
        "best_validation_score": best[0], "train_windows": len(train),
        "validation_windows": len(validation),
        "validation_source": validation_source,
        "loss_configuration": {
            "state_weights": STATE_WEIGHTS,
            "position_weight": POSITION_WEIGHT,
            "yaw_weight": YAW_WEIGHT,
            "tail_position_weight": TAIL_POSITION_WEIGHT,
            "tail_yaw_weight": TAIL_YAW_WEIGHT,
            "tail_fraction": TAIL_FRACTION,
            "checkpoint_p95_weight": CHECKPOINT_P95_WEIGHT}}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
