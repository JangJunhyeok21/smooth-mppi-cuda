#!/usr/bin/env python3
"""Train a causal end-to-end transition model against future KF states.

The model starts at every real odom callback timestamp.  It predicts at the
MPPI knot interval while the targets are interpolated from the causal Step-1
KF fields, not from ``callback_future_states`` (which contain raw pose and
velocity interpolation for backward compatibility).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# User-editable experiment settings
# ---------------------------------------------------------------------------
DATA = ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
OUTPUT = ROOT / "model_tuning/results/e2e_kf_predictor"
MODEL_DT_S = 0.04
ROLLOUT_HORIZON_S = 1.20
USE_VALIDATION_TEST_SPLIT = True
TRAIN_BAG_FRACTION = 0.70
VALIDATION_BAG_FRACTION = 0.15
HIDDEN_WIDTH = 128
HIDDEN_LAYERS = 3
EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-5
ONE_STEP_LOSS_WEIGHT = 1.0
RECURSIVE_STATE_LOSS_WEIGHT = 1.0
RECURSIVE_POSE_LOSS_WEIGHT = 1.0
POSE_XY_ERROR_SCALE_M = 0.20
POSE_YAW_ERROR_SCALE_RAD = 0.15
GRADIENT_CLIP_NORM = 5.0
SEED = 31
USE_PLOT = True

STATE_NAMES = ("vx", "vy", "yaw_rate")
POSE_NAMES = ("x", "y", "yaw")
HISTORY_NAMES = tuple(
    value for k in range(4, -1, -1)
    for value in ((f"steer_t-{k}", f"speed_t-{k}") if k else ("steer_t", "speed_t")))
INPUT_NAMES = STATE_NAMES + ("steer_cmd", "speed_cmd") + HISTORY_NAMES


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def interp_columns(t, values, query, yaw_column=None):
    result = np.empty((len(query), values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        source = np.unwrap(values[:, j]) if j == yaw_column else values[:, j]
        result[:, j] = np.interp(query, t, source, left=np.nan, right=np.nan)
    return result


def load_bag(path: Path, horizon: int):
    with np.load(path, allow_pickle=False) as data:
        required = {"samples", "columns", "callback_inputs", "callback_input_columns",
                    "callback_future_commands", "callback_future_offsets_s"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise RuntimeError(f"{path}: rerun Step 1; missing {missing}")
        columns = {str(v): k for k, v in enumerate(data["columns"])}
        callback_columns = {str(v): k for k, v in enumerate(data["callback_input_columns"])}
        kf_names = ("kf_x", "kf_y", "kf_yaw", "kf_vx", "kf_vy", "kf_yaw_rate")
        absent = [name for name in kf_names if name not in columns]
        if absent:
            raise RuntimeError(f"{path}: missing causal KF fields {absent}")

        samples = np.asarray(data["samples"], np.float64)
        callbacks = np.asarray(data["callback_inputs"], np.float64)
        future_commands_20ms = np.asarray(data["callback_future_commands"], np.float64)
        offsets = np.asarray(data["callback_future_offsets_s"], np.float64)

    target_offsets = MODEL_DT_S * np.arange(1, horizon + 1)
    target_indices = np.array([int(np.argmin(np.abs(offsets - value))) for value in target_offsets])
    if np.max(np.abs(offsets[target_indices] - target_offsets)) > 1e-6:
        raise RuntimeError(f"{path}: callback targets do not contain {MODEL_DT_S:g} s knots")

    sample_t = samples[:, columns["t"]]
    order = np.argsort(sample_t)
    sample_t = sample_t[order]
    kf = samples[order][:, [columns[name] for name in kf_names]]
    keep = np.r_[True, np.diff(sample_t) > 1e-9]
    sample_t, kf = sample_t[keep], kf[keep]

    anchor_t = callbacks[:, callback_columns["t"]]
    initial = interp_columns(sample_t, kf, anchor_t, yaw_column=2)
    future_t = (anchor_t[:, None] + target_offsets[None, :]).reshape(-1)
    target = interp_columns(sample_t, kf, future_t, yaw_column=2).reshape(-1, horizon, 6)

    command_now = callbacks[:, [callback_columns["steer_cmd"], callback_columns["speed_cmd"]]]
    # command[k] is the causal command at the beginning of transition k.
    commands = np.empty((len(callbacks), horizon, 2), np.float64)
    commands[:, 0] = command_now
    if horizon > 1:
        commands[:, 1:] = future_commands_20ms[:, target_indices[:-1], :]
    history = callbacks[:, [callback_columns[name] for name in HISTORY_NAMES]]

    finite = (np.isfinite(initial).all(1) & np.isfinite(target).all((1, 2))
              & np.isfinite(commands).all((1, 2)) & np.isfinite(history).all(1))
    imu_names = ("imu_wz", "imu_ax", "imu_ay")
    imu = callbacks[:, [callback_columns[name] for name in imu_names]]
    actuator = callbacks[:, [callback_columns["applied_steer"],
                              callback_columns["speed_reference"]]]
    finite &= np.isfinite(imu).all(1) & np.isfinite(actuator).all(1)
    return {"path": str(path), "initial_pose": initial[finite, :3],
            "initial_state": initial[finite, 3:], "target": target[finite],
            "commands": commands[finite], "history": history[finite],
            "imu": imu[finite], "actuator": actuator[finite]}


def concatenate(bags, bag_indices):
    keys = ("initial_pose", "initial_state", "target", "commands", "history",
            "imu", "actuator")
    if not bag_indices:
        raise RuntimeError("empty dataset split")
    output = {key: np.concatenate([bags[k][key] for k in bag_indices]) for key in keys}
    output["bag"] = np.concatenate([
        np.full(len(bags[k]["initial_state"]), k, np.int64) for k in bag_indices])
    return output


class E2EKFTransition(nn.Module):
    """Causal transition: KF body state + command history -> state derivative."""

    def __init__(self, input_mean, input_std, derivative_mean, derivative_std):
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_std", torch.as_tensor(input_std, dtype=torch.float32))
        self.register_buffer("derivative_mean", torch.as_tensor(derivative_mean, dtype=torch.float32))
        self.register_buffer("derivative_std", torch.as_tensor(derivative_std, dtype=torch.float32))
        layers = []
        width = len(input_mean)
        for _ in range(HIDDEN_LAYERS):
            layers.extend((nn.Linear(width, HIDDEN_WIDTH), nn.SiLU()))
            width = HIDDEN_WIDTH
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, state, command, history):
        features = torch.cat((state, command, history), dim=-1)
        normalized = (features - self.input_mean) / self.input_std
        return self.net(normalized) * self.derivative_std + self.derivative_mean


def rollout(model, pose, state, commands, history):
    poses, states = [], []
    for k in range(commands.shape[1]):
        derivative = model(state, commands[:, k], history)
        state = state + MODEL_DT_S * derivative
        yaw = pose[:, 2]
        world_vx = state[:, 0] * torch.cos(yaw) - state[:, 1] * torch.sin(yaw)
        world_vy = state[:, 0] * torch.sin(yaw) + state[:, 1] * torch.cos(yaw)
        pose = torch.stack((pose[:, 0] + MODEL_DT_S * world_vx,
                            pose[:, 1] + MODEL_DT_S * world_vy,
                            pose[:, 2] + MODEL_DT_S * state[:, 2]), dim=1)
        poses.append(pose)
        states.append(state)
        history = torch.cat((history[:, 2:], commands[:, k]), dim=1)
    return torch.stack(poses, 1), torch.stack(states, 1)


def tensors(data, device="cpu"):
    return tuple(torch.as_tensor(data[name], dtype=torch.float32, device=device)
                 for name in ("initial_pose", "initial_state", "target", "commands", "history"))


def loss_terms(model, batch, state_scale):
    initial_pose, initial_state, target, commands, history = batch
    predicted_pose, predicted_state = rollout(model, initial_pose, initial_state, commands, history)
    gt_pose, gt_state = target[:, :, :3], target[:, :, 3:]
    state_error = (predicted_state - gt_state) / state_scale
    xy_error = (predicted_pose[:, :, :2] - gt_pose[:, :, :2]) / POSE_XY_ERROR_SCALE_M
    yaw_error = wrap(predicted_pose[:, :, 2] - gt_pose[:, :, 2]) / POSE_YAW_ERROR_SCALE_RAD
    one_step = torch.mean(state_error[:, 0] ** 2)
    recursive_state = torch.mean(state_error ** 2)
    recursive_pose = torch.mean(xy_error ** 2) + torch.mean(yaw_error ** 2)
    total = (ONE_STEP_LOSS_WEIGHT * one_step
             + RECURSIVE_STATE_LOSS_WEIGHT * recursive_state
             + RECURSIVE_POSE_LOSS_WEIGHT * recursive_pose)
    return total, one_step, recursive_state, recursive_pose


@torch.no_grad()
def evaluate(model, data, device, state_scale, batch_size=512):
    dataset = TensorDataset(*tensors(data))
    predicted_pose, predicted_state, target = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        batch = tuple(value.to(device) for value in batch)
        pose, state = rollout(model, batch[0], batch[1], batch[3], batch[4])
        predicted_pose.append(pose.cpu().numpy())
        predicted_state.append(state.cpu().numpy())
        target.append(batch[2].cpu().numpy())
    pp, ps, gt = map(np.concatenate, (predicted_pose, predicted_state, target))
    state_error = ps - gt[:, :, 3:]
    pose_error = pp - gt[:, :, :3]
    pose_error[:, :, 2] = np.arctan2(np.sin(pose_error[:, :, 2]), np.cos(pose_error[:, :, 2]))
    trajectory_error = np.linalg.norm(pose_error[:, :, :2], axis=2)
    metrics = {
        "one_step_state_rmse": dict(zip(STATE_NAMES, np.sqrt(np.mean(state_error[:, 0] ** 2, axis=0)).tolist())),
        "rollout_state_rmse": dict(zip(STATE_NAMES, np.sqrt(np.mean(state_error ** 2, axis=(0, 1))).tolist())),
        "rollout_pose_rmse": dict(zip(POSE_NAMES, np.sqrt(np.mean(pose_error ** 2, axis=(0, 1))).tolist())),
        "trajectory_error_mean_m": float(np.mean(trajectory_error)),
        "trajectory_error_p95_m": float(np.percentile(trajectory_error, 95)),
        "final_trajectory_error_mean_m": float(np.mean(trajectory_error[:, -1])),
    }
    score = np.sqrt(np.mean(trajectory_error ** 2, axis=1))
    return metrics, pp, ps, gt, score


def plot_rollouts(data, predicted_pose, predicted_state, target, score, output):
    order = np.argsort(score)
    selections = (order[0], order[min(len(order)-1, int(.95*(len(order)-1)))], order[-1])
    labels = ("best", "p95", "worst")
    time = MODEL_DT_S * np.arange(1, target.shape[1] + 1)
    fig, axes = plt.subplots(3, 5, figsize=(22, 13), constrained_layout=True)
    for row, (index, label) in enumerate(zip(selections, labels)):
        gt_pose, gt_state = target[index, :, :3], target[index, :, 3:]
        axes[row, 0].plot(gt_pose[:, 0], gt_pose[:, 1], "k-", label="KF GT")
        axes[row, 0].plot(predicted_pose[index, :, 0], predicted_pose[index, :, 1], "C3--", label="E2E")
        axes[row, 0].scatter(data["initial_pose"][index, 0], data["initial_pose"][index, 1], c="C2", s=35)
        axes[row, 0].set_title(f"{label}: trajectory, RMSE={score[index]:.3f} m")
        axes[row, 0].axis("equal"); axes[row, 0].legend()
        for column, name in enumerate(STATE_NAMES, start=1):
            j = column - 1
            axes[row, column].plot(time, gt_state[:, j], "k-", label="KF GT")
            axes[row, column].plot(time, predicted_state[index, :, j], "C3--", label="E2E")
            axes[row, column].set_title(name); axes[row, column].set_xlabel("future time [s]")
        axes[row, 4].plot(time, gt_pose[:, 2] - gt_pose[0, 2], "k-", label="KF GT")
        axes[row, 4].plot(time, predicted_pose[index, :, 2] - predicted_pose[index, 0, 2], "C3--", label="E2E")
        axes[row, 4].set_title("yaw change"); axes[row, 4].set_xlabel("future time [s]")
        for ax in axes[row, 1:]: ax.grid(True, alpha=.25)
    axes[0, 1].legend()
    fig.suptitle("E2E causal predictor vs future KF state")
    fig.savefig(output, dpi=150)
    return fig


def split_bags(count):
    indices = np.arange(count)
    if not USE_VALIDATION_TEST_SPLIT or count < 3:
        return indices.tolist(), indices.tolist(), indices.tolist()
    rng = np.random.default_rng(SEED); rng.shuffle(indices)
    n_train = max(1, int(round(count * TRAIN_BAG_FRACTION)))
    n_validation = max(1, int(round(count * VALIDATION_BAG_FRACTION)))
    if n_train + n_validation >= count:
        n_train, n_validation = count - 2, 1
    return (indices[:n_train].tolist(), indices[n_train:n_train+n_validation].tolist(),
            indices[n_train+n_validation:].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="debug cap per split; 0 uses every callback anchor")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    paths = sorted(args.data.glob("*.npz")) if args.data.is_dir() else [args.data]
    if not paths: raise RuntimeError(f"no NPZ files under {args.data}")
    horizon = int(round(ROLLOUT_HORIZON_S / MODEL_DT_S))
    bags = [load_bag(path, horizon) for path in paths]
    bags = [bag for bag in bags if len(bag["initial_state"])]
    train_ids, validation_ids, test_ids = split_bags(len(bags))
    train, validation, test = (concatenate(bags, ids) for ids in (train_ids, validation_ids, test_ids))
    if args.max_samples:
        rng = np.random.default_rng(SEED)
        for data in (train, validation, test):
            choice = rng.choice(len(data["initial_state"]), min(args.max_samples, len(data["initial_state"])), replace=False)
            for key in ("initial_pose", "initial_state", "target", "commands", "history",
                        "imu", "actuator", "bag"):
                data[key] = data[key][choice]

    one_step_derivative = (train["target"][:, 0, 3:] - train["initial_state"]) / MODEL_DT_S
    features = np.c_[train["initial_state"], train["commands"][:, 0], train["history"]]
    input_mean, input_std = features.mean(0), np.maximum(features.std(0), 1e-4)
    derivative_mean = one_step_derivative.mean(0)
    derivative_std = np.maximum(one_step_derivative.std(0), 1e-3)
    state_scale = np.maximum(train["target"][:, :, 3:].std((0, 1)), np.array([.2, .1, .2]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2EKFTransition(input_mean, input_std, derivative_mean, derivative_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    state_scale_tensor = torch.as_tensor(state_scale, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(*tensors(train)), batch_size=BATCH_SIZE, shuffle=True)
    best_state, best_validation = None, float("inf")
    print(f"E2E KF data: bags={len(bags)}, train/val/test anchors="
          f"{len(train['initial_state'])}/{len(validation['initial_state'])}/{len(test['initial_state'])}")
    print(f"device={device}, dt={MODEL_DT_S:.3f}s, recursive steps={horizon}")
    for epoch in range(1, args.epochs + 1):
        model.train(); totals = np.zeros(4); batches = 0
        for batch in loader:
            batch = tuple(value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            terms = loss_terms(model, batch, state_scale_tensor)
            terms[0].backward(); nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step(); totals += np.array([float(value.detach()) for value in terms]); batches += 1
        model.eval()
        with torch.no_grad():
            val_batch = tensors(validation, device)
            validation_total = float(loss_terms(model, val_batch, state_scale_tensor)[0])
        if np.isfinite(validation_total) and validation_total < best_validation:
            best_validation = validation_total
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            mean = totals / max(batches, 1)
            print(f"epoch {epoch:4d}: total={mean[0]:.6f}, one_step={mean[1]:.6f}, "
                  f"state={mean[2]:.6f}, pose={mean[3]:.6f}, val={validation_total:.6f}")
    if best_state is not None: model.load_state_dict(best_state)

    metrics = {}
    results = {}
    for name, data in (("train", train), ("validation", validation), ("test", test)):
        metrics[name], *results[name] = evaluate(model, data, device, state_scale_tensor)
        print(f"{name}: {json.dumps(metrics[name], ensure_ascii=False)}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "model_class": "E2EKFTransition",
                "dt_s": MODEL_DT_S, "input_names": INPUT_NAMES, "output_names": STATE_NAMES,
                "hidden_width": HIDDEN_WIDTH, "hidden_layers": HIDDEN_LAYERS}, args.out / "model.pt")
    np.savez(args.out / "normalization.npz", input_mean=input_mean, input_std=input_std,
             derivative_mean=derivative_mean, derivative_std=derivative_std, state_scale=state_scale)
    contract = {"semantics": "causal recursive E2E transition trained on Step-1 KF GT",
                "pose_update": "semi-implicit body velocity integration",
                "dt_s": MODEL_DT_S, "horizon_s": ROLLOUT_HORIZON_S,
                "inputs": INPUT_NAMES, "outputs": ("dvx_dt", "dvy_dt", "dyaw_rate_dt"),
                "gt_fields": ("kf_x", "kf_y", "kf_yaw", "kf_vx", "kf_vy", "kf_yaw_rate"),
                "warning": "experimental model; not deployed to CUDA MPPI by this script"}
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pp, ps, gt, score = results["test"]
    fig = plot_rollouts(test, pp, ps, gt, score, args.out / "rollouts.png")
    print(f"saved model and evaluation to {args.out}")
    if USE_PLOT and not args.no_show: plt.show()
    else: plt.close(fig)


if __name__ == "__main__":
    main()
