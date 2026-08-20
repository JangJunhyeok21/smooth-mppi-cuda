#!/usr/bin/env python3
"""CUDA DYNAMIC_IMU_RECURSIVE와 동일한 20 ms E2E MLP를 학습한다."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "model_tuning/data/dynamic_0817_0820_inertial_ekf_bias_40ms.npz"
DEFAULT_OUT = ROOT / "model_tuning/results/e2e_0817_0820_inertial_ekf_bias_seed31"
FEATURE_NAMES = ("vx", "vy", "yaw_rate", "imu_ax", "imu_ay", "steer_cmd",
                 "speed_cmd", "base_ax", "base_ay", "base_yaw_rate",
                 "steer_t-4", "speed_t-4", "steer_t-3", "speed_t-3",
                 "steer_t-2", "speed_t-2", "steer_t-1", "speed_t-1",
                 "steer_t", "speed_t")


class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x): return self.net(x)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT); p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=31); p.add_argument("--device", default="cuda"); a = p.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    d = np.load(a.dataset); source = d["source_features"].astype(np.float32)
    obs = d["source_observations"].astype(np.float32); bag = d["source_bag_id"]
    split = d["source_split"][:-1]; valid = d["source_valid"][:-1] & d["source_valid"][1:] & (bag[:-1] == bag[1:])
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    vx = source[:-1, 0]; steer_cmd = source[:-1, 3]; speed_cmd = source[:-1, 4]
    steer = np.clip(float(cfg["kinematic_steer_scale"]) * steer_cmd + float(cfg["kinematic_steer_bias"]), -.55, .55)
    base_ax = np.clip(float(cfg["speed_servo_kp"]) * (speed_cmd - vx), float(cfg["min_accel"]), float(cfg["max_accel"]))
    base_yaw = vx * np.tan(steer) / (float(cfg["l_f"]) + float(cfg["l_r"]))
    x = np.c_[source[:-1, :3], obs[:-1, :2], steer_cmd, speed_cmd, base_ax,
              vx * base_yaw, base_yaw, source[:-1, 10:20]].astype(np.float32)
    y = obs[1:, [0, 1, 2]].astype(np.float32)  # CUDA output: next ax, ay, yaw-rate
    valid &= np.isfinite(x).all(1) & np.isfinite(y).all(1)
    train, validation = valid & (split == 0), valid & (split == 1)
    xm, xs = x[train].mean(0), np.maximum(x[train].std(0), 1e-4)
    ym, ys = y[train].mean(0), np.maximum(y[train].std(0), 1e-3)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy((x - xm) / xs).to(device); yt = torch.from_numpy((y - ym) / ys).to(device)
    net = Net().to(device); opt = torch.optim.AdamW(net.parameters(), 8e-4, weight_decay=1e-4)
    weights = torch.tensor((1., 2., 2.), device=device); indices = np.flatnonzero(train)
    counts = {int(q): max(1, int(np.sum(bag[:-1][indices] == q))) for q in np.unique(bag[:-1][indices])}
    probability = np.array([1 / np.sqrt(counts[int(bag[i])]) for i in indices])
    probability *= np.where(x[indices, 0] >= 3., 4., 1.); probability /= probability.sum()
    best = (np.inf, None, 0); stale = 0
    for epoch in range(a.epochs):
        net.train(); sampled = rng.choice(indices, len(indices), replace=True, p=probability)
        for batch in np.array_split(sampled, max(1, len(sampled) // 1024)):
            loss = (nn.functional.smooth_l1_loss(net(xt[batch]), yt[batch], reduction="none") * weights).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step()
        net.eval()
        with torch.no_grad(): score = float(nn.functional.smooth_l1_loss(net(xt[validation]), yt[validation]))
        if score < best[0] - 1e-5:
            best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch + 1); stale = 0
        else: stale += 1
        if epoch % 25 == 0: print(f"epoch={epoch+1} validation={score:.6f} best={best[0]:.6f}", flush=True)
        if stale >= 60: break
    net.load_state_dict(best[1]); net.cpu()
    with torch.no_grad():
        net.net[4].weight.mul_(torch.from_numpy(ys)[:, None]); net.net[4].bias.mul_(torch.from_numpy(ys)).add_(torch.from_numpy(ym))
        pred = net(torch.from_numpy((x - xm) / xs)).numpy()
    a.out.mkdir(parents=True, exist_ok=True); torch.save(net.state_dict(), a.out / "model.pt")
    layers = (net.net[0], net.net[2], net.net[4]); blob = np.concatenate(
        [z.detach().numpy().ravel() for layer in layers for z in (layer.weight, layer.bias)] + [xm, xs]).astype("<f4")
    assert len(blob) == 3563; blob.tofile(a.out / "e2e_20ms.bin")
    metrics = {"model": "DYNAMIC_IMU_RECURSIVE", "seed": a.seed, "best_epoch": best[2],
               "model_dt": .02, "input_features": list(FEATURE_NAMES), "outputs": ["next_ax", "next_ay", "next_yaw_rate"]}
    for sid, name in enumerate(("train", "validation", "test")):
        mask = valid & (split == sid); error = abs(pred[mask] - y[mask])
        metrics[name] = {"n": int(mask.sum()), "mae": error.mean(0).tolist(), "p95": np.quantile(error, .95, axis=0).tolist()}
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (a.out / "contract.json").write_text(json.dumps({"model": "DYNAMIC_IMU_RECURSIVE", "model_dt": .02,
        "features": list(FEATURE_NAMES), "outputs": metrics["outputs"], "classic_pacejka_used": False,
        "cuda_function": "update_dynamic_imu_recursive"}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
