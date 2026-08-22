#!/usr/bin/env python3
"""0817~0820 실차 기록 재현을 최우선으로 simulator GRU를 의도적으로 과적합한다."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experimental_train_simulator_gru import GRUPlant, InferencePlant

DATA = ROOT / "model_tuning/data/dynamic_0817_0820_inertial_ekf_bias_40ms.npz"
BASE = ROOT / "model_tuning/results/simulator_gru_0817_0820_seed31"
OUT = ROOT / "model_tuning/results/simulator_gru_0817_0820_overfit_seed31"


def starts_for_all(bag, valid, warm, horizon, stride):
    return np.asarray([
        i for i in range(warm, len(bag) - horizon, stride)
        if valid[i-warm:i+horizon+1].all()
        and np.all(bag[i-warm:i+horizon+1] == bag[i])
    ], dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    torch.manual_seed(31)
    np.random.seed(31)
    d = np.load(a.data)
    s = d["source_features"].astype(np.float32)
    obs = d["source_observations"].astype(np.float32)
    sr = d["source_speed_reference"].astype(np.float32)
    bag = d["source_bag_id"]
    valid = d["source_valid"]

    base_checkpoint = a.base / "model_recursive.pt"
    if not base_checkpoint.exists():
        base_checkpoint = a.base / "model.pt"
    ck = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    warm = int(ck["history"])
    feature = np.c_[s[:, :3], obs[:, :2], s[:, 3:5], s[:, 5], sr].astype(np.float32)
    starts = starts_for_all(bag, valid, warm, a.horizon, a.stride)
    if not len(starts):
        raise RuntimeError("학습 가능한 연속 구간이 없습니다")

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise RuntimeError("과적합 학습은 CUDA GPU가 필요합니다")
    net = GRUPlant()
    net.load_state_dict(ck["state_dict"])
    net.to(dev)
    xm = torch.as_tensor(ck["x_mean"], device=dev)
    xs = torch.as_tensor(ck["x_std"], device=dev)
    ym = torch.as_tensor(ck["y_mean"], device=dev)
    ys = torch.as_tensor(ck["y_std"], device=dev)
    # 과적합이 목적이므로 regularization은 쓰지 않는다.
    opt = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=2e-6)
    rng = np.random.default_rng(31)
    best = (float("inf"), None, 0)

    def batch_loss(ids):
        ids = np.asarray(ids)
        warm_np = np.stack([feature[i-warm:i] for i in ids])
        warm_x = (torch.from_numpy(warm_np).to(dev) - xm) / xs
        _, hidden = net.gru(warm_x)
        state = torch.from_numpy(s[ids, :3]).to(dev)
        accel = torch.from_numpy(obs[ids, :2]).to(dev)
        pred_pose = torch.zeros((len(ids), 3), device=dev)
        gt_pose = torch.zeros_like(pred_pose)
        loss = torch.zeros((), device=dev)
        state_scale = torch.tensor((0.60, 0.25, 0.40), device=dev)
        for k in range(a.horizon):
            controls = torch.from_numpy(
                np.c_[s[ids+k, 3:5], s[ids+k, 5], sr[ids+k]].astype(np.float32)
            ).to(dev)
            raw = torch.cat((state, accel, controls), dim=1)
            out, hidden = net.gru(((raw-xm)/xs)[:, None], hidden)
            pred_norm = net.head(out[:, 0])
            pred = pred_norm*ys + ym
            ax, ay, next_r = pred[:, 0], pred[:, 1], pred[:, 2]
            vx, vy = state[:, 0], state[:, 1]
            next_state = torch.stack((
                vx + (ax + vy*next_r)*.02,
                vy + (ay - vx*next_r)*.02,
                next_r,
            ), dim=1)
            target_state = torch.from_numpy(s[ids+k+1, :3]).to(dev)
            target_obs = torch.from_numpy(obs[ids+k+1, :3]).to(dev)

            pyaw = pred_pose[:, 2]
            gyaw = gt_pose[:, 2]
            pred_pose = torch.stack((
                pred_pose[:, 0] + (next_state[:, 0]*torch.cos(pyaw)-next_state[:, 1]*torch.sin(pyaw))*.02,
                pred_pose[:, 1] + (next_state[:, 0]*torch.sin(pyaw)+next_state[:, 1]*torch.cos(pyaw))*.02,
                pyaw + next_state[:, 2]*.02,
            ), dim=1)
            gt_pose = torch.stack((
                gt_pose[:, 0] + (target_state[:, 0]*torch.cos(gyaw)-target_state[:, 1]*torch.sin(gyaw))*.02,
                gt_pose[:, 1] + (target_state[:, 0]*torch.sin(gyaw)+target_state[:, 1]*torch.cos(gyaw))*.02,
                gyaw + target_state[:, 2]*.02,
            ), dim=1)
            tail = 0.35 + 1.65*(k+1)/a.horizon
            derivative = nn.functional.smooth_l1_loss(pred_norm, (target_obs-ym)/ys)
            state_error = nn.functional.smooth_l1_loss((next_state-target_state)/state_scale, torch.zeros_like(state))
            pose_error = nn.functional.smooth_l1_loss((pred_pose-gt_pose)/torch.tensor((.20, .20, .15), device=dev), torch.zeros_like(pred_pose))
            loss = loss + tail*(0.20*derivative + 1.25*state_error + 1.50*pose_error)
            state, accel = next_state, pred[:, :2]
        return loss/a.horizon

    for epoch in range(a.epochs):
        net.train()
        losses = []
        order = rng.permutation(starts)
        for begin in range(0, len(order), a.batch_size):
            loss = batch_loss(order[begin:begin+a.batch_size])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        score = float(np.mean(losses))
        if score < best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch+1)
        print(f"epoch={epoch+1} all_data={score:.6f} best={best[0]:.6f}", flush=True)

    net.load_state_dict(best[1])
    net.cpu().eval()
    a.out.mkdir(parents=True, exist_ok=True)
    ck["state_dict"] = net.state_dict()
    ck["overfit_best_epoch"] = best[2]
    ck["overfit_all_splits"] = True
    torch.save(ck, a.out / "model_overfit.pt")
    wrapper = InferencePlant(net, xm.cpu(), xs.cpu(), ym.cpu(), ys.cpu()).eval()
    torch.jit.script(wrapper).save(str(a.out / "simulator_gru.ts"))
    shutil.copy2(a.base / "rollout_60step_metrics.json", a.out / "baseline_rollout_60step_metrics.json")
    (a.out / "overfit_metrics.json").write_text(json.dumps({
        "best_epoch": best[2], "training_loss": best[0], "horizon_steps": a.horizon,
        "windows": len(starts), "stride": a.stride, "all_splits_used_for_training": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
