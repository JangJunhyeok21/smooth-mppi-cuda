#!/usr/bin/env python3
"""Visualize the deployed 20-D baseline against the causal-IMU 22-D model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD = Path("/tmp/deployed_eval/rollout_30step_metrics.npz")
DEFAULT_NEW = ROOT / "model_tuning/results/dynamic_40ms_imu22_recursive_vy_seed31/rollout_30step_metrics.npz"
DEFAULT_DATA = ROOT / "model_tuning/data/dynamic_40ms_residual_imu22.npz"
DEFAULT_OUT = ROOT / "model_tuning/results/dynamic_40ms_imu22_recursive_vy_seed31"


def stats(values):
    return {"mean": float(np.mean(values)), "p95": float(np.quantile(values, .95)),
            "worst": float(np.max(values))}


def errors(archive):
    predicted, truth = archive["predicted"], archive["ground_truth"]
    return {
        "trajectory_m": np.linalg.norm(predicted[:, -1, :2] - truth[:, -1, :2], axis=1),
        "yaw_rad": np.abs(np.arctan2(np.sin(predicted[:, -1, 2] - truth[:, -1, 2]),
                                     np.cos(predicted[:, -1, 2] - truth[:, -1, 2]))),
        "vx_mps": np.abs(predicted[:, -1, 3] - truth[:, -1, 3]),
        "vy_mps": np.abs(predicted[:, -1, 4] - truth[:, -1, 4]),
        "yaw_rate_rps": np.abs(predicted[:, -1, 5] - truth[:, -1, 5]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    old, new, data = np.load(args.old), np.load(args.new), np.load(args.data)
    if not np.array_equal(old["starts"], new["starts"]):
        raise RuntimeError("comparison archives do not contain identical windows")
    starts = new["starts"].astype(int); old_e, new_e = errors(old), errors(new)
    source = data["source_features"]
    rows = starts[:, None] + 2 * np.arange(30)[None]
    max_steer = np.max(np.abs(source[rows, 3]), axis=1)
    initial_speed = source[starts, 0]
    steer_edges = np.array([0., .15, .30, .40, .479])
    report = {"windows": int(len(starts)), "horizon_s": 1.2, "models": {},
              "steer_bins": {}, "speed_bins": {}}
    for label, values in (("deployed_20d", old_e), ("causal_imu_22d", new_e)):
        report["models"][label] = {key: stats(value) for key, value in values.items()}
    for lo, hi in zip(steer_edges[:-1], steer_edges[1:]):
        mask = (max_steer >= lo) & (max_steer < hi + 1e-8)
        key = f"{lo:.2f}-{hi:.3f}"
        report["steer_bins"][key] = {"windows": int(mask.sum()),
            "old_trajectory_m": stats(old_e["trajectory_m"][mask]) if mask.any() else None,
            "new_trajectory_m": stats(new_e["trajectory_m"][mask]) if mask.any() else None}
    for lo, hi in ((0., 2.), (2., 3.), (3., 4.), (4., 20.)):
        mask = (initial_speed >= lo) & (initial_speed < hi)
        key = f"{lo:.0f}-{hi:.0f}"
        report["speed_bins"][key] = {"windows": int(mask.sum()),
            "old_trajectory_m": stats(old_e["trajectory_m"][mask]) if mask.any() else None,
            "new_trajectory_m": stats(new_e["trajectory_m"][mask]) if mask.any() else None}
    report["trajectory_improvement_percent"] = {
        key: 100 * (report["models"]["deployed_20d"]["trajectory_m"][key]
                    - report["models"]["causal_imu_22d"]["trajectory_m"][key])
                    / report["models"]["deployed_20d"]["trajectory_m"][key]
        for key in ("mean", "p95", "worst")}
    (args.out / "imu22_overall_comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for label, value, color in (("20D baseline", old_e["trajectory_m"], "tab:orange"),
                                ("22D causal IMU", new_e["trajectory_m"], "tab:blue")):
        ordered = np.sort(value); axes[0, 0].plot(ordered, np.linspace(0, 1, len(ordered)), label=label, color=color)
    axes[0, 0].set(xlabel="1.2 s endpoint error [m]", ylabel="CDF", title="Trajectory endpoint error")
    labels = ["trajectory", "yaw", "vx", "vy", "yaw-rate"]
    keys = ["trajectory_m", "yaw_rad", "vx_mps", "vy_mps", "yaw_rate_rps"]
    xloc = np.arange(len(keys)); width = .36
    axes[0, 1].bar(xloc-width/2, [np.quantile(old_e[k], .95) for k in keys], width, label="20D")
    axes[0, 1].bar(xloc+width/2, [np.quantile(new_e[k], .95) for k in keys], width, label="22D IMU")
    axes[0, 1].set_xticks(xloc, labels, rotation=20); axes[0, 1].set_title("P95 endpoint errors")
    bin_centers = np.arange(len(steer_edges)-1)
    for model, err, offset, color in (("20D", old_e, -.18, "tab:orange"), ("22D IMU", new_e, .18, "tab:blue")):
        vals=[]
        for lo,hi in zip(steer_edges[:-1],steer_edges[1:]):
            mask=(max_steer>=lo)&(max_steer<hi+1e-8); vals.append(np.quantile(err["trajectory_m"][mask],.95) if mask.any() else np.nan)
        axes[0, 2].bar(bin_centers+offset, vals, .36, label=model, color=color)
    axes[0, 2].set_xticks(bin_centers, [f"{a:.2f}-{b:.2f}" for a,b in zip(steer_edges[:-1],steer_edges[1:])], rotation=20)
    axes[0, 2].set(title="Trajectory P95 by max steer", ylabel="error [m]")
    axes[0, 3].scatter(old_e["trajectory_m"], new_e["trajectory_m"], c=max_steer, s=10, alpha=.5, cmap="viridis")
    limit=max(old_e["trajectory_m"].max(),new_e["trajectory_m"].max());axes[0,3].plot([0,limit],[0,limit],"k--")
    axes[0, 3].set(xlabel="20D error [m]", ylabel="22D error [m]", title="Per-window comparison")

    order=np.argsort(new_e["trajectory_m"]); selected=(order[0],order[len(order)//2],order[-1])
    for axis,index,title in zip(axes[1,:3],selected,("Best 22D window","Median 22D window","Worst 22D window")):
        truth=new["ground_truth"][index]; op=old["predicted"][index]; npred=new["predicted"][index]
        axis.plot(truth[:,0],truth[:,1],"k-",lw=2,label="GT");axis.plot(op[:,0],op[:,1],"--",color="tab:orange",label="20D")
        axis.plot(npred[:,0],npred[:,1],"-",color="tab:blue",label="22D IMU");axis.axis("equal");axis.set_title(f"{title}\nold={old_e['trajectory_m'][index]:.3f} m, new={new_e['trajectory_m'][index]:.3f} m")
    worst=selected[-1];time=np.arange(new["predicted"].shape[1])*.04
    axes[1,3].plot(time,new["ground_truth"][worst,:,5],"k-",lw=2,label="GT")
    axes[1,3].plot(time,old["predicted"][worst,:,5],"--",color="tab:orange",label="20D")
    axes[1,3].plot(time,new["predicted"][worst,:,5],color="tab:blue",label="22D IMU")
    axes[1,3].set(xlabel="time [s]",ylabel="yaw-rate [rad/s]",title="Worst-window yaw-rate")
    for axis in axes.flat: axis.grid(alpha=.25); axis.legend(fontsize=8)
    fig.suptitle("Dynamic residual model: deployed 20D vs causal-IMU 22D",fontsize=15)
    fig.tight_layout();fig.savefig(args.out/"visualize_imu22_overall_comparison.png",dpi=180);plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
