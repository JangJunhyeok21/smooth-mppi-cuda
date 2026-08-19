#!/usr/bin/env python3
"""Visualize identical aggressive-test rollouts for a candidate and baseline model."""
import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LATEST = ROOT / "model_tuning/results/dynamic_40ms_new_vy_stage2_gpu/rollout_metrics_latest.npz"
DEPLOYED = ROOT / "model_tuning/results/vy_input_ablation_old_runtime/rollout_metrics_latest_comparison.npz"
OUT = ROOT / "model_tuning/results/dynamic_40ms_new_vy_stage2_gpu/trajectory_visualization"
DATA = ROOT / "model_tuning/data/dynamic_40ms_residual.npz"
DT = 0.04


def load_pair(latest_path, deployed_path):
    latest, deployed = np.load(latest_path), np.load(deployed_path)
    if not np.array_equal(latest["starts"], deployed["starts"]):
        raise RuntimeError("latest/deployed evaluation windows do not match")
    if not np.allclose(latest["ground_truth"], deployed["ground_truth"]):
        raise RuntimeError("latest/deployed ground-truth trajectories do not match")
    return (latest["starts"], latest["predicted"],
            deployed["predicted"], latest["ground_truth"])


def final_xy_error(trace, truth):
    return np.linalg.norm(trace[:, -1, :2] - truth[:, -1, :2], axis=1)


def draw(indices, labels, starts, latest, deployed, truth, speed_command, filename, title,
         out_dir, candidate_label, baseline_label):
    time = np.arange(truth.shape[1]) * DT
    latest_error = final_xy_error(latest, truth)
    deployed_error = final_xy_error(deployed, truth)
    fig, axes = plt.subplots(5, len(indices), figsize=(5.3*len(indices), 17), constrained_layout=True)
    axes = np.atleast_2d(axes)
    signals = ((3, r"$v_x$ [m/s]"), (4, r"$v_y$ [m/s]"),
               (5, "yaw rate [rad/s]"), (2, "relative yaw [rad]"))
    for col, (index, label) in enumerate(zip(indices, labels)):
        ax = axes[0, col]
        ax.plot(truth[index, :, 0], truth[index, :, 1], "k-", lw=2.4, label="GT")
        ax.plot(deployed[index, :, 0], deployed[index, :, 1], "C3:", lw=2.2,
                label=baseline_label)
        ax.plot(latest[index, :, 0], latest[index, :, 1], "C0--", lw=2.2,
                label=candidate_label)
        ax.scatter(truth[index, 0, 0], truth[index, 0, 1], c="C2", s=35, zorder=4)
        ax.axis("equal")
        ax.set_title(f"{label} · source row {starts[index]}\n"
                     f"deployed {deployed_error[index]:.3f} m → latest {latest_error[index]:.3f} m")
        ax.set_xlabel("relative x [m]"); ax.set_ylabel("relative y [m]")
        for row, (channel, ylabel) in enumerate(signals, 1):
            ax = axes[row, col]
            values = (np.unwrap(truth[index, :, channel]) if channel == 2
                      else truth[index, :, channel])
            old_values = (np.unwrap(deployed[index, :, channel]) if channel == 2
                          else deployed[index, :, channel])
            new_values = (np.unwrap(latest[index, :, channel]) if channel == 2
                          else latest[index, :, channel])
            ax.plot(time, values, "k-", lw=2.2, label="GT")
            ax.plot(time, old_values, "C3:", lw=2, label=baseline_label)
            ax.plot(time, new_values, "C0--", lw=2, label=candidate_label)
            if channel == 3:
                ax.step(time, speed_command[index], where="post", color="C2", ls="-.",
                        lw=1.8, label=r"$v_{cmd}$")
            ax.set_xlabel("future time [s]"); ax.set_ylabel(ylabel)
        for ax in axes[:, col]:
            ax.grid(alpha=.28); ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=16)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=LATEST)
    parser.add_argument("--baseline", type=Path, default=DEPLOYED)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--candidate-label", default="candidate MLP")
    parser.add_argument("--baseline-label", default="current deployed MLP")
    args = parser.parse_args()
    starts, latest, deployed, truth = load_pair(args.candidate, args.baseline)
    source = np.load(args.data)["source_features"]
    speed_command = source[starts[:, None]+2*np.arange(truth.shape[1])[None, :], 4]
    latest_error = final_xy_error(latest, truth)
    deployed_error = final_xy_error(deployed, truth)
    order = np.argsort(latest_error)
    representative = (order[0], order[len(order)//2], order[-1])
    delta = deployed_error-latest_error
    comparisons = (int(np.argmax(delta)), int(np.argsort(np.abs(delta))[0]), int(np.argmin(delta)))
    draw(representative, ("best", "median", "worst"), starts, latest, deployed, truth,
         speed_command,
         "latest_best_median_worst.png", "Aggressive held-out 1.2 s rollout: candidate cases",
         args.out, args.candidate_label, args.baseline_label)
    draw(comparisons, ("largest improvement", "similar", "largest regression"),
         starts, latest, deployed, truth, speed_command, "latest_improvement_and_regression.png",
         "Aggressive held-out 1.2 s rollout: baseline vs candidate",
         args.out, args.candidate_label, args.baseline_label)
    selected = {}
    for label, index in zip(("best", "median", "worst"), representative):
        selected[label] = {"source_row": int(starts[index]),
                           "deployed_final_error_m": float(deployed_error[index]),
                           "latest_final_error_m": float(latest_error[index]),
                           "initial_vx_mps": float(truth[index,0,3]),
                           "final_vx_mps": float(truth[index,-1,3]),
                           "initial_speed_command_mps": float(speed_command[index,0]),
                           "final_speed_command_mps": float(speed_command[index,-1])}
    for label, index in zip(("largest_improvement", "similar", "largest_regression"), comparisons):
        selected[label] = {"source_row": int(starts[index]),
                           "deployed_final_error_m": float(deployed_error[index]),
                           "latest_final_error_m": float(latest_error[index]),
                           "initial_vx_mps": float(truth[index,0,3]),
                           "final_vx_mps": float(truth[index,-1,3]),
                           "initial_speed_command_mps": float(speed_command[index,0]),
                           "final_speed_command_mps": float(speed_command[index,-1])}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selected_cases.json").write_text(json.dumps(selected, indent=2)+"\n")
    print(json.dumps({"output_dir": str(args.out), "cases": selected}, indent=2))


if __name__ == "__main__":
    main()
