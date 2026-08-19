#!/usr/bin/env python3
"""Plot old/new residual performance on the two unseen 08-08 bag segments."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "model_tuning/results/failed_bag_replay"
OUTPUT_PATH = RESULT_ROOT / "failed_bag_old_vs_regressed_kf.png"

CASES = [
    ("22:10:38 pre-collision\n0.6 s", "old_kf_221038_h06", "new_kf_221038_h06", "0.60s"),
    ("22:11:08 recovered\n1.0 s", "old_kf_221108", "new_kf_221108", "1.00s"),
    ("22:11:08 recovered\n1.6 s", "old_kf_221108", "new_kf_221108", "1.60s"),
]
KEYS = [
    ("trajectory_mean_m", "Trajectory mean [m]"),
    ("trajectory_p95_m", "Trajectory P95 [m]"),
    ("speed_mae_mps", "Speed MAE [m/s]"),
    ("yaw_rate_mae_radps", "Yaw-rate MAE [rad/s]"),
    ("yaw_mae_deg", "Yaw MAE [deg]"),
]


def load(directory, horizon):
    data = json.loads((RESULT_ROOT / directory / "metrics.json").read_text())
    return data["current_cuda_hard_state_clamp"][horizon]


def main():
    old = [[load(old_dir, horizon)[key] for _, old_dir, _, horizon in CASES]
           for key, _ in KEYS]
    new = [[load(new_dir, horizon)[key] for _, _, new_dir, horizon in CASES]
           for key, _ in KEYS]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(len(CASES)); width = .36
    labels = [case[0] for case in CASES]
    for axis, (_, title), old_values, new_values in zip(axes.flat, KEYS, old, new):
        axis.bar(x-width/2, old_values, width, label="Old KF + old MLP")
        axis.bar(x+width/2, new_values, width, label="Regressed KF + new MLP")
        axis.set_title(title); axis.set_xticks(x, labels); axis.grid(axis="y", alpha=.25)
        axis.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Unseen failed-bag replay: collision/reverse samples excluded")
    fig.tight_layout(); fig.savefig(OUTPUT_PATH, dpi=180); plt.close(fig)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
