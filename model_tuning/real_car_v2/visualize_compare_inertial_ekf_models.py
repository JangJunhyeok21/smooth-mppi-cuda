#!/usr/bin/env python3
"""현재 배포 모델과 MCL+odom+IMU EKF 재학습 모델의 rollout 비교."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "model_tuning/results/compare_0817_0820_inertial_ekf_bias"


def load(name: str):
    with (RESULTS / f"{name}.json").open() as f:
        metrics = json.load(f)["test_aggressive"]
    rollout = np.load(RESULTS / f"{name}.npz")
    return metrics, rollout


def main() -> None:
    old_m, old_z = load("deployed_lateral_only")
    new_m, new_z = load("new_lateral_only")
    classic_m, _ = load("new_classic_only")

    labels = ["trajectory [m]", "vy [m/s]", "yaw-rate [rad/s]"]
    keys = ["trajectory_m", "vy_mps", "yaw_rate_rps"]
    x = np.arange(len(keys))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for offset, name, metrics in [
        (-width, "deployed MLP", old_m),
        (0.0, "new classic", classic_m),
        (width, "new EKF+MLP", new_m),
    ]:
        ax.bar(x + offset, [metrics[k]["mean"] for k in keys], width, label=name)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("30-step mean absolute error")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # 새 모델에서 가장 큰 최종 위치오차 window를 동일 index의 기존 모델과 비교한다.
    new_final = np.linalg.norm(new_z["predicted"][:, -1, :2] - new_z["ground_truth"][:, -1, :2], axis=1)
    worst = int(np.argmax(new_final))
    ax = axes[1]
    gt = new_z["ground_truth"][worst]
    old = old_z["predicted"][worst]
    new = new_z["predicted"][worst]
    ax.plot(gt[:, 0], gt[:, 1], "k-o", ms=2, label="GT (new EKF state)")
    ax.plot(old[:, 0], old[:, 1], "--", label="deployed MLP")
    ax.plot(new[:, 0], new[:, 1], "-", label="new EKF+MLP")
    ax.scatter(new[-1, 0], new[-1, 1], c="red", s=35, zorder=5)
    ax.set_title(f"New-model worst window (final error {new_final[worst]:.3f} m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("0817-0820 held-out rollout comparison (dt=40 ms, 30 steps)")
    fig.tight_layout()
    out = RESULTS / "model_comparison.png"
    fig.savefig(out, dpi=180)
    print(out)
    plt.show()


if __name__ == "__main__":
    main()
