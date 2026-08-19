#!/usr/bin/env python3
"""Locate the latest 40 ms model's worst aggressive windows on Map1."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "model_tuning/results/effective_vs_dynamic_0813"
MAP_YAML = ROOT / "f1tenth_gym_ros/src/f1tenth_gym_ros/maps/map1/map1.yaml"
OUTPUT = RESULT / "latest40_worst_locations_map1.png"
RUNS = ("aggressive_boundary_run1", "aggressive_boundary_run2")
GLOBAL_OFFSETS = {"aggressive_boundary_run1": 22229,
                  "aggressive_boundary_run2": 24342}


def load_map():
    meta = yaml.safe_load(MAP_YAML.read_text())
    image = plt.imread(MAP_YAML.parent / meta["image"])
    resolution = float(meta["resolution"])
    ox, oy, _ = meta["origin"]
    extent = (ox, ox + image.shape[1] * resolution,
              oy, oy + image.shape[0] * resolution)
    return image, extent


def main():
    map_image, extent = load_map()
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), constrained_layout=True)
    for ax, run in zip(axes, RUNS):
        raw = np.load(RESULT / "data" / f"{run}.npz")
        samples = raw["samples"]
        columns = {str(name): i for i, name in enumerate(raw["columns"])}
        replay = np.load(RESULT / f"{run}_latest40.npz")
        predicted, ground_truth = replay["predicted"], replay["ground_truth"]
        final_error = np.linalg.norm(predicted[:, -1, :2] - ground_truth[:, -1, :2], axis=1)
        worst_window = int(np.argmax(final_error))
        global_start = int(replay["starts"][worst_window])
        local_start = global_start - GLOBAL_OFFSETS[run]
        local_end = min(local_start + 60, len(samples) - 1)  # 1.2 s at 50 Hz
        x, y = samples[:, columns["x"]], samples[:, columns["y"]]
        yaw = samples[:, columns["yaw"]]
        t = samples[:, columns["t"]] - samples[0, columns["t"]]

        ax.imshow(map_image, cmap="gray", origin="lower", extent=extent, alpha=.72)
        points = ax.scatter(x, y, c=t, s=5, cmap="viridis", label="GT full run", zorder=2)
        ax.plot(x[local_start:local_end + 1], y[local_start:local_end + 1],
                color="red", linewidth=4, label="Worst 1.2 s GT segment", zorder=4)
        ax.scatter(x[local_start], y[local_start], marker="*", s=220,
                   color="yellow", edgecolor="black", label="Worst start", zorder=6)
        ax.scatter(x[local_end], y[local_end], marker="X", s=120,
                   color="red", edgecolor="black", label="Worst end", zorder=6)
        ax.arrow(x[local_start], y[local_start], .45*np.cos(yaw[local_start]),
                 .45*np.sin(yaw[local_start]), width=.025, head_width=.16,
                 color="orange", zorder=7)
        ax.annotate(f"start: t={t[local_start]:.2f} s\n"
                    f"({x[local_start]:.2f}, {y[local_start]:.2f}) m\n"
                    f"final error={final_error[worst_window]:.3f} m",
                    (x[local_start], y[local_start]), xytext=(12, 14),
                    textcoords="offset points", fontsize=9,
                    bbox=dict(facecolor="white", alpha=.9), zorder=8)
        ax.set_title(run.replace("_", " "))
        ax.set_xlabel("Map x [m]")
        ax.set_ylabel("Map y [m]")
        ax.set_aspect("equal")
        ax.set_xlim(-4.1, 3.5)
        ax.set_ylim(-10.7, 1.6)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=8)
        fig.colorbar(points, ax=ax, shrink=.72, label="Time from bag start [s]")
        print(f"{run}: replay_window={worst_window}, global_start={global_start}, "
              f"local=[{local_start},{local_end}], time=[{t[local_start]:.3f},{t[local_end]:.3f}] s, "
              f"start_xy=({x[local_start]:.4f},{y[local_start]:.4f}), "
              f"end_xy=({x[local_end]:.4f},{y[local_end]:.4f}), error={final_error[worst_window]:.4f} m")
    fig.suptitle("Latest 40 ms dynamic residual: worst 1.2 s windows on Map1", fontsize=15)
    fig.savefig(OUTPUT, dpi=200)
    print(OUTPUT)


if __name__ == "__main__":
    main()
