#!/usr/bin/env python3
"""F5로 튜닝된 MPPI의 모든 optimal trajectory와 시간별 state를 시각화한다."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "model_tuning/results/current_gru_lane_cost_diagnosis"


def stack(data, key):
    return np.stack([np.asarray(row, dtype=float) for row in data[key]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--stride", type=int, default=8,
                        help="map overlay에서 표시할 trajectory 간격")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    data = np.load(args.run / "map1_lap_data.npz", allow_pickle=True)
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    ref = np.genfromtxt(ROOT / cfg["csv_file_path"], delimiter=",", names=True)
    run_time = np.asarray(data["prediction_t"], float)
    state = {
        "x": stack(data, "prediction_x"),
        "y": stack(data, "prediction_y"),
        "yaw": stack(data, "prediction_yaw"),
        "vx": stack(data, "prediction_vx"),
        "vy": stack(data, "prediction_vy"),
        "yaw_rate": stack(data, "prediction_yaw_rate"),
        "steer": stack(data, "prediction_steer"),
        "speed_cmd": stack(data, "prediction_speed_cmd"),
    }
    horizon_time = np.arange(state["x"].shape[1]) * float(cfg["model_dt"])
    odom = np.asarray(data["odom"], float)

    # 1. Every Nth optimal path, colored by controller run time.
    fig, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.plot(ref["left_x_m"], ref["left_y_m"], "k-", lw=1.2, label="lane boundaries")
    axis.plot(ref["right_x_m"], ref["right_y_m"], "k-", lw=1.2)
    axis.plot(ref["x_m"], ref["y_m"], "k--", lw=.8, alpha=.65, label="centerline")
    axis.plot(odom[:, 1], odom[:, 2], color="white", lw=3.8, zorder=3)
    actual = axis.scatter(odom[:, 1], odom[:, 2], c=odom[:, 0], cmap="turbo", s=7,
                          zorder=4, label="GRU simulator actual")
    indices = np.arange(0, len(run_time), max(1, args.stride))
    segments = [np.column_stack((state["x"][index], state["y"][index])) for index in indices]
    collection = LineCollection(segments, cmap="turbo", linewidths=.75, alpha=.32)
    collection.set_array(run_time[indices]); collection.set_clim(run_time.min(), run_time.max())
    axis.add_collection(collection)
    axis.set_aspect("equal", adjustable="datalim"); axis.grid(alpha=.25)
    axis.set(xlabel="map x [m]", ylabel="map y [m]",
             title=f"Tuned MPPI optimal trajectories · every {max(1,args.stride)}th solve")
    axis.legend(loc="best", fontsize=8)
    fig.colorbar(actual, ax=axis, label="run time [s]")
    path_map = args.run / "tuned_mppi_all_trajectories.png"
    fig.savefig(path_map, dpi=180); plt.close(fig)

    # 2. Every solve on x axis, every future knot on y axis.
    channels = (
        ("vx", "$v_x$ [m/s]", "viridis"),
        ("vy", "$v_y$ [m/s]", "coolwarm"),
        ("yaw_rate", "yaw rate [rad/s]", "coolwarm"),
        ("yaw", "yaw [rad]", "twilight"),
        ("steer", "steer [rad]", "coolwarm"),
        ("speed_cmd", "speed command [m/s]", "viridis"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(16, 13), constrained_layout=True)
    heatmap_ranges = {}
    for axis, (key, label, cmap) in zip(axes.flat, channels):
        values = state[key]
        if key in ("vy", "yaw_rate", "steer"):
            limit = float(np.quantile(np.abs(values), .995)); vmin, vmax = -limit, limit
        else:
            vmin, vmax = float(np.quantile(values, .005)), float(np.quantile(values, .995))
        image = axis.imshow(values.T, origin="lower", aspect="auto", cmap=cmap,
                            extent=(run_time[0], run_time[-1], horizon_time[0], horizon_time[-1]),
                            vmin=vmin, vmax=vmax)
        axis.set(xlabel="controller run time [s]", ylabel="future trajectory time [s]",
                 title=label)
        fig.colorbar(image, ax=axis, label=label)
        heatmap_ranges[key] = [vmin, vmax]
    fig.suptitle("Every published optimal trajectory state · run time × future horizon", fontsize=16)
    path_heatmap = args.run / "tuned_mppi_trajectory_state_heatmaps.png"
    fig.savefig(path_heatmap, dpi=180); plt.close(fig)

    # 3. Five representative solves with their full horizon state traces.
    fractions = np.linspace(0, 1, 5)
    selected = np.unique(np.rint(fractions * (len(run_time) - 1)).astype(int))
    fig, axes = plt.subplots(4, len(selected), figsize=(4 * len(selected), 13),
                             constrained_layout=True)
    cases = []
    for column, index in enumerate(selected):
        color = plt.cm.turbo(index / max(1, len(run_time) - 1))
        axes[0, column].plot(state["x"][index], state["y"][index], color=color, lw=2)
        axes[0, column].scatter(state["x"][index, 0], state["y"][index, 0], c="k", s=24)
        axes[0, column].set_aspect("equal", adjustable="datalim")
        axes[0, column].set(xlabel="x [m]", ylabel="y [m]",
                            title=f"run t={run_time[index]:.2f} s")
        axes[0, column].grid(alpha=.3)
        for row, key, label in ((1, "vx", "$v_x$ [m/s]"),
                                (2, "vy", "$v_y$ [m/s]"),
                                (3, "yaw_rate", "yaw rate [rad/s]")):
            axes[row, column].plot(horizon_time, state[key][index], color=color, lw=2)
            axes[row, column].set(xlabel="future time [s]", ylabel=label)
            axes[row, column].grid(alpha=.3)
        cases.append({"index": int(index), "run_time_s": float(run_time[index]),
                      "vx_min_max": [float(state["vx"][index].min()), float(state["vx"][index].max())],
                      "max_abs_vy": float(np.abs(state["vy"][index]).max()),
                      "max_abs_yaw_rate": float(np.abs(state["yaw_rate"][index]).max())})
    fig.suptitle("Representative optimal trajectories and predicted states", fontsize=16)
    path_cases = args.run / "tuned_mppi_representative_trajectory_states.png"
    fig.savefig(path_cases, dpi=180); plt.close(fig)

    report = {
        "status": str(data["status"]), "run_duration_s": float(odom[-1, 0]),
        "trajectory_messages": int(len(run_time)), "knots_per_trajectory": int(len(horizon_time)),
        "model_dt_s": float(cfg["model_dt"]), "trajectory_horizon_s": float(horizon_time[-1]),
        "actual": {"vx_mean_mps": float(odom[:, 4].mean()), "vx_max_mps": float(odom[:, 4].max()),
                   "max_abs_vy_mps": float(np.abs(odom[:, 5]).max()),
                   "max_abs_yaw_rate_rps": float(np.abs(odom[:, 6]).max())},
        "prediction_ranges": heatmap_ranges, "representative_cases": cases,
    }
    path_json = args.run / "tuned_mppi_trajectory_state_summary.json"
    path_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); print(path_map); print(path_heatmap); print(path_cases)
    if args.show: plt.show()


if __name__ == "__main__":
    main()
