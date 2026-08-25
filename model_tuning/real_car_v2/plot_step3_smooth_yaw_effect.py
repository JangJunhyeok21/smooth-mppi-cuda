#!/usr/bin/env python3
"""Visualize how Step 3 pose smoothing changes yaw and reconstructed XY."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import step_3_identify_classic_model as step3


def reconstruct_segment(samples, columns, source):
    names = {str(name): index for index, name in enumerate(columns)}
    time = samples[:, names["t"]].astype(float)
    time -= time[0]
    median_dt = float(np.median(np.diff(time)))
    window = max(5, int(round(
        step3.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S / median_dt)) | 1)
    window = min(window, len(samples) if len(samples) % 2 else len(samples) - 1)
    if window < 5:
        return None
    order = min(3, window - 2)
    raw_x = samples[:, names["x"]].astype(float)
    raw_y = samples[:, names["y"]].astype(float)
    raw_yaw = np.unwrap(samples[:, names["yaw"]].astype(float))
    smooth_x = savgol_filter(raw_x, window, order, mode="interp")
    smooth_y = savgol_filter(raw_y, window, order, mode="interp")
    smooth_yaw = savgol_filter(raw_yaw, window, order, mode="interp")
    dx = np.gradient(smooth_x, time, edge_order=2)
    dy = np.gradient(smooth_y, time, edge_order=2)
    vx = dx * np.cos(smooth_yaw) + dy * np.sin(smooth_yaw)
    vy = -dx * np.sin(smooth_yaw) + dy * np.cos(smooth_yaw)
    yaw_rate = np.gradient(smooth_yaw, time, edge_order=2)

    # Match Step 3's 40 ms recursive-knot sampling and rectangular integration.
    knot = np.arange(0, len(samples), 2)
    knot_time = time[knot]
    reconstructed = np.empty((len(knot), 3), float)
    reconstructed[0] = (raw_x[0], raw_y[0], raw_yaw[0])
    for index in range(1, len(knot)):
        row = knot[index]
        dt = knot_time[index] - knot_time[index - 1]
        yaw = reconstructed[index - 1, 2]
        reconstructed[index, 0] = reconstructed[index - 1, 0] + (
            vx[row] * np.cos(yaw) - vy[row] * np.sin(yaw)) * dt
        reconstructed[index, 1] = reconstructed[index - 1, 1] + (
            vx[row] * np.sin(yaw) + vy[row] * np.cos(yaw)) * dt
        reconstructed[index, 2] = yaw + yaw_rate[row] * dt

    yaw_difference = smooth_yaw - raw_yaw
    integrated_difference = reconstructed[:, 2] - raw_yaw[knot]
    xy_difference = np.linalg.norm(
        reconstructed[:, :2] - np.c_[raw_x[knot], raw_y[knot]], axis=1)
    return {
        "source": source, "time": time, "knot": knot,
        "raw_x": raw_x, "raw_y": raw_y, "raw_yaw": raw_yaw,
        "smooth_x": smooth_x, "smooth_y": smooth_y,
        "smooth_yaw": smooth_yaw, "yaw_rate": yaw_rate,
        "reconstructed": reconstructed, "window": window,
        "smooth_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_difference ** 2))),
        "smooth_yaw_max_rad": float(np.max(np.abs(yaw_difference))),
        "integrated_yaw_rmse_rad": float(np.sqrt(np.mean(integrated_difference ** 2))),
        "integrated_yaw_max_rad": float(np.max(np.abs(integrated_difference))),
        "reconstructed_xy_rmse_m": float(np.sqrt(np.mean(xy_difference ** 2))),
        "reconstructed_xy_max_m": float(np.max(xy_difference)),
    }


def main():
    records = []
    for path in sorted(Path(step3.DATA_PATH).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            samples = np.asarray(archive["samples"], float)
            columns = np.asarray(archive["columns"])
        names = {str(name): index for index, name in enumerate(columns)}
        bag_ids = (samples[:, names["bag_id"]].astype(int)
                   if "bag_id" in names else np.zeros(len(samples), int))
        for bag_id in np.unique(bag_ids):
            record = reconstruct_segment(
                samples[bag_ids == bag_id], columns,
                f"{path.name} / bag_id={bag_id}")
            if record is not None:
                records.append(record)
    if not records:
        raise RuntimeError(f"No usable Step 3 NPZ found in {step3.DATA_PATH}")

    ordered = sorted(records, key=lambda item: item["smooth_yaw_rmse_rad"])
    positions = (len(ordered) // 2,
                 min(len(ordered) - 1, int(np.ceil(.95 * len(ordered))) - 1),
                 len(ordered) - 1)
    labels = ("median", "P95", "worst")
    selected = [ordered[index] for index in positions]
    figure, axes = plt.subplots(3, 2, figsize=(15, 15), constrained_layout=True)
    for row, (label, record) in enumerate(zip(labels, selected)):
        xy = axes[row, 0]
        xy.plot(record["raw_x"], record["raw_y"], "k-", linewidth=2,
                label="raw MCL XY")
        xy.plot(record["smooth_x"], record["smooth_y"], color="tab:green",
                linestyle=":", linewidth=1.5, label="smoothed XY")
        xy.plot(record["reconstructed"][:, 0], record["reconstructed"][:, 1],
                color="tab:red", linewidth=1.5,
                label="40 ms integration of pose-derived states")
        xy.set_title(f"{label}: XY trajectory\n{record['source']}")
        xy.set_xlabel("global x [m]"); xy.set_ylabel("global y [m]")
        xy.axis("equal"); xy.grid(alpha=.25); xy.legend(fontsize=8)

        yaw = axes[row, 1]
        yaw.plot(record["time"], record["raw_yaw"], "k-", linewidth=2,
                 label="raw MCL yaw")
        yaw.plot(record["time"], record["smooth_yaw"], color="tab:green",
                 linestyle="--", linewidth=1.5, label="smoothed yaw")
        knot = record["knot"]
        yaw.plot(record["time"][knot], record["reconstructed"][:, 2],
                 color="tab:red", linewidth=1.4,
                 label="40 ms integral of derived yaw-rate")
        yaw.set_title(
            f"smooth RMSE/max={record['smooth_yaw_rmse_rad']:.4f}/"
            f"{record['smooth_yaw_max_rad']:.4f} rad | integrated RMSE/max="
            f"{record['integrated_yaw_rmse_rad']:.4f}/"
            f"{record['integrated_yaw_max_rad']:.4f} rad")
        yaw.set_xlabel("time [s]"); yaw.set_ylabel("unwrapped yaw [rad]")
        yaw.grid(alpha=.25); yaw.legend(fontsize=8)
    figure.suptitle(
        "Step 3 adjust_states_to_pose: smoothing and 40 ms reintegration effect\n"
        f"Savitzky-Golay target window={step3.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S:g} s",
        fontsize=15)

    output_dir = Path(step3.OUTPUT_DIR) / "smooth_yaw_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "smooth_yaw_xy_and_time_comparison.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    summary_rows = [{key: value for key, value in record.items()
                     if isinstance(value, (str, int, float))}
                    for record in records]
    aggregate = {
        "data_path": str(Path(step3.DATA_PATH).resolve()),
        "segments": len(records),
        "smoothing_window_s": step3.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S,
        "smooth_yaw_rmse_median_rad": float(np.median(
            [item["smooth_yaw_rmse_rad"] for item in records])),
        "smooth_yaw_rmse_p95_rad": float(np.quantile(
            [item["smooth_yaw_rmse_rad"] for item in records], .95)),
        "integrated_yaw_rmse_median_rad": float(np.median(
            [item["integrated_yaw_rmse_rad"] for item in records])),
        "integrated_yaw_rmse_p95_rad": float(np.quantile(
            [item["integrated_yaw_rmse_rad"] for item in records], .95)),
        "reconstructed_xy_rmse_median_m": float(np.median(
            [item["reconstructed_xy_rmse_m"] for item in records])),
        "reconstructed_xy_rmse_p95_m": float(np.quantile(
            [item["reconstructed_xy_rmse_m"] for item in records], .95)),
        "representatives": {label: row["source"]
                            for label, row in zip(labels, selected)},
        "per_segment": summary_rows,
    }
    summary_path = output_dir / "smooth_yaw_effect.json"
    summary_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: value for key, value in aggregate.items()
                      if key != "per_segment"}, indent=2))
    print(f"plot: {plot_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
