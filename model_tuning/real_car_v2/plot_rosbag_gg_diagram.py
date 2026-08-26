#!/usr/bin/env python3
"""Create a bias/sign-corrected GG diagram directly from a rosbag2 bag."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "model_tuning/results/gg_diagrams"


def stamp_seconds(msg, record_ns):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
        return record_ns * 1.0e-9
    return stamp.sec + stamp.nanosec * 1.0e-9


def storage_file(path):
    path = path.resolve()
    if path.is_file():
        return path
    files = sorted(path.glob("*.mcap"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one MCAP in {path}, found {len(files)}")
    return files[0]


def read_bag(path, imu_topic, odom_topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage = storage_file(path)
    storage_id = "mcap" if storage.suffix == ".mcap" else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(storage), storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    missing = [topic for topic in (imu_topic, odom_topic) if topic not in types]
    if missing:
        raise RuntimeError(f"Missing topics {missing}; available={sorted(types)}")
    message_types = {topic: get_message(types[topic])
                     for topic in (imu_topic, odom_topic)}
    imu = []
    odom = []
    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in message_types:
            continue
        msg = deserialize_message(raw, message_types[topic])
        timestamp = stamp_seconds(msg, record_ns)
        if topic == imu_topic:
            imu.append((timestamp, msg.linear_acceleration.x,
                        msg.linear_acceleration.y))
        else:
            velocity = msg.twist.twist.linear
            odom.append((timestamp, velocity.x, velocity.y))
    return storage, np.asarray(imu, dtype=float), np.asarray(odom, dtype=float)


def causal_ema(values, alpha):
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def causal_speed(imu_time, odom):
    indices = np.searchsorted(odom[:, 0], imu_time, side="right") - 1
    valid = indices >= 0
    speed = np.full(len(imu_time), np.nan)
    speed[valid] = np.hypot(odom[indices[valid], 1], odom[indices[valid], 2])
    return speed


def metrics(acceleration):
    resultant = np.linalg.norm(acceleration, axis=1)
    return {
        "samples": int(len(acceleration)),
        "ax_min_mps2": float(np.min(acceleration[:, 0])),
        "ax_max_mps2": float(np.max(acceleration[:, 0])),
        "ay_min_mps2": float(np.min(acceleration[:, 1])),
        "ay_max_mps2": float(np.max(acceleration[:, 1])),
        "abs_ay_p95_mps2": float(np.quantile(np.abs(acceleration[:, 1]), 0.95)),
        "resultant_p95_mps2": float(np.quantile(resultant, 0.95)),
        "resultant_p99_mps2": float(np.quantile(resultant, 0.99)),
        "resultant_max_mps2": float(np.max(resultant)),
    }


def draw(axis, acceleration, title, limit):
    values = acceleration / 9.81
    image = axis.hexbin(values[:, 1], values[:, 0], gridsize=65, mincnt=1,
                        bins="log", cmap="turbo",
                        extent=(-limit, limit, -limit, limit))
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius, style in ((0.5, "--"), (1.0, "-")):
        axis.plot(radius * np.cos(angle), radius * np.sin(angle), style,
                  color="black", linewidth=0.8, alpha=0.6, label=f"{radius:g} g")
    axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.4)
    axis.set_title(f"{title}\nN={len(acceleration):,}")
    axis.set_xlabel("lateral acceleration ay [g]")
    axis.set_ylabel("longitudinal acceleration ax [g]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right", fontsize=8)
    return image


def draw_time_trajectory(axis, acceleration, timestamps, title, limit):
    values = acceleration / 9.81
    elapsed = timestamps - timestamps[0]
    points = np.column_stack((values[:, 1], values[:, 0]))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    # Do not connect across a recording/timestamp discontinuity.
    dt = np.diff(timestamps)
    valid_segment = (dt > 0.0) & (dt < 0.2)
    collection = LineCollection(
        segments[valid_segment], cmap="viridis", linewidths=0.45, alpha=0.65)
    collection.set_array(elapsed[:-1][valid_segment])
    axis.add_collection(collection)
    scatter = axis.scatter(points[:, 0], points[:, 1], c=elapsed, cmap="viridis",
                           s=3.0, linewidths=0, alpha=0.8)
    axis.scatter(points[0, 0], points[0, 1], marker="o", s=55,
                 facecolor="lime", edgecolor="black", linewidth=0.8,
                 label="start", zorder=4)
    axis.scatter(points[-1, 0], points[-1, 1], marker="X", s=65,
                 facecolor="red", edgecolor="black", linewidth=0.8,
                 label="end", zorder=4)
    arrow_step = max(1, len(points) // 35)
    starts = points[:-1:arrow_step]
    deltas = points[1::arrow_step] - starts[:len(points[1::arrow_step])]
    useful = np.linalg.norm(deltas, axis=1) > 0.015
    axis.quiver(starts[useful, 0], starts[useful, 1],
                deltas[useful, 0], deltas[useful, 1],
                angles="xy", scale_units="xy", scale=1.0, width=0.0025,
                headwidth=4.0, color="black", alpha=0.55, zorder=3)
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius, style in ((0.5, "--"), (1.0, "-")):
        axis.plot(radius * np.cos(angle), radius * np.sin(angle), style,
                  color="gray", linewidth=0.8, alpha=0.7)
    axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.35)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{title}\ncolor: elapsed time, arrows: temporal direction")
    axis.set_xlabel("lateral acceleration ay [g]")
    axis.set_ylabel("longitudinal acceleration ax [g]")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right", fontsize=8)
    return scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--moving-speed", type=float, default=0.5)
    parser.add_argument("--ema-alpha", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    storage, imu, odom = read_bag(args.bag, args.imu_topic, args.odom_topic)
    if not len(imu) or not len(odom):
        raise RuntimeError(f"No data: imu={len(imu)}, odom={len(odom)}")

    signs = np.array([float(config.get("imu_ax_sign", 1.0)),
                      float(config.get("imu_ay_sign", 1.0))])
    biases = np.array([float(config.get("imu_ax_bias", 0.0)),
                       float(config.get("imu_ay_bias", 0.0))])
    corrected = imu[:, 1:3] * signs - biases
    filtered = causal_ema(corrected, args.ema_alpha)
    speed = causal_speed(imu[:, 0], odom)
    finite = np.isfinite(filtered).all(axis=1)
    moving = finite & np.isfinite(speed) & (speed >= args.moving_speed)
    if not np.any(moving):
        raise RuntimeError("No moving IMU samples pass the speed threshold")

    plotted = filtered[finite]
    moving_values = filtered[moving]
    abs_g = np.abs(plotted / 9.81)
    limit = max(1.0, float(np.quantile(abs_g, 0.997)))
    limit = np.ceil(limit * 4.0) / 4.0
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    image0 = draw(axes[0], plotted, "All bias-corrected IMU samples", limit)
    image1 = draw(axes[1], moving_values,
                  f"Moving samples (speed >= {args.moving_speed:g} m/s)", limit)
    figure.colorbar(image0, ax=axes[0], label="log sample density")
    figure.colorbar(image1, ax=axes[1], label="log sample density")
    figure.suptitle(f"GG diagram: {storage.parent.name}", fontsize=14)

    output_dir = args.output.resolve() / storage.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "gg_diagram.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    moving_time = imu[moving, 0]
    trajectory_limit = max(
        1.0, float(np.quantile(np.abs(moving_values / 9.81), 0.99)))
    trajectory_limit = np.ceil(trajectory_limit * 4.0) / 4.0
    trajectory_figure, trajectory_axes = plt.subplots(
        2, 2, figsize=(14, 12), constrained_layout=True)
    split_indices = np.linspace(0, len(moving_values), 5, dtype=int)
    trajectory_image = None
    for panel, axis in enumerate(trajectory_axes.flat):
        begin, end = split_indices[panel], split_indices[panel + 1]
        segment_time = moving_time[begin:end]
        absolute_begin = segment_time[0] - moving_time[0]
        absolute_end = segment_time[-1] - moving_time[0]
        trajectory_image = draw_time_trajectory(
            axis, moving_values[begin:end], segment_time,
            f"Segment {panel + 1}: t={absolute_begin:.1f}-{absolute_end:.1f} s",
            trajectory_limit)
    trajectory_figure.colorbar(
        trajectory_image, ax=trajectory_axes, label="time within each segment [s]",
        shrink=0.85)
    trajectory_figure.suptitle(f"Time-ordered GG trajectory: {storage.parent.name}",
                               fontsize=13)
    trajectory_path = output_dir / "gg_time_trajectory.png"
    trajectory_figure.savefig(trajectory_path, dpi=200)
    plt.close(trajectory_figure)
    report = {
        "bag": str(storage),
        "imu_topic": args.imu_topic,
        "odom_topic": args.odom_topic,
        "imu_samples": int(len(imu)),
        "odom_samples": int(len(odom)),
        "imu_signs_ax_ay": signs.tolist(),
        "imu_biases_ax_ay_mps2": biases.tolist(),
        "ema_alpha": args.ema_alpha,
        "moving_speed_threshold_mps": args.moving_speed,
        "all": metrics(plotted),
        "moving": metrics(moving_values),
    }
    report_path = output_dir / "gg_diagram.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"GG diagram: {plot_path}")
    print(f"time-ordered GG trajectory: {trajectory_path}")
    print(f"summary: {report_path}")


if __name__ == "__main__":
    main()
