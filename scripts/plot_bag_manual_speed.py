#!/usr/bin/env python3
"""Plot odometry and drive speed against recorded manual-mode state."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = {"/odom", "/drive", "/manual_control", "/joy"}


def read_bag(uri: str):
    reader = rosbag2_py.SequentialReader()
    storage_id = "mcap" if any(Path(uri).glob("*.mcap")) else "sqlite3"
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    data = {topic: [] for topic in TOPICS}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic not in data:
            continue
        msg = deserialize_message(raw, get_message(types[topic]))
        if topic == "/odom":
            value = msg.twist.twist.linear.x
        elif topic == "/drive":
            value = msg.drive.speed
        elif topic == "/manual_control":
            value = bool(msg.data)
        else:
            value = bool(len(msg.buttons) > 4 and msg.buttons[4])
        data[topic].append((stamp * 1e-9, value))
    return {key: np.asarray(value) for key, value in data.items()}


def state_at(query_t, samples, default=False):
    if len(samples) == 0:
        return np.full(query_t.shape, default, dtype=bool)
    indices = np.searchsorted(samples[:, 0], query_t, side="right") - 1
    valid = indices >= 0
    result = np.full(query_t.shape, default, dtype=bool)
    result[valid] = samples[indices[valid], 1].astype(bool)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--output", default="bag_manual_speed.png")
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    data = read_bag(args.bag)
    nonempty = [samples[0, 0] for samples in data.values() if len(samples)]
    t0 = min(nonempty)
    for samples in data.values():
        if len(samples):
            samples[:, 0] -= t0

    odom, drive = data["/odom"], data["/drive"]
    manual, joy = data["/manual_control"], data["/joy"]
    manual_at_drive = state_at(drive[:, 0], manual)
    if len(manual):
        idx = np.searchsorted(manual[:, 0], drive[:, 0], side="right") - 1
        heartbeat_ok = (idx >= 0) & ((drive[:, 0] - manual[np.maximum(idx, 0), 0]) <= args.timeout)
    else:
        heartbeat_ok = np.zeros(len(drive), dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1]})
    axes[0].plot(odom[:, 0], odom[:, 1], lw=1.0, label="odom vx")
    axes[0].plot(drive[:, 0], drive[:, 1], lw=0.8, alpha=0.85, label="drive speed")
    axes[0].set_ylabel("speed [m/s]")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    auto = (~manual_at_drive) & heartbeat_ok
    stopped = manual_at_drive | (~heartbeat_ok)
    axes[1].scatter(drive[auto, 0], drive[auto, 1], s=3, label="auto + heartbeat")
    axes[1].scatter(drive[stopped, 0], drive[stopped, 1], s=5,
                    label="manual or heartbeat timeout")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("drive speed [m/s]")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].step(manual[:, 0], manual[:, 1], where="post", label="manual_control")
    if len(joy):
        axes[2].step(joy[:, 0], joy[:, 1], where="post", alpha=0.7, label="Joy LB")
    axes[2].set_yticks([0, 1], ["auto", "manual"])
    axes[2].set_xlabel("time from bag start [s]")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)
    if args.start is not None or args.end is not None:
        axes[2].set_xlim(args.start, args.end)
    fig.suptitle(Path(args.bag).name)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    manual_drive = drive[stopped, 1]
    explicit_manual_drive = drive[manual_at_drive, 1]
    odom_indices = np.clip(np.searchsorted(odom[:, 0], drive[:, 0]), 0, len(odom) - 1)
    previous_odom_indices = np.maximum(odom_indices - 1, 0)
    use_previous = np.abs(odom[previous_odom_indices, 0] - drive[:, 0]) < np.abs(
        odom[odom_indices, 0] - drive[:, 0])
    odom_indices[use_previous] = previous_odom_indices[use_previous]
    stationary_manual = manual_at_drive & (np.abs(odom[odom_indices, 1]) < 0.1)
    print(f"bag={args.bag}")
    print(f"odom_samples={len(odom)} drive_samples={len(drive)} manual_samples={len(manual)} joy_samples={len(joy)}")
    print(f"manual_or_timeout_drive_samples={len(manual_drive)}")
    if len(manual_drive):
        print(f"manual_or_timeout_abs_speed_max={np.max(np.abs(manual_drive)):.6f}")
        print(f"manual_or_timeout_nonzero_gt_1e-3={np.count_nonzero(np.abs(manual_drive) > 1e-3)}")
    if len(explicit_manual_drive):
        print(f"explicit_manual_drive_samples={len(explicit_manual_drive)}")
        print(f"explicit_manual_abs_speed_max={np.max(np.abs(explicit_manual_drive)):.6f}")
    if np.any(stationary_manual):
        print(f"stationary_manual_drive_samples={np.count_nonzero(stationary_manual)}")
        print(f"stationary_manual_abs_speed_max={np.max(np.abs(drive[stationary_manual, 1])):.6f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
