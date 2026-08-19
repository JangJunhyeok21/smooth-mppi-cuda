#!/usr/bin/env python3
"""Compare upstream /ackermann_cmd and actuator-facing /drive commands."""
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-16_54_33/rosbag2_2026_08_08-16_54_33_0.db3")
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/results/ifac0808_ackermann_vs_drive"
MAX_HOLD_AGE_S = 0.20
DELAY_MIN_S = -0.50
DELAY_MAX_S = 0.50
DELAY_STEP_S = 0.002
SHOW_PLOTS = False

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
import matplotlib.pyplot as plt
import numpy as np


def read_commands(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topics = ("/ackermann_cmd", "/drive", "/teleop")
    streams = {topic: [] for topic in topics}
    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in streams:
            continue
        message = deserialize_message(raw, get_message(types[topic]))
        stamp = message.header.stamp
        header_time = stamp.sec + stamp.nanosec * 1e-9
        if header_time == 0.0:
            header_time = record_ns * 1e-9
        drive = message.drive
        streams[topic].append((header_time, record_ns * 1e-9,
                               drive.steering_angle, drive.speed,
                               drive.acceleration, drive.steering_angle_velocity,
                               drive.jerk))
    return {key: np.asarray(value, np.float64) for key, value in streams.items()}


def causal_error(source, destination, column, delay):
    # destination(t) is compared with newest source whose stamp <= t-delay.
    query = destination[:, 0] - delay
    index = np.searchsorted(source[:, 0], query, side="right") - 1
    clipped = np.maximum(index, 0)
    valid = ((index >= 0) & ((query-source[clipped, 0]) <= MAX_HOLD_AGE_S))
    error = destination[valid, column] - source[index[valid], column]
    return error, valid, index


def stats(error, reference, estimate):
    correlation = (float(np.corrcoef(reference, estimate)[0, 1])
                   if np.std(reference) > 1e-10 and np.std(estimate) > 1e-10
                   else None)
    return {"samples": int(len(error)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias_drive_minus_ackermann": float(np.mean(error)),
            "p95_abs": float(np.quantile(np.abs(error), .95)),
            "max_abs": float(np.max(np.abs(error))), "correlation": correlation,
            "exact_match_percent": float(100*np.mean(np.abs(error) < 1e-6))}


def main():
    streams = read_commands(BAG_PATH)
    ack, drive = streams["/ackermann_cmd"], streams["/drive"]
    teleop = streams["/teleop"]
    fields = ((2, "steering_angle", "rad"), (3, "speed", "m/s"),
              (4, "acceleration", "m/s^2"))
    delays = np.arange(DELAY_MIN_S, DELAY_MAX_S + .5*DELAY_STEP_S, DELAY_STEP_S)
    result = {"bag": str(BAG_PATH), "counts": {"ackermann_cmd": len(ack), "drive": len(drive),
                                                  "teleop": len(teleop)},
              "median_period_s": {"ackermann_cmd": float(np.median(np.diff(ack[:, 0]))),
                                  "drive": float(np.median(np.diff(drive[:, 0])))},
              "header_minus_record_time_ms": {
                  "ackermann_cmd_median": float(1000*np.median(ack[:, 0]-ack[:, 1])),
                  "drive_median": float(1000*np.median(drive[:, 0]-drive[:, 1]))},
              "fields": {}}
    best_delays = {}
    for column, name, unit in fields:
        scores = []
        for delay in delays:
            error, _, _ = causal_error(ack, drive, column, delay)
            scores.append(np.mean(np.abs(error)) if len(error) else np.inf)
        best_delay = float(delays[int(np.argmin(scores))])
        error, valid, index = causal_error(ack, drive, column, best_delay)
        reference = ack[index[valid], column]
        estimate = drive[valid, column]
        item = stats(error, reference, estimate)
        item.update({"best_drive_delay_s": best_delay, "unit": unit,
                     "ackermann_range": [float(ack[:, column].min()), float(ack[:, column].max())],
                     "drive_range": [float(drive[:, column].min()), float(drive[:, column].max())]})
        result["fields"][name] = item
        best_delays[name] = best_delay

    # Nearest stamp separation is independent of command values.
    right = np.searchsorted(ack[:, 0], drive[:, 0]); right = np.clip(right, 1, len(ack)-1)
    left = right-1
    nearest = np.where(np.abs(ack[right, 0]-drive[:, 0]) <
                       np.abs(ack[left, 0]-drive[:, 0]), right, left)
    separation = np.abs(ack[nearest, 0]-drive[:, 0])
    result["nearest_stamp_separation_ms"] = {
        "median": float(1000*np.median(separation)),
        "p95": float(1000*np.quantile(separation, .95)),
        "max": float(1000*np.max(separation))}

    # Identify the command source reaching actuator-facing /ackermann_cmd.
    # Match on [steering, speed]; acceleration is derived differently by nodes.
    def nearest_indices(source, query_time):
        right = np.searchsorted(source[:, 0], query_time)
        right = np.clip(right, 1, len(source)-1); left = right-1
        return np.where(np.abs(source[right, 0]-query_time) <
                        np.abs(source[left, 0]-query_time), right, left)
    drive_index = nearest_indices(drive, ack[:, 0])
    teleop_index = nearest_indices(teleop, ack[:, 0])
    drive_value_error = np.max(np.abs(ack[:, [2,3]]-drive[drive_index][:, [2,3]]), axis=1)
    teleop_value_error = np.max(np.abs(ack[:, [2,3]]-teleop[teleop_index][:, [2,3]]), axis=1)
    drive_age = np.abs(ack[:, 0]-drive[drive_index, 0])
    teleop_age = np.abs(ack[:, 0]-teleop[teleop_index, 0])
    from_drive = (drive_value_error < 1e-6) & (drive_age < .10)
    from_teleop = (teleop_value_error < 1e-6) & (teleop_age < .10) & ~from_drive
    other = ~(from_drive | from_teleop)
    result["ackermann_source_classification"] = {
        "matches_drive_percent": float(100*np.mean(from_drive)),
        "matches_teleop_percent": float(100*np.mean(from_teleop)),
        "unclassified_percent": float(100*np.mean(other)),
        "negative_speed_samples": int(np.sum(ack[:,3] < 0)),
        "negative_speed_matches_teleop_percent": float(
            100*np.mean(from_teleop[ack[:,3] < 0])) if np.any(ack[:,3] < 0) else 0.0}

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    start = max(ack[0, 0], drive[0, 0]); relative_ack = ack[:, 0]-start; relative_drive = drive[:, 0]-start
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for axis, (column, name, unit) in zip(axes, fields):
        axis.step(relative_ack, ack[:, column], where="post", lw=1, label=f"/ackermann_cmd {name}")
        axis.step(relative_drive, drive[:, column], where="post", lw=.9, label=f"/drive {name}")
        metric = result["fields"][name]
        axis.set_ylabel(f"{name} [{unit}]"); axis.grid(alpha=.25); axis.legend()
        axis.set_title(f"best delay={metric['best_drive_delay_s']*1000:.1f} ms, "
                       f"MAE={metric['mae']:.4f}, exact={metric['exact_match_percent']:.1f}%")
    axes[-1].set_xlabel("time from common start [s]")
    fig.tight_layout(); fig.savefig(OUTPUT_PATH / "ackermann_cmd_vs_drive.png", dpi=180)
    if SHOW_PLOTS: plt.show()
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
