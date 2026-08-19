#!/usr/bin/env python3
"""Extract and time-align Ackermann commands and mocap odometry from rosbag2."""

import argparse
from pathlib import Path

import numpy as np


def _stamp(msg, fallback_ns):
    stamp = msg.header.stamp
    ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return ns if ns else fallback_ns


def _yaw(q):
    return np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_bag(uri, odom_topic, drive_topic, time_source):
    # ROS Humble's Python packages must be run with its supported Python (normally 3.10).
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise SystemExit(
            "rosbag2 Python bindings are unavailable. Source /opt/ros/humble/setup.bash "
            "and use the ROS Python interpreter. Original error: " + str(exc))

    storage_id = "mcap" if str(uri).endswith(".mcap") else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    for topic in (odom_topic, drive_topic):
        if topic not in types:
            raise SystemExit(f"topic {topic!r} not found; available: {sorted(types)}")
    msg_types = {t: get_message(types[t]) for t in (odom_topic, drive_topic)}
    odom, drive = [], []
    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(raw, msg_types[topic])
        t_ns = record_ns if time_source == "record" else _stamp(msg, record_ns)
        t = t_ns * 1e-9
        if topic == odom_topic:
            p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
            odom.append((t, p.x, p.y, _yaw(q), tw.linear.x, tw.linear.y, tw.angular.z))
        else:
            d = msg.drive
            drive.append((t, d.steering_angle, d.acceleration, d.speed))
    return np.asarray(odom, dtype=np.float64), np.asarray(drive, dtype=np.float64)


def align(odom, drive, dt, max_control_age, accel_source, control_delay):
    if len(odom) < 2 or len(drive) < 2:
        raise SystemExit("bag does not contain enough odometry/control samples")
    odom, drive = odom[np.argsort(odom[:, 0])], drive[np.argsort(drive[:, 0])]
    # Drop duplicate timestamps before interpolation.
    odom = odom[np.r_[True, np.diff(odom[:, 0]) > 1e-9]]
    drive = drive[np.r_[True, np.diff(drive[:, 0]) > 1e-9]]
    start, end = max(odom[0, 0], drive[0, 0]), min(odom[-1, 0], drive[-1, 0])
    if end <= start:
        raise SystemExit(
            f"odom/control timestamps do not overlap: odom=[{odom[0,0]:.3f}, {odom[-1,0]:.3f}], "
            f"control=[{drive[0,0]:.3f}, {drive[-1,0]:.3f}]. Use --time-source record.")
    t = np.arange(start, end, dt)
    out = np.empty((len(t), 10), dtype=np.float64)
    out[:, 0] = t - t[0]
    for col in range(1, 7):
        values = odom[:, col].copy()
        if col == 3:
            values = np.unwrap(values)
        out[:, col] = np.interp(t, odom[:, 0], values)
    # Zero-order hold is correct for commands; interpolation invents actuator inputs.
    command_time = t - control_delay
    idx = np.searchsorted(drive[:, 0], command_time, side="right") - 1
    valid = (idx >= 0) & ((command_time - drive[np.maximum(idx, 0), 0]) <= max_control_age)
    idx = np.maximum(idx, 0)
    out[:, 7] = drive[idx, 1]
    out[:, 8] = drive[idx, 2]
    out[:, 9] = drive[idx, 3]
    if accel_source == "speed-derivative":
        out[:, 8] = np.gradient(out[:, 9], dt)
    return out[valid]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag", help=".mcap file or rosbag2 directory")
    p.add_argument("-o", "--output", default="model_tuning/data/aligned.npz")
    p.add_argument("--odom-topic", default="/mocap_odom")
    p.add_argument(
        "--drive-topic",
        default="/drive",
        help="actual manual/autonomous vehicle command topic (default: /drive)",
    )
    p.add_argument("--dt", type=float, default=0.02, help="uniform sample period [s]")
    p.add_argument("--max-control-age", type=float, default=0.1)
    p.add_argument("--control-delay", type=float, default=0.0,
                   help="command-to-motion delay [s]; compare e.g. 0.0, 0.04, 0.08")
    p.add_argument("--time-source", choices=("record", "header"), default="record",
                   help="record time avoids mixed ROS clock domains (default: record)")
    p.add_argument("--accel-source", choices=("acceleration", "speed-derivative"),
                   default="acceleration")
    args = p.parse_args()
    odom, drive = read_bag(args.bag, args.odom_topic, args.drive_topic, args.time_source)
    samples = align(odom, drive, args.dt, args.max_control_age, args.accel_source,
                    args.control_delay)
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, samples=samples, dt=args.dt,
                        columns=np.array(["t", "x", "y", "yaw", "vx", "vy", "omega",
                                          "steer", "accel", "speed_cmd"]),
                        odom_topic=args.odom_topic, drive_topic=args.drive_topic)
    print(f"saved {len(samples)} aligned samples to {path}")


if __name__ == "__main__":
    main()
