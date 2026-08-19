#!/usr/bin/env python3
"""Copy a ROS 2 bag and add offline/causal lateral-velocity topics for Foxglove."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from helper_lateral_velocity_kf import LateralVelocityKFParams, estimate_dataset
from offline_lateral_velocity_smoother import smooth_segment_vy


def ema(values, alpha):
    result = values.copy()
    for index in range(1, len(result)):
        result[index] = alpha*values[index] + (1-alpha)*result[index-1]
    return result


def estimates(npz_path):
    archive = np.load(npz_path)
    samples = archive["samples"].astype(float)
    columns = archive["columns"]
    names = {str(name): index for index, name in enumerate(columns)}
    dt = float(archive["dt"])
    cfg = yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    signs = archive["imu_axis_signs"].astype(float)
    alpha = float(archive["imu_ema_alpha"])
    params = LateralVelocityKFParams(
        mass=float(cfg["mass"]), yaw_inertia=float(cfg["I_z"]),
        l_f=float(cfg["l_f"]), l_r=float(cfg["l_r"]), dt=dt,
        min_longitudinal_speed=float(cfg["kf_min_vx"]),
        low_speed_threshold=float(cfg["kf_low_speed_threshold"]),
        max_abs_vy=float(cfg["kf_max_abs_vy"]),
        process_var_vy=float(cfg["kf_q_vy"]),
        process_var_yaw_rate=float(cfg["kf_q_yaw_rate"]),
        measurement_var_lateral_accel=float(cfg["kf_r_lateral_accel"]),
        measurement_var_yaw_rate=float(cfg["kf_r_yaw_rate"]),
        initial_var_vy=float(cfg["kf_initial_p_vy"]),
        initial_var_yaw_rate=float(cfg["kf_initial_p_yaw_rate"]),
        imu_lateral_accel_sign=float(cfg["imu_lateral_accel_sign"]),
        process_var_ay_bias=float(cfg["kf_q_ay_bias"]),
        initial_var_ay_bias=float(cfg["kf_initial_p_ay_bias"]),
        max_abs_ay_bias=float(cfg["kf_max_abs_ay_bias"]),
        measurement_var_pose_vy=float(cfg["kf_r_pose_vy"]),
        pose_vy_gate=float(cfg["kf_pose_vy_gate"]))
    causal_vy, yaw_rate = estimate_dataset(
        samples, columns, dt, params,
        steer_scale=float(cfg["kf_steer_scale"]),
        steer_bias=float(cfg["kf_steer_bias"]), max_steer=float(cfg["kf_max_steer"]),
        imu_ema_alpha=alpha, imu_wz_sign=float(signs[0]), imu_ay_sign=float(signs[2]),
        use_pose_vy=bool(cfg["kf_pose_vy_enabled"]),
        pose_window_s=float(cfg["kf_pose_vy_window_s"]))
    lateral_accel = ema(float(signs[2])*samples[:, names["imu_ay"]], alpha)
    offline_vy = np.empty(len(samples))
    diagnostics = []
    for segment in np.unique(samples[:, names["bag_id"]].astype(int)):
        indices = np.flatnonzero(samples[:, names["bag_id"]].astype(int) == segment)
        part = samples[indices]
        value, report = smooth_segment_vy(
            part[:, names["x"]], part[:, names["y"]], part[:, names["yaw"]],
            part[:, names["vx"]], yaw_rate[indices], lateral_accel[indices], dt)
        if not report["usable"]:
            value = causal_vy[indices]
        offline_vy[indices] = value
        diagnostics.append({"segment": int(segment), "samples": len(indices), **report})
    return archive, samples, names, causal_vy, offline_vy, diagnostics


def stamp(message, timestamp_ns):
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="input .db3")
    parser.add_argument("extract", type=Path, help="step_1 extracted .npz")
    parser.add_argument("output", type=Path, help="new rosbag2 directory")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    import rosbag2_py
    from geometry_msgs.msg import Quaternion
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from std_msgs.msg import Float64

    archive, samples, names, causal_vy, offline_vy, diagnostics = estimates(args.extract)
    sidecar = json.loads(args.extract.with_suffix(".json").read_text())
    starts = {int(item["bag_id"]): float(item["source_start_s"])
              for item in sidecar["segments"]}
    epoch = float(archive["alignment_start_epoch_s"])
    generated = []
    for index, row in enumerate(samples):
        segment = int(row[names["bag_id"]])
        timestamp_ns = int(round((epoch + starts[segment] + row[names["t"]])*1e9))
        offline = Float64(); offline.data = float(offline_vy[index])
        causal = Float64(); causal.data = float(causal_vy[index])
        odom = Odometry(); stamp(odom, timestamp_ns)
        odom.header.frame_id = "map"; odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(row[names["x"]])
        odom.pose.pose.position.y = float(row[names["y"]])
        half = .5*float(row[names["yaw"]])
        odom.pose.pose.orientation = Quaternion(z=float(np.sin(half)), w=float(np.cos(half)))
        odom.twist.twist.linear.x = float(row[names["vx"]])
        odom.twist.twist.linear.y = float(offline_vy[index])
        odom.twist.twist.angular.z = float(row[names["omega"]])
        generated.append((timestamp_ns, "/offline_vy", serialize_message(offline)))
        generated.append((timestamp_ns, "/causal_kf_vy", serialize_message(causal)))
        generated.append((timestamp_ns, "/offline_vy_odom", serialize_message(odom)))
    generated.sort(key=lambda item: item[0])

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(args.output), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    existing = {topic.name for topic in reader.get_all_topics_and_types()}
    overlap = existing & {"/offline_vy", "/causal_kf_vy", "/offline_vy_odom"}
    if overlap:
        raise RuntimeError(f"input already contains generated topics: {sorted(overlap)}")
    for topic in reader.get_all_topics_and_types():
        writer.create_topic(topic)
    for name, type_name in (("/offline_vy", "std_msgs/msg/Float64"),
                            ("/causal_kf_vy", "std_msgs/msg/Float64"),
                            ("/offline_vy_odom", "nav_msgs/msg/Odometry")):
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name, type=type_name, serialization_format="cdr"))
    generated_index = 0
    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        while generated_index < len(generated) and generated[generated_index][0] <= timestamp_ns:
            generated_ns, generated_topic, generated_raw = generated[generated_index]
            writer.write(generated_topic, generated_raw, generated_ns)
            generated_index += 1
        writer.write(topic, raw, timestamp_ns)
    while generated_index < len(generated):
        generated_ns, generated_topic, generated_raw = generated[generated_index]
        writer.write(generated_topic, generated_raw, generated_ns)
        generated_index += 1
    report = {"input_bag": str(args.bag.resolve()), "extract": str(args.extract.resolve()),
              "output_bag": str(args.output.resolve()), "source_messages_copied": True,
              "generated_samples_per_topic": len(samples),
              "topics_added": ["/offline_vy", "/causal_kf_vy", "/offline_vy_odom"],
              "offline_vy": {"mean": float(np.mean(offline_vy)),
                             "min": float(np.min(offline_vy)),
                             "max": float(np.max(offline_vy))},
              "segments": diagnostics}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
