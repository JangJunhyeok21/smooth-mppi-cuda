#!/usr/bin/env python3
"""Extract the exact real-car MPPI observation path from rosbag2.

Pose is taken from /newmcl_pose, body velocity from /odom, controls from the
selected Ackermann topic, and IMU from /imu/data.  Every stream is aligned by
causal hold; no future sample is used.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_tuning_utils.filter_collision_recovery_episodes import collision_recovery_mask

# USER SETTINGS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-22_10_38/rosbag2_2026_08_08-22_10_38_0.db3")
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/data/default_extracted_training_data.npz"
POSE_TOPIC = "/newmcl_pose"; VELOCITY_TOPIC = "/odom"; DRIVE_TOPIC = "/drive"; IMU_TOPIC = "/imu/data"
DT = .02; MAX_POSE_AGE = 1.0; MAX_VELOCITY_AGE = 1.0; MAX_COMMAND_AGE = 1.0; MAX_IMU_AGE = .05


def stamp_seconds(msg, record_ns):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
        return record_ns * 1e-9
    return stamp.sec + stamp.nanosec * 1e-9


def yaw(q):
    return np.arctan2(2*(q.w*q.z+q.x*q.y),
                      1-2*(q.y*q.y+q.z*q.z))


def read_streams(storage, pose_topic, velocity_topic, drive_topic, imu_topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage = Path(storage)
    storage_id = "mcap" if storage.suffix == ".mcap" else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(storage), storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    topics = (pose_topic, velocity_topic, drive_topic, imu_topic)
    missing = [x for x in topics if x not in types]
    if missing:
        raise RuntimeError(f"missing topics {missing}; available={sorted(types)}")
    msg_types = {x: get_message(types[x]) for x in topics}
    pose, velocity, drive, imu = [], [], [], []
    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(raw, msg_types[topic])
        t = stamp_seconds(msg, record_ns)
        if topic == pose_topic:
            p, q = msg.pose.position, msg.pose.orientation
            pose.append((t, p.x, p.y, yaw(q)))
        elif topic == velocity_topic:
            v = msg.twist.twist
            velocity.append((t, v.linear.x, v.linear.y, v.angular.z))
        elif topic == drive_topic:
            d = msg.drive
            drive.append((t, d.steering_angle, d.acceleration, d.speed))
        else:
            imu.append((t, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y))
    return tuple(np.asarray(x, np.float64) for x in (pose, velocity, drive, imu))


def causal_hold(stream, times, max_age):
    stream = stream[np.argsort(stream[:, 0])]
    stream = stream[np.r_[True, np.diff(stream[:, 0]) > 1e-9]]
    index = np.searchsorted(stream[:, 0], times, side="right")-1
    valid = index >= 0
    clipped = np.maximum(index, 0)
    valid &= (times-stream[clipped, 0]) <= max_age
    return stream[clipped], valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag", nargs="?", default=str(BAG_PATH), help="rosbag2 .db3/.mcap storage file")
    p.add_argument("-o", "--output", default=str(OUTPUT_PATH))
    p.add_argument("--pose-topic", default=POSE_TOPIC)
    p.add_argument("--velocity-topic", default=VELOCITY_TOPIC)
    p.add_argument("--drive-topic", default=DRIVE_TOPIC)
    p.add_argument("--imu-topic", default=IMU_TOPIC)
    p.add_argument("--dt", type=float, default=DT)
    # The node retains the latest pose/velocity/command between callbacks.
    # A generous watchdog reproduces that behavior without fragmenting a bag
    # for an occasional dropped 50 Hz message. IMU retains its strict 50 ms age.
    p.add_argument("--max-pose-age", type=float, default=MAX_POSE_AGE)
    p.add_argument("--max-velocity-age", type=float, default=MAX_VELOCITY_AGE)
    p.add_argument("--max-command-age", type=float, default=MAX_COMMAND_AGE)
    p.add_argument("--max-imu-age", type=float, default=MAX_IMU_AGE)
    args = p.parse_args()

    pose, velocity, drive, imu = read_streams(
        args.bag, args.pose_topic, args.velocity_topic,
        args.drive_topic, args.imu_topic)
    start = max(x[0, 0] for x in (pose, velocity, drive, imu))
    end = min(x[-1, 0] for x in (pose, velocity, drive, imu))
    times = np.arange(start, end, args.dt)
    pp, pv = causal_hold(pose, times, args.max_pose_age)
    vv, vvalid = causal_hold(velocity, times, args.max_velocity_age)
    dd, dvalid = causal_hold(drive, times, args.max_command_age)
    ii, ivalid = causal_hold(imu, times, args.max_imu_age)
    valid = pv & vvalid & dvalid & ivalid
    times, pp, vv, dd, ii = (x[valid] for x in (times, pp, vv, dd, ii))
    base = np.c_[times-times[0], pp[:, 1:4], vv[:, 1:4], dd[:, 1:4]]

    # Remove collision -> reverse -> stable-forward recovery, then make every
    # retained continuous run an independent episode so no rollout crosses it.
    bad, episodes = collision_recovery_mask(base, args.dt)
    kept = np.flatnonzero(~bad)
    breaks = np.flatnonzero((np.diff(kept) > 1) |
                            (np.diff(base[kept, 0]) > 1.5*args.dt))+1
    arrays = []
    segments = []
    for bag_id, run in enumerate(np.split(kept, breaks)):
        if not len(run):
            continue
        part = base[run].copy()
        source_start = float(part[0, 0])
        part[:, 0] -= source_start
        arrays.append(np.c_[part, np.ones(len(part)),
                            np.full(len(part), bag_id), ii[run, 1:4]])
        segments.append({"bag_id": bag_id, "samples": len(part),
                         "source_start_s": source_start,
                         "source_end_s": float(base[run[-1], 0])})
    samples = np.concatenate(arrays)
    columns = np.array(["t","x","y","yaw","vx","vy","omega",
                        "steer","accel","speed_cmd","split","bag_id",
                        "imu_wz","imu_ax","imu_ay"])
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, samples=samples, dt=args.dt, columns=columns,
                        pose_topic=np.array(args.pose_topic),
                        velocity_topic=np.array(args.velocity_topic),
                        drive_topic=np.array(args.drive_topic),
                        imu_topic=np.array(args.imu_topic))
    meta = {"source": str(Path(args.bag).resolve()),
            "pose_topic": args.pose_topic, "velocity_topic": args.velocity_topic,
            "drive_topic": args.drive_topic, "imu_topic": args.imu_topic,
            "alignment": "causal_hold", "raw_aligned_samples": len(base),
            "removed_collision_samples": int(bad.sum()),
            "output_samples": len(samples), "collision_episodes": episodes,
            "segments": segments,
            "split_policy": "single-bag-identical-train-test"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
