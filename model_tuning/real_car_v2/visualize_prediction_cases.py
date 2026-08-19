#!/usr/bin/env python3
"""Plot best/median/worst MPPI prediction cases against future measurements."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def stamp(msg, record_ns):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9 if s.sec or s.nanosec else record_ns * 1e-9


def quat_yaw(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def interp(t, samples, column):
    return np.interp(t, samples[:, 0], samples[:, column])


def ema(values, alpha):
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def load_bag(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    path = Path(path)
    files = sorted([*path.glob("*.db3"), *path.glob("*.mcap")])
    uri = str(files[0] if files else path)
    storage = "mcap" if uri.endswith(".mcap") else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    names = ("/newmcl_pose", "/odom", "/imu/data", "/mppi_optimal_trajectory")
    msg_types = {name: get_message(types[name]) for name in names}
    pose, odom, imu, trajectories = [], [], [], []
    while reader.has_next():
        topic, raw, ns = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(raw, msg_types[topic])
        t = stamp(msg, ns)
        if topic == "/newmcl_pose":
            pose.append((t, msg.pose.position.x, msg.pose.position.y,
                         quat_yaw(msg.pose.orientation)))
        elif topic == "/odom":
            odom.append((t, msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                         msg.twist.twist.angular.z))
        elif topic == "/imu/data":
            imu.append((t, msg.linear_acceleration.x, -msg.linear_acceleration.y,
                        -msg.angular_velocity.z))
        else:
            trajectories.append({
                "t": t,
                "x": np.asarray(msg.predicted_x, dtype=float),
                "y": np.asarray(msg.predicted_y, dtype=float),
                "yaw": np.asarray(msg.predicted_yaw, dtype=float),
                "vx": np.asarray(msg.predicted_v, dtype=float),
                "vy": np.asarray(msg.predicted_vy, dtype=float),
                "yaw_rate": np.asarray(msg.predicted_yaw_rate, dtype=float),
            })
    pose, odom, imu = map(np.asarray, (pose, odom, imu))
    pose[:, 3] = np.unwrap(pose[:, 3])
    imu[:, 1] = ema(imu[:, 1], 0.25)
    imu[:, 2] = ema(imu[:, 2], 0.25)
    imu[:, 3] = ema(imu[:, 3], 0.25)
    return pose, odom, imu, trajectories


def make_case(tr, pose, odom, imu, dt, horizon):
    count = min(len(tr["x"]), int(round(horizon / dt)))
    tau = (np.arange(count) + 1) * dt
    future = tr["t"] + tau
    if count < 2 or future[-1] >= min(pose[-1, 0], odom[-1, 0], imu[-1, 0]):
        return None
    actual = {
        "x": interp(future, pose, 1),
        "y": interp(future, pose, 2),
        "yaw": interp(future, pose, 3),
        "vx": interp(future, odom, 1),
        "vy": interp(future, odom, 2),
        "ax": interp(future, imu, 1),
        "ay": interp(future, imu, 2),
        "yaw_rate": interp(future, imu, 3),
    }
    pred = {key: tr[key][:count] for key in ("x", "y", "yaw", "vx", "vy", "yaw_rate")}
    initial_vx = interp(np.asarray([tr["t"]]), odom, 1)[0]
    initial_vy = interp(np.asarray([tr["t"]]), odom, 2)[0]
    pred["ax"] = np.diff(np.r_[initial_vx, pred["vx"]]) / dt
    pred["ay"] = (np.diff(np.r_[initial_vy, pred["vy"]]) / dt
                  + pred["vx"] * pred["yaw_rate"])
    errors = {key: pred[key] - actual[key] for key in pred}
    errors["yaw"] = wrap(errors["yaw"])
    position = np.hypot(errors["x"], errors["y"])
    metrics = {
        "start_time_s": float(tr["t"]),
        "trajectory_rmse_m": float(np.sqrt(np.mean(position ** 2))),
        "trajectory_final_error_m": float(position[-1]),
    }
    for key in ("vx", "vy", "ax", "ay", "yaw_rate", "yaw"):
        metrics[f"{key}_mae"] = float(np.mean(np.abs(errors[key])))
        metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(errors[key] ** 2)))
        metrics[f"{key}_max_abs"] = float(np.max(np.abs(errors[key])))
    return {"tau": tau, "pred": pred, "actual": actual, "errors": errors,
            "position_error": position, "metrics": metrics}


def plot_case(name, case, output):
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(case["actual"]["x"], case["actual"]["y"], "k-", lw=2.4, label="actual")
    ax.plot(case["pred"]["x"], case["pred"]["y"], "C3--", lw=2.2, label="predicted")
    ax.scatter(case["actual"]["x"][0], case["actual"]["y"][0], c="C2", s=45, zorder=3)
    ax.set_title(f"XY trajectory ({name}), RMSE={case['metrics']['trajectory_rmse_m']:.3f} m")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.axis("equal"); ax.grid(True); ax.legend()
    ax = axes[0, 1]
    ax.plot(case["tau"], case["position_error"], "C1", lw=2)
    ax.set_title("Trajectory position error"); ax.set_ylabel("error [m]"); ax.grid(True)
    fields = (("vx", "m/s"), ("vy", "m/s"), ("ax", "m/s²"), ("ay", "m/s²"),
              ("yaw_rate", "rad/s"), ("yaw", "rad"))
    for ax, (key, unit) in zip(axes.flat[2:], fields):
        actual = case["actual"][key]
        predicted = case["pred"][key]
        if key == "yaw":
            predicted = actual + case["errors"][key]
        ax.plot(case["tau"], actual, "k-", lw=2, label="actual")
        ax.plot(case["tau"], predicted, "C3--", lw=1.8, label="predicted")
        error_ax = ax.twinx()
        error_ax.plot(case["tau"], case["errors"][key], "C0", alpha=0.45, label="error")
        error_ax.axhline(0, color="C0", lw=.6, alpha=.5)
        ax.set_title(f"{key}: MAE={case['metrics'][key + '_mae']:.3f} {unit}")
        ax.set_xlabel("future time [s]"); ax.set_ylabel(unit); error_ax.set_ylabel(f"error [{unit}]", color="C0")
        ax.grid(True); ax.legend(loc="upper left"); error_ax.legend(loc="upper right")
    fig.suptitle(f"MPPI model prediction — {name.upper()} 1.2 s case", fontsize=16)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trajectory-dt", type=float, default=0.04)
    parser.add_argument("--horizon", type=float, default=1.2)
    args = parser.parse_args()
    output = Path(args.out_dir); output.mkdir(parents=True, exist_ok=True)
    pose, odom, imu, trajectories = load_bag(args.bag)
    cases = [make_case(tr, pose, odom, imu, args.trajectory_dt, args.horizon)
             for tr in trajectories]
    cases = [case for case in cases if case is not None]
    scores = np.asarray([case["metrics"]["trajectory_rmse_m"] for case in cases])
    order = np.argsort(scores)
    selected = {"best": cases[order[0]], "median": cases[order[len(order) // 2]],
                "worst": cases[order[-1]]}
    rows = []
    for name, case in selected.items():
        case["metrics"]["case"] = name
        case["metrics"]["percentile"] = float(np.mean(scores <= case["metrics"]["trajectory_rmse_m"]) * 100)
        rows.append(case["metrics"])
        plot_case(name, case, output / f"prediction_{name}.png")
    with (output / "prediction_case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"selection_metric": "1.2 s XY trajectory RMSE", "valid_predictions": len(cases),
               "rmse_distribution_m": {"min": float(scores.min()), "median": float(np.median(scores)),
                                       "p95": float(np.quantile(scores, .95)), "max": float(scores.max())},
               "cases": {row["case"]: row for row in rows}}
    (output / "prediction_case_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
