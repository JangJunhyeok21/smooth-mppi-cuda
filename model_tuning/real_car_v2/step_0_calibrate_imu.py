#!/usr/bin/env python3
"""Step 0: estimate IMU FLU signs and stationary biases from ROS 2 bags.

The runtime convention is::

    corrected = imu_sign * raw - imu_bias

Biases are robust medians from genuinely stationary intervals.  Signs are
identified independently on moving data against odometry-derived references.
No YAML is modified unless UPDATE_YAML or --update-yaml is enabled.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# USER SETTINGS (F5/direct execution)
# ---------------------------------------------------------------------------
BAG_ROOTS = (
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0817 (1)"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0818"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0819"),
)
PARAMS_YAML = ROOT / "config/params.yaml"
USE_PLOT = True
UPDATE_YAML = False
PLOT_OUTPUT = ROOT / "model_tuning/results/imu_calibration/imu_bias_sign_diagnostic.png"

IMU_TOPIC = "/imu/data"
ODOM_TOPIC = "/odom"
COMMAND_TOPICS = ("/ackermann_cmd", "/drive")

# A sample must satisfy every condition, continuously, before it is admitted
# to the stationary bias population.
STATIONARY_COMMAND_MAX = 0.05       # m/s
STATIONARY_VX_MAX = 0.05            # m/s
STATIONARY_VY_MAX = 0.05            # m/s
STATIONARY_YAW_RATE_MAX = 0.03      # rad/s
MIN_STATIONARY_DURATION_S = 0.50
STATIONARY_EDGE_TRIM_S = 0.15
MIN_STATIONARY_SAMPLES = 100

# Sign identification requires a meaningful reference excitation. If the
# absolute correlation is smaller than this, preserve the configured sign.
MIN_SIGN_SAMPLES = 100
MIN_SIGN_ABS_CORRELATION = 0.25
WZ_REFERENCE_MIN = 0.08             # rad/s
AX_REFERENCE_MIN = 0.30             # m/s^2
AY_REFERENCE_MIN = 0.30             # m/s^2
DERIVATIVE_SMOOTHING_S = 0.20


def stamp_seconds(msg, record_ns):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
        return record_ns * 1.0e-9
    return stamp.sec + stamp.nanosec * 1.0e-9


def resolve_storages(inputs):
    """Return unique db3/mcap storage files from files, bags, or parent roots."""
    found = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix in (".db3", ".mcap"):
            found.append(path.resolve())
            continue
        if not path.exists():
            print(f"SKIP missing input: {path}", file=sys.stderr)
            continue
        if (path / "metadata.yaml").exists():
            candidates = sorted(path.glob("*.mcap")) or sorted(path.glob("*.db3"))
            found.extend(candidate.resolve() for candidate in candidates)
            continue
        for metadata in path.rglob("metadata.yaml"):
            candidates = sorted(metadata.parent.glob("*.mcap")) or sorted(
                metadata.parent.glob("*.db3"))
            found.extend(candidate.resolve() for candidate in candidates)
    return sorted(dict.fromkeys(found))


def read_bag(storage):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_id = "mcap" if storage.suffix == ".mcap" else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(storage), storage_id=storage_id),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    available = {item.name: item.type for item in reader.get_all_topics_and_types()}
    command_topic = next((topic for topic in COMMAND_TOPICS if topic in available), None)
    required = (IMU_TOPIC, ODOM_TOPIC)
    missing = [topic for topic in required if topic not in available]
    if missing or command_topic is None:
        raise RuntimeError(
            f"missing {missing or list(COMMAND_TOPICS)}; available={sorted(available)}")
    selected = (*required, command_topic)
    types = {topic: get_message(available[topic]) for topic in selected}
    imu, odom, command = [], [], []
    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in types:
            continue
        msg = deserialize_message(raw, types[topic])
        time = stamp_seconds(msg, record_ns)
        if topic == IMU_TOPIC:
            imu.append((time, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y))
        elif topic == ODOM_TOPIC:
            twist = msg.twist.twist
            odom.append((time, twist.linear.x, twist.linear.y, twist.angular.z))
        else:
            command.append((time, msg.drive.speed))
    arrays = tuple(np.asarray(values, dtype=np.float64)
                   for values in (imu, odom, command))
    if any(len(values) < 2 for values in arrays):
        raise RuntimeError("one or more selected streams contain fewer than two samples")
    return (*arrays, command_topic)


def unique_sorted(stream):
    stream = stream[np.argsort(stream[:, 0])]
    return stream[np.r_[True, np.diff(stream[:, 0]) > 1.0e-9]]


def interpolate(stream, times, column):
    stream = unique_sorted(stream)
    return np.interp(times, stream[:, 0], stream[:, column], left=np.nan, right=np.nan)


def causal_hold(stream, times):
    stream = unique_sorted(stream)
    index = np.searchsorted(stream[:, 0], times, side="right") - 1
    valid = index >= 0
    index = np.maximum(index, 0)
    values = stream[index, 1]
    age = times - stream[index, 0]
    return values, valid & (age <= 0.15)


def contiguous_stationary(mask, times):
    """Keep only sufficiently long runs and trim their transition edges."""
    output = np.zeros_like(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return output
    gap = np.flatnonzero((np.diff(indices) > 1) |
                         (np.diff(times[indices]) > 0.10)) + 1
    for group in np.split(indices, gap):
        if len(group) < 2 or times[group[-1]] - times[group[0]] < MIN_STATIONARY_DURATION_S:
            continue
        keep = group[(times[group] >= times[group[0]] + STATIONARY_EDGE_TRIM_S) &
                     (times[group] <= times[group[-1]] - STATIONARY_EDGE_TRIM_S)]
        output[keep] = True
    return output


def smooth(values, times):
    if len(values) < 3:
        return values
    dt = np.median(np.diff(times))
    width = max(1, int(round(DERIVATIVE_SMOOTHING_S / max(dt, 1.0e-3))))
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def robust_correlation(raw, reference, threshold):
    mask = np.isfinite(raw) & np.isfinite(reference) & (np.abs(reference) >= threshold)
    if mask.sum() < MIN_SIGN_SAMPLES:
        return np.nan, int(mask.sum())
    x, y = raw[mask], reference[mask]
    # Remove extreme derivatives/impacts without biasing the sign decision.
    limit = np.percentile(np.abs(y), 98.0)
    keep = np.abs(y) <= limit
    x, y = x[keep], y[keep]
    x = x - np.median(x); y = y - np.median(y)
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return (float(np.dot(x, y) / denom) if denom > 1.0e-12 else np.nan,
            int(len(x)))


def yaml_parameters(path):
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("/**", {}).get("ros__parameters", {})


def replace_yaml_scalars(path, updates):
    """Atomically replace only requested scalar lines, preserving comments."""
    text = path.read_text()
    for key, value in updates.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)[^#\n]*(.*)$", re.MULTILINE)
        replacement = rf"\g<1>{value:.10g}\g<2>"
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"expected exactly one YAML key '{key}', found {count}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def analyse_bag(storage):
    imu, odom, command, command_topic = read_bag(storage)
    imu = unique_sorted(imu); odom = unique_sorted(odom)
    times = imu[:, 0]
    vx = interpolate(odom, times, 1)
    vy = interpolate(odom, times, 2)
    yaw_rate = interpolate(odom, times, 3)
    speed_cmd, command_valid = causal_hold(command, times)
    finite = np.isfinite(vx + vy + yaw_rate + speed_cmd)
    stationary = finite & command_valid
    stationary &= np.abs(speed_cmd) <= STATIONARY_COMMAND_MAX
    stationary &= np.abs(vx) <= STATIONARY_VX_MAX
    stationary &= np.abs(vy) <= STATIONARY_VY_MAX
    stationary &= np.abs(yaw_rate) <= STATIONARY_YAW_RATE_MAX
    stationary = contiguous_stationary(stationary, times)

    vx_smooth, vy_smooth = smooth(vx, times), smooth(vy, times)
    dvx = np.gradient(vx_smooth, times)
    dvy = np.gradient(vy_smooth, times)
    references = np.column_stack((yaw_rate,
                                  dvx - yaw_rate * vy,
                                  dvy + yaw_rate * vx))
    return {
        "path": storage, "time": times, "raw": imu[:, 1:4],
        "references": references, "stationary": stationary,
        "command_topic": command_topic,
    }


def concatenate_for_plot(records, field):
    values, times, offset = [], [], 0.0
    for record in records:
        local = record["time"] - record["time"][0]
        times.append(local + offset); values.append(record[field])
        offset += local[-1] + 1.0
    return np.concatenate(times), np.concatenate(values)


def plot_diagnostics(records, signs, biases, correlations, output, show):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-imu-calibration")
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("yaw rate", "longitudinal acceleration", "lateral acceleration")
    units = ("rad/s", "m/s²", "m/s²")
    time, raw = concatenate_for_plot(records, "raw")
    _, reference = concatenate_for_plot(records, "references")
    stationary = np.concatenate([record["stationary"] for record in records])
    corrected = raw * signs[None, :] - biases[None, :]
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)
    for axis in range(3):
        left, right = axes[axis]
        left.plot(time, signs[axis] * raw[:, axis], lw=0.7, color="0.55",
                  label="sign × raw IMU")
        left.scatter(time[stationary], signs[axis] * raw[stationary, axis],
                     s=3, color="tab:blue", label="accepted stationary")
        left.axhline(biases[axis], color="tab:red", lw=1.5,
                     label=f"estimated bias={biases[axis]:.6g}")
        left.set_title(f"{names[axis]} bias population")
        left.set_xlabel("concatenated bag time [s]"); left.set_ylabel(units[axis])
        zoom = max(8.0 * np.median(np.abs(
            signs[axis] * raw[stationary, axis] - biases[axis])),
            0.002 if axis == 0 else 0.04)
        left.set_ylim(biases[axis] - zoom, biases[axis] + zoom)
        left.grid(True); left.legend(loc="best")

        moving = ~stationary & np.isfinite(reference[:, axis])
        stride = max(1, int(np.count_nonzero(moving) / 5000))
        indices = np.flatnonzero(moving)[::stride]
        right.plot(time[indices], reference[indices, axis], lw=0.8,
                   color="black", label="odom-derived reference")
        right.plot(time[indices], corrected[indices, axis], lw=0.7,
                   color="tab:orange", alpha=0.8, label="corrected IMU")
        right.set_title(f"{names[axis]} sign diagnostic: corr={correlations[axis]:.3f}")
        right.set_xlabel("concatenated bag time [s]"); right.set_ylabel(units[axis])
        if len(indices):
            scale = np.percentile(np.abs(np.r_[reference[indices, axis],
                                                corrected[indices, axis]]), 98.5)
            if np.isfinite(scale) and scale > 0.0:
                right.set_ylim(-1.15 * scale, 1.15 * scale)
        right.grid(True); right.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"diagnostic plot: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", help="bag file/directory or recursive parent root")
    parser.add_argument("--params-yaml", type=Path, default=PARAMS_YAML)
    parser.add_argument("--update-yaml", action="store_true", default=UPDATE_YAML)
    parser.add_argument("--use-plot", action=argparse.BooleanOptionalAction,
                        default=USE_PLOT)
    parser.add_argument("--plot-output", type=Path, default=PLOT_OUTPUT)
    args = parser.parse_args()

    storages = resolve_storages(args.bags or BAG_ROOTS)
    if not storages:
        raise RuntimeError("No rosbag2 .db3/.mcap storage files found")
    records = []
    for index, storage in enumerate(storages, 1):
        try:
            record = analyse_bag(storage); records.append(record)
            print(f"[{index}/{len(storages)}] {storage.parent.name}: "
                  f"stationary={record['stationary'].sum()} IMU samples, "
                  f"command={record['command_topic']}")
        except Exception as error:
            print(f"[{index}/{len(storages)}] SKIP {storage}: {error}", file=sys.stderr)
    if not records:
        raise RuntimeError("No usable bags contain IMU, odometry, and command streams")

    configured = yaml_parameters(args.params_yaml)
    old_signs = np.array([configured.get("imu_wz_sign", 1.0),
                          configured.get("imu_ax_sign", 1.0),
                          configured.get("imu_ay_sign", 1.0)], dtype=float)
    all_raw = np.concatenate([record["raw"] for record in records])
    all_reference = np.concatenate([record["references"] for record in records])
    all_stationary = np.concatenate([record["stationary"] for record in records])
    if all_stationary.sum() < MIN_STATIONARY_SAMPLES:
        raise RuntimeError(
            f"Only {all_stationary.sum()} stationary IMU samples found; need at least "
            f"{MIN_STATIONARY_SAMPLES}. Check command/odom thresholds or bags.")

    thresholds = (WZ_REFERENCE_MIN, AX_REFERENCE_MIN, AY_REFERENCE_MIN)
    correlations, counts, signs = [], [], old_signs.copy()
    for axis, threshold in enumerate(thresholds):
        correlation, count = robust_correlation(
            all_raw[:, axis], all_reference[:, axis], threshold)
        correlations.append(correlation); counts.append(count)
        if np.isfinite(correlation) and abs(correlation) >= MIN_SIGN_ABS_CORRELATION:
            signs[axis] = 1.0 if correlation >= 0.0 else -1.0
    correlations = np.asarray(correlations)
    signed_stationary = all_raw[all_stationary] * signs[None, :]
    biases = np.median(signed_stationary, axis=0)
    mad = 1.4826 * np.median(np.abs(signed_stationary - biases), axis=0)

    sign_keys = ("imu_wz_sign", "imu_ax_sign", "imu_ay_sign")
    bias_keys = ("imu_wz_bias", "imu_ax_bias", "imu_ay_bias")
    print("\nIMU calibration result (runtime: corrected = sign * raw - bias)")
    for axis, (sign_key, bias_key) in enumerate(zip(sign_keys, bias_keys)):
        confidence = "accepted" if (np.isfinite(correlations[axis]) and
                    abs(correlations[axis]) >= MIN_SIGN_ABS_CORRELATION) else "weak; kept YAML sign"
        print(f"  {sign_key}: {old_signs[axis]:+.0f} -> {signs[axis]:+.0f} "
              f"(corr={correlations[axis]:+.4f}, n={counts[axis]}, {confidence})")
        print(f"  {bias_key}: {float(configured.get(bias_key, 0.0)):+.9g} -> "
              f"{biases[axis]:+.9g} (stationary robust sigma={mad[axis]:.4g})")

    updates = {**dict(zip(sign_keys, signs)), **dict(zip(bias_keys, biases))}
    if args.update_yaml:
        replace_yaml_scalars(args.params_yaml, updates)
        print(f"updated YAML: {args.params_yaml}")
    else:
        print("YAML unchanged. Re-run with --update-yaml after reviewing diagnostics.")

    plot_diagnostics(records, signs, biases, correlations,
                     args.plot_output, args.use_plot)


if __name__ == "__main__":
    main()
