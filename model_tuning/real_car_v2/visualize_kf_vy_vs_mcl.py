#!/usr/bin/env python3
"""Compare runtime 2-state KF vy against offline MCL-pose differentiated vy."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from helper_lateral_velocity_kf import (  # noqa: E402
    LateralVelocityKFParams, estimate_dataset)

DATA_DIR = ROOT / "model_tuning/data/ifac0817_0818_autonomous_physics_clean"
PARAMS = ROOT / "config/params.yaml"
OUTPUT = ROOT / "model_tuning/results/kf_vy_vs_mcl_0817_0818"
POSE_DERIVATIVE_WINDOW_S = 0.20
MIN_SEGMENT_DURATION_S = 1.20


def pose_body_vy(segment, columns, dt):
    """Centered offline derivative of map pose, rotated into the body frame."""
    x, y = segment[:, columns["x"]], segment[:, columns["y"]]
    yaw = np.unwrap(segment[:, columns["yaw"]])
    window = max(5, int(round(POSE_DERIVATIVE_WINDOW_S/dt)) | 1)
    window = min(window, len(segment)//2*2-1)
    if window < 5:
        return None, None
    order = min(3, window-2)
    world_vx = savgol_filter(x, window, order, deriv=1, delta=dt)
    world_vy = savgol_filter(y, window, order, deriv=1, delta=dt)
    body_vy = -np.sin(yaw)*world_vx + np.cos(yaw)*world_vy
    # A centered derivative is unreliable near both segment boundaries.
    valid = np.ones(len(segment), dtype=bool)
    valid[:window//2] = False; valid[-window//2:] = False
    return body_vy, valid


def stats(reference, estimate):
    error = estimate-reference
    correlation = (float(np.corrcoef(reference, estimate)[0, 1])
                   if np.std(reference) > 1e-9 and np.std(estimate) > 1e-9 else None)
    return {"samples": int(len(error)), "bias_mps": float(error.mean()),
            "mae_mps": float(np.abs(error).mean()),
            "rmse_mps": float(np.sqrt(np.mean(error**2))),
            "p95_abs_mps": float(np.quantile(np.abs(error), .95)),
            "max_abs_mps": float(np.max(np.abs(error))),
            "correlation": correlation}


def main():
    global POSE_DERIVATIVE_WINDOW_S
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA_DIR)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--pose-derivative-window",type=float,
                        default=POSE_DERIVATIVE_WINDOW_S)
    args=parser.parse_args();data_dir=args.data;output=args.output
    POSE_DERIVATIVE_WINDOW_S=args.pose_derivative_window
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(PARAMS.read_text())["/**"]["ros__parameters"]
    records = []
    for path in sorted(data_dir.glob("*.npz")):
        data = np.load(path); samples = data["samples"].astype(float)
        names = {str(name): index for index, name in enumerate(data["columns"])}
        dt = float(data["dt"])
        signs = (data["imu_axis_signs"].astype(float) if "imu_axis_signs" in data.files
                 else np.array((1., 1., 1.)))
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
            pacejka_b_front=float(cfg["dynamic_mlp_B_f"]),
            pacejka_c_front=float(cfg["dynamic_mlp_C_f"]),
            pacejka_d_front=float(cfg["dynamic_mlp_D_f"]),
            pacejka_e_front=float(cfg["dynamic_mlp_E_f"]),
            pacejka_b_rear=float(cfg["dynamic_mlp_B_r"]),
            pacejka_c_rear=float(cfg["dynamic_mlp_C_r"]),
            pacejka_d_rear=float(cfg["dynamic_mlp_D_r"]),
            pacejka_e_rear=float(cfg["dynamic_mlp_E_r"]),
            process_var_ay_bias=float(cfg["kf_q_ay_bias"]),
            initial_var_ay_bias=float(cfg["kf_initial_p_ay_bias"]),
            max_abs_ay_bias=float(cfg["kf_max_abs_ay_bias"]),
            measurement_var_pose_vy=float(cfg["kf_r_pose_vy"]),
            pose_vy_gate=float(cfg["kf_pose_vy_gate"]))
        kf_vy, _ = estimate_dataset(
            samples, data["columns"], dt, params,
            steer_scale=float(cfg["kf_steer_scale"]),
            steer_bias=float(cfg["kf_steer_bias"]),
            max_steer=float(cfg["kf_max_steer"]),
            imu_ema_alpha=float(data["imu_ema_alpha"] if "imu_ema_alpha" in data.files
                                else cfg["imu_ema_alpha"]),
            imu_wz_sign=float(signs[0]), imu_ay_sign=float(signs[2]),
            use_pose_vy=bool(cfg["kf_pose_vy_enabled"]),
            pose_window_s=float(cfg["kf_pose_vy_window_s"]))
        for segment_id in np.unique(samples[:, names["bag_id"]].astype(int)):
            indices = np.flatnonzero(samples[:, names["bag_id"]].astype(int) == segment_id)
            if len(indices) < int(round(MIN_SEGMENT_DURATION_S/dt)):
                continue
            pose_vy, valid = pose_body_vy(samples[indices], names, dt)
            if pose_vy is None:
                continue
            valid &= np.isfinite(pose_vy) & np.isfinite(kf_vy[indices])
            # Stationary pose differentiation is dominated by MCL quantization;
            # report all speeds, but retain vx for separate moving-only metrics.
            records.append({"path": path, "segment": int(segment_id),
                            "t": samples[indices, names["t"]][valid],
                            "vx": samples[indices, names["vx"]][valid],
                            "x": samples[indices, names["x"]][valid],
                            "y": samples[indices, names["y"]][valid],
                            "mcl_vy": pose_vy[valid], "kf_vy": kf_vy[indices][valid]})
    if not records:
        raise RuntimeError(f"no usable NPZ in {data_dir}")

    mcl = np.concatenate([record["mcl_vy"] for record in records])
    kf = np.concatenate([record["kf_vy"] for record in records])
    vx = np.concatenate([record["vx"] for record in records])
    moving = np.abs(vx) >= .5
    report = {"data_directory": str(data_dir), "npz_files": len({r["path"] for r in records}),
              "segments": len(records), "pose_derivative":
              f"centered Savitzky-Golay, {POSE_DERIVATIVE_WINDOW_S:.2f} s",
              "all_speed": stats(mcl, kf), "moving_vx_ge_0p5": stats(mcl[moving], kf[moving])}

    per_bag = []
    for path in sorted({record["path"] for record in records}):
        selected = [record for record in records if record["path"] == path]
        reference = np.concatenate([record["mcl_vy"] for record in selected])
        estimate = np.concatenate([record["kf_vy"] for record in selected])
        metric = stats(reference, estimate); metric["bag"] = path.stem
        per_bag.append(metric)
    report["per_bag"] = sorted(per_bag, key=lambda item: item["mae_mps"])
    (output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")

    # Aggregate distribution and bag-to-bag variation.
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    limit = max(.4, float(np.quantile(np.abs(np.r_[mcl, kf]), .995)))
    axes[0, 0].hexbin(mcl, kf, gridsize=80, mincnt=1, bins="log", cmap="viridis")
    axes[0, 0].plot([-limit, limit], [-limit, limit], "r--", label="ideal KF = MCL")
    axes[0, 0].set(xlim=(-limit, limit), ylim=(-limit, limit),
                   xlabel="MCL pose-derived vy [m/s]", ylabel="KF vy [m/s]",
                   title="All cleaned 0817/0818 samples")
    error = kf-mcl
    axes[0, 1].hist(error, bins=120, range=np.quantile(error, (.005, .995)), color="C0")
    axes[0, 1].axvline(0, color="k", ls="--")
    axes[0, 1].set(title=f"Error distribution | MAE={np.mean(abs(error)):.3f} m/s",
                   xlabel="KF vy - MCL vy [m/s]", ylabel="samples")
    ordered = sorted(per_bag, key=lambda item: item["mae_mps"], reverse=True)
    axes[1, 0].bar(np.arange(len(ordered)), [item["mae_mps"] for item in ordered])
    axes[1, 0].set(title="Bag-level lateral-velocity MAE", xlabel="bags sorted worst → best",
                   ylabel="MAE [m/s]")
    axes[1, 1].scatter(vx, np.abs(error), s=3, alpha=.12)
    speed_bins = np.arange(0, max(4.6, np.max(np.abs(vx)))+.5, .5)
    centers, medians, p95 = [], [], []
    for lo, hi in zip(speed_bins[:-1], speed_bins[1:]):
        selected = (np.abs(vx) >= lo) & (np.abs(vx) < hi)
        if selected.sum() < 10: continue
        centers.append((lo+hi)/2); medians.append(np.median(abs(error[selected])))
        p95.append(np.quantile(abs(error[selected]), .95))
    axes[1, 1].plot(centers, medians, "k-o", label="median |error|")
    axes[1, 1].plot(centers, p95, "r-o", label="p95 |error|")
    axes[1, 1].set(title="Error versus longitudinal speed", xlabel="vx [m/s]",
                   ylabel="|KF vy - MCL vy| [m/s]"); axes[1, 1].legend()
    for axis in axes.ravel(): axis.grid(alpha=.25)
    fig.suptitle("2-state runtime KF vy versus offline MCL pose-differentiated vy")
    fig.savefig(output/"aggregate_comparison.png", dpi=180); plt.close(fig)

    # Time traces for best/median/worst segment, ranked by MAE.
    ranked = sorted(records, key=lambda record: np.mean(abs(record["kf_vy"]-record["mcl_vy"])))
    cases = (("best", ranked[0]), ("median", ranked[len(ranked)//2]), ("worst", ranked[-1]))
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    for row, (label, record) in enumerate(cases):
        time = record["t"]-record["t"][0]
        mae = np.mean(abs(record["kf_vy"]-record["mcl_vy"]))
        axes[row, 0].plot(time, record["mcl_vy"], "k-", label="MCL differentiated vy")
        axes[row, 0].plot(time, record["kf_vy"], "C1--", label="runtime KF vy")
        axes[row, 0].set(title=f"{label}: {record['path'].stem}, segment {record['segment']} | MAE={mae:.3f} m/s",
                         xlabel="time [s]", ylabel="vy [m/s]")
        mismatch = abs(record["kf_vy"]-record["mcl_vy"])
        points = axes[row, 1].scatter(record["x"], record["y"], c=mismatch,
                                      cmap="magma", s=12, vmin=0,
                                      vmax=max(.1, np.quantile(mismatch, .95)))
        axes[row, 1].set(title="MCL trajectory colored by |vy error|", xlabel="x [m]", ylabel="y [m]")
        axes[row, 1].axis("equal"); fig.colorbar(points, ax=axes[row, 1], label="|error| [m/s]")
        for axis in axes[row]: axis.grid(alpha=.25)
    axes[0, 0].legend()
    fig.savefig(output/"best_median_worst_segments.png", dpi=180); plt.close(fig)
    print(json.dumps({key: value for key, value in report.items() if key != "per_bag"}, indent=2))


if __name__ == "__main__": main()
