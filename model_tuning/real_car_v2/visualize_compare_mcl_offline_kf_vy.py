#!/usr/bin/env python3
"""Compare MCL pose-derived, non-causal offline, and runtime causal KF vy."""
from pathlib import Path
import argparse, json, sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATA = (ROOT / "model_tuning/data/ifac0820_042348/"
                "rosbag2_2026_08_20-04_23_48.npz")
DEFAULT_OUTPUT = ROOT / "model_tuning/results/compare_mcl_offline_old_simple_kf_0820_042348"
sys.path.insert(0, str(HERE))
from helper_lateral_velocity_kf import (LateralVelocityKFParams, estimate_dataset,
                                        estimate_dataset_pose_only)
from offline_lateral_velocity_smoother import smooth_segment_vy


def ema(values, alpha):
    result = values.copy()
    for i in range(1, len(result)):
        result[i] = alpha * result[i] + (1.0-alpha) * result[i-1]
    return result


def pose_vy(samples, c, dt, window_s):
    window = max(7, int(round(window_s/dt)) | 1)
    window = min(window, len(samples)//2*2-1)
    if window < 7:
        return None, None
    yaw = np.unwrap(samples[:, c["yaw"]])
    vxw = savgol_filter(samples[:, c["x"]], window, min(3, window-2), deriv=1, delta=dt)
    vyw = savgol_filter(samples[:, c["y"]], window, min(3, window-2), deriv=1, delta=dt)
    result = -np.sin(yaw)*vxw + np.cos(yaw)*vyw
    valid = np.ones(len(samples), bool)
    valid[:window//2] = False
    valid[-window//2:] = False
    return result, valid


def metrics(reference, estimate):
    error = estimate-reference
    corr = float(np.corrcoef(reference, estimate)[0, 1]) if np.std(reference)>1e-9 and np.std(estimate)>1e-9 else None
    return {"samples": int(len(error)), "bias_mps": float(np.mean(error)),
            "mae_mps": float(np.mean(np.abs(error))),
            "rmse_mps": float(np.sqrt(np.mean(error**2))),
            "p95_abs_mps": float(np.quantile(np.abs(error), .95)),
            "max_abs_mps": float(np.max(np.abs(error))), "correlation": corr}


def make_params(cfg, dt):
    return LateralVelocityKFParams(
        mass=float(cfg["mass"]), yaw_inertia=float(cfg["I_z"]),
        l_f=float(cfg["l_f"]), l_r=float(cfg["l_r"]), dt=dt,
        min_longitudinal_speed=float(cfg["kf_min_vx"]),
        low_speed_threshold=float(cfg["kf_low_speed_threshold"]),
        max_abs_vy=float(cfg["kf_max_abs_vy"]),
        process_var_vy=float(cfg["kf_q_vy"]), process_var_yaw_rate=float(cfg["kf_q_yaw_rate"]),
        measurement_var_lateral_accel=float(cfg["kf_r_lateral_accel"]),
        measurement_var_yaw_rate=float(cfg["kf_r_yaw_rate"]),
        initial_var_vy=float(cfg["kf_initial_p_vy"]),
        initial_var_yaw_rate=float(cfg["kf_initial_p_yaw_rate"]),
        imu_lateral_accel_sign=float(cfg["imu_lateral_accel_sign"]),
        pacejka_b_front=float(cfg["dynamic_mlp_B_f"]), pacejka_c_front=float(cfg["dynamic_mlp_C_f"]),
        pacejka_d_front=float(cfg["dynamic_mlp_D_f"]), pacejka_e_front=float(cfg["dynamic_mlp_E_f"]),
        pacejka_b_rear=float(cfg["dynamic_mlp_B_r"]), pacejka_c_rear=float(cfg["dynamic_mlp_C_r"]),
        pacejka_d_rear=float(cfg["dynamic_mlp_D_r"]), pacejka_e_rear=float(cfg["dynamic_mlp_E_r"]),
        process_var_ay_bias=float(cfg["kf_q_ay_bias"]),
        initial_var_ay_bias=float(cfg["kf_initial_p_ay_bias"]),
        max_abs_ay_bias=float(cfg["kf_max_abs_ay_bias"]),
        measurement_var_pose_vy=float(cfg["kf_r_pose_vy"]), pose_vy_gate=float(cfg["kf_pose_vy_gate"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA,
                        help=f"step-1 NPZ (default: {DEFAULT_DATA})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"result directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--pose-window", type=float, default=.30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    z = np.load(args.data)
    all_samples = z["samples"].astype(float)
    columns = z["columns"]
    c = {str(name): i for i, name in enumerate(columns)}
    dt = float(z["dt"])
    signs = z["imu_axis_signs"].astype(float) if "imu_axis_signs" in z.files else np.ones(3)
    alpha = float(z["imu_ema_alpha"]) if "imu_ema_alpha" in z.files else float(cfg["imu_ema_alpha"])
    records = []
    for segment_id in np.unique(all_samples[:, c["bag_id"]].astype(int)):
        s = all_samples[all_samples[:, c["bag_id"]].astype(int)==segment_id]
        if len(s) < 20:
            continue
        kf, yaw_rate = estimate_dataset(
            s, columns, dt, make_params(cfg, dt),
            steer_scale=float(cfg["kf_steer_scale"]), steer_bias=float(cfg["kf_steer_bias"]),
            max_steer=float(cfg["kf_max_steer"]), imu_ema_alpha=alpha,
            imu_wz_sign=float(signs[0]), imu_ay_sign=float(signs[2]),
            use_pose_vy=bool(cfg["kf_pose_vy_enabled"]),
            pose_window_s=float(cfg["kf_pose_vy_window_s"]))
        simple_kf, _ = estimate_dataset_pose_only(
            s, columns, dt, make_params(cfg, dt),
            steer_scale=float(cfg["kf_steer_scale"]), steer_bias=float(cfg["kf_steer_bias"]),
            max_steer=float(cfg["kf_max_steer"]), imu_ema_alpha=alpha,
            imu_wz_sign=float(signs[0]), use_pose_vy=bool(cfg["kf_pose_vy_enabled"]),
            pose_window_s=float(cfg["kf_pose_vy_window_s"]))
        mcl, valid = pose_vy(s, c, dt, args.pose_window)
        if mcl is None:
            continue
        ay = ema(float(signs[2])*s[:, c["imu_ay"]], alpha)
        offline, diagnostic = smooth_segment_vy(
            s[:, c["x"]], s[:, c["y"]], s[:, c["yaw"]], s[:, c["vx"]],
            yaw_rate, ay, dt, pose_window_s=args.pose_window)
        valid &= np.isfinite(mcl) & np.isfinite(offline) & np.isfinite(kf) & np.isfinite(simple_kf)
        records.append({"id": int(segment_id), "s": s[valid], "mcl": mcl[valid],
                        "offline": offline[valid], "kf": kf[valid],
                        "simple_kf": simple_kf[valid], "diag": diagnostic})
    if not records:
        raise RuntimeError("No usable continuous segment")
    joined = {key: np.concatenate([r[key] for r in records])
              for key in ("mcl", "offline", "kf", "simple_kf")}
    vx = np.concatenate([r["s"][:, c["vx"]] for r in records])
    moving = np.abs(vx) >= .5
    report = {"source": str(args.data), "dt_s": dt, "segments": len(records),
              "mcl_definition": f"centered Savitzky-Golay pose derivative ({args.pose_window:.2f} s)",
              "offline_definition": "non-causal robust pose+IMU dynamics smoother",
              "kf_definition": "current causal runtime Pacejka EKF parameters"}
    for scope, mask in (("all", np.ones(len(vx), bool)), ("moving_vx_ge_0p5", moving)):
        report[scope] = {"offline_vs_mcl": metrics(joined["mcl"][mask], joined["offline"][mask]),
                         "kf_vs_mcl": metrics(joined["mcl"][mask], joined["kf"][mask]),
                         "kf_vs_offline": metrics(joined["offline"][mask], joined["kf"][mask]),
                         "simple_kf_vs_mcl": metrics(joined["mcl"][mask], joined["simple_kf"][mask]),
                         "simple_kf_vs_offline": metrics(joined["offline"][mask], joined["simple_kf"][mask])}
    report["offline_smoother_diagnostics"] = [r["diag"] for r in records]
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(args.output/"comparison_data.npz", **joined, vx=vx)

    fig, axes = plt.subplots(len(records), 2, figsize=(17, 5*len(records)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    vmax = max(.1, float(np.quantile(np.abs(joined["kf"]-joined["offline"]), .95)))
    for row, record in enumerate(records):
        s = record["s"]; t = s[:, c["t"]]-s[0, c["t"]]
        axes[row, 0].plot(t, record["mcl"], color=".55", lw=1, label="MCL pose derivative")
        axes[row, 0].plot(t, record["offline"], "C0", lw=2, label="offline smoother")
        axes[row, 0].plot(t, record["kf"], "C1--", lw=1.5, label="runtime KF")
        axes[row, 0].plot(t, record["simple_kf"], "C2", lw=1.5,
                          label="simple KF (1x Pacejka + MCL)")
        axes[row, 0].set(title=f"segment {record['id']}", xlabel="segment time [s]", ylabel="body vy [m/s]")
        mismatch = np.abs(record["kf"]-record["offline"])
        points = axes[row, 1].scatter(s[:, c["x"]], s[:, c["y"]], c=mismatch,
                                      cmap="Reds", vmin=0, vmax=vmax, s=15)
        axes[row, 1].set(title="trajectory: red = large |KF - offline vy|", xlabel="x [m]", ylabel="y [m]")
        axes[row, 1].axis("equal")
        fig.colorbar(points, ax=axes[row, 1], label="|KF - offline| [m/s]")
        for axis in axes[row]: axis.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("MCL vy / offline vy / runtime KF vy")
    fig.savefig(args.output/"time_and_trajectory_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"all": report["all"], "moving_vx_ge_0p5": report["moving_vx_ge_0p5"]}, indent=2))


if __name__ == "__main__":
    main()
