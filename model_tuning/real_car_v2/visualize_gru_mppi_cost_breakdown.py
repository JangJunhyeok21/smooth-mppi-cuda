#!/usr/bin/env python3
"""현재 GRU simulator 주행의 MPPI horizon 비용과 차선 이탈 원인을 재계산한다."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "model_tuning/results/current_gru_lane_cost_diagnosis"
DT = 0.04


def wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    data = np.load(args.run / "map1_lap_data.npz", allow_pickle=True)
    ref = np.genfromtxt(ROOT / cfg["csv_file_path"], delimiter=",", names=True)
    rx, ry = ref["x_m"], ref["y_m"]
    yaw = np.arctan2(np.roll(ry, -1) - np.roll(ry, 1),
                     np.roll(rx, -1) - np.roll(rx, 1))
    left = np.hypot(ref["left_x_m"] - rx, ref["left_y_m"] - ry)
    right = np.hypot(ref["right_x_m"] - rx, ref["right_y_m"] - ry)
    closed_length = np.hypot(np.roll(rx, -1) - rx, np.roll(ry, -1) - ry)

    odom = data["odom"]
    drive = data["drive"]
    times = data["prediction_t"]
    keys = ("heading", "speed_reward", "overspeed", "error_speed", "friction",
            "front_slip", "rear_slip", "steer", "control_rate", "boundary", "progress_reward")
    totals = {key: [] for key in keys}
    minimum_clearance = []
    outside_steps = []
    trajectories = []

    for n, stamp in enumerate(times):
        x = np.asarray(data["prediction_x"][n], float)
        y = np.asarray(data["prediction_y"][n], float)
        psi = np.asarray(data["prediction_yaw"][n], float)
        vx = np.asarray(data["prediction_vx"][n], float)
        vy = np.asarray(data["prediction_vy"][n], float)
        rate = np.asarray(data["prediction_yaw_rate"][n], float)
        steer = np.asarray(data["prediction_steer"][n], float)
        speed_cmd = np.asarray(data["prediction_speed_cmd"][n], float)
        nearest = np.argmin((x[:, None] - rx[None])**2 + (y[:, None] - ry[None])**2, axis=1)
        dx, dy = x - rx[nearest], y - ry[nearest]
        contour = -np.sin(yaw[nearest]) * dx + np.cos(yaw[nearest]) * dy
        lag = np.cos(yaw[nearest]) * dx + np.sin(yaw[nearest]) * dy
        heading_error = wrap(psi - yaw[nearest])
        clearance = np.minimum(left[nearest] - contour, right[nearest] + contour)
        minimum_clearance.append(float(clearance.min()))
        outside_steps.append(int(np.count_nonzero(clearance < 0)))

        oi = min(np.searchsorted(odom[:, 0], stamp), len(odom) - 1)
        di = min(np.searchsorted(drive[:, 0], stamp), len(drive) - 1)
        previous_vx, previous_vy, previous_rate = odom[oi, 4:7]
        previous_steer, previous_speed = drive[di, 1:3]
        old_vx = np.r_[previous_vx, vx[:-1]]
        old_vy = np.r_[previous_vy, vy[:-1]]
        old_rate = np.r_[previous_rate, rate[:-1]]
        ax = (vx - old_vx) / DT
        ay = (vy - old_vy) / DT + old_vx * old_rate

        forward = vx * np.cos(heading_error)
        overspeed = np.maximum(0, vx - float(cfg["max_speed"]))
        utilization = (ax / float(cfg["longitudinal_accel_soft_limit"]))**2 + \
                      (ay / float(cfg["lat_g_soft_limit"]))**2
        friction_excess = np.maximum(0, utilization - 1)
        alpha_r = -np.arctan2(vy - float(cfg["l_r"]) * rate, np.maximum(np.abs(vx), .5))
        alpha_f = steer - np.arctan2(vy + float(cfg["l_f"]) * rate,
                                     np.maximum(np.abs(vx), .5))
        front_slip_excess = np.where(np.abs(vx) >= float(cfg["front_slip_cost_min_speed"]),
            np.maximum(0, np.abs(alpha_f) - np.deg2rad(float(cfg["front_slip_soft_limit_deg"]))), 0)
        slip_excess = np.where(np.abs(vx) >= float(cfg["rear_slip_cost_min_speed"]),
            np.maximum(0, np.abs(alpha_r) - np.deg2rad(float(cfg["rear_slip_soft_limit_deg"]))), 0)
        dsteer = steer - np.r_[previous_steer, steer[:-1]]
        dspeed = speed_cmd - np.r_[previous_speed, speed_cmd[:-1]]
        slack = np.maximum(0, float(cfg["collision_radius"]) - clearance)
        boundary_weight = np.full(len(x), float(cfg["q_boundary_slack"]))
        boundary_weight[-1] = float(cfg["q_boundary_terminal_slack"])

        start_idx = int(np.argmin((odom[oi, 1] - rx)**2 + (odom[oi, 2] - ry)**2))
        end_idx = int(nearest[-1]); steps = (end_idx - start_idx) % len(rx)
        progress = 0.0 if steps > len(rx) // 2 else float(closed_length[(start_idx + np.arange(steps)) % len(rx)].sum())
        start_lag = np.clip((odom[oi, 1] - rx[start_idx]) * np.cos(yaw[start_idx]) +
                            (odom[oi, 2] - ry[start_idx]) * np.sin(yaw[start_idx]), -.15, .15)
        end_lag = np.clip(lag[-1], -.15, .15)
        progress = max(0.0, progress + end_lag - start_lag)

        values = {
            "heading": float(cfg["q_heading"]) * heading_error**2 +
                       float(cfg["q_contour"]) * contour**2 + float(cfg["q_lag"]) * lag**2,
            "speed_reward": -(float(cfg["q_v"]) * .2) * forward,
            "overspeed": float(cfg["q_v"]) * overspeed**2,
            "error_speed": float(cfg["q_error_speed"]) * vx**2 * (contour**2 + heading_error**2),
            "friction": float(cfg["q_lat_g"]) * friction_excess**2,
            "front_slip": float(cfg["q_front_slip"]) * front_slip_excess**2,
            "rear_slip": float(cfg["q_rear_slip"]) * slip_excess**2,
            "steer": float(cfg["q_steer"]) * steer**2,
            "control_rate": float(cfg["q_du"]) * (dsteer**2 + dspeed**2),
            "boundary": boundary_weight * slack**2,
        }
        for key, value in values.items():
            totals[key].append(float(np.sum(value)))
        totals["progress_reward"].append(-float(cfg["q_progress"]) * progress)
        trajectories.append((x, y, clearance))

    total_array = {key: np.asarray(value) for key, value in totals.items()}
    minimum_clearance = np.asarray(minimum_clearance)
    outside_steps = np.asarray(outside_steps)
    critical = int(np.argmin(minimum_clearance))
    penalty_keys = ("heading", "overspeed", "error_speed", "friction", "front_slip", "rear_slip",
                    "steer", "control_rate", "boundary")
    positive = sum(max(0.0, total_array[key][critical]) for key in penalty_keys)
    reward = sum(abs(min(0.0, total_array[key][critical])) for key in ("speed_reward", "progress_reward"))

    run_positive = {key: float(np.maximum(total_array[key], 0).sum()) for key in penalty_keys}
    run_rewards = {key: float(np.maximum(-total_array[key], 0).sum())
                   for key in ("speed_reward", "progress_reward")}
    run_positive_sum = sum(run_positive.values())
    run_reward_sum = sum(run_rewards.values())
    report = {
        "run_status": str(data["status"]),
        "parameters": {key: cfg[key] for key in ("max_speed", "q_v", "q_contour", "q_lag",
            "q_heading", "q_error_speed", "q_lat_g", "q_rear_slip", "rear_slip_soft_limit_deg",
            "collision_radius", "q_boundary_slack", "q_boundary_terminal_slack", "q_progress")},
        "critical_horizon": {"index": critical, "time_s": float(times[critical]),
            "minimum_boundary_clearance_m": float(minimum_clearance[critical]),
            "outside_lane_steps": int(outside_steps[critical]),
            "cost": {key: float(total_array[key][critical]) for key in keys},
            "positive_penalty_ratio_percent": {key: 100 * max(0.0, float(total_array[key][critical])) / max(positive, 1e-9) for key in penalty_keys},
            "reward_magnitude_ratio_percent": {key: 100 * abs(min(0.0, float(total_array[key][critical]))) / max(reward, 1e-9) for key in ("speed_reward", "progress_reward")},
            "positive_penalty_sum": positive, "reward_magnitude_sum": reward,
            "net_cost": float(sum(total_array[key][critical] for key in keys))},
        "run": {"horizons": len(times), "horizons_predicting_outside_lane": int(np.count_nonzero(outside_steps)),
            "minimum_predicted_clearance_m": float(minimum_clearance.min()),
            "positive_penalty_sum": run_positive_sum,
            "reward_magnitude_sum": run_reward_sum,
            "net_cost_sum": float(sum(v.sum() for v in total_array.values())),
            "positive_penalty_ratio_percent": {
                key: 100.0 * value / max(run_positive_sum, 1e-9)
                for key, value in run_positive.items()},
            "reward_magnitude_ratio_percent": {
                key: 100.0 * value / max(run_reward_sum, 1e-9)
                for key, value in run_rewards.items()}},
    }
    (args.run / "mppi_cost_breakdown.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    x, y, clearance = trajectories[critical]
    axes[0, 0].plot(rx, ry, "k--", lw=1, label="centerline")
    axes[0, 0].plot(ref["left_x_m"], ref["left_y_m"], "k", lw=1.5, label="lane boundaries")
    axes[0, 0].plot(ref["right_x_m"], ref["right_y_m"], "k", lw=1.5)
    axes[0, 0].scatter(x, y, c=clearance, cmap="RdYlGn", vmin=-.4, vmax=.4, s=28, label="optimal horizon")
    axes[0, 0].plot(odom[:, 1], odom[:, 2], color="tab:blue", lw=1.8, label="simulator actual")
    axes[0, 0].axis("equal"); axes[0, 0].grid(alpha=.3); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title(f"Most off-lane predicted horizon · clearance {minimum_clearance[critical]:.3f} m")

    axes[0, 1].plot(times, minimum_clearance, label="minimum predicted clearance")
    axes[0, 1].axhline(0, color="r", ls="--", label="lane boundary")
    axes[0, 1].axhline(float(cfg["collision_radius"]), color="orange", ls=":", label="penalty starts")
    axes[0, 1].set(xlabel="run time [s]", ylabel="clearance [m]", title="Optimal horizon lane clearance")
    axes[0, 1].grid(alpha=.3); axes[0, 1].legend()

    labels = list(keys); values = [total_array[key][critical] for key in labels]
    axes[1, 0].barh(labels, values, color=["tab:red" if value > 0 else "tab:green" for value in values])
    axes[1, 0].axvline(0, color="k", lw=.8); axes[1, 0].grid(axis="x", alpha=.3)
    axes[1, 0].set(title="Critical horizon cumulative signed costs", xlabel="cost")

    stack_keys = ("boundary", "friction", "front_slip", "rear_slip", "heading", "steer", "control_rate")
    bottom = np.zeros(len(times))
    for key in stack_keys:
        axes[1, 1].fill_between(times, bottom, bottom + total_array[key], label=key, alpha=.8)
        bottom += total_array[key]
    axes[1, 1].set(xlabel="run time [s]", ylabel="positive horizon cost", title="Penalty composition over run")
    axes[1, 1].grid(alpha=.3); axes[1, 1].legend(fontsize=8)
    fig.suptitle("GRU simulator · current MPPI lane-departure cost diagnosis", fontsize=16)
    output = args.run / "mppi_cost_breakdown.png"; fig.savefig(output, dpi=180)
    print(json.dumps(report, indent=2)); print(output)
    if args.show: plt.show()
    else: plt.close(fig)


if __name__ == "__main__":
    main()
