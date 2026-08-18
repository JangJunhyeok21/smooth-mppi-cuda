#!/usr/bin/env python3
"""Regress classic Pacejka bicycle parameters from an extracted NPZ dataset.

Required sample columns are t,x,y,yaw,vx,vy,omega,steer,accel,speed_cmd.
The lateral state is [speed, beta, yaw_rate].  `/ackermann_cmd.speed` is treated as a
setpoint through a fitted proportional longitudinal loop; it is not divided by
dt and misinterpreted as acceleration.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from model_tuning_utils.lateral_velocity_kf import estimate_dataset

VISUALIZE_PREPROCESS_DATA = True  # set True to plot the regression data and fitted derivatives, False to skip plotting
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "model_tuning/data/ifac0807_mppi_observation.npz"
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/results/suite_dynamic_regression_0807"
MASS=3.74; I_Z=.04712; LF=.163; LR=.161; STEER_SCALE=.50927964; STEER_BIAS=.01015773
DIRECT_PREVIOUS_STEER = True
MIN_ACCEL=-8.; MAX_ACCEL=8.5; SMOOTH_WINDOW=21; MIN_SPEED=.7; MAX_SPEED=10.
MAX_YAW_RATE=8.; MAX_FIT_SAMPLES=30000; MAX_NFEV=500; RANDOM_SEED=7
PARAMETER_NAMES = (
    "speed_kp", "B_f", "C_f", "D_f", "E_f",
    "B_r", "C_r", "D_r", "E_r",
)


def smooth_by_segment(values, bag_ids, window, dt):
    filtered = np.full_like(values, np.nan, dtype=np.float64)
    derivative = np.full_like(values, np.nan, dtype=np.float64)
    for bag_id in np.unique(bag_ids):
        indices = np.flatnonzero(bag_ids == bag_id)
        width = min(window, len(indices) // 2 * 2 - 1)
        if width < 5:
            continue
        for column in range(values.shape[1]):
            filtered[indices, column] = savgol_filter(values[indices, column], width, 3)
            derivative[indices, column] = savgol_filter(
                values[indices, column], width, 3, deriv=1, delta=dt
            )
    return filtered, derivative


def dynamic_derivative(state, command, parameters, fixed):
    speed, beta, yaw_rate = state.T
    steer_cmd, speed_cmd = command.T
    kp, bf, cf, df, ef, br, cr, dr, er = parameters
    mass, iz, lf, lr, gravity, steer_scale, steer_bias, min_accel, max_accel = fixed
    steer = np.clip(steer_scale * steer_cmd + steer_bias, -.55, .55)
    vx = np.maximum(speed * np.cos(beta), .5)
    vy = speed * np.sin(beta)
    alpha_f = steer - np.arctan2(vy + lf * yaw_rate, vx)
    alpha_r = -np.arctan2(vy - lr * yaw_rate, vx)
    fzf = mass * gravity * lr / (lf + lr)
    fzr = mass * gravity * lf / (lf + lr)
    fyf = fzf * df * np.sin(cf * np.arctan(
        bf * alpha_f - ef * (bf * alpha_f - np.arctan(bf * alpha_f))))
    fyr = fzr * dr * np.sin(cr * np.arctan(
        br * alpha_r - er * (br * alpha_r - np.arctan(br * alpha_r))))
    v_dot = np.clip(kp * (speed_cmd - speed), min_accel, max_accel)
    beta_dot = (fyf * np.cos(steer) + fyr) / (mass * np.maximum(speed, .5)) - yaw_rate
    yaw_rate_dot = (lf * fyf * np.cos(steer) - lr * fyr) / iz
    return np.column_stack((v_dot, beta_dot, yaw_rate_dot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",nargs="?",default=str(DATASET_PATH))
    parser.add_argument("-o", "--output", default=str(OUTPUT_PATH))
    parser.add_argument("--mass", type=float, default=MASS)
    parser.add_argument("--iz", type=float, default=I_Z,
                        help="fixed yaw moment of inertia [kg m^2]; not optimized")
    parser.add_argument("--lf", type=float, default=LF)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--steer-scale", type=float, default=STEER_SCALE)
    parser.add_argument("--steer-bias", type=float, default=STEER_BIAS)
    parser.add_argument("--min-accel", type=float, default=MIN_ACCEL)
    parser.add_argument("--max-accel", type=float, default=MAX_ACCEL)
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--min-speed", type=float, default=MIN_SPEED)
    parser.add_argument("--max-speed", type=float, default=MAX_SPEED)
    parser.add_argument("--max-yaw-rate", type=float, default=MAX_YAW_RATE)
    parser.add_argument("--max-fit-samples", type=int, default=MAX_FIT_SAMPLES)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    archive = np.load(args.dataset)
    raw = archive["samples"].astype(np.float64) # raw : t,x,y,yaw,vx,vy,omega,steer,accel,speed_cmd
    dt = float(archive["dt"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    bag_ids = raw[:, 11].astype(int) if raw.shape[1] > 11 else np.zeros(len(raw), int)
    # Offline identification target comes from /newmcl_pose derivatives.  The
    # stored odom vy is zero on this vehicle and must not be treated as GT.
    pose_values=np.c_[raw[:,1],raw[:,2],raw[:,3]].copy()
    for bag_id in np.unique(bag_ids):
        ii=np.flatnonzero(bag_ids==bag_id);pose_values[ii,2]=np.unwrap(pose_values[ii,2])
    pose_filtered,pose_derivative=smooth_by_segment(pose_values,bag_ids,args.smooth_window,dt)
    yaw_pose=pose_filtered[:,2];world_vx=pose_derivative[:,0];world_vy=pose_derivative[:,1]
    vx_pose=world_vx*np.cos(yaw_pose)+world_vy*np.sin(yaw_pose)
    vy_pose=-world_vx*np.sin(yaw_pose)+world_vy*np.cos(yaw_pose)
    speed = np.hypot(vx_pose, vy_pose)
    beta = np.arctan2(vy_pose, np.maximum(vx_pose, 1e-3))
    state, observed = smooth_by_segment(
        np.column_stack((speed, beta, pose_derivative[:,2])), bag_ids, args.smooth_window, dt
    )
    if VISUALIZE_PREPROCESS_DATA:
        try:
            vy_estimated,yaw_rate_estimated=estimate_dataset(raw,archive["columns"],dt)
            estimator_label="2-state KF vy"
        except ValueError:
            vy_estimated=raw[:,5].copy();yaw_rate_estimated=raw[:,6].copy()
            estimator_label="stored vy (no aligned IMU columns)"
        beta_estimated=np.arctan2(vy_estimated,raw[:,4])
        beta_pose=np.arctan2(vy_pose,vx_pose)
        ok=(np.all(np.isfinite(np.c_[vy_estimated,vy_pose,beta_estimated,beta_pose]),axis=1) &
            (np.abs(raw[:,4])>=args.min_speed) & (np.abs(vx_pose)>=args.min_speed))
        vy_delta=vy_estimated[ok]-vy_pose[ok]
        beta_delta=np.arctan2(np.sin(beta_estimated[ok]-beta_pose[ok]),np.cos(beta_estimated[ok]-beta_pose[ok]))
        comparison={"estimator":estimator_label,"samples":int(ok.sum()),
                    "comparison_min_abs_vx_mps":args.min_speed,
                    "vy_mae_mps":float(np.mean(np.abs(vy_delta))),
                    "vy_rmse_mps":float(np.sqrt(np.mean(vy_delta**2))),
                    "beta_mae_rad":float(np.mean(np.abs(beta_delta))),
                    "beta_rmse_rad":float(np.sqrt(np.mean(beta_delta**2)))}
        (output/"vy_estimator_vs_pose_metrics.json").write_text(json.dumps(comparison,indent=2)+"\n")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(4,1,sharex=True,figsize=(11,10))
        axes[0].plot(raw[:,0],raw[:,4],"k.",alpha=.2,label="raw odom vx")
        axes[0].plot(raw[:,0],vx_pose,"b-",label="pose-difference body vx")
        axes[0].set_ylabel("vx [m/s]");axes[0].legend()
        axes[1].plot(raw[:,0],vy_pose,"k-",alpha=.7,label="pose-difference body vy")
        axes[1].plot(raw[:,0],vy_estimated,"b-",alpha=.8,label=estimator_label)
        axes[1].set_ylabel("vy [m/s]");axes[1].legend()
        axes[2].plot(raw[:,0],beta_pose,"k-",alpha=.7,label="pose-derived beta")
        axes[2].plot(raw[:,0],beta_estimated,"b-",alpha=.8,label="estimated beta=atan2(vy,vx)")
        axes[2].set_ylabel("beta [rad]");axes[2].legend()
        axes[3].plot(raw[:,0],raw[:,6],"k.",alpha=.2,label="raw yaw rate")
        axes[3].plot(raw[:,0],yaw_rate_estimated,"b-",label="estimated yaw rate")
        axes[3].set_ylabel("yaw rate [rad/s]");axes[3].set_xlabel("time [s]");axes[3].legend()
        for boundary in np.flatnonzero(np.r_[False,bag_ids[1:]!=bag_ids[:-1]]):
            for axis in axes: axis.axvline(raw[boundary,0],color="0.7",lw=.6,ls=":")
        fig.tight_layout();fig.savefig(output/"vy_estimator_vs_pose.png",dpi=180);plt.close(fig)




    command = raw[:, [7, 9]].copy()  # steer, /ackermann_cmd.speed
    if DIRECT_PREVIOUS_STEER:
        # Match MPPI dynamic_mlp_residual exactly: at time t the physical
        # steering angle is the previous Ackermann command in this segment.
        for bag_id in np.unique(bag_ids):
            ii=np.flatnonzero(bag_ids==bag_id)
            command[ii[1:],0]=raw[ii[:-1],7]
            command[ii[0],0]=raw[ii[0],7]
    valid = (
        np.all(np.isfinite(np.c_[state, observed, command]), axis=1)
        & ((raw[:, 10] == 0) if raw.shape[1] > 10 else True)
        & (state[:, 0] >= args.min_speed) & (state[:, 0] <= args.max_speed)
        & (np.abs(state[:, 2]) <= args.max_yaw_rate)
        & (command[:, 1] >= 0.) & (command[:, 1] <= 12.)
        & (np.abs(observed[:, 0]) <= 15.)
        & (np.abs(observed[:, 1]) <= 15.)
        & (np.abs(observed[:, 2]) <= 100.)
    )
    indices = np.flatnonzero(valid)
    if len(indices) < 100:
        raise SystemExit(f"only {len(indices)} valid samples")
    if len(indices) > args.max_fit_samples:
        indices = np.random.default_rng(args.seed).choice(
            indices, args.max_fit_samples, replace=False
        )
    state, command, observed = state[indices], command[indices], observed[indices]
    regression_steer_scale=1.0 if DIRECT_PREVIOUS_STEER else args.steer_scale
    regression_steer_bias=0.0 if DIRECT_PREVIOUS_STEER else args.steer_bias
    fixed = (args.mass, args.iz, args.lf, args.lr, 9.81, regression_steer_scale,
             regression_steer_bias, args.min_accel, args.max_accel)
    initial = np.array([8., 6., 1.4, .8, .1, 6., 1.4, .8, .1])
    lower = np.array([.1, .1, .5, .05, -1., .1, .5, .05, -1.])
    upper = np.array([30., 30., 3., 2., 1., 30., 3., 2., 1.])
    scale = np.array([2., 4., 1.])
    objective = lambda p: ((dynamic_derivative(state, command, p, fixed)-observed)/scale).ravel()
    fit = least_squares(objective, initial, bounds=(lower, upper), loss="soft_l1",
                        max_nfev=args.max_nfev, verbose=1)
    prediction = dynamic_derivative(state, command, fit.x, fixed)
    rmse = np.sqrt(np.mean((prediction-observed)**2, axis=0))
    fitted_parameters = dict(zip(PARAMETER_NAMES, map(float, fit.x)))
    # Keep I_z in parameters for compatibility with archived checkpoints and MPPI
    # exporter, while explicitly recording that it was fixed during fitting.
    fitted_parameters["I_z"] = float(args.iz)
    result = {
        "parameters": fitted_parameters,
        "fixed": {"mass": args.mass, "I_z": args.iz, "l_f": args.lf, "l_r": args.lr,
                  "steer_scale": regression_steer_scale, "steer_bias": regression_steer_bias,
                  "steering_policy": ("previous_ackermann_command" if DIRECT_PREVIOUS_STEER
                                      else "mapped_current_command")},
        "rmse": dict(zip(("v_dot", "beta_dot", "yaw_rate_dot"), map(float, rmse))),
        "samples": int(len(indices)), "dt": dt, "optimizer_cost": float(fit.cost),
        "success": bool(fit.success), "message": fit.message,
    }
    (output / "dynamic_params.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output / "dynamic_regression_predictions.npz",
                        state=state, command=command, gt_derivative=observed,
                        predicted_derivative=prediction)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
