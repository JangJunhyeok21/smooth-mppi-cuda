#!/usr/bin/env python3
"""Applied wheel-angle lag regression used by numbered Step 4.

There is no steering-angle measurement.  The targets are causal KF vy/r and
MCL trajectory response.  Candidate-dependent command warm-up is performed by
``classic_model_regression.rollout_numpy``.
"""
from copy import deepcopy
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import yaml

import classic_model_regression as classic
from classic_model_regression import (
    DATA, HORIZON, NAMES, mcl_relative_pose, metrics, objective, relative_pose,
    rollout_numpy, starts)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/params.yaml"
OPPONENT_CONFIG = ROOT / "config/opponent_params.yaml"
OUT = Path(os.environ.get("STEERING_ID_OUT",
    ROOT / "model_tuning/results/steering_actuator_regression"))
FIX_STEER_TIME_CONSTANT = False
SCALE_BOUNDS = (0.5, 1.5)
BIAS_BOUNDS = (-0.15, 0.15)
TAU_BOUNDS = (0.04, 0.50)
SEED = 31
SHOW_PLOTS = True


def selected_tire_parameters(config):
    """Use the current runtime YAML, never a stale regression artifact."""
    return np.asarray([config[f"dynamic_mlp_{name}"] for name in NAMES], float)


def excitation_subsets(data, indices):
    u = data["features"][:, 3]
    static, transient = [], []
    for index in indices:
        window = u[index:index+50]
        delta = np.mean(np.abs(np.diff(window)))
        if np.mean(np.abs(window)) > .035 and delta < .012:
            static.append(index)
        if np.ptp(window) > .12 or delta > .012:
            transient.append(index)
    return (np.asarray(static if static else indices, int),
            np.asarray(transient if transient else indices, int))


def with_steering(config, scale, bias, tau):
    result = deepcopy(config)
    result["kinematic_steer_scale"] = float(scale)
    result["kinematic_steer_bias"] = float(bias)
    result["steer_servo_time_constant"] = float(tau)
    return result


def steering_state_rollout(data, window_starts, config):
    """Reproduce rollout_numpy's command mapping and applied-steer state."""
    feature=data["features"]
    starts_array=np.asarray(window_starts,int)
    tau=max(float(config["steer_servo_time_constant"]),1e-6)
    max_steer=float(config["max_steer"])
    rate_limit=float(config["actuator_max_steer_rate"])
    applied=np.clip(feature[starts_array-classic.WARMUP_SAMPLES,3],
                    -max_steer,max_steer)
    for offset in range(-classic.WARMUP_SAMPLES+1,0):
        target=np.clip(float(config["kinematic_steer_scale"])*
                       feature[starts_array+offset,3]+
                       float(config["kinematic_steer_bias"]),-max_steer,max_steer)
        steer_rate=np.clip((target-applied)/tau,-rate_limit,rate_limit)
        applied=np.clip(applied+steer_rate*.02,-max_steer,max_steer)
    applied_trace=[applied.copy()]
    raw_command=[];target_trace=[]
    for step in range(HORIZON):
        command=feature[starts_array+2*step,3]
        target=np.clip(float(config["kinematic_steer_scale"])*command+
                       float(config["kinematic_steer_bias"]),-max_steer,max_steer)
        steer_rate=np.clip((target-applied)/tau,-rate_limit,rate_limit)
        applied=np.clip(applied+steer_rate*.04,-max_steer,max_steer)
        raw_command.append(command.copy());target_trace.append(target.copy())
        applied_trace.append(applied.copy())
    # Commands are held over each interval; prepend the first command so every
    # trace shares the t=0 ... horizon plotting grid.
    raw=np.stack(raw_command,axis=1)
    target=np.stack(target_trace,axis=1)
    raw=np.concatenate((raw[:,:1],raw),axis=1)
    target=np.concatenate((target[:,:1],target),axis=1)
    return raw,target,np.stack(applied_trace,axis=1)


def update_yaml(scale, bias, tau):
    values = {"kinematic_steer_scale": scale, "kinematic_steer_bias": bias,
              "steer_servo_time_constant": tau}
    for path in (CONFIG,OPPONENT_CONFIG):
        lines = path.read_text().splitlines()
        found = set()
        for line_number, line in enumerate(lines):
            for key, value in values.items():
                if line.strip().startswith(key + ":"):
                    indent = line[:len(line)-len(line.lstrip())]
                    lines[line_number] = f"{indent}{key}: {value:.9g}  # Step 4 vehicle-response identification"
                    found.add(key)
        if found != set(values):
            raise RuntimeError(f"{path}: missing runtime YAML keys: {set(values)-found}")
        path.write_text("\n".join(lines) + "\n")


def plot_open_loop_evaluation(data, tire, config, old, fitted, validation, test):
    """Plot held-out GT against previous and fitted lateral free rollouts."""
    heldout = np.concatenate((validation, test))
    if not len(heldout):
        raise RuntimeError("no validation/test steering rollout is available for plotting")
    old_config = with_steering(config, *old)
    fitted_config = with_steering(config, *fitted)
    old_prediction, truth = rollout_numpy(tire, data, heldout, old_config)
    fitted_prediction, fitted_truth = rollout_numpy(tire, data, heldout, fitted_config)
    if not np.allclose(truth, fitted_truth, equal_nan=True):
        raise RuntimeError("previous/fitted steering rollouts do not share identical GT")
    raw_steer,old_target,old_applied=steering_state_rollout(data,heldout,old_config)
    _,fitted_target,fitted_applied=steering_state_rollout(data,heldout,fitted_config)

    # Rank by the previous model's full-horizon lateral-state RMSE. Scaling
    # keeps vy and yaw rate comparably represented in the selection metric.
    normalized_error = (old_prediction[:, :, 1:3]-truth[:, :, 1:3]) / np.array((.5, 1.0))
    ranking_error = np.sqrt(np.mean(normalized_error**2, axis=(1, 2)))
    order = np.argsort(ranking_error)
    selected = (order[0], order[int(round(.95*(len(order)-1)))], order[-1])
    labels = ("best", "p95", "worst")
    # rollout_numpy returns t=40 ms ... horizon and therefore does not contain
    # its shared t=0 initial condition.  Prepend it for an honest visual audit:
    # all three traces must start from the exact same measured state/pose.
    initial_state=data["features"][heldout,:3].astype(float).copy()
    if "teacher_state" in data.files:
        initial_state[:]=data["teacher_state"][heldout]
    old_state_trace=np.concatenate((initial_state[:,None,:],old_prediction),axis=1)
    fitted_state_trace=np.concatenate((initial_state[:,None,:],fitted_prediction),axis=1)
    truth_state_trace=np.concatenate((initial_state[:,None,:],truth),axis=1)
    time = .04*np.arange(HORIZON+1)
    old_pose = np.concatenate((np.zeros((len(heldout),1,3)),
                               relative_pose(old_prediction, 1.0)),axis=1)
    fitted_pose = np.concatenate((np.zeros((len(heldout),1,3)),
                                  relative_pose(fitted_prediction, 1.0)),axis=1)
    future_gt_pose = (mcl_relative_pose(data, heldout)
                      if "target_pose" in data.files or "mcl_pose" in data.files
                      else relative_pose(truth, 1.0))
    gt_pose = np.concatenate((np.zeros((len(heldout),1,3)),future_gt_pose),axis=1)
    fig, axes = plt.subplots(3, 4, figsize=(24, 14), constrained_layout=True)
    cases = {}
    for row, (label, index) in enumerate(zip(labels, selected)):
        axis = axes[row, 0]
        axis.plot(gt_pose[index, :, 0], gt_pose[index, :, 1], "k-", lw=2.3,
                  label="GT trajectory")
        axis.plot(old_pose[index, :, 0], old_pose[index, :, 1], "--", lw=2,
                  color="tab:blue", label="previous lateral open loop")
        axis.plot(fitted_pose[index, :, 0], fitted_pose[index, :, 1], "-", lw=1.9,
                  color="tab:red", label="fitted lateral open loop")
        axis.scatter(gt_pose[index, -1, 0], gt_pose[index, -1, 1], color="black", s=25)
        axis.set(title=f"{label.upper()} held-out trajectory", xlabel="relative x [m]",
                 ylabel="relative y [m]")
        axis.axis("equal");axis.grid(alpha=.3);axis.legend()

        axis = axes[row, 1]
        axis.plot(time, truth_state_trace[index, :, 1], "k-", lw=2.3, label="GT vy")
        axis.plot(time, old_state_trace[index, :, 1], "--", color="tab:blue", lw=2,
                  label="previous open loop")
        axis.plot(time, fitted_state_trace[index, :, 1], color="tab:red", lw=1.9,
                  label="fitted open loop")
        axis.set(title=f"{label.upper()} lateral velocity", xlabel="rollout time [s]",
                 ylabel="vy [m/s]")
        axis.grid(alpha=.3);axis.legend()

        axis=axes[row,3]
        axis.plot(time,raw_steer[index],color="0.45",ls=":",lw=1.8,
                  label="raw steering command")
        axis.plot(time,old_target[index],color="tab:cyan",ls="--",lw=1.8,
                  label="previous mapped target")
        axis.plot(time,fitted_target[index],color="tab:orange",ls="--",lw=1.8,
                  label="fitted mapped target")
        axis.plot(time,old_applied[index],color="tab:blue",lw=2,
                  label="previous predicted applied steer")
        axis.plot(time,fitted_applied[index],color="tab:red",lw=2,
                  label="fitted predicted applied steer")
        axis.set(title=f"{label.upper()} steering actuator state",
                 xlabel="rollout time [s]",ylabel="steering value [rad]")
        axis.grid(alpha=.3);axis.legend(fontsize=8,ncol=2)

        axis = axes[row, 2]
        axis.plot(time, truth_state_trace[index, :, 2], "k-", lw=2.3, label="GT yaw rate")
        axis.plot(time, old_state_trace[index, :, 2], "--", color="tab:blue", lw=2,
                  label="previous open loop")
        axis.plot(time, fitted_state_trace[index, :, 2], color="tab:red", lw=1.9,
                  label="fitted open loop")
        axis.set(title=f"{label.upper()} yaw rate", xlabel="rollout time [s]",
                 ylabel="yaw rate [rad/s]")
        axis.grid(alpha=.3);axis.legend()

        old_lateral_rmse=np.sqrt(np.mean(
            (old_prediction[index,:,1:3]-truth[index,:,1:3])**2,axis=0))
        fitted_lateral_rmse=np.sqrt(np.mean(
            (fitted_prediction[index,:,1:3]-truth[index,:,1:3])**2,axis=0))
        cases[label]={"source_row":int(heldout[index]),
            "previous_vy_rmse_mps":float(old_lateral_rmse[0]),
            "previous_yaw_rate_rmse_radps":float(old_lateral_rmse[1]),
            "fitted_vy_rmse_mps":float(fitted_lateral_rmse[0]),
            "fitted_yaw_rate_rmse_radps":float(fitted_lateral_rmse[1])}
    fig.suptitle(f"Steering actuator held-out lateral open-loop evaluation ({HORIZON*.04:.1f} s)",
                 fontsize=16)
    output = OUT / "open_loop_comparison.png"
    fig.savefig(output, dpi=180)
    (OUT / "representative_open_loop_rollouts.json").write_text(json.dumps({
        "selection":"best/p95/worst by previous-model normalized vy/yaw-rate rollout RMSE",
        "horizon_s":HORIZON*.04,"heldout_windows":int(len(heldout)),"cases":cases
    },indent=2)+"\n")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text())["/**"]["ros__parameters"]
    data, data_contract = classic.load_regression_data(DATA, config)
    tire = selected_tire_parameters(config)
    train, validation, test = (starts(data, split) for split in range(3))
    steady, transient = excitation_subsets(data, train)
    old = np.asarray((config["kinematic_steer_scale"],
                      config["kinematic_steer_bias"],
                      config["steer_servo_time_constant"]), float)
    fixed_tau=float(old[2])
    fit_bounds=(SCALE_BOUNDS,BIAS_BOUNDS) if FIX_STEER_TIME_CONSTANT else (
        SCALE_BOUNDS,BIAS_BOUNDS,TAU_BOUNDS)
    mapping_fit=differential_evolution(
        lambda values: objective(tire,data,train,with_steering(
            config,values[0],values[1],fixed_tau if FIX_STEER_TIME_CONSTANT
            else values[2])),bounds=fit_bounds,seed=SEED,popsize=18,
        maxiter=60,tol=1e-7,polish=True,workers=1,updating="immediate")
    fitted=np.asarray((mapping_fit.x[0],mapping_fit.x[1],fixed_tau
                       if FIX_STEER_TIME_CONSTANT else mapping_fit.x[2]))

    def evaluate(values, indices):
        return metrics(tire, data, indices,
            with_steering(config, values[0], values[1], values[2]))

    report = {
        "parameter_order": ["kinematic_steer_scale", "kinematic_steer_bias",
                            "steer_servo_time_constant"],
        "fixed_actuator_max_steer_rate": float(config["actuator_max_steer_rate"]),
        "input_path":str(Path(DATA).resolve()),
        "input_contract":data_contract,
        "rollout_horizon_steps":int(HORIZON),
        "rollout_horizon_s":float(HORIZON*.04),
        "gt_consistency_mode":classic.GT_CONSISTENCY_MODE,
        "loss_weights":{"vx":classic.VX_LOSS_WEIGHT,
            "vy":classic.VY_LOSS_WEIGHT,
            "yaw_rate":classic.YAW_RATE_LOSS_WEIGHT,
            "position_xy":classic.POSITION_LOSS_WEIGHT,
            "trajectory_yaw":classic.YAW_TRAJECTORY_LOSS_WEIGHT},
        "tire_parameter_source": str(CONFIG),
        "fixed_tire_parameters": dict(zip(NAMES, tire.tolist())),
        "target": ("configured teacher_state/target_pose from classic_model_regression; "
                   "no direct steering-angle GT"),
        "hidden_state_initialization": "candidate-dependent 0.8 s steering-command warm-up",
        "fixed_steer_servo_time_constant_s":(fixed_tau
                                               if FIX_STEER_TIME_CONSTANT else None),
        "steer_time_constant_bounds_s":list(TAU_BOUNDS),
        "stage_4": {"optimized": ["kinematic_steer_scale","kinematic_steer_bias"]+
                                  ([] if FIX_STEER_TIME_CONSTANT else
                                   ["steer_servo_time_constant"]),
                     "fixed": (["steer_servo_time_constant"]
                               if FIX_STEER_TIME_CONSTANT else []),
                     "train_windows":int(len(train))},
        "previous": old.tolist(), "fitted": fitted.tolist(),
        "metrics_previous": {"validation": evaluate(old, validation),
                             "test": evaluate(old, test)},
        "metrics_fitted": {"validation": evaluate(fitted, validation),
                           "test": evaluate(fitted, test)}}
    old_score=classic.validation_score(report["metrics_previous"]["validation"])
    new_score=classic.validation_score(report["metrics_fitted"]["validation"])
    report["weighted_validation_score_previous"]=float(old_score)
    report["weighted_validation_score_fitted"]=float(new_score)
    report["deployment_gate_passed"] = bool(new_score < old_score)
    (OUT / "regression.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_path = plot_open_loop_evaluation(
        data, tire, config, old, fitted, validation, test)
    if report["deployment_gate_passed"]:
        update_yaml(*fitted)
    print(json.dumps(report, indent=2))
    print("\nTuned steering-actuator parameters:")
    parameter_labels=(
        ("kinematic_steer_scale", "", old[0], fitted[0]),
        ("kinematic_steer_bias", "rad", old[1], fitted[1]),
        ("steer_servo_time_constant", "s", old[2], fitted[2]),
    )
    for name,unit,previous,tuned in parameter_labels:
        suffix=f" {unit}" if unit else ""
        change=tuned-previous
        print(f"  {name}: {previous:.9g} -> {tuned:.9g}{suffix} "
              f"(delta {change:+.9g}{suffix})")
    print("Step 4 loss weights: "
          f"vx={classic.VX_LOSS_WEIGHT:g}, vy={classic.VY_LOSS_WEIGHT:g}, "
          f"yaw_rate={classic.YAW_RATE_LOSS_WEIGHT:g}, "
          f"position_xy={classic.POSITION_LOSS_WEIGHT:g}, "
          f"trajectory_yaw={classic.YAW_TRAJECTORY_LOSS_WEIGHT:g}")
    print(f"Weighted validation score: {old_score:.6g} -> {new_score:.6g}")
    classic.print_pose_metric_change("Step 4 validation",
        report["metrics_previous"]["validation"],
        report["metrics_fitted"]["validation"])
    if report["deployment_gate_passed"]:
        print(f"Deployment gate: PASS; fitted values applied to {CONFIG}")
    else:
        print("Deployment gate: FAIL; params.yaml was not changed")
    print(f"open-loop plot: {plot_path}")


if __name__ == "__main__":
    main()
