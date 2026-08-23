#!/usr/bin/env python3
"""Joint Step 3/4 identification by alternating blocks and local polishing.

This alternates tire/inertia fitting with the applied wheel-angle lag. The
Ackermann command-to-wheel mapping is fixed to identity.
"""
from pathlib import Path
import json

import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import yaml

import classic_model_regression as classic
import steering_actuator_regression as steering


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
ROOT=Path(__file__).resolve().parents[2]
DATA_PATH=ROOT/"model_tuning/data/ifac0810_0819_autonomous_physics_clean"
OUTPUT_DIR=ROOT/"model_tuning/results/joint_step_3_4_identification"
CONFIG_PATH=ROOT/"config/params.yaml"

ROLLOUT_HORIZON_STEPS=60
MAX_WINDOWS_PER_BAG=80
ACTUATOR_WARMUP_SAMPLES=40
RANDOM_SEED=31
USE_VALIDATION_TEST_SPLIT=False
GT_CONSISTENCY_MODE="adjust_states_to_pose" # adjust_vy_to_pose or "adjust_pose_to_states" or  "adjust_states_to_pose" or "none"
POSE_DERIVATIVE_SMOOTH_WINDOW_S=.20

VX_LOSS_WEIGHT=.1
VY_LOSS_WEIGHT=.5
YAW_RATE_LOSS_WEIGHT=1.5
POSITION_LOSS_WEIGHT=8.0
YAW_TRAJECTORY_LOSS_WEIGHT=1.5
# Explicit 40 ms transition loss in addition to the 60-step recursive loss.
# 0 disables it; 1 gives the one-step aggregate its natural full weight.
ONE_STEP_LOSS_WEIGHT=1.0
# Apply x/y/yaw loss at all 60 recursive MPPI knots, not only at 40 ms and
# the final 2.4 s endpoint.
FULL_TRAJECTORY_POSE_LOSS_WEIGHT=1.0
# Penalize the worst 10% rollout endpoints during fitting.  This targets the
# P95 regressions that an all-window mean objective previously accepted.
ENDPOINT_TAIL_LOSS_WEIGHT=1.0
ENDPOINT_TAIL_QUANTILE=.90

ALTERNATING_ROUNDS=8
CONVERGENCE_RELATIVE_SCORE=1e-3
TIRE_BLOCK_MAX_ITERATIONS=15
TIRE_BLOCK_POPULATION=6
STEERING_BLOCK_MAX_ITERATIONS=20
JOINT_POLISH_MAX_ITERATIONS=180
OPTIMIZER_WINDOW_LIMIT=240

PACEJKA_B_F_BOUNDS=(.2,30.);PACEJKA_C_F_BOUNDS=(.5,2.5)
PACEJKA_D_F_BOUNDS=(.05,3.5);PACEJKA_E_F_BOUNDS=(-1.,1.)
PACEJKA_B_R_BOUNDS=(.2,30.);PACEJKA_C_R_BOUNDS=(.5,2.5)
PACEJKA_D_R_BOUNDS=(.05,3.5);PACEJKA_E_R_BOUNDS=(-1.,1.)
YAW_INERTIA_BOUNDS=(.005,.5)
STEER_TIME_CONSTANT_BOUNDS=(.01,.6)

# The fitted values are written to OUTPUT_DIR regardless.  Runtime YAML is
# updated only if the complete candidate improves both validation and test.
UPDATE_CONFIG=True
USE_PLOT=True
INTERACTIVE_BAG_INSPECTOR=True


TIRE_NAMES=classic.NAMES
PARAMETER_NAMES=(*TIRE_NAMES,"I_z","steer_servo_time_constant")


def configure_modules():
    classic.DATA=Path(DATA_PATH).resolve();classic.OUT=Path(OUTPUT_DIR).resolve()
    classic.HORIZON=ROLLOUT_HORIZON_STEPS
    classic.MAX_PER_BAG=MAX_WINDOWS_PER_BAG
    classic.WARMUP_SAMPLES=ACTUATOR_WARMUP_SAMPLES
    classic.SEED=RANDOM_SEED
    classic.USE_VALIDATION_TEST_SPLIT=USE_VALIDATION_TEST_SPLIT
    classic.GT_CONSISTENCY_MODE=GT_CONSISTENCY_MODE
    classic.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S=POSE_DERIVATIVE_SMOOTH_WINDOW_S
    classic.VX_LOSS_WEIGHT=VX_LOSS_WEIGHT;classic.VY_LOSS_WEIGHT=VY_LOSS_WEIGHT
    classic.YAW_RATE_LOSS_WEIGHT=YAW_RATE_LOSS_WEIGHT
    classic.POSITION_LOSS_WEIGHT=POSITION_LOSS_WEIGHT
    classic.YAW_TRAJECTORY_LOSS_WEIGHT=YAW_TRAJECTORY_LOSS_WEIGHT
    classic.ONE_STEP_LOSS_WEIGHT=ONE_STEP_LOSS_WEIGHT
    classic.FULL_TRAJECTORY_POSE_LOSS_WEIGHT=FULL_TRAJECTORY_POSE_LOSS_WEIGHT
    classic.ENDPOINT_TAIL_LOSS_WEIGHT=ENDPOINT_TAIL_LOSS_WEIGHT
    classic.ENDPOINT_TAIL_QUANTILE=ENDPOINT_TAIL_QUANTILE
    classic.AUTO_FIT_POSITION_SPEED_SCALE=False
    classic.SHOW_PLOTS=USE_PLOT
    classic.INTERACTIVE_BAG_INSPECTOR=INTERACTIVE_BAG_INSPECTOR
    classic.BOUNDS=np.asarray((PACEJKA_B_F_BOUNDS,PACEJKA_C_F_BOUNDS,
        PACEJKA_D_F_BOUNDS,PACEJKA_E_F_BOUNDS,PACEJKA_B_R_BOUNDS,
        PACEJKA_C_R_BOUNDS,PACEJKA_D_R_BOUNDS,PACEJKA_E_R_BOUNDS),float)
    classic.I_Z_MIN,classic.I_Z_MAX=YAW_INERTIA_BOUNDS
    steering.HORIZON=ROLLOUT_HORIZON_STEPS
    steering.SHOW_PLOTS=False
    steering.OUT=Path(OUTPUT_DIR).resolve()


def split_windows(data):
    split_starts=tuple(classic.starts(data,index) for index in range(3))
    if USE_VALIDATION_TEST_SPLIT:
        return (*split_starts,"bag-disjoint validation/test")
    nonempty=[value for value in split_starts if len(value)]
    train=np.concatenate(nonempty) if nonempty else np.empty(0,int)
    bags=np.unique(data["bag_id"][train]) if len(train) else np.empty(0,int)
    if not len(bags):raise RuntimeError("no usable bag for joint evaluation")
    # With splitting disabled the user explicitly requests an in-sample
    # diagnostic.  Score every training bag, not one arbitrarily selected bag.
    return train,train.copy(),train.copy(),(
        f"in-sample all-train-bags diagnostic: bags={bags.tolist()}")


def unpack(vector,base_config):
    vector=np.asarray(vector,float);config=dict(base_config)
    config["dynamic_mlp_I_z"]=float(vector[8])
    config["kinematic_steer_scale"]=1.0
    config["kinematic_steer_bias"]=0.0
    config["steer_servo_time_constant"]=float(vector[9])
    config["kinematic_position_speed_scale"]=1.0
    return vector[:8],config


def score(vector,data,indices,base_config,regularize=True):
    tire,config=unpack(vector,base_config)
    return classic.objective(tire,data,indices,config,regularize=regularize)


def validation_metrics(vector,data,indices,base_config):
    tire,config=unpack(vector,base_config)
    metric=classic.metrics(tire,data,indices,config)
    return metric,float(classic.validation_score(metric))


def limited(indices,limit):
    if len(indices)<=limit:return np.asarray(indices,int)
    return np.asarray(indices,int)[np.linspace(0,len(indices)-1,limit).astype(int)]


def tire_inertia_block(current,data,train,base_config,round_index):
    subset=limited(train,OPTIMIZER_WINDOW_LIMIT)
    tire,config=unpack(current,base_config)
    lower=np.maximum(classic.BOUNDS[:,0],tire*np.where(tire>=0,.7,1.3))
    upper=np.minimum(classic.BOUNDS[:,1],tire*np.where(tire>=0,1.3,.7))
    for index in (3,7):
        lower[index]=max(classic.BOUNDS[index,0],tire[index]-.3)
        upper[index]=min(classic.BOUNDS[index,1],tire[index]+.3)
    invalid=lower>=upper
    lower[invalid]=classic.BOUNDS[invalid,0];upper[invalid]=classic.BOUNDS[invalid,1]
    fit=differential_evolution(lambda value:classic.objective(
        value,data,subset,config),np.c_[lower,upper],seed=RANDOM_SEED+round_index,
        maxiter=TIRE_BLOCK_MAX_ITERATIONS,popsize=TIRE_BLOCK_POPULATION,
        polish=True,workers=1)
    inertia=minimize_scalar(lambda value:classic.objective(
        fit.x,data,subset,{**config,"dynamic_mlp_I_z":value}),
        bounds=YAW_INERTIA_BOUNDS,method="bounded",options={"xatol":1e-6})
    result=current.copy();result[:8]=fit.x;result[8]=inertia.x
    return result


def steering_block(current,data,train,base_config,round_index):
    tire,config=unpack(current,base_config)
    _,transient=steering.excitation_subsets(data,train)
    transient=limited(transient,OPTIMIZER_WINDOW_LIMIT)
    lag=minimize_scalar(lambda value:classic.objective(
        tire,data,transient,steering.with_steering(config,1.0,0.0,value)),
        bounds=STEER_TIME_CONSTANT_BOUNDS,method="bounded",options={"xatol":1e-6})
    result=current.copy();result[9]=lag.x
    return result


def joint_polish(current,data,train,base_config):
    subset=limited(train,OPTIMIZER_WINDOW_LIMIT)
    bounds=[*map(tuple,classic.BOUNDS),YAW_INERTIA_BOUNDS,
            STEER_TIME_CONSTANT_BOUNDS]
    result=minimize(lambda value:score(value,data,subset,base_config),current,
        method="Powell",bounds=bounds,options={"maxiter":JOINT_POLISH_MAX_ITERATIONS,
        "xtol":1e-5,"ftol":1e-7})
    return np.asarray(result.x,float)


def update_yaml(vector):
    values=dict(zip(PARAMETER_NAMES,vector))
    yaml_keys={**{name:f"dynamic_mlp_{name}" for name in TIRE_NAMES},
               "I_z":"dynamic_mlp_I_z",
               "steer_servo_time_constant":"steer_servo_time_constant"}
    reverse={yaml_key:values[name] for name,yaml_key in yaml_keys.items()}
    lines=CONFIG_PATH.read_text().splitlines();found=set()
    for index,line in enumerate(lines):
        for key,value in reverse.items():
            if line.strip().startswith(key+":"):
                indent=line[:len(line)-len(line.lstrip())]
                lines[index]=f"{indent}{key}: {float(value):.9g}  # joint Step 3/4 identification"
                found.add(key)
    if found!=set(reverse):raise RuntimeError(f"missing YAML keys: {set(reverse)-found}")
    CONFIG_PATH.write_text("\n".join(lines)+"\n")


def main():
    configure_modules();OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    base_config=yaml.safe_load(CONFIG_PATH.read_text())["/**"]["ros__parameters"]
    data,data_contract=classic.load_regression_data(DATA_PATH,base_config)
    train,validation,test,evaluation_contract=split_windows(data)
    if min(len(train),len(validation),len(test))==0:
        raise RuntimeError("joint Step 3/4 has an empty train/validation/test window set")
    current=np.asarray([*[base_config[f"dynamic_mlp_{name}"] for name in TIRE_NAMES],
        base_config["dynamic_mlp_I_z"],base_config["steer_servo_time_constant"]],float)
    accepted=current.copy();baseline_metric,baseline_score=validation_metrics(
        current,data,validation,base_config)
    baseline_test_metric,baseline_test_score=validation_metrics(
        current,data,test,base_config)
    accepted_score=baseline_score;trace=[]
    print(f"Joint Step 3/4 baseline weighted validation score: {baseline_score:.8g}")
    for round_index in range(ALTERNATING_ROUNDS):
        candidate=tire_inertia_block(accepted,data,train,base_config,round_index)
        candidate=steering_block(candidate,data,train,base_config,round_index)
        candidate_metric,candidate_score=validation_metrics(
            candidate,data,validation,base_config)
        improved=candidate_score<accepted_score
        trace.append({"round":round_index+1,"candidate_score":candidate_score,
                      "accepted":bool(improved),"parameters":dict(zip(
                          PARAMETER_NAMES,candidate.tolist()))})
        print(f"Round {round_index+1}: {accepted_score:.8g} -> "
              f"{candidate_score:.8g} ({'accepted' if improved else 'rejected'})")
        if improved:
            relative=(accepted_score-candidate_score)/max(abs(accepted_score),1e-12)
            accepted=candidate;accepted_score=candidate_score
            if relative<CONVERGENCE_RELATIVE_SCORE:break
    polished=joint_polish(accepted,data,train,base_config)
    polished_metric,polished_score=validation_metrics(polished,data,validation,base_config)
    polish_accepted=polished_score<accepted_score
    if polish_accepted:accepted=polished;accepted_score=polished_score
    fitted_tire,fitted_config=unpack(accepted,base_config)
    fitted_metric=classic.metrics(fitted_tire,data,validation,fitted_config)
    baseline_loss=classic.objective_breakdown(
        current[:8],data,validation,unpack(current,base_config)[1])
    fitted_loss=classic.objective_breakdown(
        fitted_tire,data,validation,fitted_config)
    fitted_test_metric=classic.metrics(fitted_tire,data,test,fitted_config)
    fitted_test_score=float(classic.validation_score(fitted_test_metric))
    # Do not deploy a candidate that merely overfits the validation windows.
    p95_gate=(fitted_metric["trajectory_p95_m"]
              <=baseline_metric["trajectory_p95_m"]
              and fitted_metric["trajectory_yaw_p95_rad"]
              <=baseline_metric["trajectory_yaw_p95_rad"]
              and fitted_test_metric["trajectory_p95_m"]
              <=baseline_test_metric["trajectory_p95_m"]
              and fitted_test_metric["trajectory_yaw_p95_rad"]
              <=baseline_test_metric["trajectory_yaw_p95_rad"])
    gate=(accepted_score<baseline_score and
          fitted_test_score<baseline_test_score and p95_gate)
    report={"input_contract":data_contract,"evaluation_contract":evaluation_contract,
        "gt_consistency_mode":GT_CONSISTENCY_MODE,
        "one_step_loss_weight":ONE_STEP_LOSS_WEIGHT,
        "full_trajectory_pose_loss_weight":FULL_TRAJECTORY_POSE_LOSS_WEIGHT,
        "endpoint_tail_loss_weight":ENDPOINT_TAIL_LOSS_WEIGHT,
        "endpoint_tail_quantile":ENDPOINT_TAIL_QUANTILE,
        "parameter_names":list(PARAMETER_NAMES),
        "previous":dict(zip(PARAMETER_NAMES,current.tolist())),
        "fitted":dict(zip(PARAMETER_NAMES,accepted.tolist())),
        "weighted_validation_score_previous":baseline_score,
        "weighted_validation_score_fitted":accepted_score,
        "weighted_test_score_previous":baseline_test_score,
        "weighted_test_score_fitted":fitted_test_score,
        "metrics_previous":baseline_metric,"metrics_fitted":fitted_metric,
        "loss_breakdown_previous":baseline_loss,
        "loss_breakdown_fitted":fitted_loss,
        "test_metrics_previous":baseline_test_metric,
        "test_metrics_fitted":fitted_test_metric,
        "alternating_trace":trace,"joint_polish_score":polished_score,
        "joint_polish_accepted":bool(polish_accepted),
        "p95_gate_passed":bool(p95_gate),
        "deployment_gate_passed":bool(gate),"config_updated":bool(gate and UPDATE_CONFIG)}
    (OUTPUT_DIR/"joint_regression.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUTPUT_DIR/"params.json").write_text(json.dumps({
        "expanded_fitted":{**dict(zip(TIRE_NAMES,accepted[:8].tolist())),
            "I_z":float(accepted[8]),"kinematic_steer_scale":1.0,
            "kinematic_steer_bias":0.0,
            "steer_servo_time_constant":float(accepted[9]),
            "kinematic_position_speed_scale":1.0},**report},indent=2)+"\n")
    if gate and UPDATE_CONFIG:update_yaml(accepted)
    previous_tire,previous_config=unpack(current,base_config)
    steering.plot_open_loop_evaluation(data,fitted_tire,base_config,
        np.asarray((1.0,0.0,current[9])),
        np.asarray((1.0,0.0,accepted[9])),validation,test)
    classic.plot_open_loop_evaluation(data,previous_tire,fitted_tire,
        previous_config,fitted_config,validation,test)
    print("\nJoint Step 3/4 parameter changes:")
    for name,old,new in zip(PARAMETER_NAMES,current,accepted):
        print(f"  {name}: {old:.9g} -> {new:.9g} (delta {new-old:+.9g})")
    classic.print_pose_metric_change("Joint Step 3/4 validation",
                                     baseline_metric,fitted_metric)
    print("\nJoint Step 3/4 validation loss components:")
    ordered=("recursive_vx","recursive_vy","recursive_yaw_rate",
             "one_step_vx","one_step_vy","one_step_yaw_rate",
             "full_trajectory_position_xy","full_trajectory_yaw",
             "position_xy","trajectory_yaw","one_step_position_xy",
             "one_step_yaw","endpoint_tail","regularization_or_residual","total")
    for name in ordered:
        if name in baseline_loss or name in fitted_loss:
            old=float(baseline_loss.get(name,0.));new=float(fitted_loss.get(name,0.))
            reduction=100.*(old-new)/max(abs(old),1e-12)
            print(f"  {name}: {old:.8g} -> {new:.8g} "
                  f"(reduction {reduction:+.2f}%)")
    print(f"Weighted validation score: {baseline_score:.8g} -> {accepted_score:.8g}")
    print(f"Weighted test score: {baseline_test_score:.8g} -> {fitted_test_score:.8g}")
    print(f"Deployment gate: {'PASS' if gate else 'FAIL'}; "
          f"params.yaml {'updated' if gate and UPDATE_CONFIG else 'not changed'}")


if __name__=="__main__":main()
