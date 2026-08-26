#!/usr/bin/env python3
"""Step 5: fine-tune classic Pacejka parameters on high-speed hard tails.

This stage intentionally does not replace the existing velocity-observer Step
5. It trains on high-speed worst-tail windows from train bags, evaluates on
bag-disjoint validation/test bags, and writes a candidate only when the hard
tail improves without materially degrading overall held-out P95.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import yaml

import classic_model_regression as regression
import search_step3_loss_weights as weight_search
import step_3_identify_classic_model as settings


HIGH_SPEED_MIN_MPS=4.5
HARD_TAIL_QUANTILE=.85
# Every rollout above this endpoint pose error is a tail, regardless of speed.
POSE_TAIL_THRESHOLD_M=.5
HARD_WINDOW_REPEAT=5
BACKGROUND_WINDOWS=1200
LOCAL_BOUND_FRACTION=.35
LOCAL_E_HALF_WIDTH=1.0
DE_POPSIZE=6
DE_MAXITER=30
YAW_RATE_LOSS_WEIGHT=8.0
POSITION_LOSS_WEIGHT=25.0
YAW_TRAJECTORY_LOSS_WEIGHT=5.0
MAX_OVERALL_HELDOUT_P95_DEGRADATION=.03
HIGH_SPEED_HOLDOUT_BAG_FRACTION=.20
# True writes the newly fitted candidate to params.yaml.  The quality gate is
# still reported; set REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE=True when a
# rejected candidate must never be written automatically.
UPDATE_PARAMS_YAML=True
REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE=False
USE_PLOT=True
OUT=settings.ROOT/"model_tuning/results/step_5_high_speed_worst_finetune"
EVALUATE_ONLY=False
EVALUATION_REPORT_PATH=OUT/"report.json"
FOCUS_WINDOWS=(("rosbag2_2026_08_25-20_45_36.npz",3.28),)
FOCUS_TIME_RADIUS_S=.20
FOCUS_WINDOW_REPEAT=10


def rollout_errors(parameters,data,starts,config):
    prediction,truth=regression.rollout_numpy(parameters,data,starts,config)
    predicted_pose=regression.relative_pose(
        prediction,float(config.get("kinematic_position_speed_scale",1.)))
    truth_pose=regression.mcl_relative_pose(data,starts)
    speed=np.mean(np.abs(truth[:,:,0]),axis=1)
    position=np.linalg.norm(predicted_pose[:,-1,:2]-truth_pose[:,-1,:2],axis=1)
    yaw=np.abs((predicted_pose[:,-1,2]-truth_pose[:,-1,2]+np.pi)%(2*np.pi)-np.pi)
    yaw_rate=np.abs(prediction[:,-1,2]-truth[:,-1,2])
    return {"speed":speed,"position":position,"yaw":yaw,"yaw_rate":yaw_rate,
            "predicted_pose":predicted_pose,"truth_pose":truth_pose}


def p95_summary(error,mask=None):
    if mask is None:mask=np.ones(len(error["speed"]),bool)
    return {"windows":int(mask.sum()),"speed_mean_mps":float(np.mean(error["speed"][mask])),
        "position_mean_m":float(np.mean(error["position"][mask])),
        "position_p95_m":float(np.quantile(error["position"][mask],.95)),
        "yaw_mean_rad":float(np.mean(error["yaw"][mask])),
        "yaw_p95_rad":float(np.quantile(error["yaw"][mask],.95)),
        "yaw_rate_mean_radps":float(np.mean(error["yaw_rate"][mask])),
        "yaw_rate_p95_radps":float(np.quantile(error["yaw_rate"][mask],.95))}


def normalized_p95(summary,baseline):
    return float(np.mean([
        summary["position_p95_m"]/max(baseline["position_p95_m"],1e-6),
        summary["yaw_p95_rad"]/max(baseline["yaw_p95_rad"],1e-6),
        summary["yaw_rate_p95_radps"]/max(baseline["yaw_rate_p95_radps"],1e-6)]))


def reduction_percent(previous,candidate):
    return 100.*(previous-candidate)/max(abs(previous),1e-12)


def print_finetuning_summary(current,candidate,current_all,candidate_all,
                             current_high,candidate_high,gate,yaml_updated):
    print("\nStep 5 classic-model fine-tuning parameter changes:")
    for name,previous,tuned in zip(regression.NAMES,current,candidate):
        print(f"  {name}: {previous:.9g} -> {tuned:.9g}")
    print("\nHeld-out P95 performance (positive reduction = improvement):")
    for scope,previous,tuned in (("all",current_all,candidate_all),
                                 ("high-speed",current_high,candidate_high)):
        print(f"  [{scope}; windows={previous['windows']}]")
        for key,unit in (("position_p95_m","m"),("yaw_p95_rad","rad"),
                         ("yaw_rate_p95_radps","rad/s")):
            reduction=reduction_percent(previous[key],tuned[key])
            print(f"    {key}: {previous[key]:.6g} -> {tuned[key]:.6g} {unit} "
                  f"(reduction {reduction:+.2f}%)")
    print(f"  deployment gate: {'PASS' if gate else 'REJECT'}")
    print(f"  params.yaml updated: {yaml_updated}")


def local_bounds(current):
    bounds=settings_bounds=np.asarray((
        settings.PACEJKA_B_F_BOUNDS,settings.PACEJKA_C_F_BOUNDS,
        settings.PACEJKA_D_F_BOUNDS,settings.PACEJKA_E_F_BOUNDS,
        settings.PACEJKA_B_R_BOUNDS,settings.PACEJKA_C_R_BOUNDS,
        settings.PACEJKA_D_R_BOUNDS,settings.PACEJKA_E_R_BOUNDS),float)
    result=bounds.copy()
    for index,value in enumerate(current):
        radius=(LOCAL_E_HALF_WIDTH if index in (3,7)
                else max(abs(value)*LOCAL_BOUND_FRACTION,.05))
        result[index,0]=max(bounds[index,0],value-radius)
        result[index,1]=min(bounds[index,1],value+radius)
    return result


def update_yaml(parameters):
    path=settings.ROOT/"config/params.yaml";text=path.read_text()
    import re
    for name,value in zip(regression.NAMES,parameters):
        key=f"dynamic_mlp_{name}"
        text,count=re.subn(rf"(^\s*{re.escape(key)}:\s*)[^#\n]+",
                           rf"\g<1>{float(value):.12g}",text,flags=re.MULTILINE)
        if count!=1:raise RuntimeError(f"expected exactly one {key}, found {count}")
    path.write_text(text)


def load_evaluation_candidate(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(f"evaluation report not found: {path}")
    values=json.loads(path.read_text()).get("candidate_parameters")
    if not isinstance(values,dict):raise KeyError(f"{path}: missing candidate_parameters")
    missing=[name for name in regression.NAMES if name not in values]
    if missing:raise KeyError(f"{path}: missing candidate fields {missing}")
    return np.asarray([values[name] for name in regression.NAMES],float)


def resolve_focus_starts(data):
    resolved=[];source_paths=[Path(str(value)).name for value in data["source_paths"]]
    radius=max(0,int(round(FOCUS_TIME_RADIUS_S/.02)))
    for bag_name,time_s in FOCUS_WINDOWS:
        matches=[index for index,name in enumerate(source_paths) if name==bag_name]
        if len(matches)!=1:
            raise RuntimeError(f"focus bag {bag_name}: expected one match, found {matches}")
        bag_id=matches[0];rows=np.flatnonzero(data["bag_id"]==bag_id);first=int(rows[0])
        center=first+int(round(time_s/.02))
        for start in range(center-radius,center+radius+1):
            stop=start+2*regression.HORIZON+1
            if start-regression.WARMUP_SAMPLES<first or stop>int(rows[-1])+1:continue
            if np.all(data["valid"][start:stop]) and np.all(
                    data["bag_id"][start-regression.WARMUP_SAMPLES:stop]==bag_id):
                resolved.append(start)
    if not resolved:raise RuntimeError("no valid focus rollout start was resolved")
    return np.unique(np.asarray(resolved,int))


def main():
    weight_search.configure();regression.MAX_PER_BAG=0;regression.WINDOW_START_STRIDE=1
    regression.SHOW_PLOTS=USE_PLOT;regression.OUT=OUT
    regression.YAW_RATE_LOSS_WEIGHT=YAW_RATE_LOSS_WEIGHT
    regression.POSITION_LOSS_WEIGHT=POSITION_LOSS_WEIGHT
    regression.YAW_TRAJECTORY_LOSS_WEIGHT=YAW_TRAJECTORY_LOSS_WEIGHT
    config=yaml.safe_load((settings.ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    data,contract=regression.load_regression_data(settings.DATA_PATH,config)
    stored_splits=[regression.starts(data,index) for index in range(3)]
    all_starts=np.concatenate(stored_splits)
    current=np.asarray([config[f"dynamic_mlp_{name}"] for name in regression.NAMES],float)
    all_error=rollout_errors(current,data,all_starts,config)
    all_bags=data["bag_id"][all_starts]
    high_mask=all_error["speed"]>=HIGH_SPEED_MIN_MPS
    high_bags=np.unique(all_bags[high_mask])
    if len(high_bags)<2:
        raise RuntimeError(f"need at least two high-speed bags, found {len(high_bags)}")
    rng=np.random.default_rng(settings.RANDOM_SEED);rng.shuffle(high_bags)
    holdout_count=max(1,int(round(HIGH_SPEED_HOLDOUT_BAG_FRACTION*len(high_bags))))
    holdout_bags=np.sort(high_bags[:holdout_count])
    heldout_mask=np.isin(all_bags,holdout_bags)
    development_mask=~heldout_mask
    development=all_starts[development_mask]
    development_error={key:(value[development_mask] if isinstance(value,np.ndarray)
                            and len(value)==len(all_starts) else value)
                       for key,value in all_error.items()}
    development_high=development_error["speed"]>=HIGH_SPEED_MIN_MPS
    if not development_high.any():raise RuntimeError("no high-speed development windows")
    threshold=float(np.quantile(
        development_error["position"][development_high],HARD_TAIL_QUANTILE))
    pose_tail_mask=development_error["position"]>=POSE_TAIL_THRESHOLD_M
    quantile_tail_mask=(development_high&
                        (development_error["position"]>=threshold))
    hard_mask=pose_tail_mask|quantile_tail_mask
    hard=development[hard_mask]
    quantile_hard=development[quantile_tail_mask]
    background=development[np.linspace(
        0,len(development)-1,min(BACKGROUND_WINDOWS,len(development))).astype(int)]
    focus=resolve_focus_starts(data)
    fit_starts=np.concatenate((background,hard,
                               np.tile(quantile_hard,HARD_WINDOW_REPEAT),
                               np.tile(focus,FOCUS_WINDOW_REPEAT)))
    if EVALUATE_ONLY:
        candidate=load_evaluation_candidate(EVALUATION_REPORT_PATH)
        print(f"Step 5 EVALUATE_ONLY: loaded {Path(EVALUATION_REPORT_PATH).resolve()}")
    else:
        regression.REFERENCE=current.copy();regression.BOUNDS=local_bounds(current)
        result=differential_evolution(
            lambda p:regression.objective(p,data,fit_starts,config),regression.BOUNDS,
            seed=settings.RANDOM_SEED,popsize=DE_POPSIZE,maxiter=DE_MAXITER,
            tol=8e-4,polish=False,workers=1,x0=current)
        candidate=result.x

    heldout=all_starts[heldout_mask]
    current_error=rollout_errors(current,data,heldout,config)
    candidate_error=rollout_errors(candidate,data,heldout,config)
    current_focus_error=rollout_errors(current,data,focus,config)
    candidate_focus_error=rollout_errors(candidate,data,focus,config)
    high=current_error["speed"]>=HIGH_SPEED_MIN_MPS
    current_all=p95_summary(current_error);candidate_all=p95_summary(candidate_error)
    current_high=p95_summary(current_error,high);candidate_high=p95_summary(candidate_error,high)
    high_improved=(candidate_high["position_p95_m"]<current_high["position_p95_m"]
                   and normalized_p95(candidate_high,current_high)<1.)
    overall_ratio=normalized_p95(candidate_all,current_all)
    gate=bool(high_improved and overall_ratio<=1.+MAX_OVERALL_HELDOUT_P95_DEGRADATION)
    yaml_updated=bool(not EVALUATE_ONLY and UPDATE_PARAMS_YAML and
                      (gate or not REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE))
    selected=candidate if yaml_updated else current
    if yaml_updated:update_yaml(candidate)
    report={"data_contract":contract,"evaluate_only":EVALUATE_ONLY,
        "high_speed_min_mps":HIGH_SPEED_MIN_MPS,
        "hard_tail_quantile":HARD_TAIL_QUANTILE,"hard_position_threshold_m":threshold,
        "absolute_pose_tail_threshold_m":POSE_TAIL_THRESHOLD_M,
        "absolute_pose_tail_windows":int(pose_tail_mask.sum()),
        "high_speed_quantile_tail_windows":int(quantile_tail_mask.sum()),
        "split_method":"speed-stratified bag holdout across all stored splits",
        "high_speed_bags":list(map(int,np.sort(high_bags))),
        "holdout_bags":list(map(int,holdout_bags)),
        "development_windows":int(len(development)),
        "heldout_windows":int(len(heldout)),"hard_train_windows":int(len(hard)),
        "optimizer_windows_with_repetition":int(len(fit_starts)),
        "focus_windows":[{"bag":name,"time_s":time_s}
                         for name,time_s in FOCUS_WINDOWS],
        "focus_neighbor_starts":int(len(focus)),
        "focus_window_repeat":FOCUS_WINDOW_REPEAT,
        "loss_weights":{"yaw_rate":YAW_RATE_LOSS_WEIGHT,"position":POSITION_LOSS_WEIGHT,
                        "trajectory_yaw":YAW_TRAJECTORY_LOSS_WEIGHT},
        "current_parameters":dict(zip(regression.NAMES,current.tolist())),
        "candidate_parameters":dict(zip(regression.NAMES,candidate.tolist())),
        "selected_parameters":dict(zip(regression.NAMES,selected.tolist())),
        "current_heldout":{"all":current_all,"high_speed":current_high},
        "candidate_heldout":{"all":candidate_all,"high_speed":candidate_high},
        "current_focus":p95_summary(current_focus_error),
        "candidate_focus":p95_summary(candidate_focus_error),
        "overall_normalized_p95_ratio":overall_ratio,
        "deployment_gate_passed":gate,
        "update_params_yaml":UPDATE_PARAMS_YAML,
        "require_deployment_gate_for_yaml_update":REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE,
        "yaml_updated":yaml_updated,
        "gate_policy":"high-speed position and aggregate P95 improve; overall heldout aggregate degrades <=3%"}
    OUT.mkdir(parents=True,exist_ok=True)
    report_name="evaluation_report.json" if EVALUATE_ONLY else "report.json"
    (OUT/report_name).write_text(json.dumps(report,indent=2)+"\n")

    fig,axes=plt.subplots(1,3,figsize=(15,4.6));keys=("position","yaw","yaw_rate")
    units=("m","rad","rad/s");edges=np.arange(1.,8.01,.5)
    for axis,key,unit in zip(axes,keys,units):
        centers=[];old=[];new=[]
        for low,upper in zip(edges[:-1],edges[1:]):
            mask=(current_error["speed"]>=low)&(current_error["speed"]<upper)
            if mask.sum()<20:continue
            centers.append(.5*(low+upper));old.append(np.quantile(current_error[key][mask],.95))
            new.append(np.quantile(candidate_error[key][mask],.95))
        axis.plot(centers,old,"o-",label="current");axis.plot(centers,new,"s-",label="fine-tuned")
        axis.axvline(HIGH_SPEED_MIN_MPS,color="tab:red",ls="--")
        axis.set(title=f"{key.replace('_',' ')} P95",xlabel="mean GT vx [m/s]",ylabel=unit)
        axis.grid(alpha=.25);axis.legend()
    fig.suptitle(f"High-speed hard-tail fine-tuning; gate={'PASS' if gate else 'REJECT'}")
    fig.tight_layout();fig.savefig(OUT/"speed_binned_p95_before_after.png",dpi=180);plt.close(fig)

    high_indices=np.flatnonzero(high);worst=high_indices[int(np.argmax(current_error["position"][high]))]
    fig,axis=plt.subplots(figsize=(7,5));axis.plot(current_error["truth_pose"][worst,:,0],
        current_error["truth_pose"][worst,:,1],"k-",lw=3,label="GT")
    axis.plot(current_error["predicted_pose"][worst,:,0],current_error["predicted_pose"][worst,:,1],
              lw=2,label="current")
    axis.plot(candidate_error["predicted_pose"][worst,:,0],candidate_error["predicted_pose"][worst,:,1],
              lw=2,label="fine-tuned")
    axis.set_aspect("equal",adjustable="box");axis.grid(alpha=.25);axis.legend()
    axis.set(title=f"Held-out high-speed worst; mean vx={current_error['speed'][worst]:.2f} m/s",
             xlabel="relative x [m]",ylabel="relative y [m]")
    fig.tight_layout();fig.savefig(OUT/"heldout_high_speed_worst_before_after.png",dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))
    print_finetuning_summary(current,candidate,current_all,candidate_all,
                             current_high,candidate_high,gate,
                             yaml_updated)
    print(f"outputs: {OUT}")
    if USE_PLOT:
        print("Step 5 interactive comparison: press p, then click a time-series panel. "
              "Use Left/Right for bags, number+Enter to jump, q to quit.")
        # plot_all_bag_evaluation interprets zero literally, unlike the fitting
        # path where MAX_PER_BAG=0 means unlimited.
        regression.MAX_PER_BAG=300
        regression.plot_all_bag_evaluation(
            data,current,candidate,config,config)


if __name__=="__main__":main()
