#!/usr/bin/env python3
"""Compare Step-1 and tuned MPPI-model EKFs plus classic rollouts."""
from pathlib import Path
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from classic_model_kalman_filter import filter_classic_segment,rollout_classic_segment,wrap
from contract import ClassicModelParameters
from step_1_extract_data import causal_ema,causal_mcl_body_vy

DATA=Path(os.environ.get("KF_EVALUATION_DATA",
    ROOT/"model_tuning/data/ifac0810_0819_autonomous_physics_clean"))
OUT=ROOT/"model_tuning/results/classic_model_kf_evaluation"
CFG=ROOT/"config/params.yaml"
STATE_NAMES=("x","y","yaw","vx","vy","yaw_rate")
MODES=("ekf_closed_loop","classic_closed_loop_one_step","classic_open_loop")
use_plot=True
OPEN_LOOP_HORIZON_S=1.2


def metric(error):
    value=np.asarray(error,float);value=value[np.isfinite(value)]
    if not len(value):return {"rmse":None,"mae":None,"p95_abs":None}
    absolute=np.abs(value)
    return {"rmse":float(np.sqrt(np.mean(value**2))),"mae":float(absolute.mean()),
            "p95_abs":float(np.quantile(absolute,.95))}


def state_error(value,raw):
    error=np.asarray(value)-raw;error[:,2]=wrap(error[:,2]);return error


def run_model(part,names,dt,signs,cfg):
    pose_vy=causal_mcl_body_vy(part[:,names["t"]],part[:,names["x"]],
        part[:,names["y"]],part[:,names["yaw"]],float(cfg.get("kf_pose_vy_window_s",.12)))
    alpha=float(cfg.get("imu_ema_alpha",.25))
    gyro=causal_ema(signs[0]*part[:,names["imu_wz"]]-float(cfg.get("imu_wz_bias",0.)),alpha)
    ax=causal_ema(signs[1]*part[:,names["imu_ax"]]-float(cfg.get("imu_ax_bias",0.)),alpha)
    ay=causal_ema(signs[2]*part[:,names["imu_ay"]]-float(cfg.get("imu_ay_bias",0.)),alpha)
    raw=np.c_[part[:,names["x"]],part[:,names["y"]],part[:,names["yaw"]],
              part[:,names["vx"]],pose_vy,gyro]
    filtered=filter_classic_segment(raw[:,0],raw[:,1],raw[:,2],raw[:,3],raw[:,4],
        gyro,ax,ay,part[:,names["steer"]],part[:,names["speed_cmd"]],dt,cfg)
    # Evaluate fixed 1.2 s free-rollout horizons so long bags do not receive a
    # larger error merely because they contain more elapsed open-loop time.
    opened=np.empty_like(raw);horizon=max(2,int(round(OPEN_LOOP_HORIZON_S/dt)))
    for start in range(0,len(raw),horizon):
        stop=min(len(raw),start+horizon)
        opened[start:stop]=rollout_classic_segment(filtered["state"][start],
            raw[start:stop,3],part[start:stop,names["steer"]],
            part[start:stop,names["speed_cmd"]],dt,cfg)["state"]
    return {"raw":raw,"ekf_closed_loop":filtered["state"],
            "classic_closed_loop_one_step":filtered["predicted_state"],
            "classic_open_loop":opened}


def evaluate_segment(part,names,dt,signs,before_cfg,after_cfg):
    before=run_model(part,names,dt,signs,before_cfg)
    after=run_model(part,names,dt,signs,after_cfg)
    warmup=min(len(part),max(5,int(round(.2/dt))))
    result={"time":part[warmup:,names["t"]],"raw":after["raw"][warmup:]}
    for label,model in (("before",before),("after",after)):
        for mode in MODES:
            result[f"{label}_{mode}"]=model[mode][warmup:]
            result[f"{label}_{mode}_error"]=state_error(model[mode][warmup:],result["raw"])
    return result


def aggregate(errors):
    joined=np.concatenate(errors)
    return {name:metric(joined[:,index]) for index,name in enumerate(STATE_NAMES)}


def plot_diagnostics(selected,summary):
    import matplotlib.pyplot as plt
    labels={"ekf_closed_loop":"EKF corrected","classic_closed_loop_one_step":"classic 1-step",
            "classic_open_loop":"classic free rollout"}
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    fig.canvas.manager.set_window_title("Initial vs tuned observer summary")
    for index,(axis,name) in enumerate(zip(axes.flat,STATE_NAMES)):
        values=[summary[x]["after"][name]["rmse"] for x in MODES]
        old=[summary[x]["before"][name]["rmse"] for x in MODES]
        xx=np.arange(len(MODES));axis.bar(xx-.18,old,.36,label="Step-1 initial")
        axis.bar(xx+.18,values,.36,label="current tuned")
        axis.set_xticks(xx,[labels[x] for x in MODES],rotation=15,ha="right")
        axis.set_title(f"{name} RMSE");axis.grid(axis="y",alpha=.25)
    axes.flat[0].legend()

    fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    fig.canvas.manager.set_window_title("GT / initial / tuned EKF trajectories")
    for axis,(rank,segment) in zip(axes,selected):
        data=segment["_plot"]
        axis.plot(data["raw"][:,0],data["raw"][:,1],"k",lw=2,label="raw MCL trajectory")
        axis.plot(data["before_ekf_closed_loop"][:,0],data["before_ekf_closed_loop"][:,1],
                  "--",color="C1",label="Step-1 initial EKF")
        axis.plot(data["after_ekf_closed_loop"][:,0],data["after_ekf_closed_loop"][:,1],
                  color="C3",label="tuned EKF")
        axis.set(title=f"{rank}: {segment['file']} bag={segment['bag_id']}",xlabel="x [m]",ylabel="y [m]")
        axis.axis("equal");axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)

    for rank,segment in selected:
        data=segment["_plot"]
        fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
        fig.canvas.manager.set_window_title(f"Classic open/closed loop: {rank}")
        for axis,prefix,title in ((axes[0],"before","Step-1 initial parameters"),
                                  (axes[1],"after","current tuned parameters")):
            axis.plot(data["raw"][:,0],data["raw"][:,1],"k",lw=2,label="raw MCL")
            axis.plot(data[f"{prefix}_classic_closed_loop_one_step"][:,0],
                      data[f"{prefix}_classic_closed_loop_one_step"][:,1],
                      color="C0",label="closed-loop one-step")
            axis.plot(data[f"{prefix}_classic_open_loop"][:,0],
                      data[f"{prefix}_classic_open_loop"][:,1],
                      color="C3",label="open-loop free rollout")
            axis.set(title=title,xlabel="x [m]",ylabel="y [m]");axis.axis("equal");axis.grid(alpha=.25)
        axes[0].legend();fig.suptitle(f"{rank}: {segment['file']} bag={segment['bag_id']}")
    plt.show(block=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--plot",dest="plot",action="store_true")
    parser.add_argument("--no-plot",dest="plot",action="store_false")
    parser.set_defaults(plot=use_plot);args=parser.parse_args()
    tuned=yaml.safe_load(CFG.read_text())["/**"]["ros__parameters"]
    errors={mode:{"before":[],"after":[]} for mode in MODES};segments=[];snapshots=set()
    for path in sorted(args.data.glob("*.npz")):
        with np.load(path) as archive:
            if "kf_config_snapshot_json" not in archive.files:
                raise RuntimeError(f"{path}: no Step-1 parameter snapshot; rerun Step 1 once before tuning")
            snapshot=str(archive["kf_config_snapshot_json"]);snapshots.add(snapshot)
            before={**tuned,**json.loads(snapshot)}
            samples=np.asarray(archive["samples"],float);columns=list(map(str,archive["columns"]))
            names={name:i for i,name in enumerate(columns)}
            required=("t","x","y","yaw","vx","steer","speed_cmd","bag_id","imu_wz","imu_ax","imu_ay")
            if any(name not in names for name in required):continue
            dt=float(archive["dt"]);signs=np.asarray(archive.get("imu_axis_signs",np.ones(3)),float)
            ids=samples[:,names["bag_id"]].astype(int)
            for bag_id in np.unique(ids):
                part=samples[ids==bag_id]
                if len(part)<20:continue
                diagnostic=evaluate_segment(part,names,dt,signs,before,tuned)
                for mode in MODES:
                    errors[mode]["before"].append(diagnostic[f"before_{mode}_error"])
                    errors[mode]["after"].append(diagnostic[f"after_{mode}_error"])
                vy=metric(diagnostic["after_ekf_closed_loop_error"][:,4])["rmse"]
                segments.append({"file":path.name,"bag_id":int(bag_id),"samples":len(diagnostic["raw"]),
                                 "after_ekf_vy_rmse":vy,"_plot":diagnostic})
    if not segments:raise RuntimeError(f"no usable segments in {args.data}")
    summary={mode:{label:aggregate(errors[mode][label]) for label in ("before","after")}
             for mode in MODES}
    ranked=sorted(segments,key=lambda item:item["after_ekf_vy_rmse"])
    selected=(("best",ranked[0]),("median",ranked[len(ranked)//2]),("worst",ranked[-1]))
    clean=lambda item:{key:value for key,value in item.items() if key!="_plot"}
    initial_snapshots=[json.loads(value) for value in sorted(snapshots)]
    report={"comparison":"Step-1 parameter snapshot vs current tuned params",
            "initial_parameter_snapshots":initial_snapshots,
            "current_tuned_parameters":ClassicModelParameters.from_mapping(tuned).runtime_updates(),
            "semantics":{"ekf_closed_loop":"prediction followed by measurement correction",
                "classic_closed_loop_one_step":"one-step prediction from previous corrected EKF state",
                "classic_open_loop":f"{OPEN_LOOP_HORIZON_S:.1f} s free rollout; no measurement correction"},
            "independent_gt_warning":"MCL-derived vy is diagnostic, not independent GT",
            "snapshot_variants":len(snapshots),"segments":len(segments),"metrics":summary,
            **{rank:clean(segment) for rank,segment in selected}}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if args.plot:plot_diagnostics(selected,summary)


if __name__=="__main__":main()
