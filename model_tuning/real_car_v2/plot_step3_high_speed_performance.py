#!/usr/bin/env python3
"""Visualize current Step-3 classic-model accuracy versus rollout speed."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import classic_model_regression as regression
import search_step3_loss_weights as weight_search
import step_3_identify_classic_model as settings


BIN_WIDTH_MPS=.5
MIN_WINDOWS_PER_BIN=20
HIGH_SPEED_QUANTILE=.75
OUT=settings.OUTPUT_DIR/"high_speed_performance"


def errors(parameters,data,starts,config):
    prediction,truth=regression.rollout_numpy(parameters,data,starts,config)
    predicted_pose=regression.relative_pose(
        prediction,float(config.get("kinematic_position_speed_scale",1.)))
    truth_pose=regression.mcl_relative_pose(data,starts)
    position=np.linalg.norm(predicted_pose[:,-1,:2]-truth_pose[:,-1,:2],axis=1)
    yaw=np.abs((predicted_pose[:,-1,2]-truth_pose[:,-1,2]+np.pi)%(2*np.pi)-np.pi)
    yaw_rate=np.abs(prediction[:,-1,2]-truth[:,-1,2])
    # Mean GT vx over the complete rollout is less sensitive than one initial
    # sample and answers whether the model remains accurate while travelling fast.
    speed=np.mean(np.abs(truth[:,:,0]),axis=1)
    return speed,position,yaw,yaw_rate,predicted_pose,truth_pose


def summary(values):
    return {"mean":float(np.mean(values)),"p95":float(np.quantile(values,.95)),
            "maximum":float(np.max(values))}


def main():
    weight_search.configure()
    regression.MAX_PER_BAG=0;regression.WINDOW_START_STRIDE=1
    config=yaml.safe_load((settings.ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    data,contract=regression.load_regression_data(settings.DATA_PATH,config)
    validation=regression.starts(data,1);test=regression.starts(data,2)
    starts=np.concatenate((validation,test))
    parameters=np.asarray([config[f"dynamic_mlp_{name}"] for name in regression.NAMES],float)
    speed,position,yaw,yaw_rate,predicted_pose,truth_pose=errors(
        parameters,data,starts,config)
    high_threshold=float(np.quantile(speed,HIGH_SPEED_QUANTILE))
    high=speed>=high_threshold
    edges=np.arange(np.floor(speed.min()/BIN_WIDTH_MPS)*BIN_WIDTH_MPS,
                    np.ceil(speed.max()/BIN_WIDTH_MPS)*BIN_WIDTH_MPS+BIN_WIDTH_MPS,
                    BIN_WIDTH_MPS)
    rows=[]
    for low,high_edge in zip(edges[:-1],edges[1:]):
        mask=(speed>=low)&(speed<high_edge)
        if int(mask.sum())<MIN_WINDOWS_PER_BIN:continue
        rows.append({"low_mps":float(low),"high_mps":float(high_edge),"windows":int(mask.sum()),
            "position":summary(position[mask]),"yaw":summary(yaw[mask]),
            "yaw_rate":summary(yaw_rate[mask])})
    report={"data_contract":contract,"evaluation_split":"validation + test bags",
        "windows":int(len(starts)),"rollout_horizon_s":float(regression.HORIZON*.04),
        "speed_definition":"mean absolute GT vx over rollout",
        "speed_range_mps":[float(speed.min()),float(speed.max())],
        "high_speed_threshold_mps":high_threshold,"high_speed_windows":int(high.sum()),
        "all":{"position":summary(position),"yaw":summary(yaw),"yaw_rate":summary(yaw_rate)},
        "high_speed":{"position":summary(position[high]),"yaw":summary(yaw[high]),
                      "yaw_rate":summary(yaw_rate[high])},"bins":rows}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")

    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for axis,key,label in zip(axes,("position","yaw","yaw_rate"),
                             ("position error [m]","yaw error [rad]","yaw-rate error [rad/s]")):
        centers=[.5*(row["low_mps"]+row["high_mps"]) for row in rows]
        mean=[row[key]["mean"] for row in rows];p95=[row[key]["p95"] for row in rows]
        axis.plot(centers,mean,"o-",label="mean");axis.plot(centers,p95,"s-",label="P95")
        axis.axvline(high_threshold,color="tab:red",ls="--",label="high-speed Q75")
        axis.set(xlabel="mean GT vx over rollout [m/s]",ylabel=label,title=key.replace("_"," "))
        axis.grid(alpha=.25);axis.legend()
    fig.suptitle("Current YAML classic model: held-out error versus speed")
    fig.tight_layout();fig.savefig(OUT/"error_vs_speed.png",dpi=180);plt.close(fig)

    high_indices=np.flatnonzero(high);high_error=position[high]
    selected_local=(int(np.argmin(high_error)),
                    int(np.argmin(np.abs(high_error-np.quantile(high_error,.95)))),
                    int(np.argmax(high_error)))
    selected=high_indices[np.asarray(selected_local)]
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for axis,index,title in zip(axes,selected,("high-speed best","high-speed P95","high-speed worst")):
        axis.plot(truth_pose[index,:,0],truth_pose[index,:,1],"k-",lw=3,label="GT")
        axis.plot(predicted_pose[index,:,0],predicted_pose[index,:,1],color="tab:orange",lw=2,label="classic")
        axis.scatter([0.],[0.],c="black",s=25);axis.set_aspect("equal",adjustable="box")
        axis.set(title=f"{title}\nmean vx={speed[index]:.2f} m/s, end error={position[index]:.2f} m",
                 xlabel="relative x [m]",ylabel="relative y [m]")
        axis.grid(alpha=.25)
    axes[0].legend();fig.suptitle("Held-out high-speed open-loop trajectories")
    fig.tight_layout();fig.savefig(OUT/"high_speed_best_p95_worst.png",dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2));print(f"outputs: {OUT}")


if __name__=="__main__":main()
