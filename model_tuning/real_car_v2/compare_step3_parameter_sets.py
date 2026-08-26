#!/usr/bin/env python3
"""Compare requested classic parameter sets on the current Step-3 dataset."""
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


PARAMETER_1=np.asarray((
    .3366075474431282,.6356924636110722,1.0411165641262696,-14231.435412582694,
    .8670431599215196,1.078260951127556,1.3367398542655606,-5504.52844896071))
PARAMETER_2=np.asarray((
    1.2016504839806836,1.9280513533201524,.9774102510587146,.7797639293027882,
    3.5057642812107153,1.264811851706697,1.1550628424637372,.9190111472321836))
SEARCH_REPORT=settings.OUTPUT_DIR/"loss_weight_global_search.json"
OUTPUT_DIR=settings.OUTPUT_DIR/"parameter_set_comparison"


def predicted_and_truth(parameters,data,starts,config):
    prediction,truth=regression.rollout_numpy(parameters,data,starts,config)
    predicted=regression.relative_pose(
        prediction,float(config.get("kinematic_position_speed_scale",1.)))
    truth_pose=regression.mcl_relative_pose(data,starts)
    return predicted,truth_pose


def main():
    weight_search.configure()
    regression.MAX_PER_BAG=settings.MAX_WINDOWS_PER_BAG
    config=yaml.safe_load((settings.ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    config["load_transfer_h_cg_m"]=float(settings.LOAD_TRANSFER_H_CG_M)
    data,contract=regression.load_regression_data(settings.DATA_PATH,config)
    report=json.loads(SEARCH_REPORT.read_text())
    parameter_3=np.asarray([report["fitted_parameters"][name]
                            for name in regression.NAMES])
    models={"1: extreme negative E":PARAMETER_1,
            "2: current YAML":PARAMETER_2,
            "3: global-search final":parameter_3}
    split_starts={name:regression.starts(data,index)
                  for index,name in enumerate(("train","validation","test"))}
    metrics={label:{split:regression.metrics(parameters,data,starts,config)
                    for split,starts in split_starts.items()}
             for label,parameters in models.items()}
    output={"data_contract":contract,
            "parameter_2_equals_parameter_3":bool(np.allclose(PARAMETER_2,parameter_3,
                                                               rtol=0.,atol=1e-9)),
            "parameters":{label:dict(zip(regression.NAMES,value.tolist()))
                          for label,value in models.items()},
            "metrics":metrics}
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUTPUT_DIR/"comparison.json").write_text(json.dumps(output,indent=2)+"\n")

    fields=(("trajectory_p95_m","position P95 [m]"),
            ("trajectory_yaw_p95_rad","yaw P95 [rad]"))
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    labels=list(models);x=np.arange(len(labels));width=.25
    for axis,(field,title) in zip(axes[:2],fields):
        for offset,split in enumerate(("train","validation","test")):
            axis.bar(x+(offset-1)*width,
                     [metrics[label][split][field] for label in labels],width,label=split)
        axis.set_xticks(x,labels,rotation=15,ha="right");axis.set_title(title)
        axis.grid(axis="y",alpha=.25);axis.legend()
    for offset,split in enumerate(("train","validation","test")):
        axes[2].bar(x+(offset-1)*width,
            [metrics[label][split]["state_p95"][2] for label in labels],width,label=split)
    axes[2].set_xticks(x,labels,rotation=15,ha="right")
    axes[2].set_title("yaw-rate P95 [rad/s]");axes[2].grid(axis="y",alpha=.25)
    axes[2].legend();fig.suptitle("Current Step-3 dataset: parameter-set P95 comparison")
    fig.tight_layout();fig.savefig(OUTPUT_DIR/"p95_metrics.png",dpi=180);plt.close(fig)

    starts=split_starts["validation"]
    _,truth_pose=predicted_and_truth(PARAMETER_2,data,starts,config)
    reference_prediction,_=predicted_and_truth(PARAMETER_2,data,starts,config)
    error=np.linalg.norm(reference_prediction[:,-1,:2]-truth_pose[:,-1,:2],axis=1)
    chosen=(int(np.argmin(error)),int(np.argmin(np.abs(error-np.quantile(error,.95)))),
            int(np.argmax(error)))
    predictions={label:predicted_and_truth(value,data,starts,config)[0]
                 for label,value in models.items()}
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for axis,index,title in zip(axes,chosen,("best","P95","worst")):
        axis.plot(truth_pose[index,:,0],truth_pose[index,:,1],"k-",lw=3,label="GT")
        for label,prediction in predictions.items():
            axis.plot(prediction[index,:,0],prediction[index,:,1],lw=1.7,label=label)
        axis.scatter([0.],[0.],c="black",s=25);axis.set_aspect("equal",adjustable="box")
        axis.set(title=f"validation {title}; start index={int(starts[index])}",
                 xlabel="relative x [m]",ylabel="relative y [m]")
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle("Open-loop trajectory comparison")
    fig.tight_layout();fig.savefig(OUTPUT_DIR/"best_p95_worst_trajectories.png",dpi=180);plt.close(fig)
    print(json.dumps(output,indent=2))
    print(f"outputs: {OUTPUT_DIR}")


if __name__=="__main__":main()
