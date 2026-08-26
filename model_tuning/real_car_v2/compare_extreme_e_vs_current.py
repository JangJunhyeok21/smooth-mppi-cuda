#!/usr/bin/env python3
"""Compare the requested extreme-E Pacejka set against current YAML."""
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
from step_5_finetune_high_speed_worst import rollout_errors


REQUESTED=np.asarray((
    .3366075474431282,.6356924636110722,1.0411165641262696,-14231.435412582694,
    .8670431599215196,1.078260951127556,1.3367398542655606,-5504.52844896071))
OUT=settings.ROOT/"model_tuning/results/extreme_e_vs_current"


def summary(error):
    result={"windows":len(error["speed"]),"speed_mean_mps":float(np.mean(error["speed"]))}
    for key,unit in (("position","m"),("yaw","rad"),("yaw_rate","radps")):
        values=error[key]
        result[f"{key}_mean_{unit}"]=float(np.mean(values))
        result[f"{key}_rmse_{unit}"]=float(np.sqrt(np.mean(values**2)))
        result[f"{key}_p95_{unit}"]=float(np.quantile(values,.95))
        result[f"{key}_worst_{unit}"]=float(np.max(values))
    return result


def main():
    weight_search.configure();regression.MAX_PER_BAG=0;regression.WINDOW_START_STRIDE=1
    config=yaml.safe_load((settings.ROOT/"config/params.yaml").read_text())[
        "/**"]["ros__parameters"]
    data,contract=regression.load_regression_data(settings.DATA_PATH,config)
    starts=np.concatenate([regression.starts(data,index) for index in range(3)])
    current=np.asarray([config[f"dynamic_mlp_{name}"] for name in regression.NAMES],float)
    models={"requested_extreme_E":REQUESTED,"current_yaml":current}
    errors={name:rollout_errors(value,data,starts,config) for name,value in models.items()}
    metrics={name:summary(value) for name,value in errors.items()}
    improvements={}
    for field in ("position_p95_m","yaw_p95_rad","yaw_rate_p95_radps"):
        old=metrics["requested_extreme_E"][field];new=metrics["current_yaml"][field]
        improvements[field]=100.*(old-new)/max(abs(old),1e-12)
    report={"data_contract":contract,"parameters":{
        name:dict(zip(regression.NAMES,value.tolist())) for name,value in models.items()},
        "metrics":metrics,"current_reduction_vs_requested_percent":improvements}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")

    fig,axes=plt.subplots(1,3,figsize=(16,4.8));edges=np.arange(1.,8.01,.5)
    for axis,key,unit in zip(axes,("position","yaw","yaw_rate"),("m","rad","rad/s")):
        for name,error in errors.items():
            x=[];values=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                mask=(error["speed"]>=lo)&(error["speed"]<hi)
                if mask.sum()<20:continue
                x.append((lo+hi)/2);values.append(float(np.quantile(error[key][mask],.95)))
            axis.plot(x,values,"o-",label=name)
        axis.set(title=f"{key.replace('_',' ')} P95",xlabel="mean GT vx [m/s]",ylabel=unit)
        axis.grid(alpha=.25);axis.legend()
    fig.suptitle("Requested extreme-E vs current YAML: speed-binned P95")
    fig.tight_layout();fig.savefig(OUT/"speed_binned_p95.png",dpi=180);plt.close(fig)

    # Show best/P95/worst selected independently for each parameter set.
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for row,(name,error) in enumerate(errors.items()):
        order=np.argsort(error["position"])
        chosen=(order[0],order[int(round(.95*(len(order)-1)))],order[-1])
        for axis,index,title in zip(axes[row],chosen,("best","P95","worst")):
            axis.plot(error["truth_pose"][index,:,0],error["truth_pose"][index,:,1],
                      "k-",lw=3,label="GT")
            for other,other_error in errors.items():
                axis.plot(other_error["predicted_pose"][index,:,0],
                          other_error["predicted_pose"][index,:,1],lw=1.8,label=other)
            axis.set_aspect("equal",adjustable="box");axis.grid(alpha=.25)
            axis.set(title=f"{name} {title} | pos={error['position'][index]:.3f} m",
                     xlabel="relative x [m]",ylabel="relative y [m]")
        axes[row,0].legend(fontsize=8)
    fig.suptitle("Each model's independently ranked best / P95 / worst trajectories")
    fig.tight_layout();fig.savefig(OUT/"best_p95_worst.png",dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2));print(f"outputs: {OUT}")


if __name__=="__main__":main()
