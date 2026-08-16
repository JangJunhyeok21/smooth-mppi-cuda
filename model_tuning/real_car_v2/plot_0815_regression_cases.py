#!/usr/bin/env python3
"""Visualize held-out 0815 windows where retraining degraded trajectory error."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/"model_tuning/results/dynamic_40ms_yaw_preserved_stage2_pre0815"
NEW=ROOT/"model_tuning/results/dynamic_40ms_yaw_preserved_0815_stage2"
DATA=ROOT/"model_tuning/data/dynamic_40ms_residual.npz"
OUT=NEW/"comparison_with_pre0815/regression_cases.png"
BAGS=(15,16,17);NUMBER_OF_CASES=5;DT=.04

def load(root):
    starts=[];pred=[];gt=[];bags=[]
    for bag_id in BAGS:
        archive=np.load(root/f"bag_{bag_id}.npz")
        starts.append(archive["starts"]);pred.append(archive["predicted"]);gt.append(archive["ground_truth"])
        bags.append(np.full(len(archive["starts"]),bag_id))
    return np.concatenate(starts),np.concatenate(pred),np.concatenate(gt),np.concatenate(bags)

def main():
    old_starts,old_pred,old_gt,old_bags=load(OLD);starts,new_pred,gt,bags=load(NEW)
    if not (np.array_equal(starts,old_starts) and np.array_equal(bags,old_bags) and np.allclose(gt,old_gt)):
        raise RuntimeError("old/new windows are not identical")
    old_error=np.linalg.norm(old_pred[:,-1,:2]-gt[:,-1,:2],axis=1)
    new_error=np.linalg.norm(new_pred[:,-1,:2]-gt[:,-1,:2],axis=1);degradation=new_error-old_error
    worse=np.flatnonzero(degradation>0);selected=worse[np.argsort(degradation[worse])[-NUMBER_OF_CASES:][::-1]]
    source=np.load(DATA)["source_features"];time=np.arange(new_pred.shape[1])*DT
    fig,axes=plt.subplots(6,len(selected),figsize=(4.4*len(selected),20),squeeze=False)
    rows=((3,r"$v_x$ [m/s]"),(4,r"$v_y$ [m/s]"),(5,"yaw rate [rad/s]"))
    report=[]
    for column,index in enumerate(selected):
        axis=axes[0,column];axis.plot(gt[index,:,0],gt[index,:,1],"k-",lw=2,label="GT")
        axis.plot(old_pred[index,:,0],old_pred[index,:,1],"C3:",lw=2,label="Pre-0815 SOTA")
        axis.plot(new_pred[index,:,0],new_pred[index,:,1],"C1--",lw=2,label="0815 augmented")
        axis.set_title(f"Regression #{column+1} · bag {bags[index]} · row {starts[index]}\nold {old_error[index]:.3f} → new {new_error[index]:.3f} m (Δ +{degradation[index]:.3f})")
        axis.set_xlabel("x [m]");axis.set_ylabel("y [m]");axis.axis("equal")
        for row,(signal,ylabel) in enumerate(rows,1):
            axis=axes[row,column];axis.plot(time,gt[index,:,signal],"k-",lw=2,label="GT")
            axis.plot(time,old_pred[index,:,signal],"C3:",lw=2,label="Pre-0815 SOTA")
            axis.plot(time,new_pred[index,:,signal],"C1--",lw=2,label="0815 augmented");axis.set_ylabel(ylabel)
        axes[4,column].plot(time,np.unwrap(gt[index,:,2]),"k-",lw=2,label="GT")
        axes[4,column].plot(time,np.unwrap(old_pred[index,:,2]),"C3:",lw=2,label="Pre-0815 SOTA")
        axes[4,column].plot(time,np.unwrap(new_pred[index,:,2]),"C1--",lw=2,label="0815 augmented");axes[4,column].set_ylabel("relative yaw [rad]")
        command_rows=starts[index]+2*np.arange(new_pred.shape[1]-1)
        command_time=time[:-1];axes[5,column].plot(command_time,source[command_rows,3],"C0-",label="steer cmd [rad]")
        speed_axis=axes[5,column].twinx();speed_axis.plot(command_time,source[command_rows,4],"C2--",label="speed cmd [m/s]")
        axes[5,column].set_ylabel("steer cmd [rad]",color="C0");speed_axis.set_ylabel("speed cmd [m/s]",color="C2")
        lines,labels=axes[5,column].get_legend_handles_labels();lines2,labels2=speed_axis.get_legend_handles_labels();axes[5,column].legend(lines+lines2,labels+labels2,fontsize=8)
        for row,axis in enumerate(axes[:,column]):
            axis.grid(alpha=.3)
            if row not in (0,5):axis.legend(fontsize=8)
            if row>0:axis.set_xlabel("time [s]")
        report.append({"rank":column+1,"bag_id":int(bags[index]),"source_row":int(starts[index]),
                       "old_error_m":float(old_error[index]),"new_error_m":float(new_error[index]),
                       "degradation_m":float(degradation[index]),"initial_vx_mps":float(gt[index,0,3]),
                       "initial_vy_mps":float(gt[index,0,4]),"initial_yaw_rate_rps":float(gt[index,0,5]),
                       "steer_abs_max_rad":float(np.max(np.abs(source[command_rows,3]))),
                       "speed_command_min_mps":float(np.min(source[command_rows,4])),
                       "speed_command_max_mps":float(np.max(source[command_rows,4]))})
    fig.suptitle("Largest trajectory regressions on held-out 0815 high-speed windows",y=.998)
    fig.tight_layout();OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=180);plt.close(fig)
    OUT.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"plot":str(OUT),"cases":report},indent=2))

if __name__=="__main__":main()
