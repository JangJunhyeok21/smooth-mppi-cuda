#!/usr/bin/env python3
"""Plot identical held-out 0815 best/median/worst windows for old and new models."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/"model_tuning/results/dynamic_40ms_yaw_preserved_stage2_pre0815"
NEW=ROOT/"model_tuning/results/dynamic_40ms_yaw_preserved_0815_stage2"
OUT=NEW/"comparison_with_pre0815/old_new_best_median_worst.png"
BAGS=(15,16,17)
DT=.04

def load(root):
    starts=[];pred=[];gt=[];bag=[]
    for bag_id in BAGS:
        archive=np.load(root/f"bag_{bag_id}.npz")
        starts.append(archive["starts"]);pred.append(archive["predicted"])
        gt.append(archive["ground_truth"]);bag.append(np.full(len(archive["starts"]),bag_id))
    return np.concatenate(starts),np.concatenate(pred),np.concatenate(gt),np.concatenate(bag)

def main():
    old_starts,old_pred,old_gt,old_bag=load(OLD)
    starts,pred,gt,bag=load(NEW)
    if not (np.array_equal(starts,old_starts) and np.array_equal(bag,old_bag) and np.allclose(gt,old_gt)):
        raise RuntimeError("old/new evaluation windows do not match")
    old_error=np.linalg.norm(old_pred[:,-1,:2]-gt[:,-1,:2],axis=1)
    new_error=np.linalg.norm(pred[:,-1,:2]-gt[:,-1,:2],axis=1)
    order=np.argsort(new_error);indices=(order[0],order[len(order)//2],order[-1])
    labels=("Best","Median","Worst");time=np.arange(pred.shape[1])*DT
    fig,axes=plt.subplots(5,3,figsize=(16,17))
    for column,(index,label) in enumerate(zip(indices,labels)):
        axis=axes[0,column];axis.plot(gt[index,:,0],gt[index,:,1],"k-",lw=2,label="GT")
        axis.plot(old_pred[index,:,0],old_pred[index,:,1],"C3:",lw=2,label="Pre-0815 SOTA")
        axis.plot(pred[index,:,0],pred[index,:,1],"C1--",lw=2,label="0815 augmented")
        axis.set_title(f"{label} · bag {bag[index]} · row {starts[index]}\nold {old_error[index]:.3f} m → new {new_error[index]:.3f} m")
        axis.axis("equal")
        signals=((3,r"$v_x$ [m/s]"),(4,r"$v_y$ [m/s]"),(5,"yaw rate [rad/s]"))
        for row,(signal,ylabel) in enumerate(signals,1):
            axis=axes[row,column];axis.plot(time,gt[index,:,signal],"k-",lw=2,label="GT")
            axis.plot(time,old_pred[index,:,signal],"C3:",lw=2,label="Pre-0815 SOTA")
            axis.plot(time,pred[index,:,signal],"C1--",lw=2,label="0815 augmented");axis.set_ylabel(ylabel)
        axes[4,column].plot(time,np.unwrap(gt[index,:,2]),"k-",lw=2,label="GT")
        axes[4,column].plot(time,np.unwrap(old_pred[index,:,2]),"C3:",lw=2,label="Pre-0815 SOTA")
        axes[4,column].plot(time,np.unwrap(pred[index,:,2]),"C1--",lw=2,label="0815 augmented");axes[4,column].set_ylabel("relative yaw [rad]")
        for row,axis in enumerate(axes[:,column]):
            axis.grid(alpha=.3);axis.legend(fontsize=8)
            axis.set_xlabel("x [m]" if row==0 else "time [s]")
        axes[0,column].set_ylabel("y [m]")
    fig.suptitle("Held-out 0815 high-speed: pre-0815 vs sign-aware augmented model",y=.995)
    fig.tight_layout();OUT.parent.mkdir(parents=True,exist_ok=True);fig.savefig(OUT,dpi=180);plt.close(fig)
    selected=[]
    for index,label in zip(indices,labels):
        selected.append({"case":label,"bag_id":int(bag[index]),"source_row":int(starts[index]),
                         "old_trajectory_error_m":float(old_error[index]),
                         "new_trajectory_error_m":float(new_error[index])})
    OUT.with_suffix(".json").write_text(json.dumps(selected,indent=2)+"\n")
    print(json.dumps({"plot":str(OUT),"selected":selected},indent=2))

if __name__=="__main__":main()
