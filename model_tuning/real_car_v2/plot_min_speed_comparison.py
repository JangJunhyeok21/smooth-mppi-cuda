#!/usr/bin/env python3
"""Visual comparison of deployed and min-speed-zero residual checkpoints."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/"model_tuning/results/real_car_v2_dynamic_residual_all6_seed31"
NEW=ROOT/"model_tuning/results/real_car_v2_dynamic_residual_min0_kf05_seed31"
OUTPUT=NEW/"old_vs_min0_best_median_worst.png"
DT=.02; SCALE=.8633491306389823

def load(result):
    return np.load(result/"rollout_60step_predictions_kf_0p5.npz")

def relative_pose(states):
    # states[...,0:3]=body vx, body vy, yaw-rate. Integrate exactly as MPPI
    # position update, starting every window at (0,0,0).
    shape=states.shape[:-1]+(3,);pose=np.zeros(shape,float)
    for k in range(states.shape[1]):
        previous=pose[:,k-1] if k else np.zeros((len(states),3))
        vx,vy,r=states[:,k,0],states[:,k,1],states[:,k,2]
        yaw=previous[:,2]+r*DT
        pose[:,k,0]=previous[:,0]+SCALE*(vx*np.cos(previous[:,2])-vy*np.sin(previous[:,2]))*DT
        pose[:,k,1]=previous[:,1]+SCALE*(vx*np.sin(previous[:,2])+vy*np.cos(previous[:,2]))*DT
        pose[:,k,2]=yaw
    return pose

def metrics(pred,gt):
    e=np.abs(pred-gt);p=relative_pose(pred);g=relative_pose(gt)
    distance=np.linalg.norm(p[:,-1,:2]-g[:,-1,:2],axis=1)
    return {"final_state_mae":e[:,-1].mean(0).tolist(),"final_state_p95":np.quantile(e[:,-1],.95,axis=0).tolist(),
            "trajectory_final_mean_m":float(distance.mean()),"trajectory_final_median_m":float(np.median(distance)),
            "trajectory_final_p95_m":float(np.quantile(distance,.95)),"trajectory_final_max_m":float(distance.max())},distance,p,g

def main():
    old,new=load(OLD),load(NEW);assert np.array_equal(old["starts"],new["starts"])
    gt=old["gt"];om,oe,op,gp=metrics(old["residual"],gt);nm,ne,np_,_=metrics(new["residual"],gt)
    report={"states":["vx_mps","vy_mps","yaw_rate_radps"],"horizon_s":1.2,"deployed_all6_seed31":om,"min0_seed31":nm}
    (NEW/"old_vs_min0_metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    # Rank by new-model final trajectory error; show its best/median/worst,
    # while both models and the identical GT are plotted in every panel.
    order=np.argsort(ne);chosen=[order[0],order[len(order)//2],order[-1]];labels=["Best","Median","Worst"]
    t=np.arange(1,gt.shape[1]+1)*DT
    fig,axes=plt.subplots(3,4,figsize=(20,13),constrained_layout=True)
    colors={"GT":"black","Deployed":"tab:blue","Min-speed 0":"tab:red"}
    for row,(idx,label) in enumerate(zip(chosen,labels)):
        ax=axes[row,0];ax.plot(gp[idx,:,0],gp[idx,:,1],color=colors["GT"],lw=2.5,label="GT")
        ax.plot(op[idx,:,0],op[idx,:,1],"--",color=colors["Deployed"],lw=2,label=f"Deployed ({oe[idx]:.3f} m)")
        ax.plot(np_[idx,:,0],np_[idx,:,1],"--",color=colors["Min-speed 0"],lw=2,label=f"Min-speed 0 ({ne[idx]:.3f} m)")
        ax.scatter([gp[idx,-1,0]],[gp[idx,-1,1]],c="black",s=25);ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.3);ax.legend(fontsize=8);ax.set_title(f"{label}: relative trajectory")
        for col,(state,ylabel) in enumerate(((0,"vx [m/s]"),(1,"vy [m/s]"),(2,"yaw rate [rad/s]")),1):
            a=axes[row,col];a.plot(t,gt[idx,:,state],color="black",lw=2,label="GT");a.plot(t,old["residual"][idx,:,state],"--",color=colors["Deployed"],label="Deployed");a.plot(t,new["residual"][idx,:,state],"--",color=colors["Min-speed 0"],label="Min-speed 0");a.grid(alpha=.3);a.set(xlabel="time [s]",ylabel=ylabel);a.legend(fontsize=8)
    fig.suptitle("Deployed all6_seed31 vs retrained min_speed=0 (KF vy reset below 0.5 m/s)",fontsize=16)
    fig.savefig(OUTPUT,dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2));print(OUTPUT)
if __name__=="__main__":main()
