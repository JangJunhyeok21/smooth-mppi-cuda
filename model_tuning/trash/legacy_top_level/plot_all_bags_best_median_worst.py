#!/usr/bin/env python3
"""Plot final-error best/median/worst segments from all-bag replay."""
import json
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"model_tuning/results/all_bags_dynamic_residual_full_open_loop"
OUTPUT=RESULTS/"all_bags_best_median_worst.png"
SHOW_PLOTS=False
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-smppi")
import matplotlib.pyplot as plt


def main():
    metrics=json.loads((RESULTS/"metrics.json").read_text());ranked=[]
    for bag,value in metrics.items():
        if bag.startswith("_"):continue
        for segment in value["segments"]:
            ranked.append((segment["trajectory_final_m"],bag,segment["segment"],segment["duration_s"]))
    ranked.sort();chosen=(ranked[0],ranked[len(ranked)//2],ranked[-1]);case_names=("Best","Median","Worst")
    cases=[]
    for item in chosen:
        error,bag,segment,duration=item;z=np.load(RESULTS/f"{bag}_interactive_replay.npz")
        cases.append((item,z[f"prediction_{segment}"],z[f"target_{segment}"],float(z["dt"])))
    signals=((3,r"$v_x$ [m/s]"),(4,r"$v_y$ [m/s]"),(6,r"$a_x$ [m/s²]"),
             (7,r"$a_y$ [m/s²]"),(5,"yaw-rate [rad/s]"),(2,"yaw [rad]"))
    fig,axes=plt.subplots(7,3,figsize=(17,24))
    for col,(case_name,case) in enumerate(zip(case_names,cases)):
        (error,bag,segment,duration),prediction,target,dt=case;t=np.arange(len(prediction))*dt
        ax=axes[0,col];ax.plot(target[:,0],target[:,1],"k-",lw=2,label="GT")
        ax.plot(prediction[:,0],prediction[:,1],"--",color="tab:orange",lw=2,label="Prediction")
        ax.plot(target[0,0],target[0,1],"s",color="tab:green",ms=8,label="Common start")
        ax.plot(target[-1,0],target[-1,1],"ko",ms=6,label="GT end")
        ax.plot(prediction[-1,0],prediction[-1,1],"o",color="tab:orange",ms=6,label="Prediction end")
        ax.set_title(f"{case_name}: {error:.3f} m, {duration:.2f} s\n{bag.replace('rosbag2_2026_','')}, segment {segment}")
        ax.axis("equal");ax.grid(alpha=.25);ax.legend(fontsize=8)
        for row,(index,label) in enumerate(signals,1):
            ax=axes[row,col];ax.plot(t,target[:,index],"k-",lw=1.7,label="GT")
            ax.plot(t,prediction[:,index],"--",color="tab:orange",lw=1.7,label="Prediction")
            mae=np.mean(np.abs(prediction[:,index]-target[:,index]))
            if index==2:
                delta=np.arctan2(np.sin(prediction[:,index]-target[:,index]),np.cos(prediction[:,index]-target[:,index]))
                mae=np.degrees(np.mean(np.abs(delta)));suffix="deg"
            else:suffix=label.split("[")[-1].rstrip("]") if "[" in label else ""
            ax.set_title(f"{label}, MAE={mae:.3f} {suffix}");ax.set_xlabel("time [s]")
            ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("dynamic_mlp_residual — all-bag full open-loop best / median / worst",y=.997)
    fig.subplots_adjust(left=.07,right=.98,bottom=.04,top=.97,hspace=.45,wspace=.25)
    fig.savefig(OUTPUT,dpi=180);print(f"saved: {OUTPUT}")
    for name,item in zip(case_names,chosen):print(name,item)
    if SHOW_PLOTS:plt.show()
    else:plt.close(fig)


if __name__=="__main__":main()
