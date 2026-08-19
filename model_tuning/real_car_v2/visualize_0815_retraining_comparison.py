#!/usr/bin/env python3
"""Compare pre-0815 and sign-aware 0815-augmented checkpoints."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NEW=ROOT/"model_tuning/results/dynamic_40ms_yaw_preserved_0815_stage2"
OLD=ROOT/"model_tuning/results/dynamic_40ms_yaw_preserved_stage2_pre0815"
OUT=NEW/"comparison_with_pre0815"
BAGS=(15,16,17)

def load(root,bag):
    report=json.loads((root/f"bag_{bag}.json").read_text())
    return report[f"bag_{bag}"]

def weighted(root):
    reports=[load(root,b) for b in BAGS];count=sum(r["windows"] for r in reports)
    result={"windows":count}
    for signal in ("trajectory_m","yaw_rad","vx_mps","vy_mps","yaw_rate_rps"):
        result[signal]={"mean":sum(r["windows"]*r[signal]["mean"] for r in reports)/count,
                        "weighted_segment_p95":sum(r["windows"]*r[signal]["p95"] for r in reports)/count,
                        "max":max(r[signal]["max"] for r in reports)}
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);old,new=weighted(OLD),weighted(NEW)
    comparison={"held_out_0815_highspeed_segments":list(BAGS),"old":old,"new":new,"improvement_percent":{}}
    for signal in ("trajectory_m","yaw_rad","vx_mps","vy_mps","yaw_rate_rps"):
        comparison["improvement_percent"][signal]={key:100*(old[signal][key]-new[signal][key])/old[signal][key]
                                                   for key in ("mean","weighted_segment_p95","max")}
    (OUT/"metrics.json").write_text(json.dumps(comparison,indent=2)+"\n")
    signals=("trajectory_m","yaw_rad","vx_mps","yaw_rate_rps")
    titles=("1.2 s trajectory [m]","1.2 s yaw [rad]","1.2 s vx [m/s]","1.2 s yaw-rate [rad/s]")
    fig,axes=plt.subplots(2,2,figsize=(11,8));x=np.arange(3);width=.36
    for axis,signal,title in zip(axes.flat,signals,titles):
        before=[load(OLD,b)[signal]["mean"] for b in BAGS];after=[load(NEW,b)[signal]["mean"] for b in BAGS]
        axis.bar(x-width/2,before,width,label="Pre-0815 SOTA");axis.bar(x+width/2,after,width,label="0815 augmented")
        axis.set_xticks(x,[f"highspeed seg {b-14}" for b in BAGS]);axis.set_title(title+" mean")
        axis.grid(axis="y",alpha=.3);axis.legend()
    fig.tight_layout();fig.savefig(OUT/"heldout_0815_highspeed_comparison.png",dpi=180);plt.close(fig)
    print(json.dumps(comparison,indent=2))

if __name__=="__main__":main()
