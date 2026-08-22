#!/usr/bin/env python3
"""Fit Step 3 independently across CG heights and compare open-loop metrics."""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUTPUT_ROOT=ROOT/"model_tuning/results/step3_load_transfer_sweep"
H_CG_VALUES_M=(0.0,0.065,0.070,0.074,0.075,0.080)

# A balanced first-pass global fit for every height. Increase these environment
# variables to repeat the sweep with the full Step 3 search budget.
DE_MAXITER=int(os.environ.get("SWEEP_DE_MAXITER","18"))
DE_POPSIZE=int(os.environ.get("SWEEP_DE_POPSIZE","5"))
ADAM_STEPS=int(os.environ.get("SWEEP_ADAM_STEPS","250"))
SURROGATE_SAMPLES=int(os.environ.get("SWEEP_SURROGATE_SAMPLES","160"))
SURROGATE_PROPOSALS=int(os.environ.get("SWEEP_SURROGATE_PROPOSALS","12000"))


def validation_score(metric,weights):
    state_scale=(.4,2.,1.5);state_weight=(weights["vx"],weights["vy"],weights["yaw_rate"])
    denominator=max(sum(state_weight),1e-12)
    state_mean=sum(a*b*c for a,b,c in zip(state_scale,state_weight,metric["state_mae"]))/denominator
    state_p95=sum(a*b*c for a,b,c in zip(state_scale,state_weight,metric["state_p95"]))/denominator
    return (weights["position_xy"]*(metric["trajectory_mean_m"]+.5*metric["trajectory_p95_m"])
            +weights["trajectory_yaw"]*(metric["trajectory_yaw_mean_rad"]
                                         +.5*metric["trajectory_yaw_p95_rad"])
            +.2*state_mean+.1*state_p95)


def main():
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True);rows=[]
    for h_cg in H_CG_VALUES_M:
        output=OUTPUT_ROOT/f"hcg_{h_cg:.3f}m"
        env={**os.environ,
            "LOAD_TRANSFER_H_CG_M":str(h_cg),"DYNAMIC_REGRESSION_OUT":str(output),
            "STEP3_USE_PLOT":"0","STEP3_INTERACTIVE_PLOT":"0","STEP3_APPLY_TO_YAML":"0",
            "CLASSIC_DE_MAXITER":str(DE_MAXITER),"CLASSIC_DE_POPSIZE":str(DE_POPSIZE),
            "CLASSIC_ADAM_RESTARTS":"1","CLASSIC_ADAM_STEPS":str(ADAM_STEPS),
            "CLASSIC_SURROGATE_SAMPLES":str(SURROGATE_SAMPLES),
            "CLASSIC_SURROGATE_PROPOSALS":str(SURROGATE_PROPOSALS)}
        report_path=output/"params.json"
        if report_path.is_file():
            print(f"\n=== reusing completed h_cg={h_cg:.3f} m ===",flush=True)
        else:
            print(f"\n=== fitting h_cg={h_cg:.3f} m ===",flush=True)
            with (OUTPUT_ROOT/f"hcg_{h_cg:.3f}m.log").open("w") as log:
                subprocess.run([sys.executable,str(HERE/"step_3_identify_classic_model.py")],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads(report_path.read_text())
        metric=report["metrics_fitted"]["validation"];weights=report["loss_weights"]
        row={"h_cg_m":h_cg,"weighted_score":validation_score(metric,weights),
             "position_mean_m":metric["trajectory_mean_m"],
             "position_p95_m":metric["trajectory_p95_m"],
             "yaw_mean_rad":metric["trajectory_yaw_mean_rad"],
             "yaw_p95_rad":metric["trajectory_yaw_p95_rad"],
             "selected_method":report["selected_method"],
             "parameters":report["expanded_fitted"]}
        rows.append(row);print(json.dumps(row,indent=2),flush=True)
    winner=min(rows,key=lambda row:row["weighted_score"])
    summary={"search_budget":{"de_maxiter":DE_MAXITER,"de_popsize":DE_POPSIZE,
        "adam_steps":ADAM_STEPS,"surrogate_samples":SURROGATE_SAMPLES,
        "surrogate_proposals":SURROGATE_PROPOSALS},"results":rows,"winner":winner}
    (OUTPUT_ROOT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    x=[row["h_cg_m"]*100 for row in rows]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for axis,key,label in ((axes[0],"weighted_score","weighted score"),
                           (axes[1],"position_p95_m","position P95 [m]"),
                           (axes[2],"yaw_p95_rad","yaw P95 [rad]")):
        axis.plot(x,[row[key] for row in rows],"o-");axis.set_xlabel("h_cg [cm]")
        axis.set_ylabel(label);axis.grid(alpha=.3);axis.axvline(winner["h_cg_m"]*100,
            color="tab:green",ls="--",label="best weighted score");axis.legend()
    fig.suptitle("Step 3 longitudinal load-transfer sweep")
    fig.tight_layout();fig.savefig(OUTPUT_ROOT/"load_transfer_comparison.png",dpi=180)
    print(f"\nWinner: h_cg={winner['h_cg_m']:.3f} m, score={winner['weighted_score']:.6g}")


if __name__=="__main__":main()
