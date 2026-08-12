#!/usr/bin/env python3
"""Compare the deployed effective model with the fitted 40 ms dynamic residual."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "model_tuning/results/effective_vs_dynamic_0813"
COMPARED_SUFFIX = "yawrate_dense"
COMPARED_LABEL = "Dynamic 40 ms, dense yaw-rate"
OUTPUT_PREFIX = "effective_vs_yawrate_dense"
RUNS = [
    ("effective_speed20_run2", "speed 2.0"),
    ("effective_speed25_run1", "speed 2.5"),
    ("effective_speed30_run1", "speed 3.0"),
    ("aggressive_boundary_run1", "aggressive 1"),
    ("aggressive_boundary_run2", "aggressive 2"),
]
MODELS = [("effective_recorded", "Effective", "#f28e2b"),
          (COMPARED_SUFFIX, COMPARED_LABEL, "#4e79a7")]


def load_metric(run, model):
    data = json.loads((RESULT / f"{run}_{model}.json").read_text())
    if model != "effective_recorded":
        data = next(iter(data.values()))
        return [data[k]["mean"] for k in
                ("trajectory_m", "yaw_rad", "vx_mps", "vy_mps", "yaw_rate_rps")]
    return [data[k][0] for k in
            ("trajectory_mean_p95_max_m", "yaw_mean_p95_max_rad",
             "vx_mae_p95_max_mps", "vy_mae_p95_max_mps",
             "yaw_rate_mae_p95_max_rps")]


def metric_figure():
    vals = {m: np.array([load_metric(r, m) for r, _ in RUNS]) for m, _, _ in MODELS}
    titles = ["Final trajectory error [m]", "Final yaw error [deg]",
              "Final vx error [m/s]", "Final vy error [m/s]",
              "Final yaw-rate error [rad/s]"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    x = np.arange(len(RUNS)); width = .36
    for j, ax in enumerate(axes.flat[:5]):
        for i, (model, label, color) in enumerate(MODELS):
            y = vals[model][:, j].copy()
            if j == 1: y = np.rad2deg(y)
            bars = ax.bar(x + (i-.5)*width, y, width, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
        ax.set_title(titles[j]); ax.set_xticks(x, [x[1] for x in RUNS], rotation=20, ha="right")
        ax.grid(axis="y", alpha=.3)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(.98, .08))
    fig.suptitle("1.2 s open-loop mean error: same bags and recorded /drive commands", fontsize=15)
    out = RESULT / f"{OUTPUT_PREFIX}_state_metrics.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    return vals, out


def qualitative_figure():
    # Use both aggressive runs; window order is identical after dropping the one extra tail window.
    pairs=[]
    for run, _ in RUNS[-2:]:
        e=np.load(RESULT/f"{run}_effective_recorded.npz")
        d=np.load(RESULT/f"{run}_{COMPARED_SUFFIX}.npz")
        n=min(len(e["predicted"]),len(d["predicted"]))
        for i in range(n): pairs.append((run,i,e["predicted"][i],d["predicted"][i],d["ground_truth"][i]))
    score=np.array([np.linalg.norm(q[3][-1,:2]-q[4][-1,:2]) for q in pairs])
    picks=[int(np.argmin(score)),int(np.argsort(score)[len(score)//2]),int(np.argmax(score))]
    names=["Best", "Median", "Worst"]
    t=np.arange(31)*.04
    fig,ax=plt.subplots(5,3,figsize=(16,18),constrained_layout=True)
    for col,(name,idx) in enumerate(zip(names,picks)):
        run,wi,e,d,g=pairs[idx]
        ax[0,col].plot(g[:,0],g[:,1],'k-',lw=2,label='GT')
        ax[0,col].plot(e[:,0],e[:,1],'--',color='#f28e2b',label='Effective')
        ax[0,col].plot(d[:,0],d[:,1],'--',color='#4e79a7',label=COMPARED_LABEL)
        ax[0,col].set_aspect('equal',adjustable='datalim'); ax[0,col].set_title(f"{name}: {run}\nDynamic final traj error={score[idx]:.3f} m")
        for row,(state,ylabel,scale) in enumerate([(3,'vx [m/s]',1),(4,'vy [m/s]',1),(5,'yaw rate [rad/s]',1),(2,'yaw [deg]',180/np.pi)],start=1):
            ax[row,col].plot(t,g[:,state]*scale,'k-',lw=2,label='GT')
            ax[row,col].plot(t,e[:,state]*scale,'--',color='#f28e2b',label='Effective')
            ax[row,col].plot(t,d[:,state]*scale,'--',color='#4e79a7',label=COMPARED_LABEL)
            ax[row,col].set_ylabel(ylabel); ax[row,col].set_xlabel('time [s]'); ax[row,col].grid(alpha=.3)
        ax[0,col].grid(alpha=.3); ax[0,col].set_xlabel('relative x [m]'); ax[0,col].set_ylabel('relative y [m]')
    for a in ax.flat: a.legend(fontsize=8)
    fig.suptitle("Aggressive bags: best / median / worst 1.2 s open-loop predictions",fontsize=15)
    out=RESULT/f'{OUTPUT_PREFIX}_best_median_worst.png';fig.savefig(out,dpi=180);plt.close(fig)
    return out


if __name__ == '__main__':
    vals,p1=metric_figure();p2=qualitative_figure()
    print(p1);print(p2)
    for key,label,_ in MODELS: print(label, np.mean(vals[key],axis=0))
