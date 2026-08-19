#!/usr/bin/env python3
"""Step through one all-bag open-loop replay with Space/arrow keys."""
from pathlib import Path
import os
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "model_tuning/results/all_bags_dynamic_residual_full_open_loop"

# Select "best", "median", "worst", or "custom". Ranking uses final
# trajectory error across every continuous segment in metrics.json.
CASE = "worst"
# These two are used only when CASE="custom".
BAG_NAME = "rosbag2_2026_08_10-21_45_06"
SEGMENT_INDEX = 0

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
import matplotlib.pyplot as plt


def main():
    bag_name,segment_index=BAG_NAME,SEGMENT_INDEX
    if CASE != "custom":
        if CASE not in ("best","median","worst"):
            raise SystemExit('CASE must be "best", "median", "worst", or "custom"')
        metrics=json.loads((RESULTS_DIR/"metrics.json").read_text())
        ranked=[]
        for name,value in metrics.items():
            if name.startswith("_"):continue
            for segment in value["segments"]:
                ranked.append((segment["trajectory_final_m"],name,segment["segment"]))
        ranked.sort()
        selected={"best":ranked[0],"median":ranked[len(ranked)//2],"worst":ranked[-1]}[CASE]
        _,bag_name,segment_index=selected
    path = RESULTS_DIR / f"{bag_name}_interactive_replay.npz"
    if not path.exists():
        raise SystemExit(f"Missing {path}\nRun: python model_tuning/evaluate_all_bags_full_open_loop.py")
    archive = np.load(path)
    count = int(archive["segment_count"])
    if not 0 <= segment_index < count:
        raise SystemExit(f"SEGMENT_INDEX must be in [0, {count-1}]")
    prediction = archive[f"prediction_{segment_index}"]
    target = archive[f"target_{segment_index}"]
    control = archive[f"control_{segment_index}"]
    dt = float(archive["dt"])
    initial_difference = prediction[0, :3]-target[0, :3]
    if not np.allclose(initial_difference, 0., rtol=0., atol=1e-12):
        raise RuntimeError(f"Initial [x,y,yaw] mismatch: {initial_difference}")

    labels = ((0, "x [m]"),(1, "y [m]"),(3, r"$v_x$ [m/s]"),
              (4, r"$v_y$ [m/s]"),(6, r"$a_x$ [m/s²]"),(7, r"$a_y$ [m/s²]"),
              (5, "yaw-rate [rad/s]"),(2, "yaw [rad]"))
    figure, axes = plt.subplots(3, 3, figsize=(16, 12))
    trajectory_axis = axes[0, 0]
    signal_axes = list(axes.flat[1:])
    time = np.arange(len(prediction))*dt
    all_x = np.r_[target[:,0],prediction[:,0]];all_y = np.r_[target[:,1],prediction[:,1]]
    margin = .05*max(np.ptp(all_x),np.ptp(all_y),1.)
    trajectory_axis.set_xlim(all_x.min()-margin,all_x.max()+margin)
    trajectory_axis.set_ylim(all_y.min()-margin,all_y.max()+margin)
    trajectory_axis.set_aspect("equal",adjustable="box")
    gt_trajectory, = trajectory_axis.plot([],[],"k-",lw=2,label="GT")
    pred_trajectory, = trajectory_axis.plot([],[],"--",color="tab:orange",lw=2,label="Prediction")
    gt_marker, = trajectory_axis.plot([],[],"ko",ms=7,label="GT current")
    pred_marker, = trajectory_axis.plot([],[],"o",color="tab:orange",ms=7,label="Prediction current")
    trajectory_axis.plot(target[0,0],target[0,1],"s",color="tab:green",ms=9,label="Common start")
    trajectory_axis.set_title("x-y trajectory");trajectory_axis.legend(fontsize=8);trajectory_axis.grid(alpha=.25)
    signal_lines=[]
    for axis,(column,title) in zip(signal_axes,labels):
        axis.set_xlim(0,time[-1] if time[-1]>0 else dt)
        values=np.r_[target[:,column],prediction[:,column]];pad=.08*max(np.ptp(values),1e-3)
        axis.set_ylim(values.min()-pad,values.max()+pad)
        gt_line,=axis.plot([],[],"k-",lw=1.8,label="GT")
        pred_line,=axis.plot([],[],"--",color="tab:orange",lw=1.8,label="Prediction")
        axis.set_title(title);axis.set_xlabel("time [s]");axis.grid(alpha=.25);axis.legend(fontsize=8)
        signal_lines.append((column,gt_line,pred_line))

    state={"step":0}
    def draw():
        step=state["step"];end=step+1
        gt_trajectory.set_data(target[:end,0],target[:end,1]);pred_trajectory.set_data(prediction[:end,0],prediction[:end,1])
        gt_marker.set_data([target[step,0]],[target[step,1]]);pred_marker.set_data([prediction[step,0]],[prediction[step,1]])
        for column,gt_line,pred_line in signal_lines:
            gt_line.set_data(time[:end],target[:end,column]);pred_line.set_data(time[:end],prediction[:end,column])
        position_error=np.linalg.norm(prediction[step,:2]-target[step,:2])
        yaw_error=np.degrees(np.arctan2(np.sin(prediction[step,2]-target[step,2]),np.cos(prediction[step,2]-target[step,2])))
        figure.suptitle(
            f"{CASE.upper()} | {bag_name} segment {segment_index} | step {step}/{len(time)-1}, t={time[step]:.2f}s | "
            f"position error={position_error:.4f}m, yaw error={yaw_error:.2f}° | "
            f"command=[steer {control[step,0]:.3f}, speed {control[step,1]:.3f}]\n"
            "Space/Right: +1 step, Left: -1 step, Home: reset, End: final, Q/Esc: close")
        figure.canvas.draw_idle()
    def key_press(event):
        if event.key in (" ","space","right"):
            state["step"]=min(state["step"]+1,len(time)-1)
        elif event.key=="left":state["step"]=max(state["step"]-1,0)
        elif event.key=="home":state["step"]=0
        elif event.key=="end":state["step"]=len(time)-1
        elif event.key in ("q","escape"):
            plt.close(figure);return
        draw()
    figure.canvas.mpl_connect("key_press_event",key_press)
    figure.tight_layout(rect=(0,0,.98,.92));draw();plt.show()


if __name__ == "__main__":
    main()
