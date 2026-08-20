#!/usr/bin/env python3
"""Tune selected KF Q/R values against offline vy and visualize before/after."""
from pathlib import Path
import argparse, copy, json, sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATA = (ROOT / "model_tuning/data/ifac0820_042348/"
                "rosbag2_2026_08_20-04_23_48.npz")
DEFAULT_OUTPUT = ROOT / "model_tuning/results/tune_kf_qr_0820_042348_with_mcl"
sys.path.insert(0, str(HERE))
from helper_lateral_velocity_kf import estimate_dataset
from offline_lateral_velocity_smoother import smooth_segment_vy
from visualize_compare_mcl_offline_kf_vy import ema, make_params, metrics, pose_vy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=DEFAULT_DATA,
                        help=f"step-1 NPZ (default: {DEFAULT_DATA})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"result directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    z = np.load(args.data); samples = z["samples"].astype(float); columns = z["columns"]
    c = {str(name): i for i, name in enumerate(columns)}; dt = float(z["dt"])
    signs = z["imu_axis_signs"].astype(float); alpha = float(z["imu_ema_alpha"])
    base_params = make_params(cfg, dt)
    fixed = dict(steer_scale=float(cfg["kf_steer_scale"]), steer_bias=float(cfg["kf_steer_bias"]),
                 max_steer=float(cfg["kf_max_steer"]), imu_ema_alpha=alpha,
                 imu_wz_sign=float(signs[0]), imu_ay_sign=float(signs[2]),
                 use_pose_vy=bool(cfg["kf_pose_vy_enabled"]),
                 pose_window_s=float(cfg["kf_pose_vy_window_s"]))
    records = []
    for sid in np.unique(samples[:, c["bag_id"]].astype(int)):
        s = samples[samples[:, c["bag_id"]].astype(int)==sid]
        old, yaw_rate = estimate_dataset(s, columns, dt, copy.copy(base_params), **fixed)
        ay = ema(float(signs[2])*s[:, c["imu_ay"]], alpha)
        target, diag = smooth_segment_vy(s[:, c["x"]], s[:, c["y"]], s[:, c["yaw"]],
                                         s[:, c["vx"]], yaw_rate, ay, dt)
        mcl, mcl_valid = pose_vy(s, c, dt, .30)
        valid = np.abs(s[:, c["vx"]]) >= .5
        valid &= mcl_valid & np.isfinite(mcl)
        records.append({"id": int(sid), "s": s, "old": old, "target": target,
                        "mcl": mcl, "valid": valid, "diag": diag})

    original = np.array([base_params.process_var_vy, base_params.measurement_var_lateral_accel,
                         base_params.measurement_var_pose_vy, base_params.process_var_ay_bias])

    def evaluate(log_values, retain=False):
        values = 10.0**np.asarray(log_values)
        predictions=[]; targets=[]
        for record in records:
            p=copy.copy(base_params)
            p.process_var_vy, p.measurement_var_lateral_accel = values[:2]
            p.measurement_var_pose_vy, p.process_var_ay_bias = values[2:]
            prediction,_=estimate_dataset(record["s"], columns, dt, p, **fixed)
            mask=record["valid"]; predictions.append(prediction[mask]); targets.append(record["target"][mask])
            if retain: record["new"] = prediction
        error=np.concatenate(predictions)-np.concatenate(targets)
        # Worst-sensitive objective; the bias term prevents a one-sided vy estimate.
        score=np.mean(np.abs(error)) + .45*np.quantile(np.abs(error),.95) + 1.5*abs(np.mean(error))
        return float(score)

    bounds=[(np.log10(original[0]/30),np.log10(original[0])),
            (np.log10(original[1]),np.log10(original[1]*30)),
            (np.log10(original[2]/20),np.log10(original[2])),
            (np.log10(original[3]/10),np.log10(original[3]*30))]
    result=differential_evolution(evaluate,bounds,seed=args.seed,popsize=10,maxiter=24,
                                  polish=True,tol=2e-3,workers=1,updating="immediate")
    tuned=10.0**result.x; evaluate(result.x,retain=True)
    old=np.concatenate([r["old"][r["valid"]] for r in records])
    new=np.concatenate([r["new"][r["valid"]] for r in records])
    target=np.concatenate([r["target"][r["valid"]] for r in records])
    mcl=np.concatenate([r["mcl"][r["valid"]] for r in records])
    report={"source":str(args.data),"objective":"MAE + 0.45*P95 + 1.5*|bias|",
            "fixed_model":True,"optimized_samples":int(len(target)),
            "parameters":{"kf_q_vy":{"old":original[0],"tuned":tuned[0]},
                          "kf_r_lateral_accel":{"old":original[1],"tuned":tuned[1]},
                          "kf_r_pose_vy":{"old":original[2],"tuned":tuned[2]},
                          "kf_q_ay_bias":{"old":original[3],"tuned":tuned[3]}},
            "old_vs_offline":metrics(target,old),"tuned_vs_offline":metrics(target,new),
            "offline_vs_mcl":metrics(mcl,target), "old_kf_vs_mcl":metrics(mcl,old),
            "tuned_kf_vs_mcl":metrics(mcl,new),
            "optimizer_score":float(result.fun),"optimizer_success":bool(result.success),
            "optimizer_message":str(result.message)}
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")

    fig,axes=plt.subplots(len(records),2,figsize=(17,5*len(records)),constrained_layout=True)
    axes=np.atleast_2d(axes)
    max_change=max(.05,float(np.quantile(np.abs(new-target)-np.abs(old-target),.98)))
    for row,r in enumerate(records):
        s=r["s"];t=s[:,c["t"]]-s[0,c["t"]]
        axes[row,0].plot(t,r["mcl"],color=".60",lw=1,alpha=.8,label="MCL pose derivative")
        axes[row,0].plot(t,r["target"],"k",lw=2,label="offline target")
        axes[row,0].plot(t,r["old"],"C1--",alpha=.8,label="old KF")
        axes[row,0].plot(t,r["new"],"C0",lw=1.5,label="tuned KF")
        axes[row,0].set(title=f"segment {r['id']}",xlabel="segment time [s]",ylabel="vy [m/s]")
        improvement=np.abs(r["old"]-r["target"])-np.abs(r["new"]-r["target"])
        points=axes[row,1].scatter(s[:,c["x"]],s[:,c["y"]],c=improvement,cmap="RdBu",
                                   vmin=-max_change,vmax=max_change,s=15)
        axes[row,1].axis("equal"); axes[row,1].set(title="blue: tuned KF improved, red: worsened",xlabel="x [m]",ylabel="y [m]")
        fig.colorbar(points,ax=axes[row,1],label="old |error| - tuned |error| [m/s]")
        for ax in axes[row]:ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle("Fixed Pacejka model: KF Q/R tuning result")
    # plot
    plt.show()
    # save
    fig.savefig(args.output/"old_vs_tuned_kf.png",dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))



if __name__=="__main__":main()
