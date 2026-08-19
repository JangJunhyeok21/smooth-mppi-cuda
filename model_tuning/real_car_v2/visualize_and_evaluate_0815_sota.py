#!/usr/bin/env python3
"""Evaluate the deployed 40 ms dynamic residual model on sign-aligned 0815 bags."""
from pathlib import Path
import json, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from contract import Contract, actuator_step, longitudinal_actuator_step, residual_gates
from helper_lateral_velocity_kf import LateralVelocityKFParams, estimate_dataset

DATA_DIR = ROOT / "model_tuning/data/ifac0815_autonomous_physics_clean"
WEIGHTS = ROOT / "config/dynamic_40ms_residual_servo_lag.bin"
CLASSIC_PARAMS = ROOT / "model_tuning/results/dynamic_40ms_regression/params.json"
OUTPUT_DIR = ROOT / "model_tuning/results/ifac0815_sota_evaluation"
HORIZON_STEPS = 30                 # 30 x 40 ms = 1.2 s
WINDOW_STRIDE_20MS = 10            # report windows every 0.2 s
IMU_WZ_SIGN = 1.0                  # 0815 sensor is already in MPPI body convention
IMU_AX_SIGN = 1.0
IMU_AY_SIGN = 1.0


def load_network(path):
    packed = np.fromfile(path, dtype="<f4")
    if packed.size != 3563:
        raise ValueError(f"{path}: expected 3563 float32, got {packed.size}")
    offset = 0
    def take(count):
        nonlocal offset
        value = packed[offset:offset + count]
        offset += count
        return value
    return (take(64*20).reshape(64,20), take(64),
            take(32*64).reshape(32,64), take(32),
            take(3*32).reshape(3,32), take(3), take(20), take(20))


def infer(feature, network):
    w1,b1,w2,b2,w3,b3,mean,std = network
    hidden = np.maximum(w1 @ ((feature-mean)/std) + b1, 0.0)
    hidden = np.maximum(w2 @ hidden + b2, 0.0)
    return np.clip(w3 @ hidden + b3, (-8.,-8.,-30.), (8.,8.,30.))


def wrapped(value):
    return (value + np.pi) % (2*np.pi) - np.pi


def percentile_summary(value):
    value=np.asarray(value,float)
    return {"mean":float(value.mean()), "median":float(np.median(value)),
            "p95":float(np.quantile(value,.95)), "worst":float(value.max())}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    fit=json.loads(CLASSIC_PARAMS.read_text())["expanded_fitted"]
    network=load_network(WEIGHTS)
    contract=Contract(dt=.04, steer_scale=float(cfg["kinematic_steer_scale"]),
        steer_bias=float(cfg["kinematic_steer_bias"]),
        steer_tau=float(cfg["steer_servo_time_constant"]),
        max_steer_rate=float(cfg["actuator_max_steer_rate"]),
        speed_kp=float(cfg["speed_servo_kp"]),
        speed_accel_tau=float(cfg["speed_reference_accel_time_constant"]),
        speed_brake_tau=float(cfg["speed_reference_brake_time_constant"]),
        max_speed_reference_rate=float(cfg["actuator_max_speed_reference_rate"]),
        position_speed_scale=float(cfg["kinematic_position_speed_scale"]),
        min_accel=float(cfg["min_accel"]), max_accel=float(cfg["max_accel"]),
        low_speed_center=float(cfg["dynamic_mlp_min_speed"]))
    lf,lr,mass,iz=(float(cfg[k]) for k in ("l_f","l_r","mass","dynamic_mlp_I_z"))
    fzf=mass*9.81*lr/(lf+lr); fzr=mass*9.81*lf/(lf+lr)
    all_cases=[]; per_bag={}; sign_checks={}
    for path in sorted(DATA_DIR.glob("*.npz")):
        z=np.load(path); samples=z["samples"].astype(float)
        names={str(v):i for i,v in enumerate(z["columns"])}
        bag_cases=[]; correlations=[]
        for segment in np.unique(samples[:,names["bag_id"]].astype(int)):
            rows=np.flatnonzero(samples[:,names["bag_id"]].astype(int)==segment)
            s=samples[rows]; n=len(s)
            if n < 2*HORIZON_STEPS+6: continue
            kfp=LateralVelocityKFParams(
                cornering_stiffness_front=float(cfg["kf_cornering_stiffness_front"]),
                cornering_stiffness_rear=float(cfg["kf_cornering_stiffness_rear"]),
                mass=float(cfg["mass"]), yaw_inertia=float(cfg["I_z"]),
                l_f=lf,l_r=lr,dt=.02,min_longitudinal_speed=float(cfg["kf_min_vx"]),
                low_speed_threshold=float(cfg["kf_low_speed_threshold"]))
            vy,yaw_rate=estimate_dataset(s,z["columns"],.02,kfp,
                steer_scale=float(cfg["kf_steer_scale"]),steer_bias=float(cfg["kf_steer_bias"]),
                max_steer=float(cfg["kf_max_steer"]),imu_ema_alpha=float(cfg["imu_ema_alpha"]),
                imu_wz_sign=IMU_WZ_SIGN,imu_ay_sign=IMU_AY_SIGN)
            pose_rate=np.gradient(np.unwrap(s[:,names["yaw"]]),.02)
            raw_w=s[:,names["imu_wz"]]
            good=np.isfinite(pose_rate)&np.isfinite(raw_w)&(np.abs(s[:,names["vx"]])>.5)
            if good.sum()>20:
                correlations.append((np.corrcoef(pose_rate[good],raw_w[good])[0,1],
                                     np.corrcoef(pose_rate[good],-raw_w[good])[0,1]))
            vx=s[:,names["vx"]]; command=s[:,[names["steer"],names["speed_cmd"]]]
            # Reconstruct persistent actuator states at 20 ms, matching node initialization.
            applied=np.empty(n); applied[0]=np.clip(contract.steer_scale*command[0,0]+contract.steer_bias,-.55,.55)
            speed_reference=np.empty(n); speed_reference[0]=vx[0]
            c20=Contract(**{**contract.__dict__,"dt":.02})
            for i in range(1,n):
                applied[i],_=actuator_step(applied[i-1],command[i,0],command[i,1],vx[i],c20)
                speed_reference[i],_=longitudinal_actuator_step(speed_reference[i-1],command[i,1],np.hypot(vx[i],vy[i]),c20)
            for start in range(5,n-2*HORIZON_STEPS,WINDOW_STRIDE_20MS):
                initial_xy=s[start,[names["x"],names["y"]]]; initial_yaw=s[start,names["yaw"]]
                state=np.array((vx[start],vy[start],yaw_rate[start]),float)
                pose=np.zeros(3); steer=float(applied[start]); speed_ref=float(speed_reference[start])
                hist=command[start-4:start+1].copy()
                trace=[np.r_[pose,state]]; classic_trace=[np.r_[pose,state]]
                cstate=state.copy(); cpose=pose.copy(); csteer=steer; cspeed_ref=speed_ref; chist=hist.copy()
                for step in range(HORIZON_STEPS):
                    source=start+2*step; cmd=command[source]
                    for use_mlp in (True,False):
                        st,po,delta,sref,history=(state,pose,steer,speed_ref,hist) if use_mlp else (cstate,cpose,csteer,cspeed_ref,chist)
                        previous_command=history[-1,0]
                        history=np.vstack((history[1:],cmd))
                        delta,_=actuator_step(delta,cmd[0],cmd[1],st[0],contract)
                        sref,base_ax=longitudinal_actuator_step(sref,cmd[1],np.hypot(st[0],st[1]),contract)
                        safe=max(abs(st[0]),.5)
                        alpha_f=delta-np.arctan2(st[1]+lf*st[2],safe)
                        alpha_r=-np.arctan2(st[1]-lr*st[2],safe)
                        bf=fit["B_f"]*alpha_f; br=fit["B_r"]*alpha_r
                        fyf=fzf*fit["D_f"]*np.sin(fit["C_f"]*np.arctan(bf-fit["E_f"]*(bf-np.arctan(bf))))
                        fyr=fzr*fit["D_r"]*np.sin(fit["C_r"]*np.arctan(br-fit["E_r"]*(br-np.arctan(br))))
                        base_ay=(fyf*np.cos(delta)+fyr)/mass
                        base_rdot=(lf*fyf*np.cos(delta)-lr*fyr)/iz
                        base=np.array((st[0]+(base_ax+st[1]*st[2])*.04,
                            st[1]+(base_ay-st[0]*st[2])*.04,st[2]+base_rdot*.04))
                        feature=np.r_[st,cmd,delta,cmd[0]-previous_command,base,history.ravel()]
                        residual=infer(feature,network)*residual_gates(st[0],contract) if use_mlp else np.zeros(3)
                        new_state=base+residual*.04
                        yaw=po[2]; new_pose=np.array((po[0]+contract.position_speed_scale*(new_state[0]*np.cos(yaw)-new_state[1]*np.sin(yaw))*.04,
                            po[1]+contract.position_speed_scale*(new_state[0]*np.sin(yaw)+new_state[1]*np.cos(yaw))*.04,
                            yaw+new_state[2]*.04))
                        if use_mlp: state,pose,steer,speed_ref,hist=new_state,new_pose,delta,sref,history; trace.append(np.r_[pose,state])
                        else: cstate,cpose,csteer,cspeed_ref,chist=new_state,new_pose,delta,sref,history; classic_trace.append(np.r_[cpose,cstate])
                gt_rows=start+2*np.arange(HORIZON_STEPS+1)
                dx=s[gt_rows,names["x"]]-initial_xy[0]; dy=s[gt_rows,names["y"]]-initial_xy[1]
                gt_pose=np.c_[dx*np.cos(initial_yaw)+dy*np.sin(initial_yaw),-dx*np.sin(initial_yaw)+dy*np.cos(initial_yaw),wrapped(s[gt_rows,names["yaw"]]-initial_yaw)]
                gt_state=np.c_[vx[gt_rows],vy[gt_rows],yaw_rate[gt_rows]]
                mcl_path_length=float(np.linalg.norm(np.diff(gt_pose[:,:2],axis=0),axis=1).sum())
                odom_path_length=float(np.sum(np.abs(vx[gt_rows[:-1]]))*.04)
                path_ratio=mcl_path_length/max(odom_path_length,.05)
                case={"bag":path.stem,"segment":int(segment),"start_s":float(s[start,names["t"]]),
                      "pred":np.asarray(trace),"classic":np.asarray(classic_trace),"gt":np.c_[gt_pose,gt_state],
                      "mcl_path_m":mcl_path_length,"odom_integrated_path_m":odom_path_length,
                      "mcl_to_odom_path_ratio":path_ratio,
                      # At meaningful speed, a near-frozen MCL pose with fast odometry is an
                      # unobservable wheel-spin/collision/localization case, not a rollout error.
                      "observation_consistent":bool(odom_path_length<.5 or (.65<=path_ratio<=1.3))}
                case["traj_error"]=float(np.linalg.norm(case["pred"][-1,:2]-gt_pose[-1,:2]))
                case["classic_traj_error"]=float(np.linalg.norm(case["classic"][-1,:2]-gt_pose[-1,:2]))
                bag_cases.append(case); all_cases.append(case)
        if correlations:
            sign_checks[path.stem]={"corr_pose_rate_vs_raw_imu_wz":float(np.nanmean(np.asarray(correlations)[:,0])),
                                    "corr_pose_rate_vs_negated_imu_wz":float(np.nanmean(np.asarray(correlations)[:,1]))}
        if bag_cases:
            valid_bag_cases=[q for q in bag_cases if q["observation_consistent"]]
            per_bag[path.stem]={"windows":len(bag_cases),"sota_trajectory_m":percentile_summary([q["traj_error"] for q in bag_cases]),
                                "classic_trajectory_m":percentile_summary([q["classic_traj_error"] for q in bag_cases])}
            if valid_bag_cases:
                per_bag[path.stem]["valid_autonomous_windows"]=len(valid_bag_cases)
                per_bag[path.stem]["valid_sota_trajectory_m"]=percentile_summary([q["traj_error"] for q in valid_bag_cases])
    if not all_cases: raise RuntimeError("no valid evaluation windows")
    clean_cases=[q for q in all_cases if q["observation_consistent"]]
    collision_free_names=set()
    for metadata_path in DATA_DIR.glob("*.json"):
        metadata=json.loads(metadata_path.read_text())
        if not metadata.get("collision_episodes"):
            collision_free_names.add(metadata_path.stem)
    collision_free_cases=[q for q in all_cases if q["bag"] in collision_free_names]
    def endpoint(signal_index, angle=False, cases=all_cases):
        values=[]
        for q in cases:
            delta=q["pred"][-1,signal_index]-q["gt"][-1,signal_index]
            values.append(abs(wrapped(delta)) if angle else abs(delta))
        return percentile_summary(values)
    report={"contract":{"model":"dynamic_40ms_yaw_preserved_stage2","weights":str(WEIGHTS),"model_dt_s":.04,
              "horizon_s":1.2,"imu_signs":{"wz":1,"ax":1,"ay":1},"windows_are_unseen_by_checkpoint":True},
            "bags_evaluated":len(per_bag),"windows":len(all_cases),"imu_wz_sign_check":sign_checks,
            "aggregate":{"sota_trajectory_m":percentile_summary([q["traj_error"] for q in all_cases]),
                         "classic_trajectory_m":percentile_summary([q["classic_traj_error"] for q in all_cases]),
                         "yaw_deg":{k:float(np.degrees(v)) for k,v in endpoint(2,True).items()},
                         "vx_mps":endpoint(3),"vy_mps":endpoint(4),"yaw_rate_radps":endpoint(5)},
            "observation_consistent_aggregate":{"windows":len(clean_cases),
                         "exclusion_rule":"if odom path >= 0.5 m, require 0.65 <= MCL_path/odom_path <= 1.3",
                         "sota_trajectory_m":percentile_summary([q["traj_error"] for q in clean_cases]),
                         "classic_trajectory_m":percentile_summary([q["classic_traj_error"] for q in clean_cases]),
                         "yaw_deg":{k:float(np.degrees(v)) for k,v in endpoint(2,True,clean_cases).items()},
                         "vx_mps":endpoint(3,False,clean_cases),"vy_mps":endpoint(4,False,clean_cases),
                         "yaw_rate_radps":endpoint(5,False,clean_cases)},
            "collision_free_session_aggregate":{"windows":len(collision_free_cases),
                         "bags":sorted(collision_free_names),
                         "sota_trajectory_m":percentile_summary([q["traj_error"] for q in collision_free_cases]),
                         "classic_trajectory_m":percentile_summary([q["classic_traj_error"] for q in collision_free_cases]),
                         "yaw_deg":{k:float(np.degrees(v)) for k,v in endpoint(2,True,collision_free_cases).items()},
                         "vx_mps":endpoint(3,False,collision_free_cases),"vy_mps":endpoint(4,False,collision_free_cases),
                         "yaw_rate_radps":endpoint(5,False,collision_free_cases)},
            "raw_worst_case":{k:v for k,v in max(all_cases,key=lambda q:q["traj_error"]).items() if k not in ("pred","classic","gt")},
            "consistent_worst_case":{k:v for k,v in max(clean_cases,key=lambda q:q["traj_error"]).items() if k not in ("pred","classic","gt")},
            "per_bag":per_bag}
    (OUTPUT_DIR/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    # Bag-level tail plot.
    labels=list(per_bag); x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(15,6))
    ax.bar(x-.2,[per_bag[k]["sota_trajectory_m"]["mean"] for k in labels],.4,label="SOTA mean")
    ax.bar(x+.2,[per_bag[k]["sota_trajectory_m"]["p95"] for k in labels],.4,label="SOTA p95")
    ax.plot(x,[per_bag[k]["classic_trajectory_m"]["p95"] for k in labels],"ro--",label="Classic p95")
    ax.set_xticks(x,[s.replace("codex_","") for s in labels],rotation=35,ha="right");ax.set_ylabel("1.2 s position error [m]");ax.grid(axis="y",alpha=.3);ax.legend();fig.tight_layout();fig.savefig(OUTPUT_DIR/"bagwise_trajectory_error.png",dpi=180);plt.close(fig)
    # Global best/median/worst with all state channels.
    ordered=sorted(clean_cases,key=lambda q:q["traj_error"]); chosen=(ordered[0],ordered[len(ordered)//2],ordered[-1])
    fig,axes=plt.subplots(4,3,figsize=(16,14)); titles=("Best","Median","Worst"); t=np.arange(HORIZON_STEPS+1)*.04
    for col,(q,title) in enumerate(zip(chosen,titles)):
        axes[0,col].plot(q["gt"][:,0],q["gt"][:,1],"k-",lw=2,label="GT MCL")
        axes[0,col].plot(q["pred"][:,0],q["pred"][:,1],"C1--",lw=2,label="SOTA")
        axes[0,col].plot(q["classic"][:,0],q["classic"][:,1],"C3:",label="Classic")
        axes[0,col].set_title(f"{title}: {q['traj_error']:.3f} m\n{q['bag']} @ {q['start_s']:.2f}s");axes[0,col].axis("equal")
        for row,(idx,label) in enumerate(((3,"vx [m/s]"),(4,"vy [m/s]"),(5,"yaw rate [rad/s]")),1):
            axes[row,col].plot(t,q["gt"][:,idx],"k-",label="GT");axes[row,col].plot(t,q["pred"][:,idx],"C1--",label="SOTA");axes[row,col].plot(t,q["classic"][:,idx],"C3:",label="Classic");axes[row,col].set_ylabel(label)
        for ax in axes[:,col]:ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/"best_median_worst.png",dpi=180);plt.close(fig)
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
