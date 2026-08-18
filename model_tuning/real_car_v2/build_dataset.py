#!/usr/bin/env python3
"""Build the audited 20-D residual-derivative dataset from real rosbag extracts.

The diagnostic reconstructed CSV is explicitly forbidden. Inputs must be NPZ
files produced directly from rosbag messages with /drive causal alignment.
"""
from pathlib import Path
import json, os, sys
import numpy as np
import yaml
from scipy.signal import savgol_filter

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from contract import Contract, FEATURES, OUTPUTS, actuator_step, longitudinal_actuator_step
from offline_lateral_velocity_smoother import smooth_segment_vy
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams,estimate_dataset

SOURCE_DIRS=(
    ROOT/"model_tuning/data/real_car_v2_drive",
    ROOT/"model_tuning/results/effective_vs_dynamic_0813/data",
    ROOT/"model_tuning/data/ifac0815_autonomous_physics_clean",
    ROOT/"model_tuning/data/ifac0817_0818_autonomous_physics_clean",
)
OUTPUT=Path(os.environ.get("DYNAMIC_SOURCE_OUTPUT",ROOT/"model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz"))
REPORT=OUTPUT.with_suffix(".json")
# Runtime-visible causal KF vy is always used by the model input.  The offline
# smoother is enabled by default only as the supervised teacher/target.
USE_OFFLINE_VY_SMOOTHER=os.environ.get("USE_OFFLINE_VY_SMOOTHER","1")=="1"
FORBIDDEN="prediction_vs_actual_run12_reconstructed.csv"
TRAINING_MAX_SPEED=4.0  # retain high-speed bag samples even during 2 m/s shakedown deployment
LEGACY_IMU_SIGNS=np.array((-1.0,1.0,-1.0),dtype=float)

def ema(x,alpha=.25):
    y=x.copy()
    for i in range(1,len(y)):y[i]=alpha*x[i]+(1-alpha)*y[i-1]
    return y

def main():
    cfg=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    files=sorted({p.resolve() for source in SOURCE_DIRS for p in source.glob("*.npz")})
    if not files:raise SystemExit(f"no direct-bag NPZ in {SOURCE_DIRS}")
    if any(FORBIDDEN in str(p) for p in files):raise RuntimeError("diagnostic reconstructed CSV/derivative is forbidden")
    features=[];targets=[];observations=[];teacher_vys=[];teacher_confidences=[];bag_ids=[];split_ids=[];valids=[];manifest=[];next_bag=0;c=Contract(
        steer_scale=float(cfg["kinematic_steer_scale"]),steer_bias=float(cfg["kinematic_steer_bias"]),
        steer_tau=float(cfg["steer_servo_time_constant"]),max_steer_rate=float(cfg["actuator_max_steer_rate"]),
        speed_kp=float(cfg["speed_servo_kp"]),min_accel=float(cfg["min_accel"]),max_accel=float(cfg["max_accel"]),
        speed_accel_tau=float(cfg["speed_reference_accel_time_constant"]),
        speed_brake_tau=float(cfg["speed_reference_brake_time_constant"]),
        max_speed_reference_rate=float(cfg["actuator_max_speed_reference_rate"]),
        position_speed_scale=float(cfg["kinematic_position_speed_scale"]),
        low_speed_center=float(cfg["dynamic_mlp_min_speed"]))
    # Source-session-disjoint split. Large, representative sessions are held
    # out; collision bags remain diagnostic-only segments after filtering.
    # The new 4 m/s aggressive runs are never shown to the optimizer.  The
    # speed30 session and an older independent bag form validation; all other
    # collision-cleaned /drive sessions are training data.
    # One aggressive run supplies the otherwise missing 3--4 m/s yaw-recovery
    # excitation; the second run remains strictly unseen for honest testing.
    # 0817/0818 oversteer holdouts are deliberately excluded from fitting.
    # They have the largest yaw-rate excess over the no-slip kinematic value,
    # so test performance measures generalization to rear-slip/oversteer rather
    # than memorization of the same run.
    test_names={"aggressive_boundary_run2.npz","codex_highspeed_run1.npz",
                "rosbag2_2026_08_17-17_31_57.npz",
                "rosbag2_2026_08_18-14_39_19.npz"}
    val_names={"effective_speed30_run1.npz","rosbag2_2026_08_08-16_54_33.npz",
               "codex_effective_history_1200_run2_60s.npz",
               "rosbag2_2026_08_18-14_55_28.npz",
               "rosbag2_2026_08_18-15_26_00.npz"}
    for source_id,path in enumerate(files):
        z=np.load(path);a=z["samples"].astype(float);names={str(x):i for i,x in enumerate(z["columns"])};dt=float(z["dt"])
        if abs(dt-c.dt)>1e-9:raise RuntimeError(f"{path}: dt={dt}")
        local_ids=a[:,names["bag_id"]].astype(int)
        for local in np.unique(local_ids):
            ii=np.flatnonzero(local_ids==local);s=a[ii];n=len(s)
            if n<12:continue
            kfp=LateralVelocityKFParams(cornering_stiffness_front=float(cfg["kf_cornering_stiffness_front"]),cornering_stiffness_rear=float(cfg["kf_cornering_stiffness_rear"]),mass=float(cfg["mass"]),yaw_inertia=float(cfg["I_z"]),l_f=float(cfg["l_f"]),l_r=float(cfg["l_r"]),dt=dt,min_longitudinal_speed=float(cfg["kf_min_vx"]),low_speed_threshold=float(cfg["kf_low_speed_threshold"]),max_abs_vy=float(cfg["kf_max_abs_vy"]),process_var_vy=float(cfg["kf_q_vy"]),process_var_yaw_rate=float(cfg["kf_q_yaw_rate"]),measurement_var_lateral_accel=float(cfg["kf_r_lateral_accel"]),measurement_var_yaw_rate=float(cfg["kf_r_yaw_rate"]),initial_var_vy=float(cfg["kf_initial_p_vy"]),initial_var_yaw_rate=float(cfg["kf_initial_p_yaw_rate"]),imu_lateral_accel_sign=float(cfg["imu_lateral_accel_sign"]),nonlinear_dvy_threshold=float(cfg["kf_nonlinear_dvy_threshold"]),nonlinear_dvy_width=float(cfg["kf_nonlinear_dvy_width"]),nonlinear_inertial_blend=float(cfg["kf_nonlinear_inertial_blend"]),nonlinear_process_noise_scale=float(cfg["kf_nonlinear_process_noise_scale"]),nonlinear_ay_noise_scale=float(cfg["kf_nonlinear_ay_noise_scale"]))
            # Pre-0815 bags used the legacy mounted-sensor convention. New
            # extracts store their own body-frame sign contract in the NPZ.
            # Never apply one global sign to a mixed-date training archive.
            # Missing metadata means a pre-0815 extract. Those bags retain the
            # old sensor mounting convention; current YAML signs describe the
            # live post-0815 node and must not be retroactively applied here.
            source_signs=(z["imu_axis_signs"].astype(float) if "imu_axis_signs" in z.files
                          else LEGACY_IMU_SIGNS.copy())
            imu_wz_sign,imu_ax_sign,imu_ay_sign=source_signs
            source_ema_alpha=(float(z["imu_ema_alpha"]) if "imu_ema_alpha" in z.files
                              else float(cfg["imu_ema_alpha"]))
            vy_input,r=estimate_dataset(s,z["columns"],dt,kfp,steer_scale=float(cfg["kf_steer_scale"]),steer_bias=float(cfg["kf_steer_bias"]),max_steer=float(cfg["kf_max_steer"]),imu_ema_alpha=source_ema_alpha,imu_wz_sign=imu_wz_sign,imu_ay_sign=imu_ay_sign)
            vy_teacher=vy_input.copy()
            vx=s[:,names["vx"]];steer_cmd=s[:,names["steer"]]
            speed_cmd=np.clip(s[:,names["speed_cmd"]],float(cfg["min_speed"]),TRAINING_MAX_SPEED)
            imu_ax=ema(imu_ax_sign*s[:,names["imu_ax"]],source_ema_alpha);imu_ay=ema(imu_ay_sign*s[:,names["imu_ay"]],source_ema_alpha)
            smoother_diagnostics=None
            if USE_OFFLINE_VY_SMOOTHER:
                smoothed_vy,smoother_diagnostics=smooth_segment_vy(
                    s[:,names["x"]],s[:,names["y"]],s[:,names["yaw"]],vx,r,imu_ay,dt)
                if smoother_diagnostics["usable"]:
                    # Non-causal smoothing is teacher/GT only. Runtime-visible
                    # features and the classic base must remain causal KF vy.
                    vy_teacher=smoothed_vy
            teacher_dvy=np.gradient(vy_teacher,dt)
            teacher_confidence=np.exp(-np.abs(teacher_dvy-(imu_ay-vx*r))/1.5)
            teacher_confidence*=np.clip(np.abs(vx)/.5,.15,1.0)
            teacher_confidence[:3]=teacher_confidence[-3:]=.1
            teacher_confidence=np.clip(teacher_confidence,.05,1.0)
            # Remove per-session longitudinal bias/gravity projection using
            # stationary samples only. Never estimate this from cornering or
            # acceleration data, which would erase real dynamics.
            stationary=(np.abs(s[:,names["vx"]])<.08)&(np.abs(s[:,names["speed_cmd"]])<.1)
            ax_bias=float(np.median(imu_ax[stationary])) if stationary.sum()>=10 else 0.0
            imu_ax-=ax_bias
            # Applied steer recursion is identical to contract.py and CUDA.
            applied=np.empty(n);applied[0]=np.clip(c.steer_scale*steer_cmd[0]+c.steer_bias,-.55,.55)
            base_ax=np.empty(n);speed_reference=float(vx[0])
            for k in range(n):
                prev=applied[k-1] if k else applied[0]
                applied[k],_=actuator_step(prev,steer_cmd[k],speed_cmd[k],vx[k],c)
                current_speed=np.hypot(vx[k],vy_input[k])
                speed_reference,base_ax[k]=longitudinal_actuator_step(speed_reference,speed_cmd[k],current_speed,c)
            Bf,Cf,Df,Ef=[float(cfg[f"dynamic_mlp_{x}"]) for x in ("B_f","C_f","D_f","E_f")]
            Br,Cr,Dr,Er=[float(cfg[f"dynamic_mlp_{x}"]) for x in ("B_r","C_r","D_r","E_r")]
            lf,lr,m,iz=float(cfg["l_f"]),float(cfg["l_r"]),float(cfg["mass"]),float(cfg["dynamic_mlp_I_z"]);wb=lf+lr
            safe=np.maximum(abs(vx),.5);af=applied-np.arctan2(vy_input+lf*r,safe);ar=-np.arctan2(vy_input-lr*r,safe)
            fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb
            fyf=fzf*Df*np.sin(Cf*np.arctan(Bf*af-Ef*(Bf*af-np.arctan(Bf*af))))
            fyr=fzr*Dr*np.sin(Cr*np.arctan(Br*ar-Er*(Br*ar-np.arctan(Br*ar))))
            base_ay=(fyf*np.cos(applied)+fyr)/m;base_rdot=(lf*fyf*np.cos(applied)-lr*fyr)/iz
            base_next_vx=vx+(base_ax+vy_input*r)*dt;base_next_vy=vy_input+(base_ay-vx*r)*dt;base_next_r=r+base_rdot*dt
            # GT derivatives follow the deployed body-axis convention. IMU ax,
            # ay are total body accelerations; yaw acceleration is a smoothed
            # derivative of signed causal-EMA IMU/KF yaw rate.
            win=min(11,n//2*2-1);gt_rdot=savgol_filter(r,win,3,deriv=1,delta=dt) if win>=5 else np.gradient(r,dt)
            target=np.c_[imu_ax-base_ax,imu_ay-base_ay,gt_rdot-base_rdot]
            history=np.zeros((n,10));
            for k in range(n):
                for h in range(5):
                    j=max(0,k-4+h);history[k,2*h:2*h+2]=(steer_cmd[j],speed_cmd[j])
            feat=np.c_[vx,vy_input,r,steer_cmd,speed_cmd,applied,steer_cmd-np.r_[steer_cmd[0],steer_cmd[:-1]],base_next_vx,base_next_vy,base_next_r,history]
            ok=np.isfinite(feat).all(1)&np.isfinite(target).all(1)&np.isfinite(vy_teacher)&(np.arange(n)>=5)&(abs(vx)<=6)&(abs(vy_input)<=2)&(abs(vy_teacher)<=2)&(abs(r)<=5)&(abs(imu_ax)<=15)&(abs(imu_ay)<=15)&(abs(gt_rdot)<=40)
            split=2 if path.name in test_names else 1 if path.name in val_names else 0
            # Every discontinuous segment needs its own id. Reusing source_id
            # made recursive windows cross localization/collision cuts.
            bag_id=next_bag
            features.append(feat.astype(np.float32));targets.append(target.astype(np.float32));observations.append(np.c_[imu_ax,imu_ay,r].astype(np.float32));teacher_vys.append(vy_teacher.astype(np.float32));teacher_confidences.append(teacher_confidence.astype(np.float32));valids.append(ok);bag_ids.append(np.full(n,bag_id));split_ids.append(np.full(n,split));manifest.append({"bag_id":bag_id,"source":str(path),"segment":int(local),"split":("train","val","test")[split],"samples":n,"valid":int(ok.sum()),"vy_input":"causal_kf","vy_teacher":("offline_mcl_imu_smoother" if smoother_diagnostics and smoother_diagnostics.get("usable") else "causal_kf_fallback"),"vy_smoother":smoother_diagnostics,"imu_ax_stationary_bias_removed":ax_bias,"imu_axis_signs":{"wz":imu_wz_sign,"ax":imu_ax_sign,"ay":imu_ay_sign},"imu_ema_alpha":source_ema_alpha,"command_topic":str(z["command_topic"])});next_bag+=1
    X=np.concatenate(features);Y=np.concatenate(targets);O=np.concatenate(observations);TV=np.concatenate(teacher_vys);TC=np.concatenate(teacher_confidences);B=np.concatenate(bag_ids);S=np.concatenate(split_ids);V=np.concatenate(valids)
    assert X.shape[1]==20 and tuple(FEATURES)==("vx","vy","yaw_rate","steer_cmd","speed_cmd","applied_steer","steer_cmd_delta","base_next_vx","base_next_vy","base_next_yaw_rate","steer_t-4","speed_t-4","steer_t-3","speed_t-3","steer_t-2","speed_t-2","steer_t-1","speed_t-1","steer_t","speed_t")
    np.savez_compressed(OUTPUT,features=X,targets=Y,observations=O,teacher_vy=TV,teacher_vy_confidence=TC,bag_id=B,split=S,valid=V,feature_names=np.array(FEATURES),target_names=np.array(OUTPUTS),observation_names=np.array(("imu_ax","imu_ay","imu_yaw_rate")),dt=c.dt,vy_input_contract="causal_kf",vy_teacher_contract="offline_smoother_with_causal_fallback")
    REPORT.write_text(json.dumps({"output":str(OUTPUT),"samples":len(X),"valid":int(V.sum()),"bags":int(len(np.unique(B))),"vy_input":"causal_kf","vy_teacher":"offline_mcl_imu_smoother_with_causal_fallback","sources":manifest,"forbidden_training_source":FORBIDDEN},indent=2)+"\n")
    print(REPORT.read_text())
if __name__=="__main__":main()
