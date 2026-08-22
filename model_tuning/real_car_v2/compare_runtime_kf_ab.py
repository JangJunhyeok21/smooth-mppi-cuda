#!/usr/bin/env python3
"""Causal A/B: deployed inertial KF versus classic-model forward EKF."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml

from classic_model_kalman_filter import smooth_classic_segment,wrap,_accelerations

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DEFAULT_DATA=ROOT/"model_tuning/data/ifac0810_0819_autonomous_physics_clean"
DEFAULT_OUT=ROOT/"model_tuning/results/runtime_kf_ab/classic_vs_inertial.json"


def inertial_forward(x,y,yaw,vx,gyro,ax,ay,dt,cfg):
    """Python parity of include/.../lateral_velocity_kf.hpp."""
    n=len(x);state=np.array((x[0],y[0],yaw[0],vx[0],0.,0.,0.));P=np.diag([float(cfg[q]) for q in
        ("kf_p0_x","kf_p0_y","kf_p0_yaw","kf_p0_vx","kf_p0_vy","kf_p0_ax_bias","kf_p0_ay_bias")])
    Q=np.diag([float(cfg[q]) for q in ("kf_q_x","kf_q_y","kf_q_yaw","kf_q_vx","kf_q_vy","kf_q_ax_bias","kf_q_ay_bias")])
    R=np.array([float(cfg[q]) for q in ("kf_r_mcl_x","kf_r_mcl_y","kf_r_mcl_yaw","kf_r_odom_vx")])
    filtered=np.empty((n,6));predicted=np.empty((n,6));I=np.eye(7)
    for k in range(n):
        if k:
            sy,cy=np.sin(state[2]),np.cos(state[2]);F=I.copy();r=gyro[k]
            F[0,2]+=dt*(-state[3]*sy-state[4]*cy);F[0,3]+=dt*cy;F[0,4]-=dt*sy
            F[1,2]+=dt*(state[3]*cy-state[4]*sy);F[1,3]+=dt*sy;F[1,4]+=dt*cy
            F[3,4]+=dt*r;F[3,5]-=dt;F[4,3]-=dt*r;F[4,6]-=dt
            state+=dt*np.array((state[3]*cy-state[4]*sy,state[3]*sy+state[4]*cy,r,
                ax[k]-state[5]+r*state[4],ay[k]-state[6]-r*state[3],0.,0.));state[2]=wrap(state[2]);P=F@P@F.T+Q
        predicted[k]=np.r_[state[:5],gyro[k]]
        for index,value,var in zip((0,1,2,3),(x[k],y[k],yaw[k],vx[k]),R):
            innovation=wrap(value-state[index]) if index==2 else value-state[index]
            gain=P[:,index]/(P[index,index]+var);old=P.copy();state+=gain*innovation;state[2]=wrap(state[2]);P=old-np.outer(gain,old[index]);P=.5*(P+P.T)
        filtered[k]=np.r_[state[:5],gyro[k]]
    return filtered,predicted


def errors(state,predicted,measurement,model_accel,imu_accel):
    post=state-measurement;pre=predicted-measurement;post[:,2]=wrap(post[:,2]);pre[:,2]=wrap(pre[:,2])
    pose_post=np.hypot(post[:,0],post[:,1]);pose_pre=np.hypot(pre[:,0],pre[:,1])
    values={"post_pose_m":pose_post,"pre_pose_m":pose_pre,"post_yaw_rad":np.abs(post[:,2]),
        "pre_yaw_rad":np.abs(pre[:,2]),"post_vx_mps":np.abs(post[:,3]),"pre_vx_mps":np.abs(pre[:,3]),
        "post_vy_mps":np.abs(post[:,4]),"pre_vy_mps":np.abs(pre[:,4]),"post_r_radps":np.abs(post[:,5]),
        "pre_r_radps":np.abs(pre[:,5]),"ax_mps2":np.abs(model_accel[:,0]-imu_accel[:,0]),
        "ay_mps2":np.abs(model_accel[:,1]-imu_accel[:,1])}
    return values


def summarize(chunks):
    out={}
    for key in chunks[0]:
        value=np.concatenate([q[key] for q in chunks]);value=value[np.isfinite(value)]
        out[key]={"mean":float(value.mean()),"p95":float(np.quantile(value,.95)),"rmse":float(np.sqrt(np.mean(value**2)))}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=DEFAULT_DATA);p.add_argument("--out",type=Path,default=DEFAULT_OUT);p.add_argument("--date-prefix",default="rosbag2_2026_08_20");a=p.parse_args()
    cfg=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    ea=[];eb=[];segments=[]
    for path in sorted(a.data.glob(a.date_prefix+"*.npz")):
        z=np.load(path);s=z["samples"].astype(float);names={str(q):i for i,q in enumerate(z["columns"])};dt=float(z["dt"]);sign=z["imu_axis_signs"].astype(float)
        ids=s[:,names["bag_id"]].astype(int)
        for sid in np.unique(ids):
            q=s[ids==sid]
            if len(q)<100:continue
            x,y,yaw,ovx,ovy=(q[:,names[k]] for k in ("x","y","yaw","vx","vy"));gyro=sign[0]*q[:,names["imu_wz"]];iax=sign[1]*q[:,names["imu_ax"]];iay=sign[2]*q[:,names["imu_ay"]]
            A,Ap=inertial_forward(x,y,yaw,ovx,gyro,iax,iay,dt,cfg)
            result=smooth_classic_segment(x,y,yaw,ovx,ovy,gyro,iax,iay,q[:,names["steer"]],q[:,names["speed_cmd"]],dt,cfg)
            B=result["filtered_state"];Bp=result["predicted_state"]
            bacc=np.asarray([_accelerations(B[k],result["applied_steer"][k],result["speed_reference"][k],cfg)[:2] for k in range(len(q))])
            # Inertial KF consumes IMU acceleration directly; report its state-implied acceleration.
            aacc=np.c_[np.gradient(A[:,3],dt)-A[:,5]*A[:,4],np.gradient(A[:,4],dt)+A[:,5]*A[:,3]]
            measurement=np.c_[x,y,yaw,ovx,ovy,gyro];imu=np.c_[iax,iay]
            ea.append(errors(A,Ap,measurement,aacc,imu));eb.append(errors(B,Bp,measurement,bacc,imu));segments.append({"file":path.name,"segment":int(sid),"samples":len(q)})
    if not ea:raise SystemExit("no eligible segments")
    report={"comparison":"causal forward filters only; RTS disabled for B metrics","A":"deployed 7-state inertial MCL+odom+IMU KF parity","B":"6-state classic MPPI/Pacejka forward EKF","files_prefix":a.date_prefix,"segments":segments,"notes":[
        "A yaw-rate equals the raw IMU input by construction, so its zero yaw-rate residual is not an estimated-model accuracy result.",
        "B also updates from odom vy and IMU yaw-rate/acceleration; post-update residuals therefore include measurement-following ability.",
        "Use pre-update residuals to judge one-step process prediction, and treat odom vx as the trusted longitudinal-speed reference.",
    ],"A_metrics":summarize(ea),"B_metrics":summarize(eb)}
    report["winner_by_mean"]={k:("A" if report["A_metrics"][k]["mean"]<report["B_metrics"][k]["mean"] else "B") for k in report["A_metrics"]}
    report["winner_by_mean"]["post_r_radps"]="not_comparable"
    report["winner_by_mean"]["pre_r_radps"]="not_comparable"
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
