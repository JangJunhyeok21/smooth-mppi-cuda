#!/usr/bin/env python3
"""Compare current vy observers with inertial-pose EKF and SLCMPC-style KF."""
from pathlib import Path
import copy, json, sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DEFAULT_DATA=ROOT/"model_tuning/data/ifac0820_042348/rosbag2_2026_08_20-04_23_48.npz"
DEFAULT_OUTPUT=ROOT/"model_tuning/results/compare_vy_kf_architectures_0820_042348"
sys.path.insert(0,str(HERE))
from helper_lateral_velocity_kf import estimate_dataset,estimate_dataset_pose_only
from offline_lateral_velocity_smoother import smooth_segment_vy
from visualize_compare_mcl_offline_kf_vy import ema,make_params,metrics,pose_vy


def wrap(value): return (value+np.pi)%(2*np.pi)-np.pi


def inertial_pose_ekf(s,c,dt,signs,alpha):
    """State [x,y,yaw,vx,vy,bax,bay], IMU ax/ay/r inputs, pose+vx measurements."""
    n=len(s);out=np.zeros(n);x=np.zeros(7);P=np.diag([.02,.02,.02,.10,.25,.10,.10])
    Q=np.diag([2e-5,2e-5,2e-5,2e-3,8e-3,2e-6,2e-6])
    R=np.diag([4e-3,4e-3,8e-3,3e-2]);H=np.zeros((4,7));H[0,0]=H[1,1]=H[2,2]=H[3,3]=1
    ax=ema(signs[1]*s[:,c["imu_ax"]],alpha);ay=ema(signs[2]*s[:,c["imu_ay"]],alpha)
    r=ema(signs[0]*s[:,c["imu_wz"]],alpha)
    x[:4]=[s[0,c["x"]],s[0,c["y"]],s[0,c["yaw"]],s[0,c["vx"]]]
    I=np.eye(7)
    for k in range(n):
        yaw,vx,vy,bax,bay=x[2],x[3],x[4],x[5],x[6];rk=r[k]
        f=np.array([vx*np.cos(yaw)-vy*np.sin(yaw),vx*np.sin(yaw)+vy*np.cos(yaw),rk,
                    ax[k]-bax+rk*vy,ay[k]-bay-rk*vx,0.,0.])
        F=I.copy();F[0,2]+=dt*(-vx*np.sin(yaw)-vy*np.cos(yaw));F[0,3]+=dt*np.cos(yaw);F[0,4]+=-dt*np.sin(yaw)
        F[1,2]+=dt*(vx*np.cos(yaw)-vy*np.sin(yaw));F[1,3]+=dt*np.sin(yaw);F[1,4]+=dt*np.cos(yaw)
        F[3,4]+=dt*rk;F[3,5]+=-dt;F[4,3]+=-dt*rk;F[4,6]+=-dt
        x+=dt*f;x[2]=wrap(x[2]);P=F@P@F.T+Q
        z=np.array([s[k,c["x"]],s[k,c["y"]],s[k,c["yaw"]],s[k,c["vx"]]])
        innovation=z-H@x;innovation[2]=wrap(innovation[2]);S=H@P@H.T+R
        K=np.linalg.solve(S,(P@H.T).T).T;x+=K@innovation;x[2]=wrap(x[2]);P=(I-K@H)@P
        if abs(x[3])<.5:x[4]=0.
        out[k]=np.clip(x[4],-2.,2.)
    return out


def slcmpc_style_kf(s,c,dt):
    """Faithful SLCMPC architecture; numerical F replaces solver-provided Ac/Bc."""
    # params_real.yaml bicycle constants used by the referenced real-car node.
    m,iz,lf,lr,Bf,Cf,Df,Br,Cr,Dr=3.84,.0855,.163,.162,1.7,1.5,33.,1.7,1.5,33.
    n=len(s);out=np.zeros(n);state=np.array([s[0,c["x"]],s[0,c["y"]],s[0,c["yaw"]],s[0,c["vx"]],s[0,c["imu_wz"]],0.,s[0,c["accel"]],s[0,c["steer"]]],float)
    P=np.zeros((8,8));Q=np.eye(8);R=np.eye(5);H=np.zeros((5,8));H[0,0]=H[1,1]=H[2,2]=H[3,6]=H[4,7]=1;I=np.eye(8)
    def step(q,ax,steer,vx_meas,r_meas):
        px,py,yaw,vx,r,vy,_,_=q;safe=vx if abs(vx)>1e-3 else np.copysign(1e-3,vx or 1.)
        af=-np.arctan2(r*lf+vy,safe)+steer;ar=np.arctan2(r*lr-vy,safe)
        fyf=Df*np.sin(Cf*np.arctan(Bf*af));fyr=Dr*np.sin(Cr*np.arctan(Br*ar))
        dvy=(fyr+fyf*np.cos(steer)-m*vx*r)/m;rdot=(fyf*lf*np.cos(steer)-fyr*lr)/iz
        result=q.copy();result[0]+=dt*(vx*np.cos(yaw)-vy*np.sin(yaw));result[1]+=dt*(vx*np.sin(yaw)+vy*np.cos(yaw));result[2]=wrap(yaw+dt*r)
        result[3]=vx_meas;result[4]=r_meas;result[5]=vy+dt*dvy;result[6]=ax;result[7]=steer
        if abs(vx_meas)<.5:result[4]=result[5]=0.
        return result
    for k in range(n):
        ax=s[k,c["accel"]];steer=s[k,c["steer"]];vx=s[k,c["vx"]];r=s[k,c["imu_wz"]]
        predicted=step(state,ax,steer,vx,r);F=np.zeros((8,8));eps=1e-5
        for j in range(8):
            pert=state.copy();pert[j]+=eps;F[:,j]=(step(pert,ax,steer,vx,r)-predicted)/eps
        P=F@P@F.T+Q;state=predicted
        z=np.array([s[k,c["x"]],s[k,c["y"]],s[k,c["yaw"]],ax,steer]);innovation=z-H@state;innovation[2]=wrap(innovation[2])
        S=H@P@H.T+R;K=np.linalg.solve(S,(P@H.T).T).T;state+=K@innovation;state[2]=wrap(state[2]);P=(I-K@H)@P
        out[k]=np.clip(state[5],-2.,2.)
    return out


def main(data=DEFAULT_DATA,output=DEFAULT_OUTPUT):
    output.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    z=np.load(data);a=z["samples"].astype(float);cols=z["columns"];c={str(v):i for i,v in enumerate(cols)};dt=float(z["dt"]);signs=z["imu_axis_signs"].astype(float);alpha=float(z["imu_ema_alpha"])
    fixed=dict(steer_scale=float(cfg["kf_steer_scale"]),steer_bias=float(cfg["kf_steer_bias"]),max_steer=float(cfg["kf_max_steer"]),imu_ema_alpha=alpha,imu_wz_sign=float(signs[0]),imu_ay_sign=float(signs[2]),use_pose_vy=True,pose_window_s=float(cfg["kf_pose_vy_window_s"]))
    records=[]
    for sid in np.unique(a[:,c["bag_id"]].astype(int)):
        s=a[a[:,c["bag_id"]].astype(int)==sid];p=make_params(cfg,dt)
        old,r=estimate_dataset(s,cols,dt,copy.copy(p),**fixed);simple,_=estimate_dataset_pose_only(s,cols,dt,copy.copy(p),**fixed)
        ay=ema(signs[2]*s[:,c["imu_ay"]],alpha);offline,_=smooth_segment_vy(s[:,c["x"]],s[:,c["y"]],s[:,c["yaw"]],s[:,c["vx"]],r,ay,dt)
        mcl,valid=pose_vy(s,c,dt,.30);valid&=np.abs(s[:,c["vx"]])>=.5
        records.append(dict(id=int(sid),s=s,valid=valid,mcl=mcl,offline=offline,old=old,simple=simple,
                            inertial=inertial_pose_ekf(s,c,dt,signs,alpha),slcmpc=slcmpc_style_kf(s,c,dt)))
    keys=("mcl","offline","old","simple","inertial","slcmpc");joined={k:np.concatenate([r[k][r["valid"]] for r in records]) for k in keys}
    report={"source":str(data),"reference_warning":"MCL and offline are proxy references, not independent GT","moving_vx_ge_0p5":{}}
    for key in ("old","simple","inertial","slcmpc"):
        report["moving_vx_ge_0p5"][key]={"vs_mcl":metrics(joined["mcl"],joined[key]),"vs_offline":metrics(joined["offline"],joined[key])}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    colors=dict(mcl=".65",offline="k",old="C1",simple="C2",inertial="C3",slcmpc="C4")
    labels=dict(mcl="MCL pose derivative",offline="offline smoother",old="old Pacejka EKF",simple="scalar Pacejka+MCL KF",inertial="MCL+odom+IMU EKF",slcmpc="SLCMPC-style 8-state KF")
    fig,axes=plt.subplots(len(records),1,figsize=(17,5*len(records)),squeeze=False,constrained_layout=True)
    for row,r in enumerate(records):
        t=r["s"][:,c["t"]]-r["s"][0,c["t"]]
        for key in keys:axes[row,0].plot(t,r[key],color=colors[key],lw=2 if key=="offline" else 1.2,ls="--" if key=="old" else "-",label=labels[key])
        axes[row,0].set(title=f"segment {r['id']}",xlabel="time [s]",ylabel="body vy [m/s]");axes[row,0].grid(alpha=.25)
    axes[0,0].legend(ncol=3);fig.suptitle("vy observer architecture comparison");fig.savefig(output/"vy_architecture_comparison.png",dpi=180);plt.close(fig)
    print(json.dumps(report["moving_vx_ge_0p5"],indent=2))


if __name__=="__main__":main()
