#!/usr/bin/env python3
"""Replay the deployed dynamic_residual_v2 contract on aligned real-car NPZs."""
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/"real_car_v2"))
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams,estimate_dataset
from contract import Contract,actuator_step,longitudinal_actuator_step,low_speed_gate

def ema(x,a=.25):
 y=x.copy()
 for i in range(1,len(y)):y[i]=a*x[i]+(1-a)*y[i-1]
 return y

def load_binary(path):
 z=np.fromfile(path,dtype="<f4");assert len(z)==3563;o=0
 def take(n):
  nonlocal o;q=z[o:o+n];o+=n;return q
 return take(1280).reshape(64,20),take(64),take(2048).reshape(32,64),take(32),take(96).reshape(3,32),take(3),take(20),take(20)

def net(x,w):
 w1,b1,w2,b2,w3,b3,mean,std=w
 h=np.maximum(((x-mean)/std)@w1.T+b1,0);h=np.maximum(h@w2.T+b2,0)
 return np.clip(h@w3.T+b3,(-8,-8,-30),(8,8,30))

def stats(x):return [float(np.mean(x)),float(np.quantile(x,.95)),float(np.max(x))]

def main():
 p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("--out",required=True);p.add_argument("--params",default=str(ROOT/"config/params.yaml"));p.add_argument("--weights",default=str(ROOT/"config/dynamic_residual_v2.bin"));a=p.parse_args()
 cfg=yaml.safe_load(Path(a.params).read_text())["/**"]["ros__parameters"];z=np.load(a.dataset);s=z["samples"].astype(float);cols={str(x):i for i,x in enumerate(z["columns"])};dt=float(z["dt"]);H=60;c=Contract(steer_scale=float(cfg["kinematic_steer_scale"]),steer_bias=float(cfg["kinematic_steer_bias"]),steer_tau=float(cfg["steer_servo_time_constant"]),max_steer_rate=float(cfg["actuator_max_steer_rate"]),speed_kp=float(cfg["speed_servo_kp"]),speed_accel_tau=float(cfg["speed_reference_accel_time_constant"]),speed_brake_tau=float(cfg["speed_reference_brake_time_constant"]),max_speed_reference_rate=float(cfg["actuator_max_speed_reference_rate"]),position_speed_scale=float(cfg["kinematic_position_speed_scale"]),min_accel=float(cfg["min_accel"]),max_accel=float(cfg["max_accel"]),low_speed_center=float(cfg["dynamic_mlp_min_speed"]));w=load_binary(a.weights)
 kfp=LateralVelocityKFParams(cornering_stiffness_front=float(cfg["kf_cornering_stiffness_front"]),cornering_stiffness_rear=float(cfg["kf_cornering_stiffness_rear"]),mass=float(cfg["mass"]),yaw_inertia=float(cfg["I_z"]),l_f=float(cfg["l_f"]),l_r=float(cfg["l_r"]),dt=dt,min_longitudinal_speed=float(cfg["kf_min_vx"]),low_speed_threshold=float(cfg["kf_low_speed_threshold"]))
 vy,r=estimate_dataset(s,z["columns"],dt,kfp,steer_scale=float(cfg["kf_steer_scale"]),steer_bias=float(cfg["kf_steer_bias"]),max_steer=float(cfg["kf_max_steer"]),imu_ema_alpha=float(cfg["imu_ema_alpha"]),imu_wz_sign=float(cfg["imu_wz_sign"]),imu_ay_sign=float(cfg["imu_ay_sign"]));vx=s[:,cols["vx"]];cmd=np.c_[s[:,cols["steer"]],np.clip(s[:,cols["speed_cmd"]],float(cfg["min_speed"]),float(cfg["max_speed"]))]
 applied=np.empty(len(s));speedref=np.empty(len(s));applied[0]=np.clip(c.steer_scale*cmd[0,0]+c.steer_bias,-.55,.55);sr=vx[0]
 for i in range(len(s)):
  applied[i],_=actuator_step(applied[i-1] if i else applied[0],cmd[i,0],cmd[i,1],vx[i],c);sr,_=longitudinal_actuator_step(sr,cmd[i,1],np.hypot(vx[i],vy[i]),c);speedref[i]=sr
 Bf,Cf,Df,Ef=[float(cfg[f"dynamic_mlp_{q}"]) for q in ("B_f","C_f","D_f","E_f")];Br,Cr,Dr,Er=[float(cfg[f"dynamic_mlp_{q}"]) for q in ("B_r","C_r","D_r","E_r")];lf,lr,m,iz=[float(cfg[q]) for q in ("l_f","l_r","mass","dynamic_mlp_I_z")];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb
 starts=np.arange(10,len(s)-H,5);poserr=[];yawerr=[];vxerr=[];rerr=[]
 for start in starts:
  state=np.array([vx[start],vy[start],r[start]],float);pose=s[start,1:4].copy();delta=applied[start];sr=speedref[start];hist=cmd[start-5:start].copy()
  for k in range(H):
   u=cmd[start+k];prev=hist[-1,0];delta,_=actuator_step(delta,u[0],u[1],state[0],c);sr,bax=longitudinal_actuator_step(sr,u[1],np.hypot(state[0],state[1]),c);sv=max(abs(state[0]),.5);af=delta-np.arctan2(state[1]+lf*state[2],sv);ar=-np.arctan2(state[1]-lr*state[2],sv);fyf=fzf*Df*np.sin(Cf*np.arctan(Bf*af-Ef*(Bf*af-np.arctan(Bf*af))));fyr=fzr*Dr*np.sin(Cr*np.arctan(Br*ar-Er*(Br*ar-np.arctan(Br*ar))));bay=(fyf*np.cos(delta)+fyr)/m;brd=(lf*fyf*np.cos(delta)-lr*fyr)/iz;base=np.array([state[0]+(bax+state[1]*state[2])*dt,state[1]+(bay-state[0]*state[2])*dt,state[2]+brd*dt]);feat=np.r_[state,u,delta,u[0]-prev,base,hist.ravel()];res=net(feat,w)*low_speed_gate(state[0],c);old=state;state=base+res*dt;yaw=pose[2];pose=np.array([pose[0]+c.position_speed_scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*dt,pose[1]+c.position_speed_scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*dt,(yaw+state[2]*dt+np.pi)%(2*np.pi)-np.pi]);hist=np.vstack((hist[1:],u))
  j=start+H;poserr.append(np.linalg.norm(pose[:2]-s[j,1:3]));yawerr.append(abs((pose[2]-s[j,3]+np.pi)%(2*np.pi)-np.pi));vxerr.append(abs(state[0]-vx[j]));rerr.append(abs(state[2]-r[j]))
 report={"windows":len(starts),"horizon_s":H*dt,"trajectory_mean_p95_max_m":stats(poserr),"yaw_mean_p95_max_rad":stats(yawerr),"vx_mae_p95_max_mps":stats(vxerr),"yaw_rate_mae_p95_max_rps":stats(rerr)};Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
