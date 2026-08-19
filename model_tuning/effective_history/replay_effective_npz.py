#!/usr/bin/env python3
"""Recorded-command replay of the deployed effective-history binary."""
import argparse,json,struct,sys
from pathlib import Path
import numpy as np,yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from check_cuda_parity import load
from model_tuning.real_car_v2.helper_lateral_velocity_kf import LateralVelocityKFParams,estimate_dataset

def ema(x,a=.25):
 y=x.copy()
 for i in range(1,len(y)):y[i]=a*x[i]+(1-a)*y[i-1]
 return y
def append(h,u):return np.r_[h[1:],np.asarray(u)[None]]
def forward(f,w):
 w1,b1,w2,b2,w3,b3,mean,std=w;h=np.maximum(w1@((f-mean)/std)+b1,0);h=np.maximum(w2@h+b2,0);return np.array((.12,.10,.25))*np.tanh(w3@h+b3)
def stat(x):return [float(np.mean(x)),float(np.quantile(x,.95)),float(np.max(x))]
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--out',required=True);p.add_argument('--weights',default=str(ROOT/'config/effective_history_state_residual.bin'));p.add_argument('--params',default=str(ROOT/'config/params.yaml'));a=p.parse_args();w=load(a.weights);z=np.load(a.dataset);s=z['samples'].astype(float);c={str(x):i for i,x in enumerate(z['columns'])};cfg=yaml.safe_load(Path(a.params).read_text())['/**']['ros__parameters'];dt=.02;mdt=.04;H=30;cmd=np.c_[s[:,c['steer']],np.clip(s[:,c['speed_cmd']],float(cfg['min_speed']),float(cfg['max_speed']))];r=ema(float(cfg['imu_wz_sign'])*s[:,c['imu_wz']]);ax=ema(float(cfg['imu_ax_sign'])*s[:,c['imu_ax']]);ay=ema(float(cfg['imu_ay_sign'])*s[:,c['imu_ay']]);kfp=LateralVelocityKFParams(cornering_stiffness_front=float(cfg['kf_cornering_stiffness_front']),cornering_stiffness_rear=float(cfg['kf_cornering_stiffness_rear']),mass=float(cfg['mass']),yaw_inertia=float(cfg['I_z']),l_f=float(cfg['l_f']),l_r=float(cfg['l_r']),dt=dt,min_longitudinal_speed=float(cfg['kf_min_vx']),low_speed_threshold=float(cfg['kf_low_speed_threshold']));vygt,_=estimate_dataset(s,z['columns'],dt,kfp,steer_scale=float(cfg['kf_steer_scale']),steer_bias=float(cfg['kf_steer_bias']),max_steer=float(cfg['kf_max_steer']),imu_ema_alpha=float(cfg['imu_ema_alpha']),imu_wz_sign=float(cfg['imu_wz_sign']),imu_ay_sign=float(cfg['imu_ay_sign']));starts=np.arange(10,len(s)-2*H,5);pe=[];ye=[];ve=[];vye=[];re=[];predicted=[];ground_truth=[]
 for start in starts:
  origin=s[start,1:4].copy();pose=np.zeros(3);body=np.array((s[start,c['vx']],0.,r[start]));acc=np.array((ax[start],ay[start]));hist=cmd[start-9:start+1].copy();trace=[np.r_[pose,body]]
  for k in range(H):
   i=start+2*k;c0=cmd[i];c1=cmd[i+1]
   if k>0:hist=append(hist,c0)
   steer,speed=hist[-1];vx,vy,rr=body;f=np.r_[body,acc,steer,speed,hist[:,0],hist[:,1],steer-hist[-2,0],steer-hist[-4,0],speed-hist[-2,1],vx*steer,vx*vx*steer,vx*rr,abs(vx)*steer];corr=forward(f,w)
   for u in (c0,c1):
    target=vx/.324*np.tan(.51*u[0]+.01);rdot=np.clip((target-rr)/.10,-15,15);aa=np.clip(.76*(u[1]-vx),-1,1);vx+=aa*dt;vy+=(-vy/.12)*dt;rr+=rdot*dt
   nb=np.array((vx,vy,rr))+corr;yaw=pose[2];pose=np.array((pose[0]+.8633491306389823*(nb[0]*np.cos(yaw)-nb[1]*np.sin(yaw))*mdt,pose[1]+.8633491306389823*(nb[0]*np.sin(yaw)+nb[1]*np.cos(yaw))*mdt,(yaw+nb[2]*mdt+np.pi)%(2*np.pi)-np.pi));acc=np.array(((nb[0]-body[0])/mdt,(nb[1]-body[1])/mdt+body[0]*body[2]));body=nb;hist=append(hist,c1);trace.append(np.r_[pose,body])
  rows=start+2*np.arange(31);gtbody=np.c_[s[rows,c['vx']],vygt[rows],r[rows]];gtpose=np.zeros((31,3))
  for k in range(1,31):q=gtbody[k];old=gtpose[k-1];gtpose[k]=old+np.array((.8633491306389823*(q[0]*np.cos(old[2])-q[1]*np.sin(old[2]))*mdt,.8633491306389823*(q[0]*np.sin(old[2])+q[1]*np.cos(old[2]))*mdt,q[2]*mdt))
  j=start+60;pe.append(np.linalg.norm(pose[:2]-gtpose[-1,:2]));ye.append(abs((pose[2]-gtpose[-1,2]+np.pi)%(2*np.pi)-np.pi));ve.append(abs(body[0]-s[j,c['vx']]));vye.append(abs(body[1]-vygt[j]));re.append(abs(body[2]-r[j]));predicted.append(trace);ground_truth.append(np.c_[gtpose,gtbody])
 report={'windows':len(starts),'horizon_s':1.2,'trajectory_mean_p95_max_m':stat(pe),'yaw_mean_p95_max_rad':stat(ye),'vx_mae_p95_max_mps':stat(ve),'vy_mae_p95_max_mps':stat(vye),'yaw_rate_mae_p95_max_rps':stat(re)};out=Path(a.out);out.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),starts=starts,predicted=np.asarray(predicted),ground_truth=np.asarray(ground_truth));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
