#!/usr/bin/env python3
"""Experimental: CUDA DYNAMIC_IMU_RECURSIVE E2E held-out rollout 평가."""
import argparse,json
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'model_tuning/results/e2e_0817_0820_inertial_ekf_bias_seed31';DATA=ROOT/'model_tuning/data/dynamic_0817_0820_inertial_ekf_bias_40ms.npz'
def load(path):
 z=np.fromfile(path,dtype='<f4');assert len(z)==3563;o=0
 def take(n):
  nonlocal o;q=z[o:o+n];o+=n;return q
 return take(1280).reshape(64,20),take(64),take(2048).reshape(32,64),take(32),take(96).reshape(3,32),take(3),take(20),take(20)
def net(x,w):
 w1,b1,w2,b2,w3,b3,m,s=w;h=np.maximum(((x-m)/s)@w1.T+b1,0);h=np.maximum(h@w2.T+b2,0);return h@w3.T+b3
def stat(x):return {'mean':float(np.mean(x)),'p95':float(np.quantile(x,.95)),'max':float(np.max(x))}
def main():
 p=argparse.ArgumentParser();p.add_argument('result',nargs='?',type=Path,default=RESULT);p.add_argument('--data',type=Path,default=DATA);p.add_argument('--out',type=Path);a=p.parse_args();out=a.out or a.result/'rollout_60step_metrics.json';w=load(a.result/'e2e_20ms.bin');d=np.load(a.data);x=d['source_features'].astype(float);obs=d['source_observations'].astype(float);b=d['source_bag_id'];sp=d['source_split'];v=d['source_valid'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];dt=.02;scale=float(cfg['kinematic_position_speed_scale']);wheelbase=float(cfg['l_f'])+float(cfg['l_r']);H=60;report={}
 for sid,name in ((1,'validation'),(2,'test_aggressive')):
  starts=np.array([i for i in range(10,len(x)-H) if sp[i]==sid and sp[i+H]==sid and v[i:i+H+1].all() and np.all(b[i:i+H+1]==b[i])])[::5];pe=[];ye=[];ve=[];vye=[];re=[];predicted=[];ground_truth=[]
  for start in starts:
   state=x[start,:3].copy();acc=obs[start,:2].copy();hist=x[start,10:20].reshape(5,2).copy();pose=np.zeros(3);trace=[np.r_[pose,state]]
   for k in range(H):
    i=start+k;cmd=x[i,3:5]
    steer=np.clip(float(cfg['kinematic_steer_scale'])*cmd[0]+float(cfg['kinematic_steer_bias']),-.55,.55);base_ax=np.clip(float(cfg['speed_servo_kp'])*(cmd[1]-state[0]),float(cfg['min_accel']),float(cfg['max_accel']));base_w=state[0]*np.tan(steer)/wheelbase;feature=np.r_[state,acc,cmd,base_ax,state[0]*base_w,base_w,hist.ravel()];ax,ay,next_r=net(feature,w);vx,vy,_=state;state=np.array((vx+(ax+vy*next_r)*dt,vy+(ay-vx*next_r)*dt,next_r));acc=np.array((ax,ay));yaw=pose[2];pose=np.array((pose[0]+scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*dt,pose[1]+scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*dt,yaw+state[2]*dt));trace.append(np.r_[pose,state])
    hist=np.vstack((hist[1:],cmd))
   gt=x[start+1:start+H+1,:3];gp=np.zeros((H,3))
   for k,q in enumerate(gt):
    old=gp[k-1] if k else np.zeros(3);gp[k,0]=old[0]+scale*(q[0]*np.cos(old[2])-q[1]*np.sin(old[2]))*dt;gp[k,1]=old[1]+scale*(q[0]*np.sin(old[2])+q[1]*np.cos(old[2]))*dt;gp[k,2]=old[2]+q[2]*dt
   pe.append(np.linalg.norm(pose[:2]-gp[-1,:2]));ye.append(abs((pose[2]-gp[-1,2]+np.pi)%(2*np.pi)-np.pi));ve.append(abs(state[0]-gt[-1,0]));vye.append(abs(state[1]-gt[-1,1]));re.append(abs(state[2]-gt[-1,2]));predicted.append(trace);ground_truth.append(np.c_[np.vstack((np.zeros(3),gp)),np.vstack((x[start,:3],gt))])
  report[name]={'windows':len(starts),'trajectory_m':stat(pe),'yaw_rad':stat(ye),'vx_mps':stat(ve),'vy_mps':stat(vye),'yaw_rate_rps':stat(re)}
  if sid==2:np.savez_compressed(out.with_suffix('.npz'),starts=starts,predicted=np.asarray(predicted),ground_truth=np.asarray(ground_truth))
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
