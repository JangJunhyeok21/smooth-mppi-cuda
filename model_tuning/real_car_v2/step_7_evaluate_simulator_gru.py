#!/usr/bin/env python3
"""시뮬레이터 GRU plant의 held-out 1.2 s batched recursive rollout 평가."""
import argparse,json
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'model_tuning/data/dynamic_0817_0820_inertial_ekf_bias_40ms.npz';RESULT=ROOT/'model_tuning/results/simulator_gru_0817_0820_seed31'
def stat(x):return {'mean':float(np.mean(x)),'p95':float(np.quantile(x,.95)),'max':float(np.max(x))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA);p.add_argument('--result',type=Path,default=RESULT);p.add_argument('--out',type=Path);p.add_argument('--device',default='cuda');a=p.parse_args();device=torch.device(a.device if torch.cuda.is_available() else 'cpu');model=torch.jit.load(str(a.result/'simulator_gru.ts')).to(device).eval();d=np.load(a.data);s=d['source_features'].astype(float);obs=d['source_observations'].astype(float);sr_source=d['source_speed_reference'].astype(float);bag=d['source_bag_id'];split=d['source_split'];valid=d['source_valid'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];dt=.02;H=60;L=50;scale=float(cfg['kinematic_position_speed_scale']);steer_scale=float(cfg['kinematic_steer_scale']);steer_bias=float(cfg['kinematic_steer_bias']);steer_tau=float(cfg['steer_servo_time_constant']);steer_rate_limit=float(cfg['actuator_max_steer_rate']);speed_accel_tau=float(cfg['speed_reference_accel_time_constant']);speed_brake_tau=float(cfg['speed_reference_brake_time_constant']);speed_rate_limit=float(cfg['actuator_max_speed_reference_rate']);source_feature=np.c_[s[:,:3],obs[:,:2],s[:,3:5],s[:,5],sr_source].astype(np.float32);report={};output=a.out or a.result/'rollout_60step_metrics.json'
 for sid,name in ((1,'validation'),(2,'test_aggressive')):
  starts=np.asarray([i for i in range(L,len(s)-H) if split[i]==sid and split[i+H]==sid and valid[i-L+1:i+H+1].all() and np.all(bag[i-L+1:i+H+1]==bag[i])])[::5];n=len(starts);state=s[starts,:3].copy();acc=obs[starts,:2].copy();ap=s[starts,5].copy();sr=sr_source[starts].copy();history=np.stack([source_feature[i-L+1:i+1] for i in starts]);pose=np.zeros((n,3));trace=np.empty((n,H+1,6));trace[:,0]=np.c_[pose,state]
  for k in range(H):
   if k>0:
    cmd=s[starts+k,3:5];target=np.clip(steer_scale*cmd[:,0]+steer_bias,-.55,.55);rate=np.clip((target-ap)/max(steer_tau,1e-3),-steer_rate_limit,steer_rate_limit);ap=np.clip(ap+rate*dt,-.55,.55);tau=np.where(cmd[:,1]>=sr,speed_accel_tau,speed_brake_tau);sr+=np.clip((cmd[:,1]-sr)/np.maximum(tau,1e-3),-speed_rate_limit,speed_rate_limit)*dt;row=np.c_[state,acc,cmd,ap,sr].astype(np.float32);history=np.concatenate((history[:,1:],row[:,None]),axis=1)
   with torch.no_grad():prediction=model(torch.from_numpy(history).to(device)).cpu().numpy()
   ax,ay,next_r=prediction.T;vx,vy=state[:,0],state[:,1];state=np.c_[vx+(ax+vy*next_r)*dt,vy+(ay-vx*next_r)*dt,next_r];acc=prediction[:,:2];yaw=pose[:,2].copy();pose[:,0]+=scale*(state[:,0]*np.cos(yaw)-state[:,1]*np.sin(yaw))*dt;pose[:,1]+=scale*(state[:,0]*np.sin(yaw)+state[:,1]*np.cos(yaw))*dt;pose[:,2]+=state[:,2]*dt;trace[:,k+1]=np.c_[pose,state]
  gt=np.stack([s[i+1:i+H+1,:3] for i in starts]);gp=np.zeros((n,H,3))
  for k in range(H):
   old=gp[:,k-1] if k else np.zeros((n,3));q=gt[:,k];gp[:,k,0]=old[:,0]+scale*(q[:,0]*np.cos(old[:,2])-q[:,1]*np.sin(old[:,2]))*dt;gp[:,k,1]=old[:,1]+scale*(q[:,0]*np.sin(old[:,2])+q[:,1]*np.cos(old[:,2]))*dt;gp[:,k,2]=old[:,2]+q[:,2]*dt
  pe=np.linalg.norm(pose[:,:2]-gp[:,-1,:2],axis=1);ye=np.abs((pose[:,2]-gp[:,-1,2]+np.pi)%(2*np.pi)-np.pi);ve=np.abs(state[:,0]-gt[:,-1,0]);vye=np.abs(state[:,1]-gt[:,-1,1]);re=np.abs(state[:,2]-gt[:,-1,2]);report[name]={'windows':n,'trajectory_m':stat(pe),'yaw_rad':stat(ye),'vx_mps':stat(ve),'vy_mps':stat(vye),'yaw_rate_rps':stat(re)}
  if sid==2:np.savez_compressed(output.with_suffix('.npz'),starts=starts,predicted=trace,ground_truth=np.concatenate((np.concatenate((np.zeros((n,1,3)),gp),axis=1),np.concatenate((s[starts,None,:3],gt),axis=1)),axis=2))
 output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
