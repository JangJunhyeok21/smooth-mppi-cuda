#!/usr/bin/env python3
"""Step 7: quantitatively evaluate held-out 1.2 s rollouts."""
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from contract import Contract,actuator_step,longitudinal_actuator_step,residual_gates
DATA=ROOT/'model_tuning/data/dynamic_40ms_residual.npz';PARAMS=ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'
# User-editable defaults. Optional CLI arguments remain available for bag-by-bag diagnostics.
RESULT_PATH=ROOT/'model_tuning/results/dynamic_40ms_recursive_stage2_seed31'
OUTPUT_PATH=RESULT_PATH/'rollout_30step_metrics.json'
DISABLE_MLP=False
BAG_ID=None
def load(path):
 z=np.fromfile(path,dtype='<f4');input_dim={3563:20,3695:22}.get(len(z));assert input_dim is not None;o=0
 def take(n):
  nonlocal o;q=z[o:o+n];o+=n;return q
 return (take(64*input_dim).reshape(64,input_dim),take(64),take(2048).reshape(32,64),take(32),take(96).reshape(3,32),take(3),take(input_dim),take(input_dim))
def net(x,w):
 w1,b1,w2,b2,w3,b3,mean,std=w;h=np.maximum(((x-mean)/std)@w1.T+b1,0);h=np.maximum(h@w2.T+b2,0);return np.clip(h@w3.T+b3,(-8,-8,-30),(8,8,30))
def stat(x):return {'mean':float(np.mean(x)),'p95':float(np.quantile(x,.95)),'max':float(np.max(x))}
def main():
 p=argparse.ArgumentParser();p.add_argument('result',nargs='?',default=str(RESULT_PATH));p.add_argument('--out',default=str(OUTPUT_PATH));p.add_argument('--data',default=str(DATA));p.add_argument('--classic-params',default=str(PARAMS));p.add_argument('--disable-mlp',action='store_true',default=DISABLE_MLP);p.add_argument('--disable-ax-residual',action='store_true');p.add_argument('--min-accel',type=float,help='평가에 사용할 종가속도 하한(미지정 시 YAML)');p.add_argument('--max-accel',type=float,help='평가에 사용할 종가속도 상한(미지정 시 YAML)');p.add_argument('--bag-id',type=int,default=BAG_ID);p.add_argument('--split-id',type=int);p.add_argument('--split-name',default='selected');p.add_argument('--use-offline-vy-gt',action='store_true');a=p.parse_args();result=Path(a.result);w=load(result/'dynamic_40ms_residual.bin');input_dim=w[0].shape[1];d=np.load(a.data);x=d['source_features'].astype(float);obs=d['source_observations'].astype(float);source_sr=d['source_speed_reference'].astype(float) if 'source_speed_reference' in d.files else x[:,0];teacher_vy=d['source_teacher_vy'].astype(float) if a.use_offline_vy_gt else x[:,1];b=d['source_bag_id'];sp=d['source_split'];v=d['source_valid'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];mlp_cfg=yaml.safe_load((ROOT/'config/MLP_params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(Path(a.classic_params).read_text())['expanded_fitted'];min_accel=float(cfg['min_accel']) if a.min_accel is None else a.min_accel;max_accel=float(cfg['max_accel']) if a.max_accel is None else a.max_accel;c=Contract(dt=.04,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']),speed_kp=float(cfg['speed_servo_kp']),speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),position_speed_scale=float(cfg['kinematic_position_speed_scale']),min_accel=min_accel,max_accel=max_accel);lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;H=30;report={'evaluation_contract':{'min_accel':min_accel,'max_accel':max_accel,'mlp_residual_limits':mlp_cfg}};res_limit=np.array((mlp_cfg['mlp_max_residual_ax'],mlp_cfg['mlp_max_residual_ay'],mlp_cfg['mlp_max_residual_yaw_accel']))
 # Match the S_v-free training contract.
 c.position_speed_scale=1.0
 selected_splits=(((a.split_id,a.split_name),) if a.split_id is not None else ((int(sp[np.flatnonzero(b==a.bag_id)[0]]),f'bag_{a.bag_id}'),) if a.bag_id is not None else ((1,'validation'),(2,'test_aggressive')))
 for split,name in selected_splits:
  starts=np.array([i for i in range(10,len(x)-2*H) if sp[i]==split and sp[i+2*H]==split and (a.bag_id is None or b[i]==a.bag_id) and v[i:i+2*H+1].all() and np.all(b[i:i+2*H+1]==b[i])])[::5];pe=[];ye=[];ve=[];vye=[];re=[];predicted=[];ground_truth=[]
  if not len(starts):continue
  for start in starts:
   state=x[start,:3].copy();acc=obs[start,:2].copy();ap=float(x[start,5]);sr=float(source_sr[start]);hist=x[start,10:20].reshape(5,2).copy();pose=np.zeros(3);trace=[np.r_[pose,state]]
   for k in range(H):
    i=start+2*k;c0=x[i,3:5]
    if k>0:hist=np.vstack((hist[1:],c0))
    previous=hist[-2,0];current=state.copy()
    cmd=c0;ap,_=actuator_step(ap,cmd[0],cmd[1],state[0],c);sr,bax=longitudinal_actuator_step(sr,cmd[1],state[0],c);vx,vy,r=state;safe=max(abs(vx),.5);af=ap-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;front_inner=bf-fit['E_f']*(bf-np.arctan(bf));rear_inner=br-fit['E_r']*(br-np.arctan(br));fyf=fzf*fit['D_f']*np.sin(fit['C_f']*np.arctan(front_inner));fyr=fzr*fit['D_r']*np.sin(fit['C_r']*np.arctan(rear_inner));ay=(fyf*np.cos(ap)+fyr)/m;rd=(lf*fyf*np.cos(ap)-lr*fyr)/iz;state=np.array((vx+(bax+vy*r)*.04,vy+(ay-vx*r)*.04,r+rd*.04))
    feat=np.r_[current,c0,ap,c0[0]-previous,state,hist.ravel()];feat=np.r_[feat,acc] if input_dim==22 else feat;raw=np.zeros(3) if a.disable_mlp else net(feat,w);res=np.clip(raw,-res_limit,res_limit);state=state+res*.04;acc=np.array((bax+res[0],ay+res[1]));yaw=pose[2];pose=np.array((pose[0]+c.position_speed_scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*.04,pose[1]+c.position_speed_scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*.04,yaw+state[2]*.04));trace.append(np.r_[pose,state])
   gt=x[start+2:start+2*H+1:2,:3].copy();gt[:,1]=teacher_vy[start+2:start+2*H+1:2];gp=np.zeros((H,3))
   for k,q in enumerate(gt):
    oldp=gp[k-1] if k else np.zeros(3);gp[k]=oldp+(c.position_speed_scale*np.array((q[0]*np.cos(oldp[2])-q[1]*np.sin(oldp[2]),q[0]*np.sin(oldp[2])+q[1]*np.cos(oldp[2]),0))*.04);gp[k,2]=oldp[2]+q[2]*.04
   initial_gt=x[start,:3].copy();initial_gt[1]=teacher_vy[start];pe.append(np.linalg.norm(pose[:2]-gp[-1,:2]));ye.append(abs((pose[2]-gp[-1,2]+np.pi)%(2*np.pi)-np.pi));ve.append(abs(state[0]-gt[-1,0]));vye.append(abs(state[1]-gt[-1,1]));re.append(abs(state[2]-gt[-1,2]));predicted.append(trace);ground_truth.append(np.c_[np.vstack((np.zeros(3),gp)),np.vstack((initial_gt,gt))])
  report[name]={'windows':len(starts),'trajectory_m':stat(pe),'yaw_rad':stat(ye),'vx_mps':stat(ve),'vy_mps':stat(vye),'yaw_rate_rps':stat(re)}
 out=Path(a.out) if a.out else result/'rollout_30step_metrics.json';out.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(out.with_suffix('.npz'),starts=starts,predicted=np.asarray(predicted),ground_truth=np.asarray(ground_truth));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
