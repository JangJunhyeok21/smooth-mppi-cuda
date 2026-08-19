#!/usr/bin/env python3
"""Ablate causal IMU, command and velocity history for the residual MLP.

The first history frame is measured. During free rollout, IMU and velocity
history are updated only from model predictions; future measurements are never
read. Command history may use future candidate commands because an MPPI rollout
already owns its complete candidate control sequence.
"""
from pathlib import Path
import json,os,sys
import numpy as np,torch,yaml
from torch import nn

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from contract import Contract,actuator_step,longitudinal_actuator_step,residual_gates

DATA=Path(os.environ.get('DYNAMIC_RESIDUAL_DATA',ROOT/'model_tuning/data/dynamic_40ms_residual.npz'))
PARAMS=Path(os.environ.get('DYNAMIC_CLASSIC_PARAMS',ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'))
OUT=Path(os.environ.get('HISTORY_ABLATION_OUT',ROOT/'model_tuning/results/residual_history_ablation'))
SEED=31;EPOCHS=100;PATIENCE=20;HISTORY=5;HORIZON=30
VARIANTS={
 'baseline_20d':(),
 'imu_history_35d':('imu_values',),
 'imu_delta_history_32d':('imu_deltas',),
 'command_delta_history_28d':('command_deltas',),
 'vx_history_25d':('speed_values',),
 'vx_delta_history_24d':('speed_deltas',),
 'imu_command_delta_history_40d':('imu_deltas','command_deltas'),
 'imu_command_vx_delta_history_44d':('imu_deltas','command_deltas','speed_deltas'),
 'imu_history_command_vx_delta_47d':('imu_values','command_deltas','speed_deltas'),
 'all_history_64d':('imu_values','imu_deltas','command_deltas','speed_values','speed_deltas'),
}
if os.environ.get('HISTORY_VARIANT'):
 VARIANTS={name:flags for name,flags in VARIANTS.items()
           if name==os.environ['HISTORY_VARIANT']}

class Net(nn.Module):
 def __init__(self,n):super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
 def forward(self,x):return self.net(x)

def extras(flags,imu,cmd,speed):
 parts=[]
 if 'imu_values' in flags:parts.append(imu.reshape(-1))
 if 'imu_deltas' in flags:parts.append(np.diff(imu,axis=0).reshape(-1))
 if 'command_deltas' in flags:parts.append(np.diff(cmd,axis=0).reshape(-1))
 if 'speed_values' in flags:parts.append(speed.reshape(-1))
 if 'speed_deltas' in flags:parts.append(np.diff(speed).reshape(-1))
 return np.concatenate(parts) if parts else np.empty(0)

def histories(source,observations,rows):
 imu=[];cmd=[];speed=[]
 for row in rows:
  ids=row-2*np.arange(HISTORY-1,-1,-1)
  imu.append(observations[ids]);cmd.append(source[ids,3:5]);speed.append(source[ids,0])
 return np.asarray(imu),np.asarray(cmd),np.asarray(speed)

def fit_variant(name,flags,d,dev):
 base=d['features'].astype(np.float32);target=d['targets'].astype(np.float32);rows=d['source_row'];source=d['source_features'];obs=d['source_observations'];bag=d['bag_id'];split=d['split'];valid=d['valid'].copy()
 valid&=rows>=2*(HISTORY-1)
 imu,cmd,speed=histories(source,obs,rows)
 augmented=np.asarray([np.r_[base[i],extras(flags,imu[i],cmd[i],speed[i])] for i in range(len(base))],np.float32)
 train=valid&(split==0);validation=valid&(split==1);mean=augmented[train].mean(0);std=np.maximum(augmented[train].std(0),1e-4);ym=target[train].mean(0);ys=np.maximum(target[train].std(0),1e-3)
 x=torch.from_numpy((augmented-mean)/std).to(dev);y=torch.from_numpy((target-ym)/ys).to(dev);confidence=torch.from_numpy(d['teacher_confidence'].astype(np.float32) if 'teacher_confidence' in d.files else np.ones(len(base),np.float32)).to(dev);net=Net(augmented.shape[1]).to(dev);opt=torch.optim.AdamW(net.parameters(),8e-4,weight_decay=1e-4);rng=np.random.default_rng(SEED);indices=np.flatnonzero(train);weights=torch.tensor((1.,2.,2.),device=dev);difficulty=1.+1.5*(np.abs(source[rows,2])>.5)+1.5*(np.abs(source[rows,0])>=3.)+1.5*(np.abs(obs[rows,1]-source[rows,0]*source[rows,2])>1.5);sample_probability=difficulty[indices]/difficulty[indices].sum();best=(1e30,None,0);stale=0
 for epoch in range(EPOCHS):
  net.train();sample=rng.choice(indices,len(indices),replace=True,p=sample_probability)
  for q in np.array_split(sample,max(1,len(sample)//1024)):
   channel_weights=weights[None].repeat(len(q),1);channel_weights[:,1]*=confidence[q];loss=(nn.functional.smooth_l1_loss(net(x[q]),y[q],reduction='none')*channel_weights).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  net.eval()
  with torch.no_grad():score=float(nn.functional.smooth_l1_loss(net(x[validation]),y[validation]))
  if score<best[0]-1e-5:best=(score,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch+1);stale=0
  else:stale+=1
  if stale>=PATIENCE:break
 net.load_state_dict(best[1]);net.cpu()
 with torch.no_grad():
  net.net[4].weight.mul_(torch.from_numpy(ys)[:,None]);net.net[4].bias.mul_(torch.from_numpy(ys)).add_(torch.from_numpy(ym))
  prediction=net(torch.from_numpy((augmented-mean)/std)).numpy()
 result={'input_dim':augmented.shape[1],'best_epoch':best[2],'one_step':{}}
 for k,label in enumerate(('train','validation','test')):
  mask=valid&(split==k);error=np.abs(prediction[mask]-target[mask]);result['one_step'][label]={'mae':error.mean(0).tolist(),'p95':np.quantile(error,.95,axis=0).tolist()}
 return net,mean,std,result

def network(net,mean,std,feature):
 with torch.no_grad():return np.clip(net(torch.from_numpy(((feature-mean)/std).astype(np.float32))[None]).numpy()[0],(-8,-8,-30),(8,8,30))

def rollout(net,mean,std,flags,d,cfg,fit,split_id):
 x=d['source_features'].astype(float);obs=d['source_observations'].astype(float);b=d['source_bag_id'];sp=d['source_split'];valid=d['source_valid'];lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;c=Contract(dt=.04,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']),speed_kp=float(cfg['speed_servo_kp']),speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),position_speed_scale=float(cfg['kinematic_position_speed_scale']),min_accel=float(cfg['min_accel']),max_accel=float(cfg['max_accel']),low_speed_center=float(cfg['dynamic_mlp_min_speed']))
 starts=np.asarray([i for i in range(10,len(x)-2*HORIZON) if sp[i]==split_id and sp[i+2*HORIZON]==split_id and valid[i-8:i+2*HORIZON+1].all() and np.all(b[i-8:i+2*HORIZON+1]==b[i])])[::5];pe=[];ye=[];ve=[];vye=[];re=[]
 for start in starts:
  state=x[start,:3].copy();ap=float(x[start,5]);sr=float(state[0]);command_history=x[start-4:start+1,3:5].copy();imu_history=obs[start-8:start+1:2].copy();speed_history=x[start-8:start+1:2,0].copy();pose=np.zeros(3)
  for k in range(HORIZON):
   row=start+2*k;command=x[row,3:5]
   if k:command_history=np.vstack((command_history[1:],command))
   previous=command_history[-2,0];current=state.copy();ap,_=actuator_step(ap,command[0],command[1],state[0],c);sr,bax=longitudinal_actuator_step(sr,command[1],state[0],c);vx,vy,r=state;safe=max(abs(vx),.5);af=ap-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;fi=bf-fit['E_f']*(bf-np.arctan(bf));ri=br-fit['E_r']*(br-np.arctan(br));fyf=fzf*fit['D_f']*np.sin(fit['C_f']*np.arctan(fi));fyr=fzr*fit['D_r']*np.sin(fit['C_r']*np.arctan(ri));base_ay=(fyf*np.cos(ap)+fyr)/m;base_rd=(lf*fyf*np.cos(ap)-lr*fyr)/iz;base_next=np.array((vx+(bax+vy*r)*.04,vy+(base_ay-vx*r)*.04,r+base_rd*.04));base=np.r_[current,command,ap,command[0]-previous,base_next,command_history.ravel()];feature=np.r_[base,extras(flags,imu_history,command_history,speed_history)];res=network(net,mean,std,feature)*residual_gates(current[0],c);state=base_next+res*.04;total_ax=bax+res[0];total_ay=base_ay+res[1];imu_history=np.vstack((imu_history[1:],(total_ax,total_ay,state[2])));speed_history=np.r_[speed_history[1:],state[0]];yaw=pose[2];pose=np.array((pose[0]+c.position_speed_scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*.04,pose[1]+c.position_speed_scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*.04,yaw+state[2]*.04))
  truth=x[start+2:start+2*HORIZON+1:2,:3].copy()
  if 'source_teacher_vy' in d.files:truth[:,1]=d['source_teacher_vy'][start+2:start+2*HORIZON+1:2]
  gt=np.zeros(3)
  for q in truth:
   yaw=gt[2];gt=np.array((gt[0]+c.position_speed_scale*(q[0]*np.cos(yaw)-q[1]*np.sin(yaw))*.04,gt[1]+c.position_speed_scale*(q[0]*np.sin(yaw)+q[1]*np.cos(yaw))*.04,yaw+q[2]*.04))
  pe.append(np.linalg.norm(pose[:2]-gt[:2]));ye.append(abs((pose[2]-gt[2]+np.pi)%(2*np.pi)-np.pi));ve.append(abs(state[0]-truth[-1,0]));vye.append(abs(state[1]-truth[-1,1]));re.append(abs(state[2]-truth[-1,2]))
 def stats(v):return {'mean':float(np.mean(v)),'p95':float(np.quantile(v,.95)),'max':float(np.max(v))}
 return {'windows':len(starts),'trajectory_m':stats(pe),'yaw_rad':stats(ye),'vx_mps':stats(ve),'vy_mps':stats(vye),'yaw_rate_rps':stats(re)}

def main():
 torch.manual_seed(SEED);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');d=np.load(DATA);assert 'source_observations' in d.files;cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];OUT.mkdir(parents=True,exist_ok=True);report={}
 for name,flags in VARIANTS.items():
  print(f'[{name}] training',flush=True);net,mean,std,item=fit_variant(name,flags,d,dev);item['flags']=list(flags);item['validation_rollout']=rollout(net,mean,std,flags,d,cfg,fit,1);item['test_rollout']=rollout(net,mean,std,flags,d,cfg,fit,2);report[name]=item;print(json.dumps(item['test_rollout']),flush=True)
  torch.save({'state_dict':net.state_dict(),'mean':mean,'std':std,'flags':flags},OUT/f'{name}.pt')
 ranking=sorted(report,key=lambda q:(report[q]['validation_rollout']['trajectory_m']['mean']+.5*report[q]['validation_rollout']['trajectory_m']['p95']+.25*report[q]['validation_rollout']['yaw_rad']['p95']))
 result={'causal_rollout_contract':'measured history only at k=0; predicted IMU/speed history thereafter; candidate commands known','ranking':ranking,'variants':report};(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'ranking':ranking,'output':str(OUT/'results.json')},indent=2))
if __name__=='__main__':main()
