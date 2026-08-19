#!/usr/bin/env python3
"""Recursive fine-tuning for the causal vx_delta_history_24d residual MLP."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np,torch,yaml
from torch import nn

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment_residual_history_features import Net

DATA=Path(os.environ.get('DYNAMIC_RESIDUAL_DATA',ROOT/'model_tuning/data/dynamic_40ms_residual.npz'));PARAMS=Path(os.environ.get('DYNAMIC_CLASSIC_PARAMS',ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'))
INITIAL=ROOT/'model_tuning/results/residual_history_ablation/vx_delta_history_24d.pt'
OUTPUT=ROOT/'model_tuning/results/vx_delta_history_24d_stage1'
EPOCHS=100;SEED=31;HORIZONS=(5,10,20,30)

def starts(x,b,s,v,split):return np.asarray([i for i in range(10,len(x)-60) if s[i]==split and s[i+60]==split and v[i-8:i+61].all() and np.all(b[i-8:i+61]==b[i])])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('initial',nargs='?',default=str(INITIAL));parser.add_argument('--out',default=str(OUTPUT));parser.add_argument('--epochs',type=int,default=EPOCHS);parser.add_argument('--seed',type=int,default=SEED);parser.add_argument('--input-dim',type=int,choices=(20,24),default=24);args=parser.parse_args();torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');d=np.load(DATA);x=d['source_features'].astype(np.float32);b=d['source_bag_id'];sp=d['source_split'];valid=d['source_valid'];train=starts(x,b,sp,valid,0);validation=starts(x,b,sp,valid,1);validation=validation[::max(1,len(validation)//256)][:256];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];initial=Path(args.initial);checkpoint=torch.load(initial,map_location='cpu',weights_only=False)
 if isinstance(checkpoint,dict) and 'state_dict' in checkpoint:
  state_dict=checkpoint['state_dict'];mean=np.asarray(checkpoint['mean']);std=np.asarray(checkpoint['std'])
 else:
  state_dict=checkpoint;raw=np.fromfile(initial.with_name('dynamic_40ms_residual.bin'),dtype='<f4');mean=raw[-40:-20];std=raw[-20:]
 mean=torch.as_tensor(mean,device=dev);std=torch.as_tensor(std,device=dev);net=Net(args.input_dim);net.load_state_dict(state_dict);net.to(dev);X=torch.from_numpy(x).to(dev);lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;scale=float(cfg['kinematic_steer_scale']);bias=float(cfg['kinematic_steer_bias']);steer_tau=float(cfg['steer_servo_time_constant']);steer_rate=float(cfg['actuator_max_steer_rate']);kp=float(cfg['speed_servo_kp']);accel_tau=float(cfg['speed_reference_accel_time_constant']);brake_tau=float(cfg['speed_reference_brake_time_constant']);reference_rate=float(cfg['actuator_max_speed_reference_rate']);min_accel=float(cfg['min_accel']);max_accel=float(cfg['max_accel']);position_scale=float(cfg['kinematic_position_speed_scale']);low_speed=float(cfg['dynamic_mlp_min_speed']);teacher=d['source_teacher_vy'] if 'source_teacher_vy' in d.files else x[:,1];difficulty=np.asarray([np.max(np.abs(x[i:i+60:2,2]))+2*np.mean(np.abs(x[i:i+60:2,1]-teacher[i:i+60:2])) for i in train]);train_probability=1.+3.*difficulty/np.maximum(np.quantile(difficulty,.9),1e-4);train_probability/=train_probability.sum()
 def rollout(ids):
  q=torch.as_tensor(ids,device=dev);state=X[q,:3];applied=X[q,5];speed_reference=state[:,0];command_history=X[q,10:20].reshape(-1,5,2);speed_history=X[q[:,None]+2*torch.arange(-4,1,device=dev)[None],0];pose=torch.zeros((len(ids),3),device=dev);states=[];poses=[];accelerations=[]
  for k in range(30):
   row=q+2*k;command=X[row,3:5]
   if k:command_history=torch.cat((command_history[:,1:],command[:,None]),1)
   current=state;previous=command_history[:,-2,0];target=torch.clamp(scale*command[:,0]+bias,-.55,.55);applied=torch.clamp(applied+torch.clamp((target-applied)/steer_tau,-steer_rate,steer_rate)*.04,-.55,.55);tau=torch.where(command[:,1]>=speed_reference,accel_tau,brake_tau);speed_reference=speed_reference+torch.clamp((command[:,1]-speed_reference)/tau,-reference_rate,reference_rate)*.04;vx,vy,yaw_rate=state.unbind(1);base_ax=torch.clamp(kp*(speed_reference-vx),min_accel,max_accel);safe=torch.clamp(torch.abs(vx),min=.5);alpha_front=applied-torch.atan2(vy+lf*yaw_rate,safe);alpha_rear=-torch.atan2(vy-lr*yaw_rate,safe);front=fit['B_f']*alpha_front;rear=fit['B_r']*alpha_rear;front_inner=front-fit['E_f']*(front-torch.atan(front));rear_inner=rear-fit['E_r']*(rear-torch.atan(rear));fy_front=fzf*fit['D_f']*torch.sin(fit['C_f']*torch.atan(front_inner));fy_rear=fzr*fit['D_r']*torch.sin(fit['C_r']*torch.atan(rear_inner));base_ay=(fy_front*torch.cos(applied)+fy_rear)/m;base_yaw_accel=(lf*fy_front*torch.cos(applied)-lr*fy_rear)/iz;base_next=torch.stack((vx+(base_ax+vy*yaw_rate)*.04,vy+(base_ay-vx*yaw_rate)*.04,yaw_rate+base_yaw_accel*.04),1);base_feature=torch.cat((current,command,applied[:,None],(command[:,0]-previous)[:,None],base_next,command_history.reshape(len(ids),-1)),1);feature=torch.cat((base_feature,torch.diff(speed_history,dim=1)),1) if args.input_dim==24 else base_feature;residual=torch.clamp(net((feature-mean)/std),torch.tensor((-8.,-8.,-30.),device=dev),torch.tensor((8.,8.,30.),device=dev));gate=torch.sigmoid((torch.abs(current[:,0])-low_speed)/.2);residual=residual*torch.stack((torch.ones_like(gate),gate,gate),1);state=base_next+residual*.04;speed_history=torch.cat((speed_history[:,1:],state[:,0,None]),1);yaw=pose[:,2];pose=torch.stack((pose[:,0]+position_scale*(state[:,0]*torch.cos(yaw)-state[:,1]*torch.sin(yaw))*.04,pose[:,1]+position_scale*(state[:,0]*torch.sin(yaw)+state[:,1]*torch.cos(yaw))*.04,yaw+state[:,2]*.04),1);states.append(state);poses.append(pose);accelerations.append(torch.stack((base_ax+residual[:,0],base_ay+residual[:,1]),1))
  return torch.stack(states,1),torch.stack(poses,1),torch.stack(accelerations,1)
 def loss(ids):
  predicted_state,predicted_pose,predicted_acceleration=rollout(ids);q=torch.as_tensor(ids,device=dev);truth=X[q[:,None]+2*torch.arange(1,31,device=dev)[None],:3].clone()
  teacher_rows=q[:,None]+2*torch.arange(1,31,device=dev)[None]
  if 'source_teacher_vy' in d.files:truth[:,:,1]=torch.from_numpy(d['source_teacher_vy']).to(dev)[teacher_rows]
  truth_pose=torch.zeros_like(predicted_pose);current=torch.zeros((len(ids),3),device=dev)
  for k in range(30):
   value=truth[:,k];yaw=current[:,2];current=torch.stack((current[:,0]+position_scale*(value[:,0]*torch.cos(yaw)-value[:,1]*torch.sin(yaw))*.04,current[:,1]+position_scale*(value[:,0]*torch.sin(yaw)+value[:,1]*torch.cos(yaw))*.04,yaw+value[:,2]*.04),1);truth_pose[:,k]=current
  total=0.
  for h,w in zip(HORIZONS,(.5,.8,1.3,2.)):
   state_error=nn.functional.smooth_l1_loss(predicted_state[:,h-1],truth[:,h-1],reduction='none')*torch.tensor((1.,2.,6.),device=dev)
   if 'source_teacher_vy_confidence' in d.files:state_error[:,1]*=torch.from_numpy(d['source_teacher_vy_confidence']).to(dev)[teacher_rows[:,h-1]]
   total=total+w*(state_error.mean()+2.*nn.functional.smooth_l1_loss(predicted_pose[:,h-1,:2],truth_pose[:,h-1,:2]))
  time=.04*torch.arange(1,31,device=dev);yaw_weight=1.+3.*torch.exp(-time/.2);dense_yaw=nn.functional.smooth_l1_loss(predicted_state[:,:,2],truth[:,:,2],reduction='none')*yaw_weight;per_window_yaw=dense_yaw.mean(1);total=total+4.*dense_yaw.mean()+2.*torch.topk(per_window_yaw,max(1,len(ids)//5)).values.mean()
  if 'source_observations' in d.files:
   observations=torch.from_numpy(d['source_observations']).to(dev);gt_acceleration=observations[q[:,None]+2*torch.arange(30,device=dev)[None],:2];accel_error=nn.functional.smooth_l1_loss(predicted_acceleration,gt_acceleration,reduction='none');total=total+(accel_error*torch.tensor((.35,1.5),device=dev)).mean()
  endpoint_position_error=torch.linalg.vector_norm(predicted_pose[:,-1,:2]-truth_pose[:,-1,:2],dim=1);tail_count=max(1,len(ids)//5);total=total+1.5*torch.topk(endpoint_position_error,tail_count).values.mean();return total
 optimizer=torch.optim.AdamW(net.parameters(),1e-4,weight_decay=1e-5);best=(1e30,None,0);stale=0
 for epoch in range(args.epochs):
  net.train();values=[]
  for _ in range(32):
   ids=rng.choice(train,min(64,len(train)),replace=True,p=train_probability);value=loss(ids);optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);optimizer.step();values.append(float(value.detach()))
  net.eval()
  with torch.no_grad():score=float(np.mean([float(loss(ids)) for ids in np.array_split(validation,max(1,len(validation)//64))]))
  print(f'epoch={epoch+1} train={np.mean(values):.6f} val={score:.6f}',flush=True)
  if score<best[0]-1e-5:best=(score,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch+1);stale=0
  else:stale+=1
  if stale>=18:break
 net.load_state_dict(best[1]);net.cpu();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);name='vx_delta_history_24d' if args.input_dim==24 else 'command_history_20d';payload={'state_dict':net.state_dict(),'mean':mean.cpu().numpy(),'std':std.cpu().numpy(),'model_name':name,'flags':(('speed_deltas',) if args.input_dim==24 else ())};torch.save(payload,out/'model.pt');layers=(net.net[0],net.net[2],net.net[4]);blob=np.concatenate([v.detach().numpy().ravel() for layer in layers for v in (layer.weight,layer.bias)]+[payload['mean'],payload['std']]).astype('<f4');blob.tofile(out/(name+'.bin'));(out/'metrics.json').write_text(json.dumps({'model_name':name,'input_dim':args.input_dim,'best_epoch':best[2],'validation_recursive_loss':best[0],'binary_floats':len(blob),'binary_bytes':int(blob.nbytes)},indent=2)+'\n')
if __name__=='__main__':main()
