#!/usr/bin/env python3
"""Recursive multi-horizon fine-tuning implementation for unified Step 6."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np,torch,yaml
from torch import nn
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from residual_mlp_training import Net
DATA=Path(os.environ.get('DYNAMIC_RESIDUAL_DATA',ROOT/'model_tuning/data/dynamic_40ms_residual.npz'));PARAMS=Path(os.environ.get('DYNAMIC_CLASSIC_PARAMS',ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'));HORIZONS=(5,10,20,30)
# User-editable defaults. Run once for the canonical recursive fine-tuning stage.
INITIAL_MODEL_PATH=ROOT/'model_tuning/results/dynamic_40ms_residual_seed31'
OUTPUT_PATH=ROOT/'model_tuning/results/dynamic_40ms_recursive_seed31'
EPOCHS=100
SEED=31
STATE_WEIGHTS=(3.0,6.0,8.0)
POSITION_LOSS_WEIGHT=3.0
POSITION_ENDPOINT_WEIGHT=4.0
POSITION_TAIL_WEIGHT=8.0
DENSE_VX_LOSS_WEIGHT=2.0
DENSE_VY_LOSS_WEIGHT=6.0
VY_ENDPOINT_WEIGHT=3.0
VY_TAIL_WEIGHT=3.0
VX_TAIL_WEIGHT=2.0
# The memoryless MLP has no elapsed-horizon feature and does not predict yaw.
# Supervise the predicted yaw-rate directly; pose position remains a rollout loss.
HORIZON_YAW_LOSS_WEIGHT=2.0
DENSE_YAW_RATE_LOSS_WEIGHT=6.0
DENSE_YAW_LOSS_WEIGHT=1.0
EARLY_YAW_EXTRA_WEIGHT=3.0
EARLY_YAW_DECAY_SECONDS=0.20
TAIL_CVAR_FRACTION=0.15
TAIL_CVAR_WEIGHT=1.0
HIGH_SPEED_SAMPLE_WEIGHT=1.5
YAW_RECOVERY_SAMPLE_WEIGHT=2.0
OVERSTEER_SAMPLE_WEIGHT=3.0
HARD_BRAKING_SAMPLE_WEIGHT=3.0
OVERSTEER_EXCESS_RPS=0.35
OVERSTEER_UNDERPREDICTION_WEIGHT=1.5
# Compatibility switch used to reconstruct the high-speed yaw checkpoint
# before fitting a separate low-speed longitudinal head.
GATE_AX_RESIDUAL=os.environ.get('GATE_AX_RESIDUAL','0')=='1'
USE_AX_RESIDUAL=os.environ.get('USE_AX_RESIDUAL','0')=='1'
BATCHES_PER_EPOCH=max(1,int(os.environ.get('RECURSIVE_BATCHES_PER_EPOCH','32')))
def load_norm(path):z=np.fromfile(path,dtype='<f4');return z[-44:-22].copy(),z[-22:].copy()
def starts(x,b,s,v,split):return np.array([i for i in range(10,len(x)-60) if s[i]==split and s[i+60]==split and v[i:i+61].all() and np.all(b[i:i+61]==b[i]) and np.max(np.linalg.norm(x[i+1:i+61,:3]-x[i:i+60,:3],axis=1))<2])
def main():
 p=argparse.ArgumentParser();p.add_argument('initial',nargs='?',default=str(INITIAL_MODEL_PATH));p.add_argument('--out',default=str(OUTPUT_PATH));p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--seed',type=int,default=SEED);p.add_argument('--lambda-pose',type=float,default=.5);p.add_argument('--lambda-gyro',type=float,default=.03);p.add_argument('--lambda-residual',type=float,default=1e-4);a=p.parse_args();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');d=np.load(DATA);x=d['source_features'].astype(np.float32);obs=d['source_observations'].astype(np.float32);mcl=d['source_mcl_pose'].astype(np.float32) if 'source_mcl_pose' in d.files else None;source_sr=d['source_speed_reference'].astype(np.float32) if 'source_speed_reference' in d.files else x[:,0].copy();b=d['source_bag_id'];sp=d['source_split'];v=d['source_valid'];tr=starts(x,b,sp,v,0);va=starts(x,b,sp,v,1);va=va[::max(1,len(va)//256)][:256];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];mlp_cfg=yaml.safe_load((ROOT/'config/MLP_params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];mean,std=load_norm(Path(a.initial)/'dynamic_40ms_residual.bin');net=Net();net.load_state_dict(torch.load(Path(a.initial)/'model.pt',map_location='cpu',weights_only=True));net.to(dev);X=torch.from_numpy(x).to(dev);OBS=torch.from_numpy(obs).to(dev);MCL=torch.from_numpy(mcl).to(dev) if mcl is not None else None;SOURCE_SR=torch.from_numpy(source_sr).to(dev);mean=torch.from_numpy(mean).to(dev);std=torch.from_numpy(std).to(dev);lf,lr,m=[float(cfg[q]) for q in ('l_f','l_r','mass')];iz=float(fit.get('I_z',cfg['dynamic_mlp_I_z']));wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;scale=float(cfg['kinematic_steer_scale']);bias=float(cfg['kinematic_steer_bias']);stau=float(cfg['steer_servo_time_constant']);srate=float(cfg['actuator_max_steer_rate']);kp=float(cfg['speed_servo_kp']);ata=float(cfg['speed_reference_accel_time_constant']);atb=float(cfg['speed_reference_brake_time_constant']);rrate=float(cfg['actuator_max_speed_reference_rate']);amin=float(cfg['min_accel']);amax=float(cfg['max_accel']);pscale=float(cfg['kinematic_position_speed_scale']);res_limit=torch.tensor((mlp_cfg['mlp_max_residual_ax'],mlp_cfg['mlp_max_residual_ay'],mlp_cfg['mlp_max_residual_yaw_accel']),device=dev)
 # S_v is intentionally absent from training: integrate measured m/s directly.
 pscale=1.0
 def rollout(ids):
  q=torch.as_tensor(ids,device=dev);state=X[q,:3];acc=OBS[q,:2];ap=X[q,5];sr=SOURCE_SR[q];hist=X[q,10:20].reshape(-1,5,2);pose=torch.zeros((len(ids),3),device=dev);outs={};resall=[];state_trace=[];pose_trace=[]
  for k in range(30):
   i=q+2*k;c0=X[i,3:5]
   if k>0:hist=torch.cat((hist[:,1:],c0[:,None]),1)
   current=state;previous=hist[:,-2,0]
   cmd=c0;target=torch.clamp(scale*cmd[:,0]+bias,-.55,.55);ap=torch.clamp(ap+torch.clamp((target-ap)/stau,-srate,srate)*.04,-.55,.55);tau=torch.where(cmd[:,1]>=sr,ata,atb);sr=sr+torch.clamp((cmd[:,1]-sr)/tau,-rrate,rrate)*.04;vx,vy,r=state.unbind(1);bax=torch.clamp(kp*(sr-vx),amin,amax);safe=torch.clamp(torch.abs(vx),min=.5);af=ap-torch.atan2(vy+lf*r,safe);ar=-torch.atan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;front_inner=bf-fit['E_f']*(bf-torch.atan(bf));rear_inner=br-fit['E_r']*(br-torch.atan(br));fyf=fzf*fit['D_f']*torch.sin(fit['C_f']*torch.atan(front_inner));fyr=fzr*fit['D_r']*torch.sin(fit['C_r']*torch.atan(rear_inner));ay=(fyf*torch.cos(ap)+fyr)/m;rd=(lf*fyf*torch.cos(ap)-lr*fyr)/iz;state=torch.stack((vx+(bax+vy*r)*.04,vy+(ay-vx*r)*.04,r+rd*.04),1)
   feat=torch.cat((current,c0,ap[:,None],(c0[:,0]-previous)[:,None],state,hist.reshape(len(ids),-1),acc),1);res=torch.clamp(net((feat-mean)/std),-res_limit,res_limit);state=state+res*.04;acc=torch.stack((bax+res[:,0],ay+res[:,1]),1);yaw=pose[:,2];pose=torch.stack((pose[:,0]+pscale*(state[:,0]*torch.cos(yaw)-state[:,1]*torch.sin(yaw))*.04,pose[:,1]+pscale*(state[:,0]*torch.sin(yaw)+state[:,1]*torch.cos(yaw))*.04,yaw+state[:,2]*.04),1);resall.append(res)
   state_trace.append(state);pose_trace.append(pose)
   if k+1 in HORIZONS:outs[k+1]=(state,pose,X[q+2*(k+1),:3])
  return outs,torch.stack(resall,1),torch.stack(state_trace,1),torch.stack(pose_trace,1)
 def loss(ids,return_per_window=False):
  out,res,predicted_states,predicted_poses=rollout(ids);total=0
  for h,w in zip(HORIZONS,(.5,.8,1.3,2.)):
   state,pose,gt=out[h];se=nn.functional.smooth_l1_loss(state,gt,reduction='none')*torch.tensor(STATE_WEIGHTS,device=dev);gp=torch.zeros_like(pose);gstate=X[torch.as_tensor(ids,device=dev)[:,None]+2*torch.arange(1,h+1,device=dev)[None],:3]
   for k in range(h):
    yaw=gp[:,2];q=gstate[:,k];gp=torch.stack((gp[:,0]+pscale*(q[:,0]*torch.cos(yaw)-q[:,1]*torch.sin(yaw))*.04,gp[:,1]+pscale*(q[:,0]*torch.sin(yaw)+q[:,1]*torch.cos(yaw))*.04,yaw+q[:,2]*.04),1)
   total=total+w*(se.mean()+POSITION_LOSS_WEIGHT*nn.functional.smooth_l1_loss(pose[:,:2],gp[:,:2])+HORIZON_YAW_LOSS_WEIGHT*nn.functional.smooth_l1_loss(pose[:,2],gp[:,2]))
  ids_tensor=torch.as_tensor(ids,device=dev);offsets=2*torch.arange(1,31,device=dev)
  gt_states=X[ids_tensor[:,None]+offsets[None],:3]
  times=.04*torch.arange(1,31,device=dev,dtype=predicted_states.dtype);early_weight=1.0+EARLY_YAW_EXTRA_WEIGHT*torch.exp(-times/EARLY_YAW_DECAY_SECONDS)
  yaw_rate_error=nn.functional.smooth_l1_loss(predicted_states[:,:,2],gt_states[:,:,2],reduction='none')
  yaw_window=(yaw_rate_error*early_weight).mean(1)
  dense_yaw_rate=DENSE_YAW_RATE_LOSS_WEIGHT*yaw_window.mean()
  gt_pose=torch.zeros((len(ids),3),device=dev)
  gt_pose_trace=[]
  for k in range(30):
   yaw=gt_pose[:,2];q=gt_states[:,k];gt_pose=torch.stack((gt_pose[:,0]+pscale*(q[:,0]*torch.cos(yaw)-q[:,1]*torch.sin(yaw))*.04,gt_pose[:,1]+pscale*(q[:,0]*torch.sin(yaw)+q[:,1]*torch.cos(yaw))*.04,yaw+q[:,2]*.04),1);gt_pose_trace.append(gt_pose)
  gt_poses=torch.stack(gt_pose_trace,1)
  if MCL is not None:
   absolute=MCL[ids_tensor[:,None]+offsets[None]];initial=MCL[ids_tensor];dx=absolute[:,:,0]-initial[:,None,0];dy=absolute[:,:,1]-initial[:,None,1];heading=initial[:,None,2];mcl_poses=torch.stack((dx*torch.cos(heading)+dy*torch.sin(heading),-dx*torch.sin(heading)+dy*torch.cos(heading),torch.atan2(torch.sin(absolute[:,:,2]-heading),torch.cos(absolute[:,:,2]-heading))),2)
  else:mcl_poses=gt_poses
  gyro=OBS[ids_tensor[:,None]+offsets[None],2]
  observation_pose_loss=nn.functional.smooth_l1_loss(predicted_poses,mcl_poses)
  gyro_loss=nn.functional.smooth_l1_loss(predicted_states[:,:,2],gyro)
  endpoint_position=torch.linalg.vector_norm(predicted_poses[:,-1,:2]-gt_poses[:,-1,:2],dim=1)
  endpoint_vx=torch.abs(predicted_states[:,-1,0]-gt_states[:,-1,0])
  endpoint_vy=torch.abs(predicted_states[:,-1,1]-gt_states[:,-1,1])
  dense_vx=nn.functional.smooth_l1_loss(predicted_states[:,:,0],gt_states[:,:,0])
  dense_vy=nn.functional.smooth_l1_loss(predicted_states[:,:,1],gt_states[:,:,1])
  dense_position=nn.functional.smooth_l1_loss(predicted_poses[:,:,:2],gt_poses[:,:,:2])
  dense_yaw=nn.functional.smooth_l1_loss(predicted_poses[:,:,2],gt_poses[:,:,2])
  count=max(1,int(np.ceil(len(ids)*TAIL_CVAR_FRACTION)))
  trajectory_tail=torch.topk(endpoint_position,count).values.mean()
  vx_tail=torch.topk(endpoint_vx,count).values.mean()
  vy_tail=torch.topk(endpoint_vy,count).values.mean()
  if TAIL_CVAR_FRACTION>0:
   dense_yaw_rate=dense_yaw_rate+TAIL_CVAR_WEIGHT*torch.topk(yaw_window,count).values.mean()
  expected_r=gt_states[:,:,0]*torch.tan(X[ids_tensor[:,None]+offsets[None],5])/(lf+lr)
  oversteer_event=(gt_states[:,:,0]>1.5)&(torch.abs(X[ids_tensor[:,None]+offsets[None],5])>.04)&((torch.abs(gt_states[:,:,2])-torch.abs(expected_r))>OVERSTEER_EXCESS_RPS)
  yaw_peak_under=torch.relu(torch.abs(gt_states[:,:,2])-torch.abs(predicted_states[:,:,2]))
  oversteer_under=(yaw_peak_under.square()*oversteer_event).sum()/torch.clamp(oversteer_event.sum(),min=1)
  value=(total+dense_yaw_rate+DENSE_VX_LOSS_WEIGHT*dense_vx
         +DENSE_VY_LOSS_WEIGHT*dense_vy
         +POSITION_LOSS_WEIGHT*dense_position
         +DENSE_YAW_LOSS_WEIGHT*dense_yaw
         +POSITION_ENDPOINT_WEIGHT*endpoint_position.mean()
         +POSITION_TAIL_WEIGHT*trajectory_tail+VX_TAIL_WEIGHT*vx_tail
         +VY_ENDPOINT_WEIGHT*endpoint_vy.mean()+VY_TAIL_WEIGHT*vy_tail
         +OVERSTEER_UNDERPREDICTION_WEIGHT*oversteer_under
         +a.lambda_pose*observation_pose_loss+a.lambda_gyro*gyro_loss
         +a.lambda_residual*(res*res).mean())
  # Checkpoint selection must include lateral-state failure. The previous
  # selector optimized position/vx/yaw only and therefore accepted a model
  # whose trajectory improved while vy P95 doubled.
  selection_window=endpoint_position+.35*endpoint_vx+.30*endpoint_vy+.15*yaw_window
  return (value,selection_window) if return_per_window else value
 opt=torch.optim.AdamW(net.parameters(),5e-5,weight_decay=1e-5);best=(1e99,None,0);stale=0
 # Oversample the exact operating regime that caused the Map1 tail failures:
 # high speed with yaw-rate decay/sign reversal during the following 1.2 s.
 future_r=x[tr[:,None]+2*np.arange(1,31)[None],2]
 initial_r=x[tr,2]
 recovery=(np.abs(initial_r)>.45)&((np.sign(future_r[:,-1])!=np.sign(initial_r))|(np.abs(future_r[:,-1])<.35*np.abs(initial_r)))
 future_rows=tr[:,None]+2*np.arange(1,31)[None]
 expected_r=x[future_rows,0]*np.tan(x[future_rows,5])/(lf+lr)
 oversteer=np.any((x[future_rows,0]>1.5)&(np.abs(x[future_rows,5])>.04)&
                  ((np.abs(future_r)-np.abs(expected_r))>OVERSTEER_EXCESS_RPS),axis=1)
 hard_braking=np.min(np.diff(x[tr[:,None]+2*np.arange(31)[None],0],axis=1)/.04,axis=1)<-2.0
 sample_probability=np.ones(len(tr));sample_probability*=np.where(x[tr,0]>=3.0,HIGH_SPEED_SAMPLE_WEIGHT,1.0);sample_probability*=np.where(recovery,YAW_RECOVERY_SAMPLE_WEIGHT,1.0);sample_probability*=np.where(oversteer,OVERSTEER_SAMPLE_WEIGHT,1.0);sample_probability*=np.where(hard_braking,HARD_BRAKING_SAMPLE_WEIGHT,1.0);sample_probability/=sample_probability.sum()
 for epoch in range(a.epochs):
  net.train();vals=[]
  for _ in range(BATCHES_PER_EPOCH):q=rng.choice(tr,min(64,len(tr)),replace=True,p=sample_probability);z=loss(q);opt.zero_grad();z.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();vals.append(float(z.detach()))
  net.eval()
  with torch.no_grad():
   validation_window=[]
   for q in np.array_split(va,max(1,len(va)//64)):
    _,per_window=loss(q,True);validation_window.extend(per_window.cpu().numpy())
   validation_window=np.asarray(validation_window);score=float(validation_window.mean()+2*np.quantile(validation_window,.95))
  print(f'epoch={epoch+1} train={np.mean(vals):.6f} val={score:.6f}',flush=True)
  if score<best[0]-1e-5:best=(score,{k:z.detach().cpu().clone() for k,z in net.state_dict().items()},epoch+1);stale=0
  else:stale+=1
  if stale>=18:break
 net.load_state_dict(best[1]);net.cpu();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),out/'model.pt');layers=(net.net[0],net.net[2],net.net[4]);blob=np.concatenate([z.detach().numpy().ravel() for layer in layers for z in (layer.weight,layer.bias)]+[mean.cpu().numpy(),std.cpu().numpy()]).astype('<f4');assert len(blob)==3695;blob.tofile(out/'dynamic_40ms_residual.bin');(out/'metrics.json').write_text(json.dumps({'seed':a.seed,'input_features':22,'supervision':'recursive open-loop KF state + actual MCL pose + weak raw gyro; classic frozen','lambda_pose':a.lambda_pose,'lambda_gyro':a.lambda_gyro,'lambda_residual':a.lambda_residual,'uses_causal_imu_ax_ay':True,'best_epoch':best[2],'batches_per_epoch':BATCHES_PER_EPOCH,'best_recursive_validation_tail_score':best[0],'train_starts':len(tr),'validation_starts':len(va),'high_speed_train_windows':int(np.sum(x[tr,0]>=3.0)),'yaw_recovery_train_windows':int(recovery.sum()),'oversteer_train_windows':int(oversteer.sum()),'hard_braking_train_windows':int(hard_braking.sum()),'oversteer_sample_weight':OVERSTEER_SAMPLE_WEIGHT,'hard_braking_sample_weight':HARD_BRAKING_SAMPLE_WEIGHT,'tail_cvar_fraction':TAIL_CVAR_FRACTION,'tail_cvar_weight':TAIL_CVAR_WEIGHT,'position_endpoint_weight':POSITION_ENDPOINT_WEIGHT,'position_tail_weight':POSITION_TAIL_WEIGHT,'dense_vx_loss_weight':DENSE_VX_LOSS_WEIGHT,'dense_vy_loss_weight':DENSE_VY_LOSS_WEIGHT,'vy_endpoint_weight':VY_ENDPOINT_WEIGHT,'vy_tail_weight':VY_TAIL_WEIGHT,'vx_tail_weight':VX_TAIL_WEIGHT},indent=2)+'\n')
if __name__=='__main__':main()
