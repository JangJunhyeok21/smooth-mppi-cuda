#!/usr/bin/env python3
"""Fully recursive multi-horizon fine-tuning for the 40 ms dynamic residual."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch,yaml
from torch import nn
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from train_dynamic_40ms import Net
DATA=ROOT/'model_tuning/data/dynamic_40ms_residual.npz';PARAMS=ROOT/'model_tuning/results/dynamic_40ms_regression/params.json';HORIZONS=(5,10,20,30)
# User-editable defaults. Run once for the canonical recursive fine-tuning stage.
INITIAL_MODEL_PATH=ROOT/'model_tuning/results/dynamic_40ms_residual_seed31'
OUTPUT_PATH=ROOT/'model_tuning/results/dynamic_40ms_recursive_seed31'
EPOCHS=100
SEED=31
STATE_WEIGHTS=(1.0,2.0,6.0)
POSITION_LOSS_WEIGHT=2.0
# The memoryless MLP has no elapsed-horizon feature and does not predict yaw.
# Supervise the predicted yaw-rate directly; pose position remains a rollout loss.
HORIZON_YAW_LOSS_WEIGHT=0.0
DENSE_YAW_RATE_LOSS_WEIGHT=4.0
DENSE_YAW_LOSS_WEIGHT=0.0
EARLY_YAW_EXTRA_WEIGHT=3.0
EARLY_YAW_DECAY_SECONDS=0.20
def load_norm(path):z=np.fromfile(path,dtype='<f4');return z[-40:-20].copy(),z[-20:].copy()
def starts(x,b,s,v,split):return np.array([i for i in range(10,len(x)-60) if s[i]==split and s[i+60]==split and v[i:i+61].all() and np.all(b[i:i+61]==b[i]) and np.max(np.linalg.norm(x[i+1:i+61,:3]-x[i:i+60,:3],axis=1))<2])
def main():
 p=argparse.ArgumentParser();p.add_argument('initial',nargs='?',default=str(INITIAL_MODEL_PATH));p.add_argument('--out',default=str(OUTPUT_PATH));p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--seed',type=int,default=SEED);a=p.parse_args();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');d=np.load(DATA);x=d['source_features'].astype(np.float32);b=d['source_bag_id'];sp=d['source_split'];v=d['source_valid'];tr=starts(x,b,sp,v,0);va=starts(x,b,sp,v,1);va=va[::max(1,len(va)//256)][:256];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];mean,std=load_norm(Path(a.initial)/'dynamic_40ms_residual.bin');net=Net();net.load_state_dict(torch.load(Path(a.initial)/'model.pt',map_location='cpu',weights_only=True));net.to(dev);X=torch.from_numpy(x).to(dev);mean=torch.from_numpy(mean).to(dev);std=torch.from_numpy(std).to(dev);lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;scale=float(cfg['kinematic_steer_scale']);bias=float(cfg['kinematic_steer_bias']);stau=float(cfg['steer_servo_time_constant']);srate=float(cfg['actuator_max_steer_rate']);kp=float(cfg['speed_servo_kp']);ata=float(cfg['speed_reference_accel_time_constant']);atb=float(cfg['speed_reference_brake_time_constant']);rrate=float(cfg['actuator_max_speed_reference_rate']);amin=float(cfg['min_accel']);amax=float(cfg['max_accel']);pscale=float(cfg['kinematic_position_speed_scale']);low=float(cfg['dynamic_mlp_min_speed'])
 def rollout(ids):
  q=torch.as_tensor(ids,device=dev);state=X[q,:3];ap=X[q,5];sr=state[:,0];hist=X[q,10:20].reshape(-1,5,2);pose=torch.zeros((len(ids),3),device=dev);outs={};resall=[];state_trace=[];pose_trace=[]
  for k in range(30):
   i=q+2*k;c0=X[i,3:5]
   if k>0:hist=torch.cat((hist[:,1:],c0[:,None]),1)
   current=state;previous=hist[:,-2,0]
   cmd=c0;target=torch.clamp(scale*cmd[:,0]+bias,-.55,.55);ap=torch.clamp(ap+torch.clamp((target-ap)/stau,-srate,srate)*.04,-.55,.55);tau=torch.where(cmd[:,1]>=sr,ata,atb);sr=sr+torch.clamp((cmd[:,1]-sr)/tau,-rrate,rrate)*.04;vx,vy,r=state.unbind(1);bax=torch.clamp(kp*(sr-torch.hypot(vx,vy)),amin,amax);safe=torch.clamp(torch.abs(vx),min=.5);af=ap-torch.atan2(vy+lf*r,safe);ar=-torch.atan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;fyf=fzf*fit['D_f']*torch.sin(fit['C_f']*torch.atan(bf));fyr=fzr*fit['D_r']*torch.sin(fit['C_r']*torch.atan(br));ay=(fyf*torch.cos(ap)+fyr)/m;rd=(lf*fyf*torch.cos(ap)-lr*fyr)/iz;state=torch.stack((vx+(bax+vy*r)*.04,vy+(ay-vx*r)*.04,r+rd*.04),1)
   feat=torch.cat((current,c0,ap[:,None],(c0[:,0]-previous)[:,None],state,hist.reshape(len(ids),-1)),1);res=torch.clamp(net((feat-mean)/std),torch.tensor((-8.,-8.,-30.),device=dev),torch.tensor((8.,8.,30.),device=dev));gate=torch.sigmoid((torch.abs(current[:,0])-low)/.2);state=state+res*gate[:,None]*.04;yaw=pose[:,2];pose=torch.stack((pose[:,0]+pscale*(state[:,0]*torch.cos(yaw)-state[:,1]*torch.sin(yaw))*.04,pose[:,1]+pscale*(state[:,0]*torch.sin(yaw)+state[:,1]*torch.cos(yaw))*.04,yaw+state[:,2]*.04),1);resall.append(res)
   state_trace.append(state);pose_trace.append(pose)
   if k+1 in HORIZONS:outs[k+1]=(state,pose,X[q+2*(k+1),:3])
  return outs,torch.stack(resall,1),torch.stack(state_trace,1),torch.stack(pose_trace,1)
 def loss(ids):
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
  dense_yaw_rate=DENSE_YAW_RATE_LOSS_WEIGHT*(yaw_rate_error*early_weight).mean()
  return total+dense_yaw_rate+1e-4*(res*res).mean()
 opt=torch.optim.AdamW(net.parameters(),1e-4,weight_decay=1e-5);best=(1e99,None,0);stale=0
 for epoch in range(a.epochs):
  net.train();vals=[]
  for _ in range(24):q=rng.choice(tr,min(48,len(tr)),replace=False);z=loss(q);opt.zero_grad();z.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();vals.append(float(z))
  net.eval()
  with torch.no_grad():score=float(np.mean([float(loss(q)) for q in np.array_split(va,max(1,len(va)//64))]))
  print(f'epoch={epoch+1} train={np.mean(vals):.6f} val={score:.6f}',flush=True)
  if score<best[0]-1e-5:best=(score,{k:z.detach().cpu().clone() for k,z in net.state_dict().items()},epoch+1);stale=0
  else:stale+=1
  if stale>=18:break
 net.load_state_dict(best[1]);net.cpu();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),out/'model.pt');layers=(net.net[0],net.net[2],net.net[4]);blob=np.concatenate([z.detach().numpy().ravel() for layer in layers for z in (layer.weight,layer.bias)]+[mean.cpu().numpy(),std.cpu().numpy()]).astype('<f4');blob.tofile(out/'dynamic_40ms_residual.bin');(out/'metrics.json').write_text(json.dumps({'seed':a.seed,'best_epoch':best[2],'best_recursive_validation_loss':best[0],'train_starts':len(tr),'validation_starts':len(va)},indent=2)+'\n')
if __name__=='__main__':main()
