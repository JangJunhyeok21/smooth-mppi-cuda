#!/usr/bin/env python3
"""Train/export the one-step 20-64-32-3 residual MLP for unified Step 6."""
import argparse,json,os,sys
from dataclasses import replace
from pathlib import Path
import numpy as np,torch,yaml
from torch import nn
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from contract import ClassicModelParameters,IMU_RESIDUAL_FEATURES,Contract,actuator_step,longitudinal_actuator_step
from callback_training_data import load_callback_archives
# User-editable defaults. The script runs without command-line arguments.
ROOT=HERE.parents[1]
DATASET_PATH=ROOT/'model_tuning/data/ifac0810_0819_autonomous_physics_clean'
OUTPUT_PATH=ROOT/'model_tuning/results/dynamic_40ms_residual_seed31'
EPOCHS=300
SEED=31
DEVICE='cuda'
class Net(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(len(IMU_RESIDUAL_FEATURES),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
 def forward(self,x):return self.net(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',default=str(DATASET_PATH),help='Step-1 bag NPZ directory');p.add_argument('--out',default=str(OUTPUT_PATH));p.add_argument('--initialize-from',help='previous iteration model.pt');p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--seed',type=int,default=SEED);p.add_argument('--horizon-steps',type=int,default=30);p.add_argument('--device',default=DEVICE);a=p.parse_args();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
 params=ClassicModelParameters.from_yaml(ROOT/'config/params.yaml');classic_path=Path(os.environ.get('DYNAMIC_CLASSIC_PARAMS',ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'))
 if classic_path.exists():
  fitted=json.loads(classic_path.read_text()).get('expanded_fitted',{});updates={k:float(v) for k,v in fitted.items() if hasattr(params,k)}
  if 'I_z' in fitted:updates['Iz']=float(fitted['I_z'])
  params=replace(params,**updates)
 d=load_callback_archives(Path(a.dataset),model_dt=.04,horizon=a.horizon_steps)
 cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];c=Contract.from_parameters(params,dt=.04)
 state=d['initial_state'];cmd=d['commands'][:,0];ap=d['actuator'][:,0].copy();sr=d['actuator'][:,1].copy();base=np.empty_like(state)
 lf,lr,m,iz=float(cfg['l_f']),float(cfg['l_r']),float(cfg['mass']),params.Iz;wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb
 for i in range(len(state)):
  ap[i],_=actuator_step(ap[i],cmd[i,0],cmd[i,1],state[i,0],c);sr[i],bax=longitudinal_actuator_step(sr[i],cmd[i,1],state[i,0],c);vx,vy,r=state[i];safe=max(abs(vx),.5);af=ap[i]-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=params.B_f*af;br=params.B_r*ar;fyf=fzf*params.D_f*np.sin(params.C_f*np.arctan(bf-params.E_f*(bf-np.arctan(bf))));fyr=fzr*params.D_r*np.sin(params.C_r*np.arctan(br-params.E_r*(br-np.arctan(br))));ay=(fyf*np.cos(ap[i])+fyr)/m;rd=(lf*fyf*np.cos(ap[i])-lr*fyr)/iz;base[i]=(vx+(bax+vy*r)*.04,vy+(ay-vx*r)*.04,r+rd*.04)
 previous=d['history'][:,-4];x=np.c_[state,cmd,ap,cmd[:,0]-previous,base,d['history'],d['imu']].astype(np.float32);y=((d['target_state'][:,0]-base)/.04).astype(np.float32);s=d['split'];v=np.isfinite(x).all(1)&np.isfinite(y).all(1);tr=v&(s==0);va=v&(s==1);metadata={'source':'Step-1 callback archives (direct, no residual NPZ)','classic_parameter_hash':params.digest(),'target':'40 ms online MPPI-model EKF transition minus classic transition'};mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),1e-4);ym=y[tr].mean(0);ys=np.maximum(y[tr].std(0),1e-3);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');xt=torch.from_numpy((x-mean)/std).to(dev);yt=torch.from_numpy((y-ym)/ys).to(dev);net=Net();
 if a.initialize_from:net.load_state_dict(torch.load(Path(a.initialize_from)/'model.pt',map_location='cpu',weights_only=True))
 net=net.to(dev);opt=torch.optim.AdamW(net.parameters(),8e-4,weight_decay=1e-4);idx=np.flatnonzero(tr);b=d['bag_name'];prob=np.array([4 if x[i,0]>=3 else 1 for i in idx],float);counts={q:max(1,np.sum(b[idx]==q)) for q in np.unique(b[idx])};prob*=np.array([1/np.sqrt(counts[b[i]]) for i in idx]);prob/=prob.sum();weights=torch.tensor((1.,2.,2.),device=dev);best=(1e99,None,0);stale=0
 for epoch in range(a.epochs):
  net.train();sample=rng.choice(idx,len(idx),replace=True,p=prob)
  for q in np.array_split(sample,max(1,len(sample)//1024)):
   pred=net(xt[q]);loss=(nn.functional.smooth_l1_loss(pred,yt[q],reduction='none')*weights).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  net.eval()
  with torch.no_grad():score=float(nn.functional.smooth_l1_loss(net(xt[va]),yt[va]))
  if score<best[0]-1e-5:best=(score,{k:z.detach().cpu().clone() for k,z in net.state_dict().items()},epoch+1);stale=0
  else:stale+=1
  if stale>=50:break
 net.load_state_dict(best[1]);net.cpu()
 with torch.no_grad():net.net[4].weight.mul_(torch.from_numpy(ys)[:,None]);net.net[4].bias.mul_(torch.from_numpy(ys)).add_(torch.from_numpy(ym));pred=net(torch.from_numpy((x-mean)/std)).numpy()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),out/'model.pt');layers=(net.net[0],net.net[2],net.net[4]);blob=np.concatenate([z.detach().numpy().ravel() for layer in layers for z in (layer.weight,layer.bias)]+[mean,std]).astype('<f4');assert len(blob)==3695;blob.tofile(out/'dynamic_40ms_residual.bin');metrics={'seed':a.seed,'best_epoch':best[2],'model_dt':.04,'control_dt':.02,'supervision':'one-step temporary pseudo-label initialization','dataset_metadata':metadata,'input_features':list(IMU_RESIDUAL_FEATURES)}
 for k,name in enumerate(('train','validation','test')):
  mask=v&(s==k);e=abs(pred[mask]-y[mask]);metrics[name]={'n':int(mask.sum()),'mae':e.mean(0).tolist(),'p95':np.quantile(e,.95,axis=0).tolist()}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(out/'contract.json').write_text(json.dumps({'model':'dynamic_mlp_residual_servo_lag_40ms','control_dt':.02,'model_dt':.04,'features':list(IMU_RESIDUAL_FEATURES),'outputs':['delta_ax','delta_ay','delta_yaw_accel'],'substeps':1,'integration':'single Euler step at 0.04 s'},indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
