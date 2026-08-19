#!/usr/bin/env python3
"""Step 5: train/export the one-step 20-64-32-3 residual MLP."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));from contract import IMU_RESIDUAL_FEATURES
# User-editable defaults. The script runs without command-line arguments.
ROOT=HERE.parents[1]
DATASET_PATH=ROOT/'model_tuning/data/dynamic_40ms_residual.npz'
OUTPUT_PATH=ROOT/'model_tuning/results/dynamic_40ms_residual_seed31'
EPOCHS=300
SEED=31
DEVICE='cuda'
class Net(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(len(IMU_RESIDUAL_FEATURES),64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
 def forward(self,x):return self.net(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',default=str(DATASET_PATH));p.add_argument('--out',default=str(OUTPUT_PATH));p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--seed',type=int,default=SEED);p.add_argument('--device',default=DEVICE);a=p.parse_args();torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);d=np.load(a.dataset);x=d['features'].astype(np.float32);y=d['targets'].astype(np.float32);b=d['bag_id'];s=d['split'];v=d['valid'];tr=v&(s==0);va=v&(s==1);mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),1e-4);ym=y[tr].mean(0);ys=np.maximum(y[tr].std(0),1e-3);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');xt=torch.from_numpy((x-mean)/std).to(dev);yt=torch.from_numpy((y-ym)/ys).to(dev);net=Net().to(dev);opt=torch.optim.AdamW(net.parameters(),8e-4,weight_decay=1e-4);idx=np.flatnonzero(tr);prob=np.array([4 if x[i,0]>=3 else 1 for i in idx],float);counts={q:max(1,np.sum(b[idx]==q)) for q in np.unique(b[idx])};prob*=np.array([1/np.sqrt(counts[b[i]]) for i in idx]);prob/=prob.sum();weights=torch.tensor((1.,2.,2.),device=dev);best=(1e99,None,0);stale=0
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
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.save(net.state_dict(),out/'model.pt');layers=(net.net[0],net.net[2],net.net[4]);blob=np.concatenate([z.detach().numpy().ravel() for layer in layers for z in (layer.weight,layer.bias)]+[mean,std]).astype('<f4');assert len(blob)==3695;blob.tofile(out/'dynamic_40ms_residual.bin');metrics={'seed':a.seed,'best_epoch':best[2],'model_dt':.04,'control_dt':.02,'input_features':list(IMU_RESIDUAL_FEATURES)}
 for k,name in enumerate(('train','validation','test')):
  mask=v&(s==k);e=abs(pred[mask]-y[mask]);metrics[name]={'n':int(mask.sum()),'mae':e.mean(0).tolist(),'p95':np.quantile(e,.95,axis=0).tolist()}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(out/'contract.json').write_text(json.dumps({'model':'dynamic_mlp_residual_servo_lag_40ms','control_dt':.02,'model_dt':.04,'features':list(IMU_RESIDUAL_FEATURES),'outputs':['delta_ax','delta_ay','delta_yaw_accel'],'substeps':1,'integration':'single Euler step at 0.04 s'},indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
