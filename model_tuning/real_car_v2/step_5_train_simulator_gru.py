#!/usr/bin/env python3
"""시뮬레이터 plant 전용 대형 causal GRU를 0817~0820 데이터로 학습한다."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'model_tuning/data/dynamic_0817_0820_inertial_ekf_bias_40ms.npz';OUT=ROOT/'model_tuning/results/simulator_gru_0817_0820_seed31'
FEATURES=('vx','vy','yaw_rate','imu_ax','imu_ay','steer_cmd','speed_cmd','applied_steer','speed_reference')
class GRUPlant(nn.Module):
 def __init__(self):
  super().__init__();self.gru=nn.GRU(9,256,3,batch_first=True,dropout=.12);self.head=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,3))
 def forward(self,x):return self.head(self.gru(x)[0])
class InferencePlant(nn.Module):
 def __init__(self,net,xm,xs,ym,ys):super().__init__();self.net=net;self.register_buffer('xm',xm);self.register_buffer('xs',xs);self.register_buffer('ym',ym);self.register_buffer('ys',ys)
 def forward(self,x):return self.net((x-self.xm)/self.xs)[:,-1]*self.ys+self.ym
def windows(features,targets,bag,split,valid,length,stride):
 starts=[]
 for i in range(0,len(features)-length,stride):
  if valid[i:i+length+1].all() and bag[i]==bag[i+length] and split[i]==split[i+length]:starts.append(i)
 idx=np.asarray(starts)[:,None]+np.arange(length)[None,:]
 return features[idx],targets[idx],split[np.asarray(starts)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=31);p.add_argument('--device',default='cuda');p.add_argument('--history',type=int,default=50);a=p.parse_args();torch.manual_seed(a.seed);d=np.load(a.data);s=d['source_features'].astype(np.float32);o=d['source_observations'].astype(np.float32);sr=d['source_speed_reference'].astype(np.float32);bag=d['source_bag_id'];sp=d['source_split'];valid=d['source_valid'];f=np.c_[s[:,:3],o[:,:2],s[:,3:5],s[:,5],sr].astype(np.float32);target=o[1:,:3].astype(np.float32);f=f[:-1];bag=bag[:-1];sp=sp[:-1];valid=valid[:-1]&d['source_valid'][1:]&(bag==d['source_bag_id'][1:]);X,Y,S=windows(f,target,bag,sp,valid,a.history,5);tr=S==0;va=S==1;xm=X[tr].reshape(-1,9).mean(0);xs=np.maximum(X[tr].reshape(-1,9).std(0),1e-4);ym=Y[tr].reshape(-1,3).mean(0);ys=np.maximum(Y[tr].reshape(-1,3).std(0),1e-3);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');xt=torch.from_numpy((X-xm)/xs);yt=torch.from_numpy((Y-ym)/ys);net=GRUPlant().to(dev);opt=torch.optim.AdamW(net.parameters(),3e-4,weight_decay=2e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs);rng=np.random.default_rng(a.seed);best=(1e9,None,0);weights=torch.tensor((1.,2.,2.),device=dev)
 for epoch in range(a.epochs):
  net.train();order=rng.permutation(np.flatnonzero(tr));losses=[]
  for ids in np.array_split(order,max(1,len(order)//128)):
   xb=xt[ids].to(dev);yb=yt[ids].to(dev);pred=net(xb);# emphasize the recent half that seeds recursive simulation
   time_weight=torch.linspace(.35,1.,a.history,device=dev)[None,:,None];loss=(nn.functional.smooth_l1_loss(pred,yb,reduction='none')*weights*time_weight).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.);opt.step();losses.append(float(loss))
  sched.step();net.eval();scores=[]
  with torch.no_grad():
   for ids in np.array_split(np.flatnonzero(va),max(1,np.sum(va)//256)):scores.append(float(nn.functional.smooth_l1_loss(net(xt[ids].to(dev)),yt[ids].to(dev))))
  score=float(np.mean(scores))
  if score<best[0]:best=(score,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch+1)
  if epoch%5==0:print(f'epoch={epoch+1} train={np.mean(losses):.6f} val={score:.6f} best={best[0]:.6f}',flush=True)
 net.load_state_dict(best[1]);net.cpu().eval();wrapper=InferencePlant(net,torch.from_numpy(xm),torch.from_numpy(xs),torch.from_numpy(ym),torch.from_numpy(ys)).eval();a.out.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':net.state_dict(),'x_mean':xm,'x_std':xs,'y_mean':ym,'y_std':ys,'history':a.history},a.out/'model.pt');torch.jit.script(wrapper).save(str(a.out/'simulator_gru.ts'))
 with torch.no_grad():pred=wrapper(torch.from_numpy(X)).numpy();err=abs(pred-Y[:,-1])
 metrics={'model':'simulator_gru','parameters':sum(q.numel() for q in net.parameters()),'history_steps':a.history,'dt':.02,'best_epoch':best[2],'features':FEATURES}
 for sid,name in ((0,'train'),(1,'validation'),(2,'test')):
  e=err[S==sid];metrics[name]={'windows':int(len(e)),'mae':e.mean(0).tolist(),'p95':np.quantile(e,.95,axis=0).tolist()}
 (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
