#!/usr/bin/env python3
"""Simulator GRU를 자신의 예측 state/IMU를 쓰는 recursive loss로 미세조정한다."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent));from experimental_train_simulator_gru import GRUPlant,InferencePlant
DATA=ROOT/'model_tuning/data/dynamic_0817_0820_inertial_ekf_bias_40ms.npz';RESULT=ROOT/'model_tuning/results/simulator_gru_0817_0820_seed31'
def make_starts(bag,split,valid,sid,warm,horizon,stride):return np.asarray([i for i in range(warm,len(bag)-horizon,stride) if split[i]==sid and split[i+horizon]==sid and valid[i-warm:i+horizon+1].all() and np.all(bag[i-warm:i+horizon+1]==bag[i])])
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA);p.add_argument('--result',type=Path,default=RESULT);p.add_argument('--epochs',type=int,default=30);p.add_argument('--horizon',type=int,default=30);p.add_argument('--device',default='cuda');a=p.parse_args();d=np.load(a.data);s=d['source_features'].astype(np.float32);obs=d['source_observations'].astype(np.float32);sr=d['source_speed_reference'].astype(np.float32);bag=d['source_bag_id'];split=d['source_split'];valid=d['source_valid'];ck=torch.load(a.result/'model.pt',map_location='cpu',weights_only=False);xm=torch.tensor(ck['x_mean']);xs=torch.tensor(ck['x_std']);ym=torch.tensor(ck['y_mean']);ys=torch.tensor(ck['y_std']);warm=int(ck['history']);feature=np.c_[s[:,:3],obs[:,:2],s[:,3:5],s[:,5],sr].astype(np.float32);train=make_starts(bag,split,valid,0,warm,a.horizon,10);val=make_starts(bag,split,valid,1,warm,a.horizon,10);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');net=GRUPlant();net.load_state_dict(ck['state_dict']);net.to(dev);xm,xs,ym,ys=[q.to(dev) for q in (xm,xs,ym,ys)];opt=torch.optim.AdamW(net.parameters(),5e-5,weight_decay=2e-4);rng=np.random.default_rng(31);best=(1e9,None,0)
 def batch_loss(ids,grad):
  ids=np.asarray(ids);warm_np=np.stack([feature[i-warm:i] for i in ids]);warm_x=(torch.from_numpy(warm_np).to(dev)-xm)/xs;_,hidden=net.gru(warm_x);state=torch.from_numpy(s[ids,:3]).to(dev);acc=torch.from_numpy(obs[ids,:2]).to(dev);loss=torch.zeros((),device=dev)
  for k in range(a.horizon):
   rows=torch.from_numpy(np.c_[s[ids+k,3:5],s[ids+k,5],sr[ids+k]].astype(np.float32)).to(dev);raw=torch.cat((state,acc,rows),1);out,hidden=net.gru(((raw-xm)/xs)[:,None],hidden);pred_norm=net.head(out[:,0]);pred=pred_norm*ys+ym;ax,ay,nr=pred[:,0],pred[:,1],pred[:,2];vx,vy=state[:,0],state[:,1];next_state=torch.stack((vx+(ax+vy*nr)*.02,vy+(ay-vx*nr)*.02,nr),1);target_state=torch.from_numpy(s[ids+k+1,:3]).to(dev);target_obs=torch.from_numpy(obs[ids+k+1,:3]).to(dev);deriv=nn.functional.smooth_l1_loss(pred_norm,(target_obs-ym)/ys);scale=torch.tensor((1.,.5,1.),device=dev);state_loss=nn.functional.smooth_l1_loss((next_state-target_state)/scale,torch.zeros_like(next_state));loss=loss+deriv+.8*state_loss;state=next_state;acc=pred[:,:2]
  return loss/a.horizon
 for epoch in range(a.epochs):
  net.train();order=rng.permutation(train);losses=[]
  for ids in np.array_split(order,max(1,len(order)//64)):
   loss=batch_loss(ids,True);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  net.eval();scores=[]
  with torch.no_grad():
   for ids in np.array_split(val,max(1,len(val)//128)):scores.append(float(batch_loss(ids,False)))
  score=float(np.mean(scores));
  if score<best[0]:best=(score,{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},epoch+1)
  print(f'epoch={epoch+1} train={np.mean(losses):.6f} val={score:.6f} best={best[0]:.6f}',flush=True)
 net.load_state_dict(best[1]);net.cpu().eval();ck['state_dict']=net.state_dict();ck['recursive_best_epoch']=best[2];torch.save(ck,a.result/'model_recursive.pt');wrapper=InferencePlant(net,xm.cpu(),xs.cpu(),ym.cpu(),ys.cpu()).eval();torch.jit.script(wrapper).save(str(a.result/'simulator_gru.ts'));(a.result/'recursive_metrics.json').write_text(json.dumps({'best_epoch':best[2],'validation_loss':best[0],'horizon_steps':a.horizon,'train_windows':len(train),'validation_windows':len(val)},indent=2)+'\n')
if __name__=='__main__':main()
