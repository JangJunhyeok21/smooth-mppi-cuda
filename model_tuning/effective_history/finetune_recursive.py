#!/usr/bin/env python3
"""Free-recursive 5/10/20/30/50/60-step fine-tuning.

Only recorded commands are supplied after the initial state. Predicted body
state, pose, acceleration features, and command history are fed recursively;
measured states are targets only and are never teacher-forced into a rollout.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contract import EffectiveContract, FEATURES
from train import Net, export, metrics

HORIZONS=(5,10,20,30,50,60)
HORIZON_WEIGHTS={5:.5,10:.8,20:1.,30:1.5,50:1.8,60:2.}
STATE_WEIGHTS=torch.tensor((1.,1.5,2.),dtype=torch.float32)

def valid_starts(d,split,horizon=60):
    ok=[]
    for i in np.flatnonzero(d["valid"]&(d["split"]==split)):
        end=i+2*(horizon-1)
        if end>=len(d["valid"]):continue
        rows=i+2*np.arange(horizon)
        if d["valid"][rows].all() and (d["split"][rows]==split).all() and (d["session"][rows]==d["session"][i]).all():ok.append(i)
    return np.asarray(ok,dtype=np.int64)

def make_features(body,accel,hist):
    s=hist[:,-1,0];vcmd=hist[:,-1,1];vx=body[:,0];r=body[:,2]
    return torch.cat((body,accel,s[:,None],vcmd[:,None],hist[:,:,0],hist[:,:,1],
      (s-hist[:,-2,0])[:,None],(s-hist[:,-4,0])[:,None],(vcmd-hist[:,-2,1])[:,None],
      (vx*s)[:,None],(vx*vx*s)[:,None],(vx*r)[:,None],(torch.abs(vx)*s)[:,None]),1)

def baseline(body,c0,c1,c):
    vx,vy,r=body.unbind(1)
    for cmd in (c0,c1):
        steer,speed=cmd.unbind(1);effective=c.effective_steer_scale*steer+c.effective_steer_bias
        target=vx/c.wheelbase*torch.tan(effective)
        rdot=torch.clamp((target-r)/c.effective_yaw_response_tau,-c.effective_max_yaw_accel,c.effective_max_yaw_accel)
        ax=torch.clamp(c.effective_speed_response_gain*(speed-vx),-c.effective_max_accel,c.effective_max_accel)
        vx=vx+ax*c.control_dt;vy=vy+(-vy/c.effective_vy_decay_tau)*c.control_dt;r=r+rdot*c.control_dt
    return torch.stack((vx,vy,r),1)

def integrate(pose,body,c):
    x,y,yaw=pose.unbind(1);vx,vy,r=body.unbind(1)
    return torch.stack((x+c.position_speed_scale*(vx*torch.cos(yaw)-vy*torch.sin(yaw))*c.model_dt,
      y+c.position_speed_scale*(vx*torch.sin(yaw)+vy*torch.cos(yaw))*c.model_dt,
      torch.atan2(torch.sin(yaw+r*c.model_dt),torch.cos(yaw+r*c.model_dt))),1)

def rollout(model,d,starts,mean,std,c,need_all=False):
    device=next(model.parameters()).device;idx=torch.as_tensor(starts,device=device);body=torch.as_tensor(d["state"][starts,3:],device=device);pose=torch.as_tensor(d["state"][starts,:3],device=device);hist=torch.as_tensor(d["command_history"][starts],device=device);accel=torch.as_tensor(d["features"][starts,3:5],device=device);out={};raws=[]
    mean=torch.as_tensor(mean,device=device);std=torch.as_tensor(std,device=device)
    for step in range(1,61):
        rows=starts+2*(step-1);c0=torch.as_tensor(d["command_t"][rows],device=device);c1=torch.as_tensor(d["command_t1"][rows],device=device)
        if step>1:hist=torch.cat((hist[:,1:,:],c0[:,None,:]),1)
        feature=make_features(body,accel,hist);res=model((feature-mean)/std);base=baseline(body,c0,c1,c);nbody=base+res;npose=integrate(pose,nbody,c)
        accel=torch.stack(((nbody[:,0]-body[:,0])/c.model_dt,(nbody[:,1]-body[:,1])/c.model_dt+body[:,0]*body[:,2]),1)
        body=nbody;pose=npose;hist=torch.cat((hist[:,1:,:],c1[:,None,:]),1);raws.append(res)
        if step in HORIZONS or need_all:
            gt=torch.as_tensor(d["next_state"][rows],device=device);out[step]=(pose,body,gt)
    return out,torch.stack(raws,1)

def rollout_loss(model,d,starts,mean,std,c):
    out,res=rollout(model,d,starts,mean,std,c);total=0.;terms={}
    for h in HORIZONS:
        pose,body,gt=out[h];body_loss=nn.functional.smooth_l1_loss(body,gt[:,3:],reduction="none")*STATE_WEIGHTS.to(body.device)
        pos=torch.linalg.vector_norm(pose[:,:2]-gt[:,:2],dim=1);yaw=torch.atan2(torch.sin(pose[:,2]-gt[:,2]),torch.cos(pose[:,2]-gt[:,2])).abs()
        term=body_loss.mean()+2.*nn.functional.smooth_l1_loss(pos,torch.zeros_like(pos))+2.*nn.functional.smooth_l1_loss(yaw,torch.zeros_like(yaw))
        total=total+HORIZON_WEIGHTS[h]*term;terms[h]=float(term.detach())
    total=total+1e-3*(res*res).mean()
    return total,terms

def validation_score(model,d,starts,mean,std,c,batch=64):
    values=[]
    model.eval()
    with torch.no_grad():
        for q in np.array_split(starts,max(1,int(np.ceil(len(starts)/batch)))):
            loss,_=rollout_loss(model,d,q,mean,std,c);values.append(float(loss))
    return float(np.mean(values))

def main():
    p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("checkpoint");p.add_argument("--out",required=True);p.add_argument("--epochs",type=int,default=80);p.add_argument("--batch-size",type=int,default=32);p.add_argument("--batches-per-epoch",type=int,default=24);p.add_argument("--validation-windows",type=int,default=192);p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--patience",type=int,default=15);p.add_argument("--seed",type=int,default=31);a=p.parse_args()
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);d=np.load(a.dataset);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);mean=np.asarray(ck["mean"],np.float32);std=np.asarray(ck["std"],np.float32);model=Net();model.load_state_dict(ck["state_dict"]);c=EffectiveContract();train=valid_starts(d,0);val=valid_starts(d,1);val=val[::max(1,len(val)//a.validation_windows)][:a.validation_windows]
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5);best=(validation_score(model,d,val,mean,std,c),{k:v.detach().clone() for k,v in model.state_dict().items()},-1);history=[];stale=0;t0=time.time()
    for epoch in range(a.epochs):
        model.train();losses=[]
        for _ in range(a.batches_per_epoch):
            starts=rng.choice(train,min(a.batch_size,len(train)),replace=False);loss,terms=rollout_loss(model,d,starts,mean,std,c);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
        score=validation_score(model,d,val,mean,std,c);history.append((epoch+1,float(np.mean(losses)),score,*[terms[h] for h in HORIZONS]));print(f"epoch={epoch+1:03d} train={np.mean(losses):.6f} val={score:.6f}",flush=True)
        if score<best[0]-1e-5:best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch);stale=0
        else:stale+=1
        if stale>=a.patience:break
    model.load_state_dict(best[1]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.save({"state_dict":model.state_dict(),"mean":mean,"std":std,"features":FEATURES,"recursive_horizons":HORIZONS,"seed":a.seed},out/"checkpoint.pt");np.savez(out/"normalization.npz",mean=mean,std=std,features=np.array(FEATURES));size=export(model,mean,std,out/"effective_history_state_residual.bin")
    x=d["features"].astype(np.float32);y=d["targets"].astype(np.float32);model.eval()
    with torch.no_grad():pred=model(torch.from_numpy((x-mean)/std)).numpy()
    report={"seed":a.seed,"best_epoch":best[2]+1,"best_validation_recursive_loss":best[0],"seconds":time.time()-t0,"train_starts":len(train),"validation_starts":len(val),"binary_bytes":size}
    for split,name in ((0,"train"),(1,"validation"),(2,"speed30_holdout")):report[name]=metrics(pred,y,d["valid"]&(d["split"]==split))
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    with open(out/"history.csv","w",newline="") as f:w=csv.writer(f);w.writerow(("epoch","train","validation",*[f"h{h}" for h in HORIZONS]));w.writerows(history)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
