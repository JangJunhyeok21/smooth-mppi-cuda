#!/usr/bin/env python3
"""Recorded-command free recursive replay at the required horizons."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contract import EffectiveContract, baseline_body_step, integrate_pose, make_features, wrap_angle
from train import Net

HORIZONS=(1,5,10,20,30,50,60)
def stats(x):return {"mean":float(np.mean(x)),"p95":float(np.quantile(x,.95)),"max":float(np.max(x))}
def main():
 p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("checkpoint");p.add_argument("--out",required=True);p.add_argument("--max-windows",type=int,default=128);a=p.parse_args();d=np.load(a.dataset);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);net=Net();net.load_state_dict(ck["state_dict"]);net.eval();mean=ck["mean"];std=ck["std"];c=EffectiveContract();report={}
 for split,name in ((1,"validation"),(2,"speed30_holdout")):
  pool=np.flatnonzero(d["valid"]&(d["split"]==split));pool=pool[pool+120<len(d["valid"])];pool=pool[::max(1,len(pool)//a.max_windows)][:a.max_windows];acc={h:[] for h in HORIZONS}
  for start in pool:
   pose=d["state"][start,:3].astype(float);body=d["state"][start,3:].astype(float);hist=d["command_history"][start].astype(float);axay=d["features"][start,3:5].astype(float)
   for step in range(1,61):
    i=start+2*(step-1)
    if i+1>=len(d["valid"]) or d["session"][i]!=d["session"][start] or not d["valid"][i]:break
    cmd0=d["command_t"][i];cmd1=d["command_t1"][i]
    # Initial history already ends at command[t]. At subsequent model steps,
    # append the new command[t+2k] before constructing that step's features.
    if step>1:hist=np.r_[hist[1:],cmd0[None]]
    feat=make_features(body,axay,hist);raw=net(torch.from_numpy(((feat-mean)/std).astype(np.float32))[None]).detach().numpy()[0]
    base=baseline_body_step(body,cmd0,cmd1,c);nb=base+raw;np_pose=integrate_pose(pose,nb,c);axay=np.array(((nb[0]-body[0])/c.model_dt,(nb[1]-body[1])/c.model_dt+body[0]*body[2]));body=nb;pose=np_pose;hist=np.r_[hist[1:],cmd1[None]]
    if step in HORIZONS:
     gt=d["next_state"][i];pos=np.linalg.norm(pose[:2]-gt[:2]);yaw=abs(wrap_angle(pose[2]-gt[2]));acc[step].append((pos,yaw,body[0]-gt[3],abs(body[1]-gt[4]),body[2]-gt[5]))
  report[name]={}
  for h,v in acc.items():
   z=np.asarray(v);report[name][str(round(h*.04,2))]={"windows":len(z),"position":stats(z[:,0]),"yaw":stats(z[:,1]),"vx_bias":float(z[:,2].mean()),"vx_mae":float(np.abs(z[:,2]).mean()),"vx_p95":float(np.quantile(np.abs(z[:,2]),.95)),"vy_mae":float(z[:,3].mean()),"yaw_rate_bias":float(z[:,4].mean()),"yaw_rate_mae":float(np.abs(z[:,4]).mean()),"yaw_rate_p95":float(np.quantile(np.abs(z[:,4]),.95))} if len(z) else {"windows":0}
 Path(a.out).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
