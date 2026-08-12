#!/usr/bin/env python3
"""Train a causal 20-64-32-3 residual MLP and export CUDA-compatible weights.

NPZ must contain features[N,20], targets[N,3], bag_id[N], valid[N].  Targets
are residual derivatives, never next-state values. Splits are by bag_id.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from contract import FEATURES, OUTPUTS, Contract

class Net(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
    def forward(self,x): return self.net(x)

def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset"); p.add_argument("--out",required=True); p.add_argument("--epochs",type=int,default=200); p.add_argument("--seed",type=int,default=7); a=p.parse_args()
    torch.manual_seed(a.seed); d=np.load(a.dataset,allow_pickle=True)
    x,y,b=np.asarray(d["features"],np.float32),np.asarray(d["targets"],np.float32),d["bag_id"]
    valid=np.asarray(d.get("valid",np.ones(len(x),bool)),bool)&np.isfinite(x).all(1)&np.isfinite(y).all(1)
    rng=np.random.default_rng(a.seed)
    if "split" in d:
        split=np.array([("train","val","test")[int(z)] for z in d["split"]])
    else:
        bags=np.unique(b); rng.shuffle(bags); n=max(1,int(.2*len(bags))); test=set(bags[:n]); val=set(bags[n:2*n]); split=np.array(["test" if z in test else "val" if z in val else "train" for z in b])
    tr=valid&(split=="train"); va=valid&(split=="val"); mean=x[tr].mean(0); std=np.maximum(x[tr].std(0),1e-4)
    target_mean=y[tr].mean(0);target_std=np.maximum(y[tr].std(0),1e-3)
    xt=torch.from_numpy((x-mean)/std); yt=torch.from_numpy((y-target_mean)/target_std); model=Net(); opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4)
    # Tail-aware Huber: emphasize large lateral/yaw events without allowing spikes to dominate.
    weights=torch.tensor([1.,2.,2.]); best=None
    train_bags=np.unique(b[tr]);train_index=np.flatnonzero(tr)
    bag_count={bid:max(1,int(np.sum(tr&(b==bid)))) for bid in train_bags}
    # Strengthen scarce 3--4 m/s states without duplicating held-out bags.
    high_speed_weight=np.where(x[train_index,0]>=3.0,4.0,1.0)
    sample_probability=np.array([1/np.sqrt(bag_count[b[i]]) for i in train_index],float)*high_speed_weight
    sample_probability/=sample_probability.sum()
    temporal_starts={}
    for horizon in (10,30,60):
        temporal_starts[horizon]=np.array([i for i in train_index if i+horizon<=len(x)
            and tr[i:i+horizon].all() and np.all(b[i:i+horizon]==b[i])],dtype=int)
    for epoch in range(a.epochs):
        model.train()
        # Equal expected contribution per source session. The old pipeline was
        # 91% one long bag and failed badly on a short held-out acceleration.
        idx=rng.choice(train_index,len(train_index),replace=True,p=sample_probability);rng.shuffle(idx)
        for q in np.array_split(idx,max(1,len(idx)//1024)):
            pred=model(xt[q]); loss=(nn.functional.smooth_l1_loss(pred,yt[q],reduction="none")*weights).mean(); opt.zero_grad(); loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        # Multi-horizon integral loss suppresses small derivative biases that
        # dominate 0.2/0.6/1.2 s open-loop state error. Feature/state feedback
        # is evaluated separately by evaluate_rollout.py.
        for horizon,temporal_weight in ((10,.08),(30,.05),(60,.03)):
            pool=temporal_starts[horizon]
            if not len(pool):continue
            starts=rng.choice(pool,min(128,len(pool)),replace=False)
            rows=(starts[:,None]+np.arange(horizon)[None,:]).reshape(-1)
            normalized_prediction=model(xt[rows]).reshape(len(starts),horizon,3)
            physical_prediction=normalized_prediction*torch.from_numpy(target_std)+torch.from_numpy(target_mean)
            physical_target=torch.from_numpy(y[rows]).reshape(len(starts),horizon,3)
            temporal_loss=(nn.functional.smooth_l1_loss(
                physical_prediction.sum(1)*.02,physical_target.sum(1)*.02,reduction="none")*weights).mean()
            opt.zero_grad();(temporal_weight*temporal_loss).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
        model.eval()
        with torch.no_grad(): score=nn.functional.smooth_l1_loss(model(xt[va]),yt[va]).item() if va.any() else loss.item()
        if best is None or score<best[0]: best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    # Fold target de-normalization into the final layer so CUDA still receives
    # physical residual derivatives with the unchanged 3563-float format.
    with torch.no_grad():
        model.net[4].weight.mul_(torch.from_numpy(target_std)[:,None])
        model.net[4].bias.mul_(torch.from_numpy(target_std)).add_(torch.from_numpy(target_mean))
    torch.save(model.state_dict(),out/"model.pt")
    layers=[model.net[0],model.net[2],model.net[4]]
    # CUDA split_mlp expects output-major contiguous Linear weights, then biases.
    blob=np.concatenate([z.detach().numpy().ravel() for l in layers for z in (l.weight,l.bias)]+[mean,std]).astype("<f4")
    assert blob.size==3563; blob.tofile(out/"dynamic_residual_v2.bin")
    pred=model(xt).detach().numpy(); metrics={}
    for s in ("train","val","test"):
        m=valid&(split==s)
        if m.any():
            e=np.abs(pred[m]-y[m]); metrics[s]={"n":int(m.sum()),"mae":e.mean(0).tolist(),"p95":np.quantile(e,.95,axis=0).tolist(),"max":e.max(0).tolist()}
    Contract().dump(out/"contract.json"); np.savez_compressed(out/"predictions.npz",prediction=pred,target=y,split=split,valid=valid)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)); (out/"split_manifest.json").write_text(json.dumps({"train":sorted(set(b[split=="train"].astype(str))),"val":sorted(set(b[split=="val"].astype(str))),"test":sorted(set(b[split=="test"].astype(str)))},indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
