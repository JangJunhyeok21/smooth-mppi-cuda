#!/usr/bin/env python3
"""Train/export the 34-64-32-3 40 ms state-residual model."""
import argparse, json, struct, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contract import EffectiveContract, FEATURES

class Net(nn.Module):
    def __init__(self,n_in=len(FEATURES)):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n_in,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,3))
        self.register_buffer("limits",torch.tensor(EffectiveContract().residual_limits,dtype=torch.float32))
    def forward(self,x): return self.limits*torch.tanh(self.layers(x))

def metrics(pred,target,mask):
    e=np.abs(pred[mask]-target[mask]);return {"n":int(mask.sum()),"mae":e.mean(0).tolist(),"p95":np.quantile(e,.95,axis=0).tolist(),"max":e.max(0).tolist(),
      "saturation_fraction":float((np.abs(pred[mask])>=.98*np.asarray(EffectiveContract().residual_limits)).mean())}

def export(model,mean,std,path):
    layers=(model.layers[0],model.layers[2],model.layers[4]); payload=np.concatenate(
      [z.detach().cpu().numpy().astype("<f4").ravel() for l in layers for z in (l.weight,l.bias)]+[mean.astype("<f4"),std.astype("<f4")])
    header=struct.pack("<8s5I5f",b"EHSR004\0",1,len(FEATURES),64,32,3,.02,.04,*EffectiveContract().residual_limits)
    with open(path,"wb") as f:f.write(header);f.write(payload.tobytes())
    return len(header)+payload.nbytes

def main():
    p=argparse.ArgumentParser();p.add_argument("dataset");p.add_argument("--out",required=True);p.add_argument("--epochs",type=int,default=100);p.add_argument("--batch-size",type=int,default=512);p.add_argument("--max-batches",type=int,default=0);p.add_argument("--seed",type=int,default=31);a=p.parse_args()
    torch.manual_seed(a.seed);np.random.seed(a.seed);d=np.load(a.dataset);x=d["features"].astype(np.float32);y=d["targets"].astype(np.float32);split=d["split"];valid=d["valid"]
    tr=valid&(split==0);va=valid&(split==1);te=valid&(split==2);mean=x[tr].mean(0);std=np.maximum(x[tr].std(0),1e-4);xn=(x-mean)/std
    model=Net();opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4);rng=np.random.default_rng(a.seed);ids=np.flatnonzero(tr);best=None;batches=0;t0=time.time()
    for epoch in range(a.epochs):
        rng.shuffle(ids);model.train()
        for q in np.array_split(ids,max(1,int(np.ceil(len(ids)/a.batch_size)))):
            xb=torch.from_numpy(xn[q]);yb=torch.from_numpy(y[q]);pred=model(xb)
            speed=x[q,0];ay=np.abs(x[q,4]);steer_rate=np.abs(x[q,27]/.02)
            w=1+2*(speed>2)+4*(speed>2.5)+2*(ay>3)+2*(steer_rate>.5);w=torch.from_numpy(np.minimum(w,9).astype(np.float32))
            loss=(nn.functional.smooth_l1_loss(pred,yb,reduction="none")*torch.tensor([1.,1.5,2.])*w[:,None]).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();batches+=1
            if a.max_batches and batches>=a.max_batches:break
        model.eval()
        with torch.no_grad():score=nn.functional.smooth_l1_loss(model(torch.from_numpy(xn[va])),torch.from_numpy(y[va])).item()
        if best is None or score<best[0]:best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()})
        if a.max_batches and batches>=a.max_batches:break
    model.load_state_dict(best[1]);model.eval()
    with torch.no_grad():pred=model(torch.from_numpy(xn)).numpy()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.save({"state_dict":model.state_dict(),"mean":mean,"std":std,"features":FEATURES},out/"checkpoint.pt")
    np.savez(out/"normalization.npz",mean=mean,std=std,features=np.array(FEATURES));size=export(model,mean,std,out/"effective_history_state_residual.bin")
    report={"seed":a.seed,"epochs_completed":epoch+1,"batches":batches,"seconds":time.time()-t0,"parameter_count":sum(q.numel() for q in model.parameters()),"binary_bytes":size,
      "train":metrics(pred,y,tr),"validation":metrics(pred,y,va),"speed30_holdout":metrics(pred,y,te)}
    (out/"metrics.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()

