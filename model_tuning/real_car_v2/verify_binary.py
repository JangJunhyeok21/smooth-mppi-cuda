#!/usr/bin/env python3
"""Verify exported float32 layout and CUDA-equivalent ReLU inference."""
import json,sys,argparse
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from train import Net

DATASET=HERE.parents[1]/"model_tuning/data/real_car_v2_dynamic_residual.npz"
DEFAULT_RESULT=HERE.parents[1]/"model_tuning/results/real_car_v2_dynamic_residual"
def main():
 parser=argparse.ArgumentParser();parser.add_argument("result",nargs="?",default=str(DEFAULT_RESULT));args=parser.parse_args();result=Path(args.result)
 d=np.load(DATASET);blob=np.fromfile(result/"dynamic_residual_v2.bin",dtype="<f4")
 assert blob.size==3563 and (result/"dynamic_residual_v2.bin").stat().st_size==14252
 off=0
 def take(n,shape):
  nonlocal off;q=blob[off:off+n].reshape(shape);off+=n;return q
 w1=take(1280,(64,20));b1=take(64,(64,));w2=take(2048,(32,64));b2=take(32,(32,));w3=take(96,(3,32));b3=take(3,(3,));mean=take(20,(20,));std=take(20,(20,));assert off==3563
 x=d["features"].astype(np.float32);idx=np.flatnonzero(d["valid"])[:4096];z=(x[idx]-mean)/std
 # Exact operation/order counterpart of CUDA output-major dot loops.
 h1=np.maximum(np.stack([b1[o]+np.sum(w1[o]*z,axis=1,dtype=np.float32) for o in range(64)],1),0)
 h2=np.maximum(np.stack([b2[o]+np.sum(w2[o]*h1,axis=1,dtype=np.float32) for o in range(32)],1),0)
 pred=np.stack([b3[o]+np.sum(w3[o]*h2,axis=1,dtype=np.float32) for o in range(3)],1)
 model=Net();model.load_state_dict(torch.load(result/"model.pt",map_location="cpu",weights_only=True));model.eval()
 with torch.no_grad():ref=model(torch.from_numpy(z)).numpy()
 err=np.abs(pred-ref);split=d["split"];test=d["valid"]&(split==2);target=d["targets"]
 physics=np.abs(target[test]);learned=np.abs(np.load(result/"predictions.npz",allow_pickle=True)["prediction"][test]-target[test])
 report={"float_count":int(blob.size),"bytes":int(blob.nbytes),"layout_ok":True,
  "python_vs_cuda_loop_max_abs":float(err.max()),"python_vs_cuda_loop_mean_abs":float(err.mean()),
  "test_physics_only_zero_residual_mae":physics.mean(0).tolist(),"test_learned_residual_mae":learned.mean(0).tolist(),
  "learned_over_physics_ratio":(learned.mean(0)/physics.mean(0)).tolist()}
 (result/"binary_verification.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
