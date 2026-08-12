#!/usr/bin/env python3
import argparse, json, numpy as np
from contract import FEATURES
p=argparse.ArgumentParser(); p.add_argument("dataset"); a=p.parse_args(); d=np.load(a.dataset,allow_pickle=True)
x=np.asarray(d["features"]); b=np.asarray(d["bag_id"]); valid=np.asarray(d.get("valid",np.ones(len(x),bool)))
assert x.ndim==2 and x.shape[1]==len(FEATURES) and len(b)==len(x)
report={"samples":len(x),"valid_fraction":float(valid.mean()),"bags":int(len(np.unique(b))),"feature_min":dict(zip(FEATURES,np.nanmin(x,0).tolist())),"feature_max":dict(zip(FEATURES,np.nanmax(x,0).tolist()))}
print(json.dumps(report,indent=2))
