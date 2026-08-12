#!/usr/bin/env python3
"""Build a non-destructive 40 ms transition dataset from 0813 aligned NPZs."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from contract import EffectiveContract, FEATURES, HISTORY_STEPS, baseline_body_step, make_features

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for q in iter(lambda:f.read(1<<20),b""): h.update(q)
    return h.hexdigest()

def centerline_lap(xy, centerline):
    d2=((xy[:,None,:]-centerline[None,:,:])**2).sum(2); idx=d2.argmin(1).astype(float); n=len(centerline)
    un=idx.copy()
    for i in range(1,len(un)):
        delta=idx[i]-idx[i-1]
        if delta < -n/2: delta += n
        elif delta > n/2: delta -= n
        un[i]=un[i-1]+delta
    return np.floor((un-un[0])/n).astype(np.int16)+1, un/n, np.sqrt(d2.min(1))

def run(args):
    c=EffectiveContract(); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    center=np.loadtxt(args.centerline,delimiter=",",skiprows=1,usecols=(0,1))
    chunks=[]; sources=[]; invalid_counts={}
    for session,path in (("speed20",args.speed20),("speed25",args.speed25),("speed30",args.speed30)):
        path=Path(path); d=np.load(path); cols=d["columns"].tolist(); a=d["samples"]
        q={name:a[:,cols.index(name)] for name in cols}; n=len(a)
        lap,progress,cte=centerline_lap(a[:,[cols.index("x"),cols.index("y")]],center)
        yaw=np.unwrap(q["yaw"]); t=q["t"]
        # Offline targets may be non-causal. Runtime input remains causal IMU/KF.
        win=min(31, n-(1-n%2)); win=max(win,5)
        sx=savgol_filter(q["x"],win,3); sy=savgol_filter(q["y"],win,3)
        wx=savgol_filter(q["x"],win,3,deriv=1,delta=.02); wy=savgol_filter(q["y"],win,3,deriv=1,delta=.02)
        # Runtime vx is odometry velocity, so its offline target must retain the
        # same scale. Pose-derived vx is useful for auditing position scale but
        # mixing it with odom vx in one transition creates a fictitious residual.
        vx_target=savgol_filter(q["vx"],win,3)
        vy_target=-wx*np.sin(yaw)+wy*np.cos(yaw)
        # IMU yaw-rate is directly observed at runtime and is markedly cleaner
        # than differentiating MCL yaw. Non-causal smoothing is target-only.
        r_target=savgol_filter(-q["imu_wz"],win,3)
        ax=savgol_filter(q["imu_ax"],11,2); ay=savgol_filter(-q["imu_ay"],11,2)
        body=np.c_[q["vx"], np.zeros(n), -q["imu_wz"]] # runtime-causal convention
        target=np.c_[vx_target,vy_target,r_target]
        commands=np.c_[q["steer"],q["speed_cmd"]]
        valid=np.ones(n,bool); reason=np.zeros(n,np.uint16)
        finite=np.isfinite(np.c_[a,target,ax,ay]).all(1); valid&=finite; reason[~finite]|=1
        gap=np.r_[False,np.diff(t)>.03]; valid[gap]=False; reason[gap]|=2
        jump=np.r_[False,np.hypot(np.diff(q["x"]),np.diff(q["y"]))>.25]; valid[jump]=False; reason[jump]|=4
        for i in range(min(HISTORY_STEPS-1,n)): valid[i]=False;reason[i]|=8
        valid[max(0,n-120):]=False;reason[max(0,n-120):]|=16
        if session=="speed30": split=np.full(n,2,np.int8)
        else: split=np.where(lap<=5,0,1).astype(np.int8)
        boundaries=np.flatnonzero(np.r_[False,np.diff(split)!=0])
        for b in boundaries:
            lo=max(0,b-120);hi=min(n,b+120);valid[lo:hi]=False;reason[lo:hi]|=32
        ids=np.arange(HISTORY_STEPS-1,n-2)
        hist=np.stack([commands[ids-k] for k in range(9,-1,-1)],1)
        feature=make_features(body[ids],np.c_[ax[ids],ay[ids]],hist)
        base=baseline_body_step(body[ids],commands[ids],commands[ids+1],c)
        residual=target[ids+2]-base
        transition_valid=valid[ids]&valid[ids+1]&valid[ids+2]
        chunks.append(dict(features=feature.astype(np.float32),targets=residual.astype(np.float32),
          state=np.c_[q["x"][ids],q["y"][ids],q["yaw"][ids],body[ids]].astype(np.float32),
          next_state=np.c_[sx[ids+2],sy[ids+2],((yaw[ids+2]+np.pi)%(2*np.pi)-np.pi),target[ids+2]].astype(np.float32),
          command_t=commands[ids].astype(np.float32),command_t1=commands[ids+1].astype(np.float32),
          command_history=hist.astype(np.float32),t=t[ids],lap_id=lap[ids],progress=progress[ids],
          centerline_error=cte[ids].astype(np.float32),split=split[ids],valid=transition_valid,
          invalid_reason=reason[ids],session=np.full(len(ids),session),bag_id=np.zeros(len(ids),np.int16)))
        rv=residual[transition_valid]
        invalid_counts[session]={"raw":n,"transitions":len(ids),"valid":int(transition_valid.sum()),
          "invalid":int((~transition_valid).sum()),"laps_detected":int(lap.max()),
          "residual_target_abs_p99":np.quantile(np.abs(rv),.99,axis=0).tolist()}
        sources.append({"session":session,"path":str(path),"sha256":sha256(path),"samples":n})
    keys=chunks[0].keys(); merged={k:np.concatenate([q[k] for q in chunks]) for k in keys}
    np.savez_compressed(out/"dataset_model_dt004.npz",**merged,feature_names=np.array(FEATURES),
      control_dt=np.float64(.02),model_dt=np.float64(.04))
    c.dump(out/"contract.json")
    manifest={"sources":sources,"counts":invalid_counts,"split":{"0":"train laps 1-5 speed20/25","1":"validation remaining speed20/25","2":"speed30 holdout"},
      "invalid_reason_bits":{"1":"nonfinite","2":"timestamp_gap","4":"pose_jump","8":"insufficient_history","16":"insufficient_2.4s_horizon","32":"2.4s_split_purge"},
      "limitations":["source NPZ omits per-topic source timestamps/ages","teleop and lpf_imu contain zero messages per rosbag metadata","raw bag re-extraction required for clock-domain and stale-topic audit"]}
    (out/"dataset_50hz_raw_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"out":str(out),"features":len(FEATURES),"counts":invalid_counts},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();root=Path("/mnt/nas_custom/F1tenth/2026 IFAC/0813/extracted")
    p.add_argument("--speed20",default=root/"speed20.npz");p.add_argument("--speed25",default=root/"speed25.npz");p.add_argument("--speed30",default=root/"speed30.npz")
    p.add_argument("--centerline",default=HERE.parents[1]/"data/map1/map1_centerline.csv");p.add_argument("--out",required=True);run(p.parse_args())
