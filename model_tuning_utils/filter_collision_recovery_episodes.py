#!/usr/bin/env python3
"""Remove collision/reverse-recovery episodes while preserving normal driving."""
import argparse, json
from pathlib import Path
import numpy as np


def collision_recovery_mask(samples, dt, reverse_speed=-.15, min_reverse_s=.06,
                            merge_gap_s=.5, lookback_s=3., stable_forward_s=.5):
    """Return removal mask and intervals using signed measured body vx."""
    vx=samples[:,4]
    minimum=max(1,round(min_reverse_s/dt))
    seed=np.convolve((vx<reverse_speed).astype(np.int16),np.ones(minimum,np.int16),mode="same")>=minimum
    indices=np.flatnonzero(seed)
    if not len(indices): return np.zeros(len(samples),bool),[]
    gap=round(merge_gap_s/dt); groups=np.split(indices,np.flatnonzero(np.diff(indices)>gap)+1)
    mask=np.zeros(len(samples),bool); intervals=[]; stable=max(1,round(stable_forward_s/dt))
    for group in groups:
        reverse_start=max(0,int(group[0])-minimum//2); reverse_end=min(len(samples)-1,int(group[-1])+minimum//2)
        # Collision onset: last healthy forward sample before the recovery reverse.
        lo=max(0,reverse_start-round(lookback_s/dt)); healthy=np.flatnonzero(vx[lo:reverse_start]>.7)
        start=(lo+int(healthy[-1])+1) if len(healthy) else max(lo,reverse_start-round(.5/dt))
        # Retain data again only after stable forward motion has resumed.
        forward=(vx>.3).astype(np.int16)
        sums=np.convolve(forward,np.ones(stable,np.int16),mode="valid")
        candidates=np.flatnonzero(sums[reverse_end:]>=stable)
        end=(reverse_end+int(candidates[0])) if len(candidates) else len(samples)
        mask[start:end]=True
        intervals.append({"start_index":start,"end_index_exclusive":end,
                          "start_time_s":float(samples[start,0]),
                          "end_time_s":float(samples[min(end,len(samples)-1),0]),
                          "reverse_start_time_s":float(samples[reverse_start,0]),
                          "reverse_end_time_s":float(samples[reverse_end,0])})
    return mask,intervals


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("dataset"); p.add_argument("-o","--output",required=True)
    p.add_argument("--reverse-speed",type=float,default=-.15); p.add_argument("--lookback",type=float,default=3.)
    p.add_argument("--stable-forward",type=float,default=.5); a=p.parse_args()
    src=Path(a.dataset); z=np.load(src); raw=z["samples"]; source_bag=raw[:,11].astype(int); dt=float(z["dt"])
    pieces=[]; segments=[]; episodes=[]; removed=0; next_id=0
    for bid in np.unique(source_bag):
        ii=np.flatnonzero(source_bag==bid); local=raw[ii]
        bad,found=collision_recovery_mask(local,dt,a.reverse_speed,lookback_s=a.lookback,stable_forward_s=a.stable_forward)
        for x in found: x["source_bag_id"]=int(bid)
        episodes.extend(found); removed+=int(bad.sum()); kept=np.flatnonzero(~bad)
        if not len(kept): continue
        breaks=np.flatnonzero((np.diff(kept)>1)|(np.diff(local[kept,0])>1.5*dt))+1
        for segment_index,run in enumerate(np.split(kept,breaks)):
            part=local[run].copy();source_time_start=float(part[0,0]);source_time_end=float(part[-1,0])
            part[:,0]-=source_time_start;part[:,11]=next_id; pieces.append(part)
            segments.append({"segment_bag_id":next_id,"source_bag_id":int(bid),"segment_index":segment_index,
                             "samples":len(part),"split":"test" if int(part[0,10]) else "train",
                             "source_time_start_s":source_time_start,"source_time_end_s":source_time_end,
                             "usable_2s_window":bool(len(part)>=100)})
            next_id+=1
    clean=np.concatenate(pieces); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    payload={k:z[k] for k in z.files if k!="samples"}; payload["samples"]=clean; np.savez_compressed(out,**payload)
    old=src.with_suffix(".manifest.json"); manifest=json.loads(old.read_text()) if old.exists() else {}
    rule={"reverse_anchor":"measured body vx < -0.15 m/s for >=0.06 s","collision_start":"after last vx > 0.7 m/s within 3 s before reverse","recovery_end":"first vx > 0.3 m/s sustained for 0.5 s"}
    manifest.update({"collision_recovery_filter":rule,"collision_recovery_episodes":episodes,"segments":segments,
                     "removed_samples":removed,"retained_samples":len(clean)})
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summary={"input_samples":len(raw),"retained_samples":len(clean),"removed_collision_recovery_samples":removed,
             "episodes":len(episodes),"affected_source_bags":sorted({x["source_bag_id"] for x in episodes}),
             "segments":len(segments),"rule":rule}
    out.with_suffix(".collision_filter.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
