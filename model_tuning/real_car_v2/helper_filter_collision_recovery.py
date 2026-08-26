#!/usr/bin/env python3
"""Remove collision/reverse-recovery episodes while preserving normal driving."""
import argparse, json
from pathlib import Path
import numpy as np


def _expanded_intervals(mask, pre_samples, post_samples):
    indices=np.flatnonzero(mask); expanded=np.zeros(len(mask),bool)
    if not len(indices): return expanded,[]
    for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
        start=max(0,int(group[0])-pre_samples)
        end=min(len(mask),int(group[-1])+1+post_samples)
        expanded[start:end]=True
    expanded_indices=np.flatnonzero(expanded)
    intervals=[(int(group[0]),int(group[-1])+1) for group in
               np.split(expanded_indices,np.flatnonzero(np.diff(expanded_indices)>1)+1)]
    return expanded,intervals


def physical_inconsistency_mask(samples, dt, pre_margin_s=1.2, post_margin_s=.5,
                                moving_vx=.7, moving_command=.7, frozen_pose_speed=.12,
                                distance_window_s=.5, min_odom_distance=.35,
                                min_pose_odom_ratio=.65, impact_decel=-8.,
                                max_pose_step=.30, max_yaw_step=.45,
                                filter_localization_jumps=True):
    """Remove impact/wheel-spin/localization states that no vehicle model can fit.

    ``samples`` follows the extractor base layout
    [t,x,y,yaw,vx,vy,omega,steer,accel,speed_cmd].  This is an offline data
    quality filter; centered windows are intentional and never used online.
    """
    count=len(samples)
    if count<2:return np.zeros(count,bool),[]
    x,y,yaw,vx,speed_command=(samples[:,i] for i in (1,2,3,4,9))
    position_step=np.hypot(np.diff(x,prepend=x[0]),np.diff(y,prepend=y[0]))
    pose_speed=position_step/dt
    absolute_vx=np.abs(vx)
    window=max(3,round(distance_window_s/dt));kernel=np.ones(window)
    pose_distance=np.convolve(position_step,kernel,mode="same")
    odom_distance=np.convolve(absolute_vx*dt,kernel,mode="same")
    command_distance=np.convolve(np.abs(speed_command)*dt,kernel,mode="same")
    # Causal-hold naturally repeats MCL samples between localization callbacks.
    # Judge a genuinely blocked car from distance accumulated over a full
    # window, never from the fraction of individual zero pose steps.
    maximum_frozen_distance=frozen_pose_speed*distance_window_s
    pose_is_frozen=pose_distance<maximum_frozen_distance
    minimum_odom_distance=max(min_odom_distance,moving_vx*distance_window_s)
    minimum_command_distance=max(min_odom_distance,moving_command*distance_window_s)
    frozen=pose_is_frozen&(odom_distance>minimum_odom_distance)
    # A command can legitimately precede vehicle motion by actuator lag.
    # Require wheel odometry to corroborate motion before declaring collision.
    commanded_frozen=(pose_is_frozen&(command_distance>minimum_command_distance)
                       &(odom_distance>minimum_odom_distance))
    ratio=pose_distance/np.maximum(odom_distance,1e-6)
    wheel_spin_or_blocked=frozen&(ratio<min_pose_odom_ratio)
    longitudinal_accel=np.r_[0.,np.diff(vx)/dt]
    # Hard braking alone is valid, especially in manual driving. Treat a large
    # vx drop as collision evidence only when pose/odom or commanded-motion
    # consistency is already broken.
    impact=(longitudinal_accel<impact_decel)&(frozen|commanded_frozen|wheel_spin_or_blocked)
    yaw_step=np.abs((np.diff(yaw,prepend=yaw[0])+np.pi)%(2*np.pi)-np.pi)
    localization_jump=(position_step>max_pose_step)|(yaw_step>max_yaw_step)
    causes={"frozen_pose_with_nonzero_vx":frozen,
            "commanded_motion_without_pose_change":commanded_frozen,
            "mcl_odom_distance_mismatch":wheel_spin_or_blocked,
            "impact_like_vx_drop":impact,"localization_jump":localization_jump}
    active_causes=(causes.values() if filter_localization_jumps else
                   (value for name,value in causes.items()
                    if name != "localization_jump"))
    seed=np.logical_or.reduce(tuple(active_causes))
    expanded,intervals=_expanded_intervals(seed,round(pre_margin_s/dt),round(post_margin_s/dt))
    events=[]
    for start,end in intervals:
        event_causes=[name for name,value in causes.items() if value[start:end].any()]
        events.append({"start_index":start,"end_index_exclusive":end,
                       "start_time_s":float(samples[start,0]),
                       "end_time_s":float(samples[min(end,count-1),0]),
                       "causes":event_causes})
    return expanded,events


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
        # Wheel odometry can remain high after impact, so vx>0.7 is not proof
        # of healthy forward motion. Conservatively remove the complete
        # pre-recovery lookback interval; otherwise the collision approach can
        # survive as a seemingly valid high-speed training transition.
        start=max(0,reverse_start-round(lookback_s/dt))
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
    rule={"reverse_anchor":"measured body vx < -0.15 m/s for >=0.06 s",
          "collision_start":"full 3 s lookback before reverse; wheel odometry is not trusted after impact",
          "recovery_end":"first vx > 0.3 m/s sustained for 0.5 s"}
    manifest.update({"collision_recovery_filter":rule,"collision_recovery_episodes":episodes,"segments":segments,
                     "removed_samples":removed,"retained_samples":len(clean)})
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    summary={"input_samples":len(raw),"retained_samples":len(clean),"removed_collision_recovery_samples":removed,
             "episodes":len(episodes),"affected_source_bags":sorted({x["source_bag_id"] for x in episodes}),
             "segments":len(segments),"rule":rule}
    out.with_suffix(".collision_filter.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
