#!/usr/bin/env python3
"""Step 1: extract the exact real-car MPPI observation path from rosbag2.

Pose is taken from /newmcl_pose, body velocity from /odom, controls from the
selected Ackermann topic, and IMU from /imu/data.  Every stream is aligned by
causal hold; no future sample is used.
"""
import argparse
import datetime as dtlib
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from helper_filter_collision_recovery import (
    collision_recovery_mask, physical_inconsistency_mask)
from helper_lateral_velocity_kf import LateralVelocityKFParams, estimate_dataset

# USER SETTINGS. Add every bag storage file or rosbag2 directory here. Running
# this script without arguments extracts them sequentially.
NEW_DATA_ROOTS = (
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0817 (1)"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0818"),
)
# Discover rosbag directories recursively. Bags without the required topics are
# reported as SKIPPED by read_streams instead of silently entering the archive.
BAG_PATH = sorted({metadata.parent for root in NEW_DATA_ROOTS
                   for metadata in root.rglob("metadata.yaml")})
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/data/ifac0817_0818_autonomous_physics_clean"
USE_PLOT = False
# The sensor/body convention changed on 2026-08-17. Before that date IMU y/z
# oppose MPPI FLU; from 0817 onward all axes already match MPPI.
IMU_WZ_SIGN = 1.0; IMU_AX_SIGN = 1.0; IMU_AY_SIGN = 1.0; IMU_EMA_ALPHA = .25
IMU_SIGN_CUTOVER = dtlib.date(2026, 8, 17)
# Match the current runtime observer in config/params.yaml.
KF_CF = 12.7222491; KF_CR = 75.0944752
KF_LOW_SPEED_THRESHOLD = 0.5
KF_STEER_SCALE = 1.1058064699; KF_STEER_BIAS = -0.0300696939; KF_MAX_STEER = .4788
POSE_TOPIC = "/newmcl_pose"; VELOCITY_TOPIC = "/odom"; COMMAND_TOPIC = "/ackermann_cmd"; IMU_TOPIC = "/imu/data"
APPLIED_COMMAND_TOPIC = "/drive"
COMMAND_STEER_MATCH_TOL = 1e-4; COMMAND_SPEED_MATCH_TOL = 1e-4
# A rollout starting shortly before a manual takeover still contains a response
# that the autonomous command cannot explain. Remove this causal context too.
MANUAL_PRE_MARGIN_S = 1.2; MANUAL_POST_MARGIN_S = .5
PHYSICS_PRE_MARGIN_S = 1.2; PHYSICS_POST_MARGIN_S = .5
PHYSICS_MOVING_VX = .7; PHYSICS_FROZEN_POSE_SPEED = .12
PHYSICS_DISTANCE_WINDOW_S = .5; PHYSICS_MIN_ODOM_DISTANCE = .35
PHYSICS_MIN_POSE_ODOM_RATIO = .65; PHYSICS_IMPACT_DECEL = -8.0
PHYSICS_MAX_POSE_STEP = .30; PHYSICS_MAX_YAW_STEP = .45
DT = .02; MAX_POSE_AGE = .10; MAX_VELOCITY_AGE = .10; MAX_COMMAND_AGE = .10; MAX_IMU_AGE = .05


def recording_date(path):
    text=str(path)
    match=re.search(r"(?:rosbag2_)?(20\d{2})[_-](\d{2})[_-](\d{2})",text)
    if match:return dtlib.date(*(int(value) for value in match.groups()))
    short=re.search(r"(?:^|[/_ (])(0[78]\d{2})(?:[/_ )]|$)",text)
    if short:
        value=short.group(1);return dtlib.date(2026,int(value[:2]),int(value[2:]))
    raise RuntimeError(f"{path}: cannot infer recording date for IMU sign convention")


def imu_signs(path):
    date=recording_date(path)
    return ((1.,1.,1.) if date>=IMU_SIGN_CUTOVER else (-1.,1.,-1.)),date


def resolve_storage(path):
    """Resolve a rosbag directory to its single .mcap/.db3 storage file."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    candidates = sorted([*path.glob("*.mcap"), *path.glob("*.db3")])
    if len(candidates) != 1:
        raise RuntimeError(f"{path}: expected one .mcap/.db3 file, found {candidates}")
    return candidates[0]


def output_for_bag(output, storage, multiple):
    output = Path(output)
    if not multiple and output.suffix == ".npz":
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{storage.parent.name}.npz"


def stamp_seconds(msg, record_ns):
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None or (stamp.sec == 0 and stamp.nanosec == 0):
        return record_ns * 1e-9
    return stamp.sec + stamp.nanosec * 1e-9


def yaw(q):
    return np.arctan2(2*(q.w*q.z+q.x*q.y),
                      1-2*(q.y*q.y+q.z*q.z))


def read_streams(storage, pose_topic, velocity_topic, drive_topic, imu_topic,
                 applied_command_topic=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage = Path(storage)
    storage_id = "mcap" if storage.suffix == ".mcap" else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(storage), storage_id=storage_id),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    topics = (pose_topic, velocity_topic, drive_topic, imu_topic)
    if applied_command_topic is not None:
        topics += (applied_command_topic,)
    missing = [x for x in topics if x not in types]
    if missing:
        raise RuntimeError(f"missing topics {missing}; available={sorted(types)}")
    msg_types = {x: get_message(types[x]) for x in topics}
    pose, velocity, drive, imu, applied = [], [], [], [], []
    while reader.has_next():
        topic, raw, record_ns = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(raw, msg_types[topic])
        t = stamp_seconds(msg, record_ns)
        if topic == pose_topic:
            p, q = msg.pose.position, msg.pose.orientation
            pose.append((t, p.x, p.y, yaw(q)))
        elif topic == velocity_topic:
            v = msg.twist.twist
            velocity.append((t, v.linear.x, v.linear.y, v.angular.z))
        elif topic == drive_topic:
            d = msg.drive
            drive.append((t, d.steering_angle, d.acceleration, d.speed))
        elif topic == applied_command_topic:
            d = msg.drive
            applied.append((t, d.steering_angle, d.acceleration, d.speed))
        else:
            imu.append((t, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y))
    result=tuple(np.asarray(x, np.float64) for x in (pose, velocity, drive, imu))
    return result if applied_command_topic is None else result+(np.asarray(applied,np.float64),)


def expand_boolean_intervals(mask, pre_samples, post_samples):
    """Expand each true interval without joining unrelated distant intervals."""
    indices=np.flatnonzero(mask)
    if not len(indices): return mask.copy()
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1);expanded=np.zeros_like(mask)
    for group in groups:
        lo=max(0,int(group[0])-pre_samples);hi=min(len(mask),int(group[-1])+1+post_samples)
        expanded[lo:hi]=True
    return expanded


def causal_hold(stream, times, max_age):
    stream = stream[np.argsort(stream[:, 0])]
    stream = stream[np.r_[True, np.diff(stream[:, 0]) > 1e-9]]
    index = np.searchsorted(stream[:, 0], times, side="right")-1
    valid = index >= 0
    clipped = np.maximum(index, 0)
    valid &= (times-stream[clipped, 0]) <= max_age
    return stream[clipped], valid


def plot_extracted(samples, columns, dt, title, command_topic, signs=(1.,1.,1.)):
    """Show one bag and block until its window is closed."""
    import matplotlib.pyplot as plt
    source_t=samples[:,0];x,y,heading=samples[:,1],samples[:,2],samples[:,3]
    vx,odom_vy,omega=samples[:,4],samples[:,5],samples[:,6]
    steer,speed_cmd=samples[:,7],samples[:,9]
    imu_wz,imu_ax,imu_ay=samples[:,12],samples[:,13],samples[:,14]
    segment=samples[:,11].astype(int);t=np.empty(len(samples));wx=np.empty(len(samples));wy=np.empty(len(samples))
    elapsed=0.
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id);local=source_t[ii]-source_t[ii[0]]
        t[ii]=elapsed+local;elapsed=t[ii[-1]]+(np.median(np.diff(local)) if len(ii)>1 else DT)
        edge=2 if len(ii)>=3 else 1
        wx[ii]=np.gradient(x[ii],local,edge_order=edge) if len(ii)>1 else 0.
        wy[ii]=np.gradient(y[ii],local,edge_order=edge) if len(ii)>1 else 0.
    pose_vx=wx*np.cos(heading)+wy*np.sin(heading)
    pose_vy=-wx*np.sin(heading)+wy*np.cos(heading)
    kf_params=LateralVelocityKFParams(cornering_stiffness_front=KF_CF,
        cornering_stiffness_rear=KF_CR,dt=dt,
        low_speed_threshold=KF_LOW_SPEED_THRESHOLD)
    wz_sign,ax_sign,ay_sign=signs
    kf_vy,kf_w=estimate_dataset(samples,columns,dt,kf_params,
        steer_scale=KF_STEER_SCALE,steer_bias=KF_STEER_BIAS,max_steer=KF_MAX_STEER,
        imu_ema_alpha=IMU_EMA_ALPHA,imu_wz_sign=wz_sign,imu_ay_sign=ay_sign)
    fig,axes=plt.subplots(5,2,figsize=(15,21));fig.suptitle(title,y=.995)
    ax=axes[0,0];ax.plot(x,y,"k-",lw=1.5,label="pose trajectory")
    stride=max(1,len(samples)//35)
    ax.quiver(x[::stride],y[::stride],np.cos(heading[::stride]),np.sin(heading[::stride]),
              heading[::stride],cmap="hsv",angles="xy",scale_units="xy",scale=3.5,width=.004,
              label="yaw direction")
    ax.set_title("x-y trajectory with yaw direction/color");ax.set_xlabel("x [m]");ax.set_ylabel("y [m]")
    ax.axis("equal");ax.grid(alpha=.25);ax.legend()
    signed_imu_wz=wz_sign*imu_wz
    axes[0,1].plot(t,np.unwrap(heading),label="pose yaw [rad]")
    axes[0,1].plot(t,signed_imu_wz,label=f"signed IMU yaw-rate (raw x {wz_sign:+.0f})")
    axes[0,1].plot(t,imu_wz,":",color="0.65",label="raw IMU yaw-rate (sensor frame)")
    axes[0,1].plot(t,omega,label="odom yaw-rate [rad/s]")
    axes[0,1].plot(t,kf_w,label="KF yaw-rate [rad/s]",alpha=.8)
    axes[0,1].set_title("Yaw and yaw-rate observations")
    axes[1,0].plot(t,x,label="MCL global x");axes[1,0].plot(t,y,label="MCL global y")
    axes[1,0].set_ylabel("position [m]");axes[1,0].set_title("/newmcl_pose x and y vs time")
    axes[1,1].plot(t,np.unwrap(heading),label="MCL yaw (unwrapped)")
    axes[1,1].plot(t,heading,"--",alpha=.6,label="MCL yaw (wrapped)")
    axes[1,1].set_ylabel("yaw [rad]");axes[1,1].set_title("/newmcl_pose yaw vs time")
    axes[2,0].plot(t,vx,label="odom vx");axes[2,0].plot(t,pose_vx,label="pose-derived vx")
    axes[2,0].plot(t,speed_cmd,label=f"{command_topic} speed command");axes[2,0].set_title("Longitudinal velocity")
    axes[2,1].plot(t,odom_vy,label="stored odom vy");axes[2,1].plot(t,pose_vy,label="pose-derived vy")
    axes[2,1].plot(t,kf_vy,label="training/runtime KF vy",lw=1.5)
    axes[2,1].set_title("Lateral velocity inputs/estimates")
    axes[3,0].plot(t,ax_sign*imu_ax,label=f"signed training IMU ax (raw x {ax_sign:+.0f})")
    axes[3,0].plot(t,imu_ay,":",color="0.65",label="raw IMU ay (sensor frame)")
    axes[3,0].plot(t,ay_sign*imu_ay,label=f"signed training IMU ay (raw x {ay_sign:+.0f})",alpha=.9)
    axes[3,0].set_title("IMU acceleration and configured ay sign")
    axes[3,1].plot(t,steer,label=f"{command_topic} steering command");axes[3,1].set_title("Steering command")
    speed=np.hypot(vx,odom_vy);beta=np.arctan2(odom_vy,np.maximum(np.abs(vx),1e-4))
    axes[4,0].plot(t,speed,label="odom speed");axes[4,0].plot(t,beta,label="raw-odom beta")
    axes[4,0].set_title("Derived speed and slip angle")
    axes[4,1].plot(t,segment,label="continuous segment/bag_id")
    axes[4,1].set_title("Collision-cleaned continuous segments")

    print(f"Showing {title}. Close this window to process the next bag.")
    for axis in axes.flat:
        axis.set_xlabel("time [s]");axis.grid(alpha=.25);axis.legend(fontsize=8)
    # Keep titles, x labels and legends from touching the neighboring row.
    fig.subplots_adjust(left=.08,right=.97,bottom=.06,top=.95,hspace=.48,wspace=.25)
    print(f"Showing {title}. Close this window to process the next bag.")
    plt.show(block=True)
    plt.close(fig)


def extract_one(storage, out, args):
    signs,date=imu_signs(storage)
    pose,velocity,drive,imu,applied=read_streams(storage,args.pose_topic,args.velocity_topic,
        args.command_topic,args.imu_topic,args.applied_command_topic)
    streams={"pose":pose,"velocity":velocity,"command":drive,"imu":imu,"applied_command":applied}
    empty=[name for name,value in streams.items() if not len(value)]
    if empty: raise RuntimeError(f"{storage}: empty streams {empty}")
    start=max(x[0,0] for x in streams.values());end=min(x[-1,0] for x in streams.values())
    if end<=start: raise RuntimeError(f"{storage}: streams have no overlapping interval")
    times=np.arange(start,end,args.dt)
    pp,pv=causal_hold(pose,times,args.max_pose_age);vv,vvalid=causal_hold(velocity,times,args.max_velocity_age)
    dd,dvalid=causal_hold(drive,times,args.max_command_age);ii,ivalid=causal_hold(imu,times,args.max_imu_age)
    aa,avalid=causal_hold(applied,times,args.max_command_age)
    valid=pv&vvalid&dvalid&ivalid&avalid;times,pp,vv,dd,ii,aa=(x[valid] for x in (times,pp,vv,dd,ii,aa))
    if not len(times): raise RuntimeError(f"{storage}: no samples survived causal alignment")
    base=np.c_[times-times[0],pp[:,1:4],vv[:,1:4],dd[:,1:4]]
    command_mismatch=(np.abs(dd[:,1]-aa[:,1])>args.command_steer_match_tol)|(np.abs(dd[:,3]-aa[:,3])>args.command_speed_match_tol)
    manual=expand_boolean_intervals(command_mismatch,round(args.manual_pre_margin/args.dt),round(args.manual_post_margin/args.dt))
    collision,episodes=collision_recovery_mask(base,args.dt)
    physical_bad,physical_events=physical_inconsistency_mask(base,args.dt,
        pre_margin_s=args.physics_pre_margin,post_margin_s=args.physics_post_margin,
        moving_vx=args.physics_moving_vx,frozen_pose_speed=args.physics_frozen_pose_speed,
        distance_window_s=args.physics_distance_window,min_odom_distance=args.physics_min_odom_distance,
        min_pose_odom_ratio=args.physics_min_pose_odom_ratio,impact_decel=args.physics_impact_decel,
        max_pose_step=args.physics_max_pose_step,max_yaw_step=args.physics_max_yaw_step)
    bad=collision|manual|physical_bad;kept=np.flatnonzero(~bad)
    if not len(kept): raise RuntimeError(f"{storage}: collision filtering removed every sample")
    breaks=np.flatnonzero((np.diff(kept)>1)|(np.diff(base[kept,0])>1.5*args.dt))+1
    arrays=[];segments=[]
    for bag_id,run in enumerate(np.split(kept,breaks)):
        if not len(run): continue
        part=base[run].copy();source_start=float(part[0,0]);part[:,0]-=source_start
        arrays.append(np.c_[part,np.ones(len(part)),np.full(len(part),bag_id),ii[run,1:4]])
        segments.append({"bag_id":bag_id,"samples":len(part),"source_start_s":source_start,
                         "source_end_s":float(base[run[-1],0])})
    samples=np.concatenate(arrays)
    columns=np.array(["t","x","y","yaw","vx","vy","omega","steer","accel","speed_cmd",
                      "split","bag_id","imu_wz","imu_ax","imu_ay"])
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,samples=samples,dt=args.dt,columns=columns,pose_topic=np.array(args.pose_topic),
                        velocity_topic=np.array(args.velocity_topic),command_topic=np.array(args.command_topic),
                        imu_topic=np.array(args.imu_topic),
                        recording_date=np.array(date.isoformat()),
                        imu_sign_cutover=np.array(IMU_SIGN_CUTOVER.isoformat()),
                        imu_axis_signs=np.array(signs,np.float32),
                        imu_ema_alpha=np.array(IMU_EMA_ALPHA,np.float32),
                        kf_cornering_stiffness=np.array([KF_CF,KF_CR],np.float32),
                        kf_low_speed_threshold=np.array(KF_LOW_SPEED_THRESHOLD,np.float32))
    meta={"source":str(storage.resolve()),"pose_topic":args.pose_topic,"velocity_topic":args.velocity_topic,
          "command_topic":args.command_topic,"imu_topic":args.imu_topic,"alignment":"causal_hold",
          "recording_date":date.isoformat(),"imu_sign_cutover":"2026-08-17",
          "imu_axis_signs":{"wz":signs[0],"ax":signs[1],"ay":signs[2]},
          "imu_ema_alpha":IMU_EMA_ALPHA,
          "kf_parameters":{"cornering_stiffness_front":KF_CF,"cornering_stiffness_rear":KF_CR,
                           "low_speed_threshold":KF_LOW_SPEED_THRESHOLD,"steer_scale":KF_STEER_SCALE,
                           "steer_bias":KF_STEER_BIAS,"max_steer":KF_MAX_STEER},
          "applied_command_topic":args.applied_command_topic,"raw_aligned_samples":len(base),
          "removed_collision_samples":int(collision.sum()),"raw_command_mismatch_samples":int(command_mismatch.sum()),
          "removed_manual_with_margin_samples":int(manual.sum()),
          "removed_physical_inconsistency_samples":int(physical_bad.sum()),
          "physical_inconsistency_events":physical_events,
          "physical_filter":{"pre_margin_s":args.physics_pre_margin,"post_margin_s":args.physics_post_margin,
              "moving_vx_mps":args.physics_moving_vx,"frozen_pose_speed_mps":args.physics_frozen_pose_speed,
              "distance_window_s":args.physics_distance_window,"min_odom_distance_m":args.physics_min_odom_distance,
              "min_mcl_odom_distance_ratio":args.physics_min_pose_odom_ratio,
              "impact_decel_mps2":args.physics_impact_decel,"max_pose_step_m":args.physics_max_pose_step,
              "max_yaw_step_rad":args.physics_max_yaw_step},
          "manual_filter":{"steer_tolerance":args.command_steer_match_tol,"speed_tolerance":args.command_speed_match_tol,
                           "pre_margin_s":args.manual_pre_margin,"post_margin_s":args.manual_post_margin},
          "output_samples":len(samples),"collision_episodes":episodes,"segments":segments,
          "split_policy":"single-bag-identical-train-test"}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({**meta,"output":str(out)},indent=2))
    return samples,columns,signs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag", nargs="*", help="optional .db3/.mcap files or rosbag2 directories; default=BAG_PATH")
    p.add_argument("-o", "--output", default=str(OUTPUT_PATH))
    p.add_argument("--pose-topic", default=POSE_TOPIC)
    p.add_argument("--velocity-topic", default=VELOCITY_TOPIC)
    p.add_argument("--command-topic", "--drive-topic", dest="command_topic", default=COMMAND_TOPIC,
                   help="Ackermann command source; default is /ackermann_cmd")
    p.add_argument("--imu-topic", default=IMU_TOPIC)
    p.add_argument("--applied-command-topic",default=APPLIED_COMMAND_TOPIC)
    p.add_argument("--command-steer-match-tol",type=float,default=COMMAND_STEER_MATCH_TOL)
    p.add_argument("--command-speed-match-tol",type=float,default=COMMAND_SPEED_MATCH_TOL)
    p.add_argument("--manual-pre-margin",type=float,default=MANUAL_PRE_MARGIN_S)
    p.add_argument("--manual-post-margin",type=float,default=MANUAL_POST_MARGIN_S)
    p.add_argument("--physics-pre-margin",type=float,default=PHYSICS_PRE_MARGIN_S)
    p.add_argument("--physics-post-margin",type=float,default=PHYSICS_POST_MARGIN_S)
    p.add_argument("--physics-moving-vx",type=float,default=PHYSICS_MOVING_VX)
    p.add_argument("--physics-frozen-pose-speed",type=float,default=PHYSICS_FROZEN_POSE_SPEED)
    p.add_argument("--physics-distance-window",type=float,default=PHYSICS_DISTANCE_WINDOW_S)
    p.add_argument("--physics-min-odom-distance",type=float,default=PHYSICS_MIN_ODOM_DISTANCE)
    p.add_argument("--physics-min-pose-odom-ratio",type=float,default=PHYSICS_MIN_POSE_ODOM_RATIO)
    p.add_argument("--physics-impact-decel",type=float,default=PHYSICS_IMPACT_DECEL)
    p.add_argument("--physics-max-pose-step",type=float,default=PHYSICS_MAX_POSE_STEP)
    p.add_argument("--physics-max-yaw-step",type=float,default=PHYSICS_MAX_YAW_STEP)
    p.add_argument("--dt", type=float, default=DT)
    # 0815 topics are near 50 Hz or faster. Reject stale held samples instead
    # of silently turning a topic dropout into apparently valid dynamics data.
    p.add_argument("--max-pose-age", type=float, default=MAX_POSE_AGE)
    p.add_argument("--max-velocity-age", type=float, default=MAX_VELOCITY_AGE)
    p.add_argument("--max-command-age", type=float, default=MAX_COMMAND_AGE)
    p.add_argument("--max-imu-age", type=float, default=MAX_IMU_AGE)
    args = p.parse_args()

    requested=[Path(x) for x in args.bag] if args.bag else list(BAG_PATH)
    if not requested: raise SystemExit("BAG_PATH is empty")
    storages=[resolve_storage(x) for x in requested]
    for number,storage in enumerate(storages,1):
        out=output_for_bag(args.output,storage,len(storages)>1)
        print(f"[{number}/{len(storages)}] Extracting {storage} -> {out}")
        try:
            samples,columns,signs=extract_one(storage,out,args)
        except RuntimeError as error:
            print(f"[{number}/{len(storages)}] SKIPPED: {error}",file=sys.stderr)
            continue
        if USE_PLOT: plot_extracted(samples,columns,args.dt,
                                    f"Bag {number}/{len(storages)}: {storage.parent.name}",
                                    args.command_topic,signs)


if __name__ == "__main__":
    main()
