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
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from helper_filter_collision_recovery import (
    collision_recovery_mask, physical_inconsistency_mask)
from classic_model_kalman_smoother import filter_classic_segment

# USER SETTINGS. Add every bag storage file or rosbag2 directory here. Running
# this script without arguments extracts them sequentially.
NEW_DATA_ROOTS = (
    # Path("/mnt/nas_custom/F1tenth/2026 IFAC/0817 (1)"),
    # Path("/mnt/nas_custom/F1tenth/2026 IFAC/0818"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0819"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0820")
)
# Discover rosbag directories recursively. Bags without the required topics are
# reported as SKIPPED by read_streams instead of silently entering the archive.
BAG_PATH = sorted({metadata.parent for root in NEW_DATA_ROOTS
                   for metadata in root.rglob("metadata.yaml")})
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
# F5/direct execution is an interactive inspection workflow.  Set this to
# False only for unattended batch extraction.
USE_PLOT = True
# Bag-to-pose verification shows that the sensor/body convention changed by
# 2026-08-15. 0810--0813 y/z oppose MPPI FLU; 0815 onward already match it.
IMU_WZ_SIGN = 1.0; IMU_AX_SIGN = 1.0; IMU_AY_SIGN = 1.0; IMU_EMA_ALPHA = .25
IMU_SIGN_CUTOVER = dtlib.date(2026, 8, 15)
POSE_TOPIC = "/newmcl_pose"; VELOCITY_TOPIC = "/odom"; COMMAND_TOPIC = "/ackermann_cmd"; IMU_TOPIC = "/imu/data"
APPLIED_COMMAND_TOPIC = "/drive"
COMMAND_STEER_MATCH_TOL = 1e-4; COMMAND_SPEED_MATCH_TOL = 1e-4
# A rollout starting shortly before a manual takeover still contains a response
# that the autonomous command cannot explain. Remove this causal context too.
MANUAL_PRE_MARGIN_S = 1.2; MANUAL_POST_MARGIN_S = .5
PHYSICS_PRE_MARGIN_S = 1.2; PHYSICS_POST_MARGIN_S = .5
COLLISION_PRE_MARGIN_S = .5
PHYSICS_MOVING_VX = .7; PHYSICS_FROZEN_POSE_SPEED = .12
PHYSICS_DISTANCE_WINDOW_S = .5; PHYSICS_MIN_ODOM_DISTANCE = .35
PHYSICS_MIN_POSE_ODOM_RATIO = .65; PHYSICS_IMPACT_DECEL = -8.0
PHYSICS_MAX_POSE_STEP = .30; PHYSICS_MAX_YAW_STEP = .45
DT = .02; MAX_POSE_AGE = .10; MAX_VELOCITY_AGE = .10; MAX_COMMAND_AGE = .10; MAX_IMU_AGE = .05
MIN_CONTINUOUS_SEGMENT_S = 2.0
PLOT_ARROW_INTERVAL_S = .10  # data remain 50 Hz; direction arrows are drawn at 10 Hz
PLOT_TIME_LABEL_INTERVAL_S = 1.0


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


def causal_mcl_body_vy(times,x,y,yaw,window_s=.12):
    """Match the runtime trailing-window MCL regression used as KF vy observation."""
    result=np.full(len(times),np.nan)
    for k in range(len(times)):
        lo=np.searchsorted(times,times[k]-window_s,side="left")
        if k-lo+1<3:continue
        local=times[lo:k+1]-times[lo];centered=local-local.mean();denominator=centered@centered
        if denominator<=1e-8:continue
        world_vx=centered@(x[lo:k+1]-x[lo:k+1].mean())/denominator
        world_vy=centered@(y[lo:k+1]-y[lo:k+1].mean())/denominator
        result[k]=-np.sin(yaw[k])*world_vx+np.cos(yaw[k])*world_vy
    return result


def causal_ema(values,alpha):
    result=np.empty_like(values,dtype=float);result[0]=values[0]
    for k in range(1,len(values)):result[k]=alpha*values[k]+(1.-alpha)*result[k-1]
    return result


def plot_extracted(samples, columns, dt, title, command_topic, signs=(1.,1.,1.)):
    """Show one bag and block until its window is closed."""
    import matplotlib.pyplot as plt
    names={str(name):index for index,name in enumerate(columns)}
    source_t=samples[:,0];x,y,heading=samples[:,1],samples[:,2],samples[:,3]
    vx,odom_vy,omega=samples[:,4],samples[:,5],samples[:,6]
    steer,speed_cmd=samples[:,7],samples[:,9]
    imu_wz,imu_ax,imu_ay=samples[:,12],samples[:,13],samples[:,14]
    segment=samples[:,11].astype(int);t=np.empty(len(samples));wx=np.empty(len(samples));wy=np.empty(len(samples));raw_yaw_rate=np.empty(len(samples))
    elapsed=0.
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id);local=source_t[ii]-source_t[ii[0]]
        t[ii]=elapsed+local;elapsed=t[ii[-1]]+(np.median(np.diff(local)) if len(ii)>1 else DT)
        edge=2 if len(ii)>=3 else 1
        wx[ii]=np.gradient(x[ii],local,edge_order=edge) if len(ii)>1 else 0.
        wy[ii]=np.gradient(y[ii],local,edge_order=edge) if len(ii)>1 else 0.
        raw_yaw_rate[ii]=np.gradient(np.unwrap(heading[ii]),local,edge_order=edge) if len(ii)>1 else 0.
    pose_vx=wx*np.cos(heading)+wy*np.sin(heading)
    pose_vy=np.empty(len(samples))
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id)
        pose_vy[ii]=causal_mcl_body_vy(source_t[ii],x[ii],y[ii],heading[ii])
    raw_mcl_speed=np.hypot(pose_vx,pose_vy)
    raw_ax=np.empty(len(samples));raw_ay=np.empty(len(samples));odom_dvx=np.empty(len(samples));odom_dvy=np.empty(len(samples))
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id);local=source_t[ii]-source_t[ii[0]]
        edge=2 if len(ii)>=3 else 1
        dvx=np.gradient(pose_vx[ii],local,edge_order=edge) if len(ii)>1 else np.zeros(len(ii))
        dvy=np.gradient(pose_vy[ii],local,edge_order=edge) if len(ii)>1 else np.zeros(len(ii))
        raw_ax[ii]=dvx-raw_yaw_rate[ii]*pose_vy[ii]
        raw_ay[ii]=dvy+raw_yaw_rate[ii]*pose_vx[ii]
        odom_dvx[ii]=np.gradient(vx[ii],local,edge_order=edge) if len(ii)>1 else 0.
        odom_dvy[ii]=np.gradient(odom_vy[ii],local,edge_order=edge) if len(ii)>1 else 0.
    wz_sign,ax_sign,ay_sign=signs
    fig,axes=plt.subplots(5,2,figsize=(16,22));fig.suptitle(title,y=.995)
    panels=axes.flat;ax=panels[0]
    # MCL samples remain at 50 Hz; only yaw-direction glyphs are decimated.
    stride=max(1,int(round(PLOT_ARROW_INTERVAL_S/dt)))
    # Original MCL samples and their measured yaw direction.
    ax.scatter(x,y,s=5,color="tab:orange",alpha=.35,label="raw MCL samples")
    ax.quiver(x[::stride],y[::stride],np.cos(heading[::stride]),np.sin(heading[::stride]),
              color="tab:orange",angles="xy",scale_units="xy",scale=3.5,width=.004,
              label="MCL yaw arrows")
    # Connect only measured samples within each retained continuous segment.
    line_label=True
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id)
        ax.plot(x[ii],y[ii],color="tab:green",lw=1.2,
                label="raw MCL connected samples" if line_label else None)
        line_label=False
    kf_x=samples[:,names["kf_x"]];kf_y=samples[:,names["kf_y"]]
    kf_yaw=samples[:,names["kf_yaw"]]
    kf_label=True
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id)
        ax.plot(kf_x[ii],kf_y[ii],color="tab:blue",lw=1.8,
                label="MPPI-model KF x/y" if kf_label else None)
        kf_label=False
    ax.quiver(kf_x[::stride],kf_y[::stride],np.cos(kf_yaw[::stride]),np.sin(kf_yaw[::stride]),
              color="tab:blue",angles="xy",scale_units="xy",scale=3.5,width=.004,
              label="MPPI-model KF yaw arrows")
    label_bucket=np.floor(t/PLOT_TIME_LABEL_INTERVAL_S).astype(int)
    time_indices=np.flatnonzero(np.r_[True,np.diff(label_bucket)>0])
    ax.scatter(kf_x[time_indices],kf_y[time_indices],s=18,color="black",zorder=5,
               label=f"KF position every {PLOT_TIME_LABEL_INTERVAL_S:g} s")
    for index in time_indices:
        ax.annotate(f"{t[index]:.0f}s",(kf_x[index],kf_y[index]),xytext=(4,4),
                    textcoords="offset points",fontsize=7,color="black",
                    bbox={"boxstyle":"round,pad=.15","fc":"white","ec":"none","alpha":.7})
    ax.set_title("Raw MCL pose/yaw arrows and MPPI-model KF trajectory");ax.set_xlabel("x [m]");ax.set_ylabel("y [m]")
    ax.axis("equal");ax.grid(alpha=.25);ax.legend()
    signed_imu_wz=wz_sign*imu_wz
    state_axes=panels[1:7]
    state_specs=(("kf_x",x,"MCL x","x","m"),("kf_y",y,"MCL y","y","m"),
        ("kf_yaw",heading,"MCL yaw","yaw","rad"),("kf_vx",vx,"odom vx","vx","m/s"),
        ("kf_vy",pose_vy,"MCL-difference body vy","vy","m/s"),("kf_yaw_rate",signed_imu_wz,"signed IMU yaw-rate","yaw-rate","rad/s"))
    for axis,(field,raw,raw_label,state_name,unit) in zip(state_axes,state_specs):
        estimate=samples[:,names[field]]
        if field=="kf_yaw":
            raw_plot=np.unwrap(raw);estimate_plot=np.unwrap(estimate)
            residual=(estimate-raw+np.pi)%(2*np.pi)-np.pi
        else:
            raw_plot=raw;estimate_plot=estimate;residual=estimate-raw
        axis.plot(t,raw_plot,color="tab:orange",alpha=.72,label=f"raw {raw_label}")
        axis.plot(t,estimate_plot,color="tab:blue",lw=1.6,label=f"MPPI-model KF {state_name}")
        # axis.plot(t,residual,color="tab:red",alpha=.45,label="KF - raw difference")
        axis.set_title(f"{state_name}: causal MPPI-model KF vs raw {raw_label}")
        axis.set_ylabel(unit)

    consistency_axes=panels[7:9]
    odom_ax=odom_dvx-signed_imu_wz*odom_vy
    consistency_axes[0].plot(t,ax_sign*imu_ax,color="tab:orange",label="signed raw IMU ax")
    consistency_axes[0].plot(t,odom_dvx,color="tab:green",alpha=.55,label="d(odom vx)/dt")
    consistency_axes[0].plot(t,odom_ax,color="tab:blue",alpha=.8,label="d(odom vx)/dt - r·odom_vy")
    consistency_axes[0].set_ylabel("m/s²");consistency_axes[0].set_title("IMU ax vs odom-vx slope")
    kf_vx=samples[:,names["kf_vx"]];kf_vy=samples[:,names["kf_vy"]]
    kf_r=samples[:,names["kf_yaw_rate"]];kf_dvy=np.empty(len(samples))
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id);local=source_t[ii]-source_t[ii[0]]
        edge=2 if len(ii)>=3 else 1
        kf_dvy[ii]=np.gradient(kf_vy[ii],local,edge_order=edge) if len(ii)>1 else 0.
    kf_state_ay=kf_dvy+kf_r*kf_vx
    consistency_axes[1].plot(t,ay_sign*imu_ay,color="tab:orange",label="signed raw IMU ay")
    consistency_axes[1].plot(t,samples[:,names["kf_ay"]],color="tab:green",alpha=.8,
                             label="KF model ay (tire-force prediction)")
    consistency_axes[1].plot(t,kf_state_ay,color="tab:blue",alpha=.8,
                             label="d(KF vy)/dt + KF yaw-rate·KF vx")
    consistency_axes[1].plot(t,odom_dvy+signed_imu_wz*vx,color="tab:purple",alpha=.65,
                             label="d(odom vy)/dt + IMU yaw-rate·odom vx")
    consistency_axes[1].set_ylabel("m/s²");consistency_axes[1].set_title("IMU ay vs KF model/state-derived ay")
    panels[9].axis("off")

    print(f"Showing {title}. Close this window to process the next bag.")
    for axis in panels[1:9]:
        axis.set_xlabel("time [s]");axis.grid(alpha=.25);axis.legend(fontsize=8)
    # Keep titles, x labels and legends from touching the neighboring row.
    fig.subplots_adjust(left=.08,right=.97,bottom=.06,top=.95,hspace=.48,wspace=.25)
    print(f"Showing {title}. Close this window to process the next bag.")
    plt.show(block=True)
    plt.close(fig)


def extract_one(storage, out, args):
    cfg=yaml.safe_load((PROJECT_ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
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
    collision,episodes=collision_recovery_mask(
        base,args.dt,lookback_s=args.collision_pre_margin)
    physical_bad,physical_events=physical_inconsistency_mask(base,args.dt,
        pre_margin_s=args.physics_pre_margin,post_margin_s=args.physics_post_margin,
        moving_vx=args.physics_moving_vx,frozen_pose_speed=args.physics_frozen_pose_speed,
        distance_window_s=args.physics_distance_window,min_odom_distance=args.physics_min_odom_distance,
        min_pose_odom_ratio=args.physics_min_pose_odom_ratio,impact_decel=args.physics_impact_decel,
        max_pose_step=args.physics_max_pose_step,max_yaw_step=args.physics_max_yaw_step)
    bad=collision|manual|physical_bad;kept=np.flatnonzero(~bad)
    if not len(kept): raise RuntimeError(f"{storage}: collision filtering removed every sample")
    breaks=np.flatnonzero((np.diff(kept)>1)|(np.diff(base[kept,0])>1.5*args.dt))+1
    arrays=[];segments=[];discarded_short_fragments=[]
    for bag_id,run in enumerate(np.split(kept,breaks)):
        if len(run)<max(10,int(round(args.min_continuous_segment/args.dt))):
            discarded_short_fragments.append({"candidate_id":bag_id,"samples":len(run),
                "duration_s":float(max(0,len(run)-1)*args.dt)})
            continue
        part=base[run].copy();source_start=float(part[0,0]);part[:,0]-=source_start
        arrays.append(np.c_[part,np.ones(len(part)),np.full(len(part),bag_id),ii[run,1:4]])
        segments.append({"bag_id":bag_id,"samples":len(part),"source_start_s":source_start,
                         "source_end_s":float(base[run[-1],0])})
    if not arrays:
        raise RuntimeError(
            f"no continuous segment >= {args.min_continuous_segment:.2f} s survived filtering")
    samples=np.concatenate(arrays)
    columns=np.array(["t","x","y","yaw","vx","vy","omega","steer","accel","speed_cmd",
                      "split","bag_id","imu_wz","imu_ax","imu_ay"])
    names={name:index for index,name in enumerate(columns)}
    kf=np.full((len(samples),8),np.nan);kf_reports=[]
    segment_ids=samples[:,names["bag_id"]].astype(int)
    for segment_id in np.unique(segment_ids):
        jj=np.flatnonzero(segment_ids==segment_id);part=samples[jj]
        mcl_vy=causal_mcl_body_vy(part[:,names["t"]],part[:,names["x"]],
                                  part[:,names["y"]],part[:,names["yaw"]],args.kf_pose_vy_window)
        imu_alpha=float(cfg.get("imu_ema_alpha",IMU_EMA_ALPHA))
        kf_gyro=causal_ema(signs[0]*part[:,names["imu_wz"]]-float(cfg.get("imu_wz_bias",0.)),imu_alpha)
        kf_ax=causal_ema(signs[1]*part[:,names["imu_ax"]]-float(cfg.get("imu_ax_bias",0.)),imu_alpha)
        kf_ay=causal_ema(signs[2]*part[:,names["imu_ay"]]-float(cfg.get("imu_ay_bias",0.)),imu_alpha)
        result=filter_classic_segment(part[:,names["x"]],part[:,names["y"]],
            part[:,names["yaw"]],part[:,names["vx"]],mcl_vy,
            kf_gyro,kf_ax,kf_ay,part[:,names["steer"]],
            part[:,names["speed_cmd"]],args.dt,cfg)
        state=result["state"]
        kf[jj,:6]=state
        kf[jj,6:8]=result["acceleration"]
        raw_reference=np.c_[part[:,names["x"]],part[:,names["y"]],part[:,names["yaw"]],
                            part[:,names["vx"]],mcl_vy,
                            kf_gyro]
        difference=state-raw_reference
        difference[:,2]=(difference[:,2]+np.pi)%(2*np.pi)-np.pi
        kf_reports.append({"segment_id":int(segment_id),"samples":len(jj),
                           "raw_difference_rmse":np.sqrt(np.nanmean(difference**2,axis=0)).tolist()})
    samples=np.c_[samples,kf]
    columns=np.r_[columns,np.array(("kf_x","kf_y","kf_yaw","kf_vx","kf_vy",
                                    "kf_yaw_rate","kf_ax","kf_ay"))]
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,samples=samples,dt=args.dt,columns=columns,
                        alignment_start_epoch_s=np.array(start,np.float64),
                        pose_topic=np.array(args.pose_topic),
                        velocity_topic=np.array(args.velocity_topic),command_topic=np.array(args.command_topic),
                        imu_topic=np.array(args.imu_topic),
                        recording_date=np.array(date.isoformat()),
                        imu_sign_cutover=np.array(IMU_SIGN_CUTOVER.isoformat()),
                        imu_axis_signs=np.array(signs,np.float32),
                        imu_ema_alpha=np.array(IMU_EMA_ALPHA,np.float32),
                        kf_state_clamp_enabled=np.array(False))
    meta={"source":str(storage.resolve()),"pose_topic":args.pose_topic,"velocity_topic":args.velocity_topic,
          "command_topic":args.command_topic,"imu_topic":args.imu_topic,"alignment":"causal_hold",
          "recording_date":date.isoformat(),"imu_sign_cutover":IMU_SIGN_CUTOVER.isoformat(),
          "imu_axis_signs":{"wz":signs[0],"ax":signs[1],"ay":signs[2]},
          "imu_ema_alpha":IMU_EMA_ALPHA,
          "applied_command_topic":args.applied_command_topic,
          "alignment_start_epoch_s":float(start),"raw_aligned_samples":len(base),
          "removed_collision_samples":int(collision.sum()),"raw_command_mismatch_samples":int(command_mismatch.sum()),
          "collision_pre_margin_s":args.collision_pre_margin,
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
          "continuous_segment_policy":{"minimum_duration_s":args.min_continuous_segment,
              "discarded_short_fragments":discarded_short_fragments,
              "mcl_alignment":"causal hold with maximum sample age"},
          "state_estimator":{"method":"causal MPPI classic-model EKF",
              "state_order":["x","y","yaw","vx","vy","yaw_rate"],
              "comparison_reference":["MCL x","MCL y","MCL yaw","odom vx","MCL-difference body vy","signed IMU yaw-rate"],
              "segments":kf_reports},
          "split_policy":"single-bag-identical-train-test"}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({**meta,"output":str(out)},indent=2))
    return samples,columns,signs


def main():
    if USE_PLOT:
        # This workstation currently defaults to the non-interactive ``agg``
        # backend even with DISPLAY available. Select a GUI backend before any
        # helper imports pyplot, otherwise plt.show() silently returns.
        import matplotlib
        try:
            matplotlib.use("TkAgg", force=True)
        except ImportError as error:
            raise RuntimeError(
                "interactive plotting requires TkAgg; install python3-tk or "
                "set USE_PLOT=False for batch extraction") from error
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
    p.add_argument("--collision-pre-margin",type=float,default=COLLISION_PRE_MARGIN_S,
                   help="접촉 직전 정상 hard-case를 보존하기 위한 제거 여유 [s]")
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
    p.add_argument("--min-continuous-segment",type=float,default=MIN_CONTINUOUS_SEGMENT_S,
                   help="discard filtered fragments shorter than this duration [s]")
    p.add_argument("--kf-pose-vy-window",type=float,default=.12,
                   help="causal trailing MCL regression window used by the runtime KF [s]")
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
