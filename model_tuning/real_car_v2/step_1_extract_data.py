#!/usr/bin/env python3
"""Step 1: extract the exact real-car MPPI observation path from rosbag2.

Pose is taken from /newmcl_pose, body velocity from /odom, controls from the
selected Ackermann topic, and IMU from /imu/data.  Every stream is aligned by
causal hold; no future sample is used.
"""
import argparse
import datetime as dtlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from helper_filter_collision_recovery import (
    collision_recovery_mask, physical_inconsistency_mask)
from classic_model_kalman_filter import filter_classic_segment
from contract import ClassicModelParameters

# USER SETTINGS. Add every bag storage file or rosbag2 directory here. Running
# this script without arguments extracts them sequentially.
NEW_DATA_ROOTS = (
    # Path("/mnt/nas_custom/F1tenth/2026 IFAC/0817 (1)"),
    # Path("/mnt/nas_custom/F1tenth/2026 IFAC/0818"),
    # Path("/mnt/nas_custom/F1tenth/2026 IFAC/0819"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/ifac2026"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/ifac2026_pratice_3th/"),
)
# Keep F5 configuration robust when a single Path is assigned without tuple
# syntax.  A pathlib.Path is path-like but is not a collection of roots.
if isinstance(NEW_DATA_ROOTS, (str, Path)):
    NEW_DATA_ROOTS = (Path(NEW_DATA_ROOTS),)
else:
    NEW_DATA_ROOTS = tuple(Path(root) for root in NEW_DATA_ROOTS)
# Discover rosbag directories recursively. Bags without the required topics are
# reported as SKIPPED by read_streams instead of silently entering the archive.
BAG_PATH = sorted({metadata.parent for root in NEW_DATA_ROOTS
                   for metadata in root.rglob("metadata.yaml")})
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/data/ifac2026"
# F5/direct execution is an interactive inspection workflow.  Set this to
# False only for unattended batch extraction.
USE_PLOT = os.environ.get("STEP1_USE_PLOT", "1") != "0"
# Set True for F5 to skip rosbag extraction and re-open the saved NPZ files.
REVIEW_SAVED_COLLISIONS = True
# Bag-to-pose verification shows that the sensor/body convention changed by
# 2026-08-15. 0810--0813 y/z oppose MPPI FLU; 0815 onward already match it.
IMU_WZ_SIGN = 1.0; IMU_AX_SIGN = 1.0; IMU_AY_SIGN = 1.0; IMU_EMA_ALPHA = .25
IMU_SIGN_CUTOVER = dtlib.date(2026, 8, 15)
POSE_TOPIC = "/newmcl_pose"; VELOCITY_TOPIC = "/odom"; COMMAND_TOPIC = "/ackermann_cmd"; IMU_TOPIC = "/imu/data"
DEFAULT_MAP_YAML = (PROJECT_ROOT / "data/ifac2026/ifac2026.yaml") # 그냥 시각화 용도임
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
# MPPI/residual-model rollout contract. Step 1 stores future GT densely at DT,
# while HORIZON_STEPS counts the coarser MODEL_DT_S prediction steps consumed
# by Steps 3/6. The default therefore stores 60 * 40 ms = 2.4 s of future GT.
MODEL_DT_S = .04
HORIZON_STEPS = 60
MIN_CONTINUOUS_SEGMENT_S = 2.0
PLOT_ARROW_INTERVAL_S = .10  # data remain 50 Hz; direction arrows are drawn at 10 Hz
PLOT_TIME_LABEL_INTERVAL_S = 1.0
# Keep the dense 4x4 diagnostic figure readable without legends covering most
# of each subplot.  Increase this value temporarily when inspecting labels.
PLOT_LEGEND_FONT_SIZE = 5.5
PLOT_LEGEND_OPTIONS = {
    "fontsize": PLOT_LEGEND_FONT_SIZE,
    "handlelength": 1.0,
    "handletextpad": .35,
    "labelspacing": .20,
    "borderpad": .25,
    "columnspacing": .55,
    "framealpha": .45,
}
REVIEW_WINDOW_S = .50
REVIEW_MOVING_SPEED = .70
REVIEW_MAX_POSE_SPEED = .12
REVIEW_MIN_POSE_ODOM_RATIO = .65


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
    if applied_command_topic is not None and applied_command_topic in types:
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
    applied_array=(np.asarray(applied,np.float64) if applied else np.empty((0,4),np.float64))
    return result if applied_command_topic is None else result+(applied_array,)


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


def interpolate_stream(stream,times,angle_columns=()):
    """Linear continuous-time interpolation without endpoint extrapolation."""
    stream=stream[np.argsort(stream[:,0])]
    stream=stream[np.r_[True,np.diff(stream[:,0])>1e-9]]
    result=np.empty((len(times),stream.shape[1]-1),float)
    for column in range(1,stream.shape[1]):
        values=np.unwrap(stream[:,column]) if column in angle_columns else stream[:,column]
        result[:,column-1]=np.interp(times,stream[:,0],values,left=np.nan,right=np.nan)
    return result


def build_callback_prediction_samples(pose,velocity,drive,imu,signs,cfg,accepted_intervals,
                                      origin,horizon_s=.0,prediction_dt=.02,
                                      max_command_age=.10,max_imu_age=.05,
                                      max_pose_step=.30,max_yaw_step=.45,
                                      command_history_steps=5):
    """Build irregular callback anchors with causal actuator state/history.

    The first callback initializes applied steer from the current steer command
    and speed reference from measured vx. No artificial history padding is
    used: with a five-callback history, anchors 0--3 are discarded.
    """
    steps=max(1,int(round(horizon_s/prediction_dt)))
    offsets=prediction_dt*np.arange(1,steps+1,dtype=float)
    inputs=[];targets=[];future_commands=[]
    velocity=velocity[np.argsort(velocity[:,0])]
    processed_imu=imu[np.argsort(imu[:,0])].copy()
    processed_imu[:,1:4]*=np.asarray(signs,float)[None,:]
    processed_imu[:,1]-=float(cfg.get("imu_wz_bias",0.))
    processed_imu[:,2]-=float(cfg.get("imu_ax_bias",0.))
    processed_imu[:,3]-=float(cfg.get("imu_ay_bias",0.))
    alpha=float(cfg.get("imu_ema_alpha",IMU_EMA_ALPHA))
    for k in range(1,len(processed_imu)):
        processed_imu[k,1:4]=alpha*processed_imu[k,1:4]+(1.-alpha)*processed_imu[k-1,1:4]
    for bag_id,(start,end) in enumerate(accepted_intervals):
        # A target horizon must stay inside one audited continuous interval.
        anchors=velocity[(velocity[:,0]>=start)&(velocity[:,0]+offsets[-1]<=end)]
        if not len(anchors):continue
        anchor_t=anchors[:,0];pose_now=interpolate_stream(pose,anchor_t,(3,))
        command,command_valid=causal_hold(drive,anchor_t,max_command_age)
        inertial,imu_valid=causal_hold(processed_imu,anchor_t,max_imu_age)
        steer_cmd=command[:,1];speed_cmd=command[:,3]
        applied_steer=np.empty(len(anchors));speed_reference=np.empty(len(anchors))
        max_steer=float(cfg.get("max_steer",.4788))
        steer_scale=float(cfg.get("kinematic_steer_scale",1.0))
        steer_bias=float(cfg.get("kinematic_steer_bias",0.0))
        applied_steer[0]=np.clip(steer_scale*steer_cmd[0]+steer_bias,
                                 -max_steer,max_steer)
        speed_reference[0]=anchors[0,1]
        steer_tau=max(float(cfg.get("steer_servo_time_constant",.08)),1e-3)
        max_steer_rate=float(cfg.get("actuator_max_steer_rate",np.inf))
        accel_tau=max(float(cfg.get("speed_reference_accel_time_constant",.05)),1e-3)
        brake_tau=max(float(cfg.get("speed_reference_brake_time_constant",.05)),1e-3)
        max_speed_rate=float(cfg.get("actuator_max_speed_reference_rate",np.inf))
        for k in range(1,len(anchors)):
            callback_dt=max(0.,anchor_t[k]-anchor_t[k-1])
            steer_target=np.clip(steer_scale*steer_cmd[k-1]+steer_bias,
                                 -max_steer,max_steer)
            steer_rate=np.clip(
                (steer_target-applied_steer[k-1])/steer_tau,
                -max_steer_rate,max_steer_rate)
            applied_steer[k]=np.clip(
                applied_steer[k-1]+steer_rate*callback_dt,
                -max_steer,max_steer)
            tau=accel_tau if speed_cmd[k-1]>=speed_reference[k-1] else brake_tau
            speed_rate=np.clip((speed_cmd[k-1]-speed_reference[k-1])/tau,
                               -max_speed_rate,max_speed_rate)
            speed_reference[k]=speed_reference[k-1]+speed_rate*callback_dt
        command_history=np.full((len(anchors),2*command_history_steps),np.nan)
        for k in range(command_history_steps-1,len(anchors)):
            command_history[k]=np.c_[
                steer_cmd[k-command_history_steps+1:k+1],
                speed_cmd[k-command_history_steps+1:k+1]].reshape(-1)
        future_t=anchor_t[:,None]+offsets[None,:]
        future_command,future_command_valid=causal_hold(
            drive,future_t.ravel(),max_command_age)
        future_command=future_command[:,[1,3]].reshape(len(anchors),steps,2)
        future_command_valid=future_command_valid.reshape(len(anchors),steps).all(1)
        future_pose=interpolate_stream(pose,future_t.ravel(),(3,)).reshape(len(anchors),steps,3)
        future_velocity=interpolate_stream(velocity,future_t.ravel()).reshape(len(anchors),steps,3)
        valid=(command_valid&imu_valid&future_command_valid&np.isfinite(pose_now).all(1)
               &np.isfinite(future_pose).all((1,2))&np.isfinite(future_velocity).all((1,2)))
        valid&=np.isfinite(command_history).all(1)
        pose_trace=np.concatenate((pose_now[:,None,:],future_pose),axis=1)
        xy_step=np.linalg.norm(np.diff(pose_trace[:,:,:2],axis=1),axis=2)
        yaw_step=np.abs(np.diff(pose_trace[:,:,2],axis=1))
        valid&=(np.max(xy_step,axis=1)<=max_pose_step)
        valid&=(np.max(yaw_step,axis=1)<=max_yaw_step)
        if not valid.any():continue
        current=np.c_[anchor_t-origin,pose_now,anchors[:,1:4],steer_cmd,speed_cmd,
                      inertial[:,1:4],applied_steer,speed_reference,command_history,
                      np.full(len(anchors),bag_id)]
        inputs.append(current[valid]);targets.append(
            np.concatenate((future_pose,future_velocity),axis=2)[valid])
        future_commands.append(future_command[valid])
    if not inputs:
        input_width=15+2*command_history_steps
        return (np.empty((0,input_width)),np.empty((0,steps,6)),
                np.empty((0,steps,2)),offsets)
    return (np.concatenate(inputs),np.concatenate(targets),
            np.concatenate(future_commands),offsets)


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


def refresh_saved_kf_from_yaml(directory):
    """Rebuild KF/Pacejka columns in trimmed Step-1 NPZs without changing rows."""
    cfg=yaml.safe_load((PROJECT_ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    parameter_set=ClassicModelParameters.from_mapping(cfg)
    snapshot={**parameter_set.runtime_updates(),
        "classic_kf_process_var":list(map(float,cfg["classic_kf_process_var"])),
        "classic_kf_measurement_var":list(map(float,cfg["classic_kf_measurement_var"])),
        "classic_kf_initial_var":list(map(float,cfg["classic_kf_initial_var"])),
        "kf_pose_vy_window_s":float(cfg.get("kf_pose_vy_window_s",.12))}
    paths=sorted(Path(directory).expanduser().resolve().glob("*.npz"))
    if not paths:raise RuntimeError(f"no Step-1 NPZ files found in {directory}")
    reports=[]
    for number,path in enumerate(paths,1):
        with np.load(path) as archive:
            if not {"samples","columns","dt"}.issubset(archive.files):
                print(f"[{number}/{len(paths)}] SKIPPED non-Step-1 NPZ: {path.name}")
                continue
            payload={key:np.asarray(archive[key]) for key in archive.files}
        samples=np.asarray(payload["samples"],float).copy()
        columns=np.asarray(payload["columns"]);names={str(v):i for i,v in enumerate(columns)}
        required=("t","x","y","yaw","vx","steer","speed_cmd","imu_wz","imu_ax","imu_ay",
                  "kf_x","kf_y","kf_yaw","kf_vx","kf_vy","kf_yaw_rate","kf_ax","kf_ay")
        missing=[name for name in required if name not in names]
        if missing:raise RuntimeError(f"{path}: missing Step-1 columns {missing}")
        old_count=len(samples);old_first=float(samples[0,names["t"]]);old_last=float(samples[-1,names["t"]])
        signs=np.asarray(payload.get("imu_axis_signs",np.ones(3)),float)
        alpha=float(payload.get("imu_ema_alpha",cfg.get("imu_ema_alpha",IMU_EMA_ALPHA)))
        sample_dt=float(payload["dt"]);window=float(cfg.get("kf_pose_vy_window_s",.12))
        segment_ids=(samples[:,names["bag_id"]].astype(int) if "bag_id" in names
                     else np.zeros(old_count,int))
        segment_reports=[]
        for segment_id in np.unique(segment_ids):
            jj=np.flatnonzero(segment_ids==segment_id);part=samples[jj]
            mcl_vy=causal_mcl_body_vy(part[:,names["t"]],part[:,names["x"]],
                part[:,names["y"]],part[:,names["yaw"]],window)
            gyro=causal_ema(signs[0]*part[:,names["imu_wz"]]-float(cfg.get("imu_wz_bias",0.)),alpha)
            ax=causal_ema(signs[1]*part[:,names["imu_ax"]]-float(cfg.get("imu_ax_bias",0.)),alpha)
            ay=causal_ema(signs[2]*part[:,names["imu_ay"]]-float(cfg.get("imu_ay_bias",0.)),alpha)
            result=filter_classic_segment(part[:,names["x"]],part[:,names["y"]],
                part[:,names["yaw"]],part[:,names["vx"]],mcl_vy,gyro,ax,ay,
                part[:,names["steer"]],part[:,names["speed_cmd"]],sample_dt,cfg)
            state=result["state"]
            samples[np.ix_(jj,[names["kf_x"],names["kf_y"],names["kf_yaw"],
                names["kf_vx"],names["kf_vy"],names["kf_yaw_rate"]])]=state
            samples[np.ix_(jj,[names["kf_ax"],names["kf_ay"]])]=result["acceleration"]
            reference=np.c_[part[:,names["x"]],part[:,names["y"]],part[:,names["yaw"]],
                            part[:,names["vx"]],mcl_vy,gyro]
            difference=state-reference;difference[:,2]=(difference[:,2]+np.pi)%(2*np.pi)-np.pi
            segment_reports.append({"segment_id":int(segment_id),"samples":len(jj),
                "raw_difference_rmse":np.sqrt(np.nanmean(difference**2,axis=0)).tolist()})
        payload["samples"]=samples
        payload["kf_parameter_hash"]=np.array(parameter_set.digest())
        payload["kf_config_snapshot_json"]=np.array(json.dumps(snapshot,sort_keys=True))
        temporary=path.with_name(path.stem+".refresh-kf.tmp.npz")
        np.savez_compressed(temporary,**payload);temporary.replace(path)
        metadata_path=path.with_suffix(".json")
        if metadata_path.is_file():
            metadata=json.loads(metadata_path.read_text())
            metadata["output_samples"]=old_count
            metadata["state_estimator"]={"method":"causal MPPI classic-model EKF",
                "classic_parameter_hash":parameter_set.digest(),
                "state_order":["x","y","yaw","vx","vy","yaw_rate"],
                "comparison_reference":["MCL x","MCL y","MCL yaw","odom vx",
                    "MCL-difference body vy","signed IMU yaw-rate"],
                "segments":segment_reports,
                "refreshed_from_trimmed_npz":True}
            metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
        if len(samples)!=old_count or samples[0,names["t"]]!=old_first or samples[-1,names["t"]]!=old_last:
            raise RuntimeError(f"{path}: trim invariant changed during KF refresh")
        reports.append((path.name,old_count,old_first,old_last))
        print(f"[{number}/{len(paths)}] refreshed current-YAML KF: {path.name}; "
              f"rows={old_count}, retained={old_first:.3f}..{old_last:.3f} s")
    print(f"Refreshed {len(reports)} Step-1 NPZ files; parameter hash={parameter_set.digest()}")
    return reports


def draw_occupancy_map(axis,map_yaml):
    """Draw a ROS occupancy map in the same world frame as MCL poses."""
    import matplotlib.image as mpimg
    from matplotlib.transforms import Affine2D
    map_yaml=Path(map_yaml)
    if not map_yaml.is_file():
        raise FileNotFoundError(f"map YAML does not exist: {map_yaml}")
    metadata=yaml.safe_load(map_yaml.read_text())
    image_path=(map_yaml.parent/metadata["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"map image from {map_yaml} does not exist: {image_path}")
    image=mpimg.imread(image_path)
    resolution=float(metadata["resolution"]);origin=np.asarray(metadata["origin"],float)
    height,width=image.shape[:2]
    extent=(origin[0],origin[0]+width*resolution,
            origin[1],origin[1]+height*resolution)
    transform=(Affine2D().rotate_around(origin[0],origin[1],origin[2])+axis.transData
               if abs(origin[2])>1e-12 else axis.transData)
    axis.imshow(image,cmap="gray",origin="upper",extent=extent,transform=transform,
                interpolation="nearest",alpha=.48,zorder=0,label="_nolegend_")
    axis.plot([],[],color="0.65",lw=6,alpha=.6,label="occupancy map")
    return {"yaml":str(map_yaml),"image":str(image_path),"resolution":resolution,
            "origin":origin.tolist(),"shape":[height,width]}


def collision_review_mask(samples, columns, window_s=REVIEW_WINDOW_S,
                          moving_speed=REVIEW_MOVING_SPEED,
                          max_pose_speed=REVIEW_MAX_POSE_SPEED,
                          min_pose_odom_ratio=REVIEW_MIN_POSE_ODOM_RATIO):
    """Mark intervals where command/odom reports motion but MCL is stuck."""
    names={str(name):index for index,name in enumerate(columns)}
    t=samples[:,names["t"]];x=samples[:,names["x"]];y=samples[:,names["y"]]
    odom=np.abs(samples[:,names["vx"]]);command=np.abs(samples[:,names["speed_cmd"]])
    moving=np.maximum(odom,command)
    suspect=np.zeros(len(samples),dtype=bool)
    pose_speed=np.full(len(samples),np.nan);ratio=np.full(len(samples),np.nan)
    for index in range(len(samples)):
        end=int(np.searchsorted(t,t[index]+window_s,side="right"))-1
        if end<=index:continue
        duration=t[end]-t[index]
        displacement=np.hypot(x[end]-x[index],y[end]-y[index])
        pose_speed[index]=displacement/max(duration,1e-6)
        odom_distance=np.trapz(odom[index:end+1],t[index:end+1])
        ratio[index]=displacement/max(odom_distance,1e-6)
        active=float(np.mean(moving[index:end+1]))>=moving_speed
        suspect[index]=active and (pose_speed[index]<max_pose_speed or
                                    ratio[index]<min_pose_odom_ratio)
    expanded=suspect.copy()
    for index in np.flatnonzero(suspect):
        expanded[index:np.searchsorted(t,t[index]+window_s,side="right")]=True
    return expanded,{"pose_speed":pose_speed,"pose_odom_ratio":ratio}


def _true_time_spans(t,mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
    return [(float(t[start]),float(t[end])) for start,end in zip(starts,ends)]


def plot_extracted(samples, columns, dt, title, command_topic, signs=(1.,1.,1.),
                   map_yaml=DEFAULT_MAP_YAML,trim_controls=False,time_offset_s=0.,
                   bag_count=None,current_bag_number=None,review_collisions=False):
    """Show one bag and block until its window is closed."""
    import matplotlib.pyplot as plt
    names={str(name):index for index,name in enumerate(columns)}
    source_t=samples[:,0];x,y,heading=samples[:,1],samples[:,2],samples[:,3]
    vx,odom_vy,omega=samples[:,4],samples[:,5],samples[:,6]
    steer,speed_cmd=samples[:,7],samples[:,9]
    imu_wz,imu_ax,imu_ay=samples[:,12],samples[:,13],samples[:,14]
    segment=samples[:,11].astype(int);t=np.empty(len(samples));wx=np.empty(len(samples));wy=np.empty(len(samples));raw_yaw_rate=np.empty(len(samples))
    elapsed=float(time_offset_s)
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
    fig,axes=plt.subplots(4,4,figsize=(22,18));fig.suptitle(title,y=.995)
    panels=axes.flat;ax=panels[0]
    if map_yaml is not None:
        draw_occupancy_map(ax,map_yaml)
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
        # wrapped=np.flatnonzero(np.abs(np.diff(heading[ii]))>np.pi)+1
        # if len(wrapped):
        #     jj=ii[wrapped]
        #     ax.scatter(x[jj],y[jj],marker="x",s=70,color="red",zorder=8,
        #                label="raw yaw wrap (unwrapped in callback GT)" if line_label else None)
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
    review_mask=np.zeros(len(samples),dtype=bool);review_spans=[]
    if review_collisions:
        review_mask,_=collision_review_mask(samples,columns)
        review_spans=_true_time_spans(t,review_mask)
        if np.any(review_mask):
            ax.scatter(x[review_mask],y[review_mask],s=24,color="red",marker="x",zorder=9,
                       label="collision/stuck review candidate")
    label_bucket=np.floor(t/PLOT_TIME_LABEL_INTERVAL_S).astype(int)
    time_indices=np.flatnonzero(np.r_[True,np.diff(label_bucket)>0])
    ax.scatter(kf_x[time_indices],kf_y[time_indices],s=18,color="black",zorder=5,
               label=f"KF position every {PLOT_TIME_LABEL_INTERVAL_S:g} s")
    for index in time_indices:
        ax.annotate(f"{t[index]:.0f}s",(kf_x[index],kf_y[index]),xytext=(4,4),
                    textcoords="offset points",fontsize=7,color="black",
                    bbox={"boxstyle":"round,pad=.15","fc":"white","ec":"none","alpha":.7})
    ax.set_title("Raw MCL pose/yaw arrows and MPPI-model KF trajectory");ax.set_xlabel("x [m]");ax.set_ylabel("y [m]")
    ax.axis("equal");ax.grid(alpha=.25);ax.legend(**PLOT_LEGEND_OPTIONS)
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
    kf_vx=samples[:,names["kf_vx"]];kf_vy=samples[:,names["kf_vy"]]
    kf_r=samples[:,names["kf_yaw_rate"]]
    kf_dvx=np.empty(len(samples));kf_dvy=np.empty(len(samples))
    for bag_id in np.unique(segment):
        ii=np.flatnonzero(segment==bag_id);local=source_t[ii]-source_t[ii[0]]
        edge=2 if len(ii)>=3 else 1
        kf_dvx[ii]=np.gradient(kf_vx[ii],local,edge_order=edge) if len(ii)>1 else 0.
        kf_dvy[ii]=np.gradient(kf_vy[ii],local,edge_order=edge) if len(ii)>1 else 0.
    # Body-frame acceleration identities:
    #   ax = dvx/dt - r*vy, ay = dvy/dt + r*vx.
    # kf_ax/kf_ay are the classic process-model predictions returned by the
    # EKF, whereas these state-derived traces differentiate the filtered state.
    kf_state_ax=kf_dvx-kf_r*kf_vy
    kf_state_ay=kf_dvy+kf_r*kf_vx
    consistency_axes[0].plot(t,ax_sign*imu_ax,color="tab:orange",label="signed raw IMU ax")
    consistency_axes[0].plot(t,odom_dvx,color="tab:gray",alpha=.5,label="d(odom vx)/dt")
    consistency_axes[0].plot(t,odom_ax,color="tab:purple",alpha=.65,
                             label="d(odom vx)/dt - IMU r·odom vy")
    consistency_axes[0].plot(t,samples[:,names["kf_ax"]],color="tab:green",alpha=.85,
                             label="KF model ax (longitudinal actuator prediction)")
    consistency_axes[0].plot(t,kf_state_ax,color="tab:blue",alpha=.85,
                             label="d(KF vx)/dt - KF yaw-rate·KF vy")
    consistency_axes[0].set_ylabel("m/s²")
    consistency_axes[0].set_title("IMU ax vs odom slope and KF model/state-derived ax")
    consistency_axes[1].plot(t,ay_sign*imu_ay,color="tab:orange",label="signed raw IMU ay")
    consistency_axes[1].plot(t,samples[:,names["kf_ay"]],color="tab:green",alpha=.8,
                             label="KF model ay (tire-force prediction)")
    consistency_axes[1].plot(t,kf_state_ay,color="tab:blue",alpha=.8,
                             label="d(KF vy)/dt + KF yaw-rate·KF vx")
    consistency_axes[1].plot(t,odom_dvy+signed_imu_wz*vx,color="tab:purple",alpha=.65,
                             label="d(odom vy)/dt + IMU yaw-rate·odom vx")
    consistency_axes[1].set_ylabel("m/s²");consistency_axes[1].set_title("IMU ay vs KF model/state-derived ay")
    command_axes=panels[9:11]
    command_axes[0].plot(t,speed_cmd,color="tab:red",lw=1.5,label="vx command")
    command_axes[0].plot(t,vx,color="tab:orange",alpha=.7,label="raw odom vx")
    command_axes[0].plot(t,kf_vx,color="tab:blue",alpha=.8,label="KF vx")
    command_axes[0].set_ylabel("m/s")
    command_axes[0].set_title("Longitudinal command and measured/estimated vx")
    command_axes[1].plot(t,steer,color="tab:red",lw=1.5,label="steer command")
    command_axes[1].axhline(0.,color="black",lw=.7,alpha=.4)
    command_axes[1].set_ylabel("rad")
    command_axes[1].set_title("Steering command")
    for axis in panels[11:]:
        axis.axis("off")

    for axis in panels[1:11]:
        for span_start,span_end in review_spans:
            axis.axvspan(span_start,span_end,color="red",alpha=.13,zorder=0)
        axis.set_xlabel("time [s]");axis.grid(alpha=.25)
        axis.legend(**PLOT_LEGEND_OPTIONS)
    # Keep titles, x labels and legends from touching the neighboring row.
    fig.subplots_adjust(left=.08,right=.97,bottom=.06,top=.95,hspace=.48,wspace=.25)
    action={"key":"q","time":None}
    if trim_controls:
        action={"key":None,"time":None}
        review_text=(f"; red={len(review_spans)} collision/stuck candidate interval(s)"
                     if review_collisions else "")
        fig.suptitle(title+review_text+"\ns/e=cut; q=save; ←=previous unsaved; →=next unsaved; 1..9=jump; j=two-digit jump",y=.997)
        time_axes=set(panels[1:11])
        def on_key(event):
            key=(event.key or "").lower()
            if key.isdigit() and key!="0":
                target=int(key)
                if bag_count is not None and target<=bag_count:
                    action.update(key="jump",time=None,bag_number=target);plt.close(fig)
                else:
                    print(f"Bag {target} is outside 1..{bag_count}.")
                return
            if key=="j":
                try:
                    target=int(input(f"Jump to bag number [1..{bag_count}]: "))
                except (ValueError,EOFError):
                    print("Invalid bag number.");return
                if bag_count is not None and 1<=target<=bag_count:
                    action.update(key="jump",time=None,bag_number=target);plt.close(fig)
                else:
                    print(f"Bag {target} is outside 1..{bag_count}.")
                return
            if key=="left":
                if current_bag_number is not None and current_bag_number<=1:
                    print("Already at the first bag.");return
                action.update(key="previous",time=None);plt.close(fig);return
            if key=="right":
                if (current_bag_number is not None and bag_count is not None
                        and current_bag_number>=bag_count):
                    print("Already at the last bag.");return
                action.update(key="next",time=None);plt.close(fig);return
            if key=="q":
                action.update(key="q",time=None);plt.close(fig);return
            if key not in {"s","e"}:
                return
            if event.inaxes not in time_axes or event.xdata is None:
                print("Place the mouse cursor over a time-series panel before pressing s/e.")
                return
            action.update(key=key,time=float(event.xdata));plt.close(fig)
        fig.canvas.mpl_connect("key_press_event",on_key)
        print("Manual trim: s/e=cut, q=save, left=previous unsaved, "
              "right=next unsaved, 1..9=jump, j=two-digit jump.")
    else:
        print(f"Showing {title}. Close this window to process the next bag.")
    plt.show(block=True)
    plt.close(fig)
    return action


def persist_manual_trim(out,samples,columns,cut_start_s,cut_end_s):
    """Atomically trim aligned samples and callback horizons in one NPZ."""
    names={str(name):index for index,name in enumerate(columns)}
    trimmed=samples.copy();trimmed[:,names["t"]]-=cut_start_s
    with np.load(out) as archive:
        payload={name:np.asarray(archive[name]) for name in archive.files}
    payload["samples"]=trimmed
    if "alignment_start_epoch_s" in payload:
        payload["alignment_start_epoch_s"]=np.asarray(
            float(payload["alignment_start_epoch_s"])+cut_start_s,np.float64)
    callback_kept=0
    if "callback_inputs" in payload:
        callback=payload["callback_inputs"].copy()
        callback_columns={str(name):index for index,name in enumerate(payload["callback_input_columns"])}
        callback_time=callback[:,callback_columns["t"]]
        horizon=(float(np.max(payload["callback_future_offsets_s"]))
                 if len(payload["callback_future_offsets_s"]) else 0.)
        keep=(callback_time>=cut_start_s)&(callback_time+horizon<=cut_end_s+1e-9)
        callback=callback[keep];callback[:,callback_columns["t"]]-=cut_start_s
        payload["callback_inputs"]=callback
        for field in ("callback_future_states","callback_future_commands"):
            if field in payload:payload[field]=payload[field][keep]
        callback_kept=int(keep.sum())
    temporary=out.with_name(out.stem+".manual-trim.tmp.npz")
    np.savez_compressed(temporary,**payload);temporary.replace(out)
    metadata_path=out.with_suffix(".json")
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    previous_epoch=float(metadata.get("alignment_start_epoch_s",0.))
    metadata["alignment_start_epoch_s"]=previous_epoch+cut_start_s
    metadata["output_samples"]=int(len(trimmed))
    metadata["manual_trim"]={"applied":True,"source_start_s":float(cut_start_s),
        "source_end_s":float(cut_end_s),"duration_s":float(cut_end_s-cut_start_s),
        "samples":int(len(trimmed)),"callback_anchors":callback_kept,
        "controls":"s=start, e=end, q=save/next bag"}
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Saved manual trim: {cut_start_s:.3f}..{cut_end_s:.3f} s, "
          f"{len(trimmed)} samples, {callback_kept} callback anchors -> {out}")
    return trimmed


def interactive_trim_saved_extract(out,samples,columns,dt,title,command_topic,signs,map_yaml,
                                   bag_count,current_bag_number,review_collisions=False):
    """Re-render immediately after each s/e cut, and persist only on q."""
    selected=samples
    while True:
        start=float(selected[0,0]);end=float(selected[-1,0])
        action=plot_extracted(selected,columns,dt,
            f"{title} | selected source {start:.2f}..{end:.2f} s",
            command_topic,signs,map_yaml,trim_controls=True,time_offset_s=start,
            bag_count=bag_count,current_bag_number=current_bag_number,
            review_collisions=review_collisions)
        if action["key"]=="jump":
            target=int(action["bag_number"])
            print(f"Jumping without saving: {title} -> bag {target}/{bag_count}")
            return None,target-1
        if action["key"] in {"previous","next"}:
            target=(current_bag_number-2 if action["key"]=="previous"
                    else current_bag_number)
            print(f"Moving {action['key']} without saving: {title} -> "
                  f"bag {target+1}/{bag_count}")
            return None,target
        if action["key"] in (None,"q"):
            return persist_manual_trim(out,selected,columns,start,end),None
        index=int(np.argmin(np.abs(selected[:,0]-action["time"])))
        candidate=(selected[index:] if action["key"]=="s" else selected[:index+1])
        if len(candidate)<10:
            print("Cut rejected: at least 10 samples must remain.")
            continue
        selected=candidate


def review_saved_extracts(directory,args):
    """Re-open all saved Step-1 NPZ files and highlight collision candidates."""
    paths=sorted(Path(directory).expanduser().glob("*.npz"))
    if not paths:
        raise SystemExit(
            f"No saved Step-1 NPZ files found in {directory}.\n"
            "REVIEW_SAVED_COLLISIONS=True (--review-saved-collisions) requests review-only "
            "mode, but there is nothing to review.\n"
            "저장된 NPZ가 없는데 saved-collision 검토 모드로 실행했습니다. "
            "이 의도가 맞는지 확인하세요. 새 bag을 추출하려면 "
            "REVIEW_SAVED_COLLISIONS=False 또는 --no-review-saved-collisions를 사용하세요.")
    index=0
    while index<len(paths):
        path=paths[index]
        with np.load(path) as archive:
            if "samples" not in archive or "columns" not in archive:
                print(f"[{index+1}/{len(paths)}] SKIPPED non-Step1 archive: {path}")
                index+=1;continue
            samples=np.asarray(archive["samples"]);columns=np.asarray(archive["columns"])
            sample_dt=float(archive["dt"]) if "dt" in archive else args.dt
        mask,_=collision_review_mask(samples,columns)
        spans=_true_time_spans(samples[:,0],mask)
        print(f"[{index+1}/{len(paths)}] {path.name}: "
              f"{len(spans)} collision/stuck candidate interval(s) {spans}")
        selected,jump_index=interactive_trim_saved_extract(
            path,samples,columns,sample_dt,
            f"Collision review {index+1}/{len(paths)}: {path.stem}",args.command_topic,
            (IMU_WZ_SIGN,IMU_AX_SIGN,IMU_AY_SIGN),args.map_yaml,len(paths),index+1,
            review_collisions=True)
        if selected is None and jump_index is not None:
            index=jump_index;continue
        index+=1


def backup_interactive_outputs(out):
    """Move existing artifacts aside until the user explicitly presses q."""
    # An interrupted F5/debug session can leave the last committed files under
    # ``.before-interactive`` and partially regenerated files at the targets.
    # Recover the committed pair first instead of forcing the user to delete a
    # backup that may contain their manual trim decisions.
    targets=(out,out.with_suffix(".json"))
    stale=[(target,target.with_name(target.name+".before-interactive"))
           for target in targets
           if target.with_name(target.name+".before-interactive").exists()]
    if stale:
        print(f"Recovering {len(stale)} stale interactive backup(s) for {out.stem}")
        for target,backup in stale:
            target.unlink(missing_ok=True)
            backup.replace(target)
    backups=[]
    for target in targets:
        backup=target.with_name(target.name+".before-interactive")
        if target.exists():
            target.replace(backup);backups.append((target,backup))
    return backups


def finish_interactive_outputs(out,backups,save):
    """Commit q output, or restore the pre-run files after an arrow-key skip."""
    targets=(out,out.with_suffix(".json"))
    if save:
        for _,backup in backups:
            backup.unlink(missing_ok=True)
        return
    for target in targets:
        target.unlink(missing_ok=True)
    for target,backup in backups:
        backup.replace(target)


def saved_absolute_time_range(out):
    """Return an existing extract's retained absolute interval for re-extraction."""
    out=Path(out)
    if not out.is_file():return None
    with np.load(out) as archive:
        if not {"samples","columns","alignment_start_epoch_s"}.issubset(archive.files):
            return None
        samples=np.asarray(archive["samples"]);columns=np.asarray(archive["columns"])
        if not len(samples):return None
        names={str(name):index for index,name in enumerate(columns)}
        start_epoch=float(archive["alignment_start_epoch_s"])
        return start_epoch+float(samples[0,names["t"]]),start_epoch+float(samples[-1,names["t"]])


def reapply_absolute_time_range(out,samples,columns,absolute_range):
    """Reapply a previously saved manual cut after rebuilding the raw extract."""
    if absolute_range is None:return samples
    with np.load(out) as archive:
        new_epoch=float(archive["alignment_start_epoch_s"])
    names={str(name):index for index,name in enumerate(columns)}
    relative_start=absolute_range[0]-new_epoch
    relative_end=absolute_range[1]-new_epoch
    time=samples[:,names["t"]]
    keep=(time>=relative_start-1e-9)&(time<=relative_end+1e-9)
    selected=samples[keep]
    if len(selected)<10:
        raise RuntimeError(
            f"cannot reapply saved trim {absolute_range}: only {len(selected)} samples remain")
    return persist_manual_trim(out,selected,columns,relative_start,relative_end)


def extract_one(storage, out, args):
    cfg=yaml.safe_load((PROJECT_ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    signs,date=imu_signs(storage)
    pose,velocity,drive,imu,applied=read_streams(storage,args.pose_topic,args.velocity_topic,
        args.command_topic,args.imu_topic,args.applied_command_topic)
    manual_bag=not len(applied)
    streams={"pose":pose,"velocity":velocity,"command":drive,"imu":imu}
    if not manual_bag:streams["applied_command"]=applied
    empty=[name for name,value in streams.items() if not len(value)]
    if empty: raise RuntimeError(f"{storage}: empty streams {empty}")
    start=max(x[0,0] for x in streams.values());end=min(x[-1,0] for x in streams.values())
    if end<=start: raise RuntimeError(f"{storage}: streams have no overlapping interval")
    times=np.arange(start,end,args.dt)
    pp,pv=causal_hold(pose,times,args.max_pose_age);vv,vvalid=causal_hold(velocity,times,args.max_velocity_age)
    dd,dvalid=causal_hold(drive,times,args.max_command_age);ii,ivalid=causal_hold(imu,times,args.max_imu_age)
    if manual_bag:
        aa=dd.copy();avalid=np.ones(len(times),bool)
    else:
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
        moving_vx=args.physics_moving_vx,moving_command=args.physics_moving_command,
        frozen_pose_speed=args.physics_frozen_pose_speed,
        distance_window_s=args.physics_distance_window,min_odom_distance=args.physics_min_odom_distance,
        min_pose_odom_ratio=args.physics_min_pose_odom_ratio,impact_decel=args.physics_impact_decel,
        max_pose_step=args.physics_max_pose_step,max_yaw_step=args.physics_max_yaw_step)
    # A bag is one physical run. Once physics corruption appears,
    # never splice a later recovery fragment back onto it. Callback jitter or
    # a temporary sensor-age gap is recorded but is not a terminal collision.
    # /ackermann_cmd and /drive are not identical actuator-stage signals, so
    # their mismatch remains a diagnostic and must not cut an autonomous bag.
    # Reverse is legitimate in manual bags. For autonomous bags only, recovery
    # reverse is an additional early collision signal; pose-motion consistency
    # remains the mode-independent collision rule.
    bad=physical_bad if manual_bag else (physical_bad|collision)
    bad_indices=np.flatnonzero(bad)
    time_gaps=np.flatnonzero(np.diff(base[:,0])>1.5*args.dt)
    bad_cutoff=int(bad_indices[0]) if len(bad_indices) else len(base)
    cutoff=bad_cutoff
    cutoff_causes=[]
    if cutoff==bad_cutoff and len(bad_indices):
        index=bad_cutoff
        if physical_bad[index]:cutoff_causes.append("physical_inconsistency")
        if not manual_bag and collision[index]:cutoff_causes.append("autonomous_reverse_recovery")
    run=np.arange(cutoff,dtype=int)
    minimum=max(10,int(round(args.min_continuous_segment/args.dt)))
    if len(run)<minimum:
        raise RuntimeError(f"{storage}: only {len(run)} prefix samples before terminal "
                           f"cutoff {cutoff_causes}; need {minimum}")
    part=base[run].copy();source_start=float(part[0,0]);part[:,0]-=source_start
    arrays=[np.c_[part,np.ones(len(part)),np.zeros(len(part)),ii[run,1:4]]]
    accepted_intervals=[(times[0]+float(base[run[0],0]),
                         times[0]+float(base[run[-1],0]))]
    segments=[{"bag_id":0,"samples":len(part),"source_start_s":source_start,
               "source_end_s":float(base[run[-1],0])}]
    discarded_short_fragments=[]
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
    kf_snapshot={**ClassicModelParameters.from_mapping(cfg).runtime_updates(),
        "classic_kf_process_var":list(map(float,cfg["classic_kf_process_var"])),
        "classic_kf_measurement_var":list(map(float,cfg["classic_kf_measurement_var"])),
        "classic_kf_initial_var":list(map(float,cfg["classic_kf_initial_var"])),
        "kf_pose_vy_window_s":float(cfg.get("kf_pose_vy_window_s",.12))}
    future_horizon_s=args.horizon_steps*args.model_dt
    callback_inputs,callback_future_states,callback_future_commands,callback_future_offsets_s=(
        build_callback_prediction_samples(pose,velocity,drive,imu,signs,cfg,
            accepted_intervals,times[0],future_horizon_s,args.dt,
            args.max_command_age,args.max_imu_age,args.physics_max_pose_step,
            args.physics_max_yaw_step))
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
                        kf_state_clamp_enabled=np.array(False),
                        kf_parameter_hash=np.array(ClassicModelParameters.from_mapping(cfg).digest()),
                        kf_config_snapshot_json=np.array(json.dumps(kf_snapshot,sort_keys=True)),
                        callback_inputs=callback_inputs,
                        callback_input_columns=np.array(("t","x","y","yaw","vx","vy",
                            "yaw_rate","steer_cmd","speed_cmd","imu_wz","imu_ax","imu_ay",
                            "applied_steer","speed_reference",
                            "steer_t-4","speed_t-4","steer_t-3","speed_t-3",
                            "steer_t-2","speed_t-2","steer_t-1","speed_t-1",
                            "steer_t","speed_t","bag_id")),
                        callback_future_states=callback_future_states,
                        callback_future_state_columns=np.array(("x","y","yaw","vx","vy","yaw_rate")),
                        callback_future_commands=callback_future_commands,
                        callback_future_command_columns=np.array(("steer_cmd","speed_cmd")),
                        callback_future_offsets_s=callback_future_offsets_s,
                        callback_anchor_source=np.array("every velocity/odom callback timestamp"),
                        callback_gt_interpolation=np.array("linear pose(unwrapped yaw)+velocity; no extrapolation"),
                        callback_actuator_initialization=np.array(
                            "first applied_steer=steer_cmd; first speed_reference=measured vx"),
                        callback_history_contract=np.array(
                            "five actual callbacks ending at anchor; first four anchors discarded"))
    meta={"source":str(storage.resolve()),"pose_topic":args.pose_topic,"velocity_topic":args.velocity_topic,
          "command_topic":args.command_topic,"imu_topic":args.imu_topic,"alignment":"causal_hold",
          "recording_date":date.isoformat(),"imu_sign_cutover":IMU_SIGN_CUTOVER.isoformat(),
          "imu_axis_signs":{"wz":signs[0],"ax":signs[1],"ay":signs[2]},
          "imu_ema_alpha":IMU_EMA_ALPHA,
          "applied_command_topic":args.applied_command_topic,
          "driving_mode":"manual" if manual_bag else "autonomous",
          "driving_mode_detection":"manual iff /drive topic is absent",
          "training_command_source":args.command_topic,
          "applied_command_topic_available":not manual_bag,
          "alignment_start_epoch_s":float(start),"raw_aligned_samples":len(base),
          "removed_collision_samples":int(collision.sum()),"raw_command_mismatch_samples":int(command_mismatch.sum()),
          "collision_pre_margin_s":args.collision_pre_margin,
          "removed_manual_with_margin_samples":0,
          "command_mismatch_with_margin_samples":int(manual.sum()),
          "removed_physical_inconsistency_samples":int(physical_bad.sum()),
          "physical_inconsistency_events":physical_events,
          "physical_filter":{"pre_margin_s":args.physics_pre_margin,"post_margin_s":args.physics_post_margin,
              "moving_vx_mps":args.physics_moving_vx,"moving_command_mps":args.physics_moving_command,
              "frozen_pose_speed_mps":args.physics_frozen_pose_speed,
              "distance_window_s":args.physics_distance_window,"min_odom_distance_m":args.physics_min_odom_distance,
              "min_mcl_odom_distance_ratio":args.physics_min_pose_odom_ratio,
              "impact_decel_mps2":args.physics_impact_decel,"max_pose_step_m":args.physics_max_pose_step,
              "max_yaw_step_rad":args.physics_max_yaw_step},
          "command_topic_comparison":{"diagnostic_only":True,
              "steer_tolerance":args.command_steer_match_tol,"speed_tolerance":args.command_speed_match_tol},
          "output_samples":len(samples),"collision_episodes":episodes,"segments":segments,
          "continuous_segment_policy":{"minimum_duration_s":args.min_continuous_segment,
              "discarded_short_fragments":discarded_short_fragments,
              "mcl_alignment":"causal hold with maximum sample age",
              "terminal_cutoff_index":cutoff,"terminal_cutoff_time_s":float(base[cutoff,0]) if cutoff<len(base) else None,
              "terminal_cutoff_causes":cutoff_causes,
              "nonterminal_sensor_gap_count":int(len(time_gaps)),
              "command_mismatch_used_for_cutoff":False,
              "reverse_recovery_used_for_cutoff":not manual_bag,
              "suffix_after_first_anomaly_reused":False},
          "state_estimator":{"method":"causal MPPI classic-model EKF",
              "classic_parameter_hash":ClassicModelParameters.from_mapping(cfg).digest(),
              "state_order":["x","y","yaw","vx","vy","yaw_rate"],
              "comparison_reference":["MCL x","MCL y","MCL yaw","odom vx","MCL-difference body vy","signed IMU yaw-rate"],
              "segments":kf_reports},
          "callback_prediction_dataset":{"anchors":int(len(callback_inputs)),
              "anchor_source":"every retained velocity/odom callback",
              "horizon_steps":args.horizon_steps,
              "model_dt_s":args.model_dt,
              "future_horizon_s":future_horizon_s,
              "interpolation_dt_s":args.dt,
              "future_offsets_s":callback_future_offsets_s.tolist(),
              "future_state_order":["x","y","unwrapped_yaw","vx","vy","yaw_rate"],
              "future_command_order":["steer_cmd","speed_cmd"],
              "continuity":"no extrapolation; no retained segment crossing; pose/yaw jump rejection"},
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
    p.add_argument("--physics-moving-command",type=float,default=PHYSICS_MOVING_VX)
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
    p.add_argument("--horizon-steps",type=int,default=HORIZON_STEPS,
                   help="future rollout horizon in MODEL_DT prediction steps")
    p.add_argument("--model-dt",type=float,default=MODEL_DT_S,
                   help="MPPI/residual-model prediction interval [s]")
    p.add_argument("--map-yaml",type=Path,default=DEFAULT_MAP_YAML,
                   help="ROS occupancy-map YAML overlaid on the trajectory panel")
    p.add_argument("--interactive-trim",action=argparse.BooleanOptionalAction,default=True,
                   help="s/e/q manual dataset trimming while each bag figure is open")
    p.add_argument("--preserve-existing-trim",action=argparse.BooleanOptionalAction,default=False,
                   help="reapply an existing NPZ's absolute retained interval after re-extraction")
    p.add_argument("--refresh-saved-kf-from-yaml",action="store_true",
                   help="recompute KF/Pacejka columns in saved trimmed NPZs using config/params.yaml")
    p.add_argument("--review-saved-collisions",action=argparse.BooleanOptionalAction,
                   default=REVIEW_SAVED_COLLISIONS,
                   help="skip extraction; review saved Step-1 NPZs and mark collision/stuck candidates")
    args = p.parse_args()

    if args.horizon_steps < 1:
        p.error("--horizon-steps must be at least 1")
    if args.model_dt <= 0. or args.dt <= 0.:
        p.error("--model-dt and --dt must be positive")
    model_stride=args.model_dt/args.dt
    if not np.isclose(model_stride,round(model_stride),rtol=0.,atol=1e-9):
        p.error("--model-dt must be an integer multiple of --dt so model targets exist exactly")

    if args.refresh_saved_kf_from_yaml:
        refresh_saved_kf_from_yaml(args.output)
        return

    if args.review_saved_collisions:
        if not USE_PLOT:raise SystemExit("--review-saved-collisions requires USE_PLOT=True")
        review_saved_extracts(args.output,args)
        return

    requested=[Path(x) for x in args.bag] if args.bag else list(BAG_PATH)
    if not requested: raise SystemExit("BAG_PATH is empty")
    storages=[resolve_storage(x) for x in requested]
    storage_index=0
    while storage_index<len(storages):
        number=storage_index+1;storage=storages[storage_index]
        out=output_for_bag(args.output,storage,len(storages)>1)
        preserved_range=(saved_absolute_time_range(out)
                         if args.preserve_existing_trim else None)
        print(f"[{number}/{len(storages)}] Extracting {storage} -> {out}")
        backups=(backup_interactive_outputs(out)
                 if USE_PLOT and args.interactive_trim else [])
        try:
            samples,columns,signs=extract_one(storage,out,args)
        except RuntimeError as error:
            if USE_PLOT and args.interactive_trim:
                finish_interactive_outputs(out,backups,save=False)
            print(f"[{number}/{len(storages)}] SKIPPED: {error}",file=sys.stderr)
            storage_index+=1
            continue
        if preserved_range is not None:
            samples=reapply_absolute_time_range(out,samples,columns,preserved_range)
        if USE_PLOT:
            title=f"Bag {number}/{len(storages)}: {storage.parent.name}"
            if args.interactive_trim:
                try:
                    selected,jump_index=interactive_trim_saved_extract(
                        out,samples,columns,args.dt,title,args.command_topic,signs,
                        args.map_yaml,len(storages),number)
                except BaseException:
                    finish_interactive_outputs(out,backups,save=False)
                    raise
                finish_interactive_outputs(out,backups,save=selected is not None)
                if selected is None:
                    if jump_index is not None:
                        storage_index=jump_index
                        continue
                    print(f"[{number}/{len(storages)}] not saved; moving to next bag")
                    storage_index+=1
                    continue
                samples=selected
            else:
                plot_extracted(samples,columns,args.dt,title,args.command_topic,
                               signs,args.map_yaml)
        storage_index+=1


if __name__ == "__main__":
    main()
