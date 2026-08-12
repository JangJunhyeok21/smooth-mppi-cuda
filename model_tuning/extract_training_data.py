#!/usr/bin/env python3
"""Extract the exact real-car MPPI observation path from rosbag2.

Pose is taken from /newmcl_pose, body velocity from /odom, controls from the
selected Ackermann topic, and IMU from /imu/data.  Every stream is aligned by
causal hold; no future sample is used.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_tuning_utils.filter_collision_recovery_episodes import collision_recovery_mask
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams, estimate_dataset

# USER SETTINGS. Add every bag storage file or rosbag2 directory here. Running
# this script without arguments extracts them sequentially.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAG_PATH = [
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0807/rosbag2_2026_08_07-19_13_58/rosbag2_2026_08_07-19_13_58_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-16_54_33/rosbag2_2026_08_08-16_54_33_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-20_19_06/rosbag2_2026_08_08-20_19_06_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-20_20_34/rosbag2_2026_08_08-20_20_34_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-20_25_26/rosbag2_2026_08_08-20_25_26_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-22_10_38/rosbag2_2026_08_08-22_10_38_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0808/rosbag2_2026_08_08-22_11_08/rosbag2_2026_08_08-22_11_08_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0810/rosbag2_2026_08_10-21_45_06/rosbag2_2026_08_10-21_45_06_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0810/rosbag2_2026_08_10-21_45_57/rosbag2_2026_08_10-21_45_57_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0810/rosbag2_2026_08_10-21_46_44/rosbag2_2026_08_10-21_46_44_0.db3"),
    Path("/mnt/nas_custom/F1tenth/2026 IFAC/0810/rosbag2_2026_08_10-21_52_23/rosbag2_2026_08_10-21_52_23_0.db3"),
]
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/data/extracted_bags"
USE_PLOT = True
# Must match the MPPI runtime observer/sign convention used for training.
IMU_WZ_SIGN = -1.0; IMU_AY_SIGN = -1.0; IMU_EMA_ALPHA = .25
KF_CF = 12.7222491; KF_CR = 75.0944752
KF_STEER_SCALE = 1.1058064699; KF_STEER_BIAS = -0.0300696939; KF_MAX_STEER = .4788
POSE_TOPIC = "/newmcl_pose"; VELOCITY_TOPIC = "/odom"; DRIVE_TOPIC = "/drive"; IMU_TOPIC = "/imu/data"
DT = .02; MAX_POSE_AGE = 1.0; MAX_VELOCITY_AGE = 1.0; MAX_COMMAND_AGE = 1.0; MAX_IMU_AGE = .05


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


def read_streams(storage, pose_topic, velocity_topic, drive_topic, imu_topic):
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
    missing = [x for x in topics if x not in types]
    if missing:
        raise RuntimeError(f"missing topics {missing}; available={sorted(types)}")
    msg_types = {x: get_message(types[x]) for x in topics}
    pose, velocity, drive, imu = [], [], [], []
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
        else:
            imu.append((t, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y))
    return tuple(np.asarray(x, np.float64) for x in (pose, velocity, drive, imu))


def causal_hold(stream, times, max_age):
    stream = stream[np.argsort(stream[:, 0])]
    stream = stream[np.r_[True, np.diff(stream[:, 0]) > 1e-9]]
    index = np.searchsorted(stream[:, 0], times, side="right")-1
    valid = index >= 0
    clipped = np.maximum(index, 0)
    valid &= (times-stream[clipped, 0]) <= max_age
    return stream[clipped], valid


def plot_extracted(samples, columns, dt, title):
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
        cornering_stiffness_rear=KF_CR,dt=dt)
    kf_vy,kf_w=estimate_dataset(samples,columns,dt,kf_params,
        steer_scale=KF_STEER_SCALE,steer_bias=KF_STEER_BIAS,max_steer=KF_MAX_STEER,
        imu_ema_alpha=IMU_EMA_ALPHA,imu_wz_sign=IMU_WZ_SIGN,imu_ay_sign=IMU_AY_SIGN)
    fig,axes=plt.subplots(5,2,figsize=(15,21));fig.suptitle(title,y=.995)
    ax=axes[0,0];ax.plot(x,y,"k-",lw=1.5,label="pose trajectory")
    stride=max(1,len(samples)//35)
    ax.quiver(x[::stride],y[::stride],np.cos(heading[::stride]),np.sin(heading[::stride]),
              heading[::stride],cmap="hsv",angles="xy",scale_units="xy",scale=3.5,width=.004,
              label="yaw direction")
    ax.set_title("x-y trajectory with yaw direction/color");ax.set_xlabel("x [m]");ax.set_ylabel("y [m]")
    ax.axis("equal");ax.grid(alpha=.25);ax.legend()
    signed_imu_wz=IMU_WZ_SIGN*imu_wz
    axes[0,1].plot(t,np.unwrap(heading),label="pose yaw [rad]")
    axes[0,1].plot(t,signed_imu_wz,label=f"signed IMU yaw-rate (raw x {IMU_WZ_SIGN:+.0f})")
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
    axes[2,0].plot(t,speed_cmd,label="/drive speed command");axes[2,0].set_title("Longitudinal velocity")
    axes[2,1].plot(t,odom_vy,label="stored odom vy");axes[2,1].plot(t,pose_vy,label="pose-derived vy")
    axes[2,1].plot(t,kf_vy,label="training/runtime KF vy",lw=1.5)
    axes[2,1].set_title("Lateral velocity inputs/estimates")
    axes[3,0].plot(t,imu_ax,label="raw IMU ax");axes[3,0].plot(t,imu_ay,":",color="0.65",label="raw IMU ay (sensor frame)")
    axes[3,0].plot(t,IMU_AY_SIGN*imu_ay,label=f"signed training IMU ay (raw x {IMU_AY_SIGN:+.0f})",alpha=.9)
    axes[3,0].set_title("IMU acceleration and configured ay sign")
    axes[3,1].plot(t,steer,label="/drive steering command");axes[3,1].set_title("Steering command")
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
    pose,velocity,drive,imu=read_streams(storage,args.pose_topic,args.velocity_topic,args.drive_topic,args.imu_topic)
    streams={"pose":pose,"velocity":velocity,"drive":drive,"imu":imu}
    empty=[name for name,value in streams.items() if not len(value)]
    if empty: raise RuntimeError(f"{storage}: empty streams {empty}")
    start=max(x[0,0] for x in streams.values());end=min(x[-1,0] for x in streams.values())
    if end<=start: raise RuntimeError(f"{storage}: streams have no overlapping interval")
    times=np.arange(start,end,args.dt)
    pp,pv=causal_hold(pose,times,args.max_pose_age);vv,vvalid=causal_hold(velocity,times,args.max_velocity_age)
    dd,dvalid=causal_hold(drive,times,args.max_command_age);ii,ivalid=causal_hold(imu,times,args.max_imu_age)
    valid=pv&vvalid&dvalid&ivalid;times,pp,vv,dd,ii=(x[valid] for x in (times,pp,vv,dd,ii))
    if not len(times): raise RuntimeError(f"{storage}: no samples survived causal alignment")
    base=np.c_[times-times[0],pp[:,1:4],vv[:,1:4],dd[:,1:4]]
    bad,episodes=collision_recovery_mask(base,args.dt);kept=np.flatnonzero(~bad)
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
                        velocity_topic=np.array(args.velocity_topic),drive_topic=np.array(args.drive_topic),
                        imu_topic=np.array(args.imu_topic))
    meta={"source":str(storage.resolve()),"pose_topic":args.pose_topic,"velocity_topic":args.velocity_topic,
          "drive_topic":args.drive_topic,"imu_topic":args.imu_topic,"alignment":"causal_hold",
          "raw_aligned_samples":len(base),"removed_collision_samples":int(bad.sum()),
          "output_samples":len(samples),"collision_episodes":episodes,"segments":segments,
          "split_policy":"single-bag-identical-train-test"}
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({**meta,"output":str(out)},indent=2))
    return samples,columns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag", nargs="*", help="optional .db3/.mcap files or rosbag2 directories; default=BAG_PATH")
    p.add_argument("-o", "--output", default=str(OUTPUT_PATH))
    p.add_argument("--pose-topic", default=POSE_TOPIC)
    p.add_argument("--velocity-topic", default=VELOCITY_TOPIC)
    p.add_argument("--drive-topic", default=DRIVE_TOPIC)
    p.add_argument("--imu-topic", default=IMU_TOPIC)
    p.add_argument("--dt", type=float, default=DT)
    # The node retains the latest pose/velocity/command between callbacks.
    # A generous watchdog reproduces that behavior without fragmenting a bag
    # for an occasional dropped 50 Hz message. IMU retains its strict 50 ms age.
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
            samples,columns=extract_one(storage,out,args)
        except RuntimeError as error:
            print(f"[{number}/{len(storages)}] SKIPPED: {error}",file=sys.stderr)
            continue
        if USE_PLOT: plot_extracted(samples,columns,args.dt,f"Bag {number}/{len(storages)}: {storage.parent.name}")


if __name__ == "__main__":
    main()
