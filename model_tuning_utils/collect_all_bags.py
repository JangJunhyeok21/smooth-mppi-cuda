#!/usr/bin/env python3
"""Discover usable bags, extract each independently, and make an 80/20 bag-level dataset."""
import argparse, json
from pathlib import Path
import numpy as np, yaml
try:
    from model_tuning_utils.extract_bag import read_bag, align
    from model_tuning_utils.filter_collision_recovery_episodes import collision_recovery_mask
except ModuleNotFoundError:  # direct execution from model_tuning_utils/
    from extract_bag import read_bag, align
    from filter_collision_recovery_episodes import collision_recovery_mask

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("root")
    p.add_argument("-o","--output",default="model_tuning/data/all_bags_80_20.npz")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--dt",type=float,default=.02)
    p.add_argument("--drive-topic",default="/drive",
                   help="required command topic; no fallback to /mpc_cmd")
    p.add_argument("--max-pose-step",type=float,default=.25,
                   help="reject a bag if any aligned x/y step exceeds this [m]")
    a=p.parse_args(); candidates=[]
    for meta in sorted(Path(a.root).rglob("metadata.yaml")):
        try:
            info=yaml.safe_load(meta.read_text())["rosbag2_bagfile_information"]
            tp={x["topic_metadata"]["name"]:x["topic_metadata"]["type"] for x in info["topics_with_message_count"]}
            od=next((n for n in ["/mocap_odom","/odom"] if tp.get(n)=="nav_msgs/msg/Odometry"),None)
            drive=(a.drive_topic
                   if tp.get(a.drive_topic)=="ackermann_msgs/msg/AckermannDriveStamped"
                   else None)
            if od and drive:
                rel=info["relative_file_paths"][0]; candidates.append((meta.parent/rel,od,drive))
        except Exception as e:
            # A partially written metadata.yaml must not hide an otherwise
            # readable MCAP. Inspect storage files directly as a recovery path.
            recovered=False
            try:
                import rosbag2_py
                for storage in sorted((*meta.parent.glob("*.mcap"),
                                       *meta.parent.glob("*.db3"))):
                    storage_id="mcap" if storage.suffix==".mcap" else "sqlite3"
                    reader=rosbag2_py.SequentialReader();reader.open(
                        rosbag2_py.StorageOptions(uri=str(storage),storage_id=storage_id),
                        rosbag2_py.ConverterOptions("cdr","cdr"))
                    tp={x.name:x.type for x in reader.get_all_topics_and_types()}
                    od=next((n for n in ["/mocap_odom","/odom"]
                             if tp.get(n)=="nav_msgs/msg/Odometry"),None)
                    drive=(a.drive_topic if tp.get(a.drive_topic)==
                           "ackermann_msgs/msg/AckermannDriveStamped" else None)
                    if od and drive:
                        candidates.append((storage,od,drive));recovered=True
                        print(f"recovered corrupt metadata via storage: {storage}")
                        break
            except Exception as recovery_error:
                print(f"storage recovery failed {meta}: {recovery_error}")
            if not recovered: print(f"skip metadata {meta}: {e}")
    rng=np.random.default_rng(a.seed); order=rng.permutation(len(candidates)); ntrain=int(.8*len(candidates))
    train_set=set(order[:ntrain].tolist()); arrays=[]; manifest=[]; segments=[]; segment_bag_id=0
    for source_bag_id,(path,od,drive) in enumerate(candidates):
        split=0 if source_bag_id in train_set else 1
        row={"bag_id":source_bag_id,"source_bag_id":source_bag_id,"path":str(path),"odom_topic":od,"drive_topic":drive,
             "split":"train" if split==0 else "test"}
        try:
            o,d=read_bag(path,od,drive,"record"); s=align(o,d,a.dt,.1,"acceleration",0.)
            pose_step=np.linalg.norm(np.diff(s[:,1:3],axis=0),axis=1)
            max_step=float(pose_step.max(initial=0.))
            row["max_pose_step_m"]=max_step
            row["pose_steps_over_limit"]=int(np.sum(pose_step>a.max_pose_step))
            if max_step>a.max_pose_step:
                raise ValueError(
                    f"corrupt odometry: max pose step {max_step:.3f} m "
                    f"> {a.max_pose_step:.3f} m"
                )
            # speed is the control setpoint; reverse recovery is identified by
            # measured signed body vx. Remove collision -> reverse -> stable
            # forward recovery as one episode, then split all discontinuities.
            collision_recovery,episodes=collision_recovery_mask(s,a.dt)
            kept=np.flatnonzero(~collision_recovery); segment_ids=[]
            breaks=np.flatnonzero((np.diff(kept)>1) | (np.diff(s[kept,0])>1.5*a.dt))+1
            for run in np.split(kept,breaks) if len(kept) else []:
                part=s[run].copy()
                if not len(part): continue
                source_time_start=float(part[0,0]);source_time_end=float(part[-1,0])
                # Each retained segment has a local clock. The manifest keeps
                # the source-bag offset used by causal IMU alignment.
                part[:,0]-=source_time_start
                extra=np.c_[np.full(len(part),split),np.full(len(part),segment_bag_id)]
                arrays.append(np.c_[part,extra]); segment_ids.append(segment_bag_id)
                segments.append({"segment_bag_id":segment_bag_id,
                    "source_bag_id":source_bag_id,"samples":len(part),
                    "split":"test" if split else "train",
                    "source_time_start_s":source_time_start,
                    "source_time_end_s":source_time_end,
                    "usable_2s_window":bool(len(part)>=100)})
                segment_bag_id+=1
            row["samples"]=len(s)
            row["removed_collision_recovery_samples"]=int(collision_recovery.sum())
            row["collision_recovery_episodes"]=episodes
            row["segment_bag_ids"]=segment_ids; row["status"]="ok"
            print(f"[{source_bag_id+1}/{len(candidates)}] {row['split']} {path.name}: "
                  f"{len(kept)}/{len(s)} retained samples, {len(episodes)} collision episodes, "
                  f"{len(segment_ids)} segments")
        except BaseException as e:
            row["status"]="failed"; row["error"]=str(e); print(f"FAILED {path}: {e}")
        manifest.append(row)
    if not arrays: raise SystemExit("no bag could be extracted")
    # Rebalance only after quality filtering. Otherwise failed/corrupt bags can
    # consume test slots and leave a severely biased effective split.
    successful_ids=np.array([int(x["source_bag_id"]) for x in manifest
                             if x.get("status")=="ok"],dtype=int)
    successful_order=rng.permutation(successful_ids)
    successful_train_count=int(.8*len(successful_ids))
    if len(successful_ids)>1:
        successful_train_count=min(len(successful_ids)-1,
                                   max(1,successful_train_count))
    final_train=set(successful_order[:successful_train_count].tolist())
    segment_to_source={int(x["segment_bag_id"]):int(x["source_bag_id"])
                       for x in segments}
    for part in arrays:
        source_id=segment_to_source[int(part[0,-1])]
        part[:,-2]=0 if source_id in final_train else 1
    for row in manifest:
        if row.get("status")=="ok":
            row["split"]="train" if int(row["source_bag_id"]) in final_train else "test"
    for segment in segments:
        segment["split"]=("train" if int(segment["source_bag_id"]) in final_train
                          else "test")
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,samples=np.concatenate(arrays),dt=a.dt,
      columns=np.array(["t","x","y","yaw","vx","vy","omega","steer","accel","speed_cmd","split","bag_id"]),
      drive_topic=np.array(a.drive_topic))
    (out.with_suffix(".manifest.json")).write_text(json.dumps({"seed":a.seed,"requested_ratio":"80/20",
      "required_drive_topic":a.drive_topic,"discovered":len(candidates),
      "successful":sum(x.get("status")=="ok" for x in manifest),"bags":manifest,
      "segments":segments},indent=2)+"\n")
    print(f"saved {sum(map(len,arrays))} samples to {out}")

if __name__=="__main__": main()
