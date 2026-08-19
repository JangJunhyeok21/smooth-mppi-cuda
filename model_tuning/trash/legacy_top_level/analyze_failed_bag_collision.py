#!/usr/bin/env python3
"""Reconstruct the 22:10:38 collision from pose, commands and MPPI best path."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BAG = Path("/home/a/Downloads/rosbag2_2026_08_08-22_1/rosbag2_2026_08_08-22_10_38/rosbag2_2026_08_08-22_10_38_0.db3")
OUT = ROOT / "model_tuning/results/failed_bag_collision_analysis"
CAR_RADIUS = 0.25
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")


def stamp(message, record_ns):
    header = getattr(message, "header", None)
    value = getattr(header, "stamp", None)
    if value is None or (value.sec == 0 and value.nanosec == 0):
        return record_ns*1e-9
    return value.sec+value.nanosec*1e-9


def nearest_distance(points, boundary):
    return np.sqrt(((points[:, None, :]-boundary[None, :, :])**2).sum(2)).min(1)


def main():
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(BAG),storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr","cdr"))
    types={item.name:item.type for item in reader.get_all_topics_and_types()}
    pose=[];odom=[];drive=[];ack=[];teleop=[];viz=[];left=right=path=None
    wanted={"/newmcl_pose","/odom","/drive","/ackermann_cmd","/teleop",
            "/mppi_viz","/mppi_left_boundary","/mppi_right_boundary","/mppi_target_path"}
    while reader.has_next():
        topic,raw,record_ns=reader.read_next()
        if topic not in wanted: continue
        msg=deserialize_message(raw,get_message(types[topic]));t=stamp(msg,record_ns)
        if topic=="/newmcl_pose":
            q=msg.pose.orientation;yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
            pose.append((t,msg.pose.position.x,msg.pose.position.y,yaw))
        elif topic=="/odom":
            v=msg.twist.twist;odom.append((t,v.linear.x,v.linear.y,v.angular.z))
        elif topic in ("/drive","/ackermann_cmd","/teleop"):
            d=msg.drive;row=(t,d.steering_angle,d.speed,d.acceleration)
            {"/drive":drive,"/ackermann_cmd":ack,"/teleop":teleop}[topic].append(row)
        elif topic=="/mppi_viz":
            best=next((m for m in msg.markers if m.ns=="best_trajectory"),None)
            if best and len(best.points)>=2:
                # LINE_LIST is p0,p1,p1,p2,...; recover the 80 unique states.
                xy=[(best.points[0].x,best.points[0].y)]
                xy.extend((best.points[i].x,best.points[i].y) for i in range(1,len(best.points),2))
                viz.append((t,np.asarray(xy)))
        else:
            xy=np.array([(p.pose.position.x,p.pose.position.y) for p in msg.poses])
            if topic=="/mppi_left_boundary":left=xy
            elif topic=="/mppi_right_boundary":right=xy
            else:path=xy
    pose,odom,drive,ack,teleop=(np.asarray(x,float) for x in (pose,odom,drive,ack,teleop))
    start=max(pose[0,0],odom[0,0],drive[0,0]);end=min(pose[-1,0],odom[-1,0],drive[-1,0])
    # Use the same collision evidence as extraction: first sustained stop before reverse.
    reverse_candidates=np.flatnonzero((ack[:,2]<0)&(ack[:,0]>=start)&(ack[:,0]<=end))
    reverse_start=ack[reverse_candidates[0],0]
    speed_index=np.searchsorted(odom[:,0],reverse_start)-1
    stopped=np.flatnonzero((odom[:speed_index,1]<.15)&(odom[:speed_index,0]>start+.5))
    collision_time=odom[stopped[0],0] if len(stopped) else reverse_start
    pvalid=(pose[:,0]>=start)&(pose[:,0]<=end);p=pose[pvalid]
    actual_clearance=np.minimum(nearest_distance(p[:,1:3],left),nearest_distance(p[:,1:3],right))-CAR_RADIUS

    records=[]
    for t,trajectory in viz:
        if t<start or t>collision_time: continue
        clearance=np.minimum(nearest_distance(trajectory,left),nearest_distance(trajectory,right))-CAR_RADIUS
        future=t+1.58
        pi=np.argmin(np.abs(pose[:,0]-future))
        endpoint_error=float(np.linalg.norm(trajectory[-1]-pose[pi,1:3]))
        records.append((t-start,float(clearance.min()),float(clearance[-1]),endpoint_error,trajectory))
    numeric=np.array([r[:4] for r in records])
    summary={"common_start_s":float(start),"collision_time_from_start_s":float(collision_time-start),
             "reverse_time_from_start_s":float(reverse_start-start),
             "minimum_actual_boundary_clearance_before_collision_m":float(actual_clearance[p[:,0]<=collision_time].min()),
             "prediction_samples_before_collision":len(records),
             "predicted_min_clearance_median_m":float(np.median(numeric[:,1])),
             "predicted_min_clearance_min_m":float(numeric[:,1].min()),
             "predicted_1p58s_endpoint_error_median_m":float(np.median(numeric[:,3])),
             "predicted_1p58s_endpoint_error_max_m":float(numeric[:,3].max())}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(OUT/"analysis.npz",time=p[:,0]-start,pose=p[:,1:4],actual_clearance=actual_clearance,
                        prediction_metrics=numeric)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    ax=axes[0,0];ax.plot(left[:,0],left[:,1],"k-");ax.plot(right[:,0],right[:,1],"k-");ax.plot(path[:,0],path[:,1],"k--",alpha=.5,label="target path");ax.plot(p[:,1],p[:,2],color="tab:blue",lw=2,label="actual")
    selected=records[max(0,len(records)-1)] if records else None
    if selected:ax.plot(selected[4][:,0],selected[4][:,1],"r--",lw=2,label=f"last pre-impact MPPI best, t={selected[0]:.2f}s")
    ax.axis("equal");ax.grid(alpha=.25);ax.legend();ax.set_title("Track, actual pose and last pre-impact MPPI best")
    axes[0,1].plot(p[:,0]-start,actual_clearance);axes[0,1].axhline(0,color="r",ls="--");axes[0,1].axvline(collision_time-start,color="k",ls=":",label="collision/stop");axes[0,1].set(title="Actual boundary clearance",xlabel="time [s]",ylabel="clearance after car radius [m]");axes[0,1].grid(alpha=.25);axes[0,1].legend()
    axes[1,0].plot(numeric[:,0],numeric[:,1],label="predicted minimum clearance");axes[1,0].axhline(0,color="r",ls="--");axes[1,0].set(title="MPPI best-trajectory boundary clearance",xlabel="prediction start [s]",ylabel="clearance [m]");axes[1,0].grid(alpha=.25);axes[1,0].legend()
    axes[1,1].plot(numeric[:,0],numeric[:,3],label="1.58 s endpoint error");axes[1,1].set(title="Recorded MPPI best vs future actual pose",xlabel="prediction start [s]",ylabel="endpoint error [m]");axes[1,1].grid(alpha=.25);axes[1,1].legend()
    fig.tight_layout();fig.savefig(OUT/"collision_timeline.png",dpi=180);plt.close(fig);print(json.dumps(summary,indent=2))

    # Show the recorded best rollout itself at representative pre-impact times.
    # A faulted CUDA rollout repeats its last state for the remaining horizon, so
    # report the number of unique points as well as drawing the line.
    requested_times = np.array([0.25, 0.75, 1.25, 1.60, 1.80, 2.00, 2.25])
    selected_records = []
    for requested in requested_times:
        selected_records.append(records[int(np.argmin(np.abs(numeric[:, 0] - requested)))])

    fig, axes = plt.subplots(2, 4, figsize=(21, 10), squeeze=False)
    table = []
    horizon_s = 1.58
    for ax, record in zip(axes.ravel(), selected_records):
        rel_t, min_clearance, end_clearance, endpoint_error, trajectory = record
        future_mask = ((p[:, 0] - start) >= rel_t) & ((p[:, 0] - start) <= rel_t + horizon_s)
        actual_future = p[future_mask, 1:3]
        rounded = np.round(trajectory, 5)
        unique_points = len(np.unique(rounded, axis=0))
        predicted_length = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
        actual_length = (float(np.linalg.norm(np.diff(actual_future, axis=0), axis=1).sum())
                         if len(actual_future) > 1 else float("nan"))
        table.append({
            "prediction_start_s": float(rel_t),
            "predicted_min_clearance_m": float(min_clearance),
            "predicted_unique_points": unique_points,
            "predicted_path_length_m": predicted_length,
            "actual_future_path_length_m": actual_length,
        })
        ax.plot(left[:, 0], left[:, 1], "k-", lw=1)
        ax.plot(right[:, 0], right[:, 1], "k-", lw=1)
        ax.plot(path[:, 0], path[:, 1], color="0.65", ls="--", lw=1, label="target path")
        if len(actual_future):
            ax.plot(actual_future[:, 0], actual_future[:, 1], color="tab:blue", lw=2.5,
                    label="future actual")
            ax.scatter(actual_future[0, 0], actual_future[0, 1], color="tab:blue", s=35)
        ax.plot(trajectory[:, 0], trajectory[:, 1], color="tab:red", ls="--", lw=2.5,
                label="recorded MPPI best")
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color="tab:red", marker="x", s=55)
        ax.set_title(f"start={rel_t:.2f}s, min clearance={min_clearance:+.3f}m\n"
                     f"unique={unique_points}/{len(trajectory)}, path={predicted_length:.2f}m")
        ax.axis("equal"); ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 3].axis("off")
    fig.suptitle("Recorded /mppi_viz best trajectory vs subsequent actual motion", fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT/"best_trajectory_snapshots.png", dpi=180)
    plt.close(fig)
    (OUT/"best_trajectory_snapshots.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))


if __name__=="__main__":main()
