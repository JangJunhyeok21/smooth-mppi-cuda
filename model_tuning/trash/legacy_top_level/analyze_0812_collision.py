#!/usr/bin/env python3
"""Diagnose the 0812 collision using deployed-model replay and recorded MPPI viz."""
import json
import os
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BAG=Path("/mnt/nas_custom/F1tenth/2026 IFAC/0812/rosbag2_2026_08_12-17_27_52/rosbag2_2026_08_12-17_27_52_0.db3")
REPLAY=ROOT/"model_tuning/results/all_bags_dynamic_residual_full_open_loop/rosbag2_2026_08_12-17_27_52_interactive_replay.npz"
EXTRACT_META=ROOT/"model_tuning/data/extracted_bags/rosbag2_2026_08_12-17_27_52.json"
OUTPUT=ROOT/"model_tuning/results/0812_collision_analysis"
POSITION_ERROR_THRESHOLD_M=.20
SHOW_PLOTS=False
os.environ.setdefault("MPLCONFIGDIR","/tmp/matplotlib-smppi")
import matplotlib.pyplot as plt


def angle_error(predicted,target):
    return np.arctan2(np.sin(predicted-target),np.cos(predicted-target))


def read_pose_and_viz():
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from visualization_msgs.msg import MarkerArray
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(BAG),storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr","cdr"))
    poses=[];visualizations=[];first_times={}
    relevant=("/newmcl_pose","/ackermann_cmd","/odom","/imu/data")
    while reader.has_next():
        topic,raw,record_ns=reader.read_next();record_time=record_ns*1e-9
        if topic in relevant and topic not in first_times:first_times[topic]=record_time
        if topic=="/newmcl_pose":
            msg=deserialize_message(raw,PoseStamped)
            poses.append((record_time,msg.pose.position.x,msg.pose.position.y))
        elif topic=="/mppi_viz":
            msg=deserialize_message(raw,MarkerArray);trajectories={}
            for marker in msg.markers:
                if marker.ns not in ("best_trajectory","weighted_control_trajectory"):continue
                # LINE_LIST is [p0,p1,p1,p2,...]. Recover one point per state.
                points=marker.points
                if len(points)>=2:
                    xy=[(points[0].x,points[0].y)]+[(points[k].x,points[k].y) for k in range(1,len(points),2)]
                    trajectories[marker.ns]=np.asarray(xy,np.float64)
            if trajectories:visualizations.append((record_time,trajectories))
    return np.asarray(poses),visualizations,max(first_times.values())


def model_replay_analysis():
    z=np.load(REPLAY);prediction=z["prediction_0"];target=z["target_0"];dt=float(z["dt"])
    time=np.arange(len(prediction))*dt;position_error=np.linalg.norm(prediction[:,:2]-target[:,:2],axis=1)
    yaw_error=np.degrees(np.abs(angle_error(prediction[:,2],target[:,2])))
    first_large=int(np.flatnonzero(position_error>=POSITION_ERROR_THRESHOLD_M)[0]) if np.any(position_error>=POSITION_ERROR_THRESHOLD_M) else len(time)-1
    max_error=int(np.argmax(position_error));collision_index=len(time)-1
    signals=((3,r"$v_x$ [m/s]","m/s"),(4,r"$v_y$ [m/s]","m/s"),
             (6,r"$a_x$ [m/s²]","m/s²"),(7,r"$a_y$ [m/s²]","m/s²"),
             (5,"yaw-rate [rad/s]","rad/s"),(2,"yaw [rad]","deg"))
    fig,axes=plt.subplots(4,2,figsize=(16,17));trajectory_axis=axes[0,0]
    trajectory_axis.plot(target[:,0],target[:,1],"k-",lw=2,label="GT")
    trajectory_axis.plot(prediction[:,0],prediction[:,1],"--",color="tab:orange",lw=2,label="Model open-loop")
    colored=trajectory_axis.scatter(target[:,0],target[:,1],c=position_error,cmap="turbo",s=18,zorder=3,label="GT colored by error")
    fig.colorbar(colored,ax=trajectory_axis,label="position error [m]")
    markers=((0,"Common start","tab:green","s"),(first_large,f"First ≥{POSITION_ERROR_THRESHOLD_M:.1f}m","tab:blue","^"),
             (max_error,"Maximum error","tab:red","X"),(collision_index,"Collision onset","tab:purple","P"))
    for index,label,color,marker in markers:
        trajectory_axis.scatter(target[index,0],target[index,1],s=110,c=color,marker=marker,edgecolors="white",linewidths=.8,label=f"{label}: t={time[index]:.2f}s")
    trajectory_axis.axis("equal");trajectory_axis.set_title("Where deployed-model error grows on the GT trajectory")
    trajectory_axis.grid(alpha=.25);trajectory_axis.legend(fontsize=8)
    axes[0,1].plot(time,position_error,label="position error [m]");axes[0,1].plot(time,yaw_error,label="|yaw error| [deg]")
    axes[0,1].axhline(POSITION_ERROR_THRESHOLD_M,color="tab:red",ls=":",label="position threshold")
    for index,_,color,_ in markers[1:]:axes[0,1].axvline(time[index],color=color,ls="--",alpha=.7)
    axes[0,1].set_title("Trajectory/yaw error growth");axes[0,1].legend(fontsize=8);axes[0,1].grid(alpha=.25)
    state_metrics={}
    for axis,(column,title,unit) in zip(axes.flat[2:],signals):
        if column==2:
            difference=np.degrees(np.abs(angle_error(prediction[:,column],target[:,column])))
        else:difference=np.abs(prediction[:,column]-target[:,column])
        state_metrics[title]={"mae":float(difference.mean()),"max_abs":float(difference.max()),"unit":unit,
                              "max_time_s":float(time[np.argmax(difference)])}
        axis.plot(time,target[:,column],"k-",lw=1.7,label="GT")
        axis.plot(time,prediction[:,column],"--",color="tab:orange",lw=1.7,label="Prediction")
        axis.axvline(time[first_large],color="tab:blue",ls="--",alpha=.6)
        axis.axvline(time[collision_index],color="tab:purple",ls="--",alpha=.6)
        axis.set_title(f"{title}: MAE={difference.mean():.3f}, max={difference.max():.3f} {unit}")
        axis.set_xlabel("time [s]");axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle("0812 collision — current dynamic_mlp_residual vs recorded state",y=.995)
    fig.subplots_adjust(left=.07,right=.97,bottom=.05,top=.96,hspace=.42,wspace=.25)
    fig.savefig(OUTPUT/"model_replay_error_at_collision.png",dpi=180);plt.close(fig)
    return {"duration_s":float(time[-1]),"position_error_mean_m":float(position_error.mean()),
            "position_error_final_m":float(position_error[-1]),"position_error_max_m":float(position_error[max_error]),
            "position_error_max_time_s":float(time[max_error]),"first_position_error_threshold_s":float(time[first_large]),
            "yaw_error_mean_deg":float(yaw_error.mean()),"yaw_error_final_deg":float(yaw_error[-1]),
            "state_errors":state_metrics}


def recorded_mppi_analysis(collision_start_s):
    poses,visualizations,aligned_start=read_pose_and_viz();collision_time=aligned_start+collision_start_s
    pose_time=poses[:,0];endpoint_records=[]
    for solve_time,trajectories in visualizations:
        weighted=trajectories.get("weighted_control_trajectory")
        best=trajectories.get("best_trajectory")
        selected=weighted if weighted is not None else best
        if selected is None:continue
        # MPPI publishes 79 line segments at dt=0.02: endpoint is about 1.58 s ahead.
        horizon_s=(len(selected)-1)*.02;future_time=solve_time+horizon_s
        if solve_time>collision_time or future_time>pose_time[-1]:continue
        actual_index=int(np.argmin(np.abs(pose_time-future_time)))
        current_index=int(np.argmin(np.abs(pose_time-solve_time)))
        endpoint_error=float(np.linalg.norm(selected[-1]-poses[actual_index,1:3]))
        endpoint_records.append((solve_time-aligned_start,endpoint_error,current_index,actual_index,selected,best,weighted))
    if not endpoint_records:return {"available":False}
    errors=np.asarray([x[1] for x in endpoint_records]);worst=endpoint_records[int(np.argmax(errors))]
    time,error,current_index,actual_index,selected,best,weighted=worst
    fig,axes=plt.subplots(1,2,figsize=(15,6.5))
    before=pose_time<=collision_time;axes[0].plot(poses[before,1],poses[before,2],"k-",lw=2,label="Actual vehicle")
    if best is not None:axes[0].plot(best[:,0],best[:,1],"--",color="tab:blue",lw=2,label="MPPI best trajectory")
    if weighted is not None:axes[0].plot(weighted[:,0],weighted[:,1],"--",color="tab:orange",lw=2,label="Weighted-control trajectory")
    axes[0].scatter(poses[current_index,1],poses[current_index,2],c="tab:green",s=90,marker="s",label="Solve position")
    axes[0].scatter(poses[actual_index,1],poses[actual_index,2],c="tab:red",s=90,marker="X",label="Actual future endpoint")
    axes[0].scatter(selected[-1,0],selected[-1,1],c="tab:orange",s=90,marker="o",label="Predicted future endpoint")
    axes[0].axis("equal");axes[0].set_title(f"Worst recorded MPPI horizon: t={time:.2f}s, endpoint error={error:.3f}m")
    axes[0].grid(alpha=.25);axes[0].legend(fontsize=8)
    times=np.asarray([x[0] for x in endpoint_records]);axes[1].plot(times,errors,".-")
    axes[1].scatter(time,error,c="tab:red",s=90,marker="X",label="Maximum")
    axes[1].axvline(collision_start_s,color="tab:purple",ls="--",label="Collision onset")
    axes[1].set_title("Recorded MPPI ~1.58 s endpoint vs actual future position")
    axes[1].set_xlabel("time from aligned bag start [s]");axes[1].set_ylabel("endpoint error [m]")
    axes[1].grid(alpha=.25);axes[1].legend()
    fig.tight_layout();fig.savefig(OUTPUT/"recorded_mppi_prediction_vs_actual.png",dpi=180);plt.close(fig)
    return {"available":True,"samples":len(errors),"horizon_s":float((len(selected)-1)*.02),
            "endpoint_error_mean_m":float(errors.mean()),"endpoint_error_p95_m":float(np.quantile(errors,.95)),
            "endpoint_error_max_m":float(errors.max()),"endpoint_error_max_s":float(time)}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);meta=json.loads(EXTRACT_META.read_text())
    collision_start=float(meta["collision_episodes"][0]["start_time_s"])
    report={"bag":str(BAG),"collision_onset_s":collision_start,
            "reverse_command_start_s":float(meta["collision_episodes"][0]["reverse_start_time_s"]),
            "model_open_loop":model_replay_analysis(),
            "recorded_mppi_future":recorded_mppi_analysis(collision_start)}
    (OUTPUT/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if SHOW_PLOTS:plt.show()


if __name__=="__main__":main()
