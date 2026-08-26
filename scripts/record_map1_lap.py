#!/usr/bin/env python3
"""Record MPPI laps on an arbitrary track (or collision/timeout)."""
import csv
import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from smppi_cuda_controller.msg import MppiTrajectory

# User-editable test settings. No command-line arguments are required.
TARGET_LAPS = 10.0
TIMEOUT_SECONDS = 180.0
OUTPUT_DIRECTORY = Path(
    "/home/a/smooth-mppi-cuda/model_tuning/results/map1_boundary_soft_10laps")
DEFAULT_TRACK = Path(__file__).resolve().parents[1] / "data/map1/map1_centerline.csv"


class Recorder(Node):
    def __init__(self, target_laps=TARGET_LAPS, timeout_seconds=TIMEOUT_SECONDS,
                 output_directory=OUTPUT_DIRECTORY, track_path=DEFAULT_TRACK):
        super().__init__("map1_lap_recorder")
        self.target_laps = target_laps
        self.timeout_seconds = timeout_seconds
        self.out = output_directory
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = None; self.start = None; self.left_start = False
        self.latest_pose = None
        self.previous_recorded_pose = None
        centerline = np.genfromtxt(track_path, delimiter=",", names=True)
        self.centerline_xy = np.column_stack((centerline["x_m"], centerline["y_m"]))
        closed = np.vstack((self.centerline_xy, self.centerline_xy[0]))
        segment_length = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
        self.centerline_s = np.r_[0.0, np.cumsum(segment_length[:-1])]
        self.track_length = float(segment_length.sum())
        self.last_progress = None
        self.accumulated_progress = 0.0
        self.odom=[]; self.drive=[]; self.pred=[]; self.obstacle=[]; self.status="timeout"
        self.create_subscription(Odometry,"/ego_racecar/odom",self.odom_cb,50)
        self.create_subscription(Odometry,"/opp_racecar/odom",self.obstacle_cb,20)
        self.create_subscription(AckermannDriveStamped,"/drive",self.drive_cb,50)
        # Prediction telemetry is optional for lap timing. Some stale ROS
        # workspaces cannot load this custom message's Python type support;
        # keep odom/drive/collision recording usable in that case.
        try:
            self.create_subscription(MppiTrajectory,"/mppi_optimal_trajectory",self.pred_cb,20)
        except Exception:
            self.get_logger().warning(
                "MPPI prediction telemetry unavailable; continuing lap timing")
        self.create_subscription(Bool,"/collision0",self.collision_cb,10)
        self.create_timer(.1,self.check)

    def now_s(self): return time.monotonic()
    def rel(self): return 0. if self.t0 is None else self.now_s()-self.t0
    def odom_cb(self,m):
        q=m.pose.pose.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        p=m.pose.pose.position; tw=m.twist.twist
        self.latest_pose = (p.x, p.y)
        if self.t0 is None:
            return
        if self.previous_recorded_pose is not None:
            pose_jump = math.hypot(
                p.x - self.previous_recorded_pose[0],
                p.y - self.previous_recorded_pose[1])
            # At 100 Hz even 5 m/s moves only 0.05 m. A 0.5 m one-frame
            # displacement is a manual relocation or simulator reset, never a
            # valid vehicle transition; invalidate the run immediately.
            # The simulator can publish one initialization/respawn transition
            # immediately after the first drive command.  Do not confuse that
            # startup hand-off with an in-run external relocation.
            if pose_jump > 0.5 and self.rel() > 2.0:
                self.finish("external_pose_reset")
                return
        self.previous_recorded_pose = (p.x, p.y)
        self.odom.append((self.rel(),p.x,p.y,yaw,tw.linear.x,tw.linear.y,tw.angular.z))
        nearest = int(np.argmin(np.sum((self.centerline_xy - np.array([p.x, p.y]))**2, axis=1)))
        progress = float(self.centerline_s[nearest])
        if self.last_progress is not None:
            delta = progress - self.last_progress
            if delta > 0.5*self.track_length: delta -= self.track_length
            if delta < -0.5*self.track_length: delta += self.track_length
            self.accumulated_progress += delta
        self.last_progress = progress
    def drive_cb(self,m):
        if self.t0 is None and abs(m.drive.speed) > .05 and self.latest_pose is not None:
            self.t0=self.now_s(); self.start=self.latest_pose
        self.drive.append((self.rel(),m.drive.steering_angle,m.drive.speed,m.drive.acceleration))
    def obstacle_cb(self,m):
        p=m.pose.pose.position
        self.obstacle.append((self.rel(),p.x,p.y))
    def pred_cb(self,m):
        self.pred.append((self.rel(),m.tracking_cost,m.friction_ellipse_cost,
                          m.front_slip_cost,m.rear_slip_cost,m.steer_cost,
                          m.rate_cost,m.boundary_cost,m.obs_cost,m.progress_cost))
    def collision_cb(self,m):
        if m.data and self.t0 is not None: self.finish("collision")
    def check(self):
        if not self.odom:return
        if abs(self.accumulated_progress) >= self.target_laps*self.track_length:
            self.finish("laps_complete")
        elif self.rel() > self.timeout_seconds:
            self.finish("timeout")
    def finish(self,status):
        if not rclpy.ok():return
        self.status=status; self.save(); self.get_logger().info(f"END {status}, t={self.rel():.2f}s")
        rclpy.shutdown()
    def save(self):
        od=np.asarray(self.odom); dr=np.asarray(self.drive)
        obstacle=np.asarray(self.obstacle)
        np.savez_compressed(self.out/"map1_lap_data.npz",odom=od,drive=dr,obstacle=obstacle,
            mppi_cost=np.asarray(self.pred,dtype=np.float32),
            status=self.status)
        minimum_obstacle_distance=float("nan")
        if len(od) and len(obstacle):
            center=np.median(obstacle[:,1:3],axis=0)
            minimum_obstacle_distance=float(np.min(np.hypot(od[:,1]-center[0],od[:,2]-center[1])))
        with open(self.out/"summary.txt","w") as f:
            lap_ratio=abs(self.accumulated_progress)/max(self.track_length,1e-9)
            f.write(f"status={self.status}\nduration_s={self.rel():.6f}\nodom_samples={len(od)}\ndrive_samples={len(dr)}\nprediction_samples={len(self.pred)}\naccumulated_progress_m={abs(self.accumulated_progress):.6f}\ntrack_length_m={self.track_length:.6f}\nlap_ratio={lap_ratio:.6f}\nminimum_obstacle_center_distance_m={minimum_obstacle_distance:.6f}\n")
        fig,ax=plt.subplots(1,2,figsize=(15,6))
        if len(od): ax[0].plot(od[:,1],od[:,2],"k",lw=2,label="Simulator actual")
        if len(obstacle):
            center=np.median(obstacle[:,1:3],axis=0)
            ax[0].scatter(center[0],center[1],s=100,color="red",marker="x",label="Static obstacle")
            ax[0].add_patch(plt.Circle(center,0.65,color="red",alpha=.18,label="0.65 m exclusion radius"))
        stride=max(1,len(self.pred)//25)
        for j,p in enumerate(self.pred[::stride]):
            ax[0].plot(p[1],p[2],color="tab:blue",alpha=.18,lw=.8,label="MPPI prediction" if j==0 else None)
        ax[0].axis("equal");ax[0].grid();ax[0].legend();ax[0].set_title(f"Map1: {self.status}");ax[0].set_xlabel("x [m]");ax[0].set_ylabel("y [m]")
        if len(dr):
            ax[1].plot(dr[:,0],dr[:,1],label="steer [rad]")
            ax[1].plot(dr[:,0],dr[:,2],label="speed cmd [m/s]")
            ax[1].plot(dr[:,0],dr[:,3],label="accel [m/s²]")
        ax[1].grid();ax[1].legend();ax[1].set_xlabel("time [s]");ax[1].set_title("MPPI commands")
        fig.tight_layout();fig.savefig(self.out/"map1_mppi_prediction_vs_simulator.png",dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--laps",type=float,default=TARGET_LAPS)
    parser.add_argument("--timeout",type=float,default=TIMEOUT_SECONDS)
    parser.add_argument("--output",type=Path,default=OUTPUT_DIRECTORY)
    parser.add_argument("--track",type=Path,default=DEFAULT_TRACK,
                        help="closed CSV containing x_m,y_m")
    args=parser.parse_args()
    rclpy.init(); n=Recorder(args.laps,args.timeout,args.output,args.track)
    try:rclpy.spin(n)
    except KeyboardInterrupt:
        if rclpy.ok(): n.finish("interrupted")

if __name__=="__main__":main()
