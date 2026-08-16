#!/usr/bin/env python3
"""Record one Map1 MPPI lap (or collision/timeout) and generate plots."""
import csv
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


class Recorder(Node):
    def __init__(self):
        super().__init__("map1_lap_recorder")
        self.out = Path("/home/a/smooth-mppi-cuda/model_tuning/map1_closed_loop_no_imu")
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = None; self.start = None; self.left_start = False
        self.latest_pose = None
        self.previous_recorded_pose = None
        centerline_path = Path(__file__).resolve().parents[1] / "data/map1/map1_centerline.csv"
        centerline = np.genfromtxt(centerline_path, delimiter=",", names=True)
        self.centerline_xy = np.column_stack((centerline["x_m"], centerline["y_m"]))
        closed = np.vstack((self.centerline_xy, self.centerline_xy[0]))
        segment_length = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
        self.centerline_s = np.r_[0.0, np.cumsum(segment_length[:-1])]
        self.track_length = float(segment_length.sum())
        self.last_progress = None
        self.accumulated_progress = 0.0
        self.odom=[]; self.drive=[]; self.pred=[]; self.status="timeout"
        self.create_subscription(Odometry,"/ego_racecar/odom",self.odom_cb,50)
        self.create_subscription(AckermannDriveStamped,"/drive",self.drive_cb,50)
        self.create_subscription(MppiTrajectory,"/mppi_optimal_trajectory",self.pred_cb,20)
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
            if pose_jump > 0.5:
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
    def pred_cb(self,m):
        self.pred.append((self.rel(),np.asarray(m.predicted_x),np.asarray(m.predicted_y),
                          np.asarray(m.steer),np.asarray(m.accel)))
    def collision_cb(self,m):
        if m.data and self.t0 is not None: self.finish("collision")
    def check(self):
        if not self.odom:return
        if abs(self.accumulated_progress) >= .98*self.track_length:
            self.finish("lap_complete")
        elif self.rel()>90: self.finish("timeout")
    def finish(self,status):
        if not rclpy.ok():return
        self.status=status; self.save(); self.get_logger().info(f"END {status}, t={self.rel():.2f}s")
        rclpy.shutdown()
    def save(self):
        od=np.asarray(self.odom); dr=np.asarray(self.drive)
        np.savez_compressed(self.out/"map1_lap_data.npz",odom=od,drive=dr,
            prediction_t=np.asarray([p[0] for p in self.pred]),
            prediction_x=np.asarray([p[1] for p in self.pred],dtype=object),
            prediction_y=np.asarray([p[2] for p in self.pred],dtype=object),status=self.status)
        with open(self.out/"summary.txt","w") as f:
            f.write(f"status={self.status}\nduration_s={self.rel():.6f}\nodom_samples={len(od)}\ndrive_samples={len(dr)}\nprediction_samples={len(self.pred)}\n")
        fig,ax=plt.subplots(1,2,figsize=(15,6))
        if len(od): ax[0].plot(od[:,1],od[:,2],"k",lw=2,label="Simulator actual")
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
    rclpy.init(); n=Recorder()
    try:rclpy.spin(n)
    except KeyboardInterrupt:
        if rclpy.ok(): n.finish("interrupted")

if __name__=="__main__":main()
