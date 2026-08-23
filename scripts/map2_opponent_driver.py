#!/usr/bin/env python3
import math
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    source_path = ROOT / path
    if source_path.exists():
        return source_path
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('smppi_cuda_controller')) / path
    except Exception:
        return source_path


def yaw(q): return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi


class OpponentDriver(Node):
    def __init__(self):
        super().__init__('map2_opponent_driver')
        self.declare_parameter('track_csv','data/map2/map2_mppi_track_optimal.csv')
        self.declare_parameter('target_speed',1.8); self.declare_parameter('lookahead_points',12)
        track_path=resolve_path(self.get_parameter('track_csv').value)
        raw=np.genfromtxt(track_path,delimiter=',',names=True)
        self.track=np.column_stack((raw['x_m'],raw['y_m']))
        self.pub=self.create_publisher(AckermannDriveStamped,'/opp_drive',10)
        self.create_subscription(Odometry,'/opp_racecar/odom',self.callback,10)
    def callback(self,msg):
        x,y=msg.pose.pose.position.x,msg.pose.pose.position.y; psi=yaw(msg.pose.pose.orientation)
        v=math.hypot(msg.twist.twist.linear.x,msg.twist.twist.linear.y)
        nearest=int(np.argmin((self.track[:,0]-x)**2+(self.track[:,1]-y)**2))
        target=self.track[(nearest+int(self.get_parameter('lookahead_points').value))%len(self.track)]
        alpha=wrap(math.atan2(target[1]-y,target[0]-x)-psi); look=max(0.2,math.hypot(target[0]-x,target[1]-y))
        steer=math.atan2(0.324*2*math.sin(alpha),look)
        out=AckermannDriveStamped(); out.header.stamp=self.get_clock().now().to_msg()
        out.drive.steering_angle=float(np.clip(steer,-0.45,0.45)); out.drive.speed=float(self.get_parameter('target_speed').value)
        out.drive.acceleration=float(np.clip(3.0*(out.drive.speed-v),-3.0,3.0)); self.pub.publish(out)

def main():
    rclpy.init(); node=OpponentDriver()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
if __name__=='__main__':main()
