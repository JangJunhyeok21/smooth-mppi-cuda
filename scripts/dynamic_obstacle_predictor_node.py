#!/home/a/anaconda3/envs/RL/bin/python
"""Separate ROS2 MDN opponent predictor for CUDA MPPI."""
from collections import defaultdict, deque
from pathlib import Path
import math

import numpy as np
import rclpy
from rclpy.node import Node
import torch
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from smppi_cuda_controller.msg import DynamicObstacleTrajectory

try:
    from f1_msgs.msg import F1stateArr
except ImportError:
    F1stateArr = None

HISTORY_STEPS, MAP_POINTS, HORIZON, DT = 6, 10, 60, 0.04


def wrap(value): return (value + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DynamicObstaclePredictor(Node):
    def __init__(self):
        super().__init__("dynamic_obstacle_predictor")
        # Sixty tiny recursive network calls are faster and more deterministic
        # with one intra-op worker than with a thread-pool launch per knot.
        torch.set_num_threads(1)
        self.declare_parameter("input_mode", "simulation")
        self.declare_parameter("simulation_odom_topic", "/opp_racecar/odom")
        self.declare_parameter("perception_topic", "/f1/perception/object/obstacles/arr")
        self.declare_parameter("output_topic", "/mppi/dynamic_obstacle_trajectory")
        self.declare_parameter("model_path", "/home/a/smooth-mppi-cuda/model_tuning/results/dynamic_obstacle_frenet_mdn/frenet_mdn.ts")
        self.declare_parameter("track_csv", "/home/a/smooth-mppi-cuda/data/map2/map2_mppi_track_optimal.csv")
        self.declare_parameter("opponent_radius", 0.24)
        self.declare_parameter("uncertainty_sigma_scale", 1.0)
        self.declare_parameter("maximum_radius", 0.75)
        self.declare_parameter("longitudinal_ellipse_gain", 3.1)
        self.declare_parameter("lateral_ellipse_gain", 2.1)
        self.declare_parameter("publish_rate_hz", 25.0)
        raw = np.genfromtxt(self.get_parameter("track_csv").value, delimiter=",", names=True)
        self.track = np.column_stack((raw["x_m"], raw["y_m"])).astype(np.float64)
        self.track_yaw = np.unwrap(np.asarray(raw["psi_rad"], float))
        self.track_kappa = np.asarray(raw["kappa_radpm"], float)
        self.track_left = np.asarray(raw["w_tr_left_m"], float)
        self.track_right = np.asarray(raw["w_tr_right_m"], float)
        segment = np.hypot(np.roll(self.track[:, 0], -1)-self.track[:, 0],
                           np.roll(self.track[:, 1], -1)-self.track[:, 1])
        self.track_s = np.r_[0.0, np.cumsum(segment[:-1])]
        self.track_length = float(segment.sum())
        model_path = Path(self.get_parameter("model_path").value)
        self.model = torch.jit.load(str(model_path), map_location="cpu") if model_path.exists() else None
        if self.model is None:
            self.get_logger().warning(f"MDN not found at {model_path}; constant-velocity fallback active")
        else:
            self.model.eval(); self.get_logger().info(f"Loaded MDN predictor: {model_path}")
        self.history = defaultdict(lambda: deque(maxlen=100))
        self.publisher = self.create_publisher(DynamicObstacleTrajectory,
            self.get_parameter("output_topic").value, 10)
        self.marker_publisher = self.create_publisher(MarkerArray,
            "/mppi/dynamic_obstacle_prediction_markers", 10)
        mode = self.get_parameter("input_mode").value
        if mode in ("simulation", "both"):
            self.create_subscription(Odometry, self.get_parameter("simulation_odom_topic").value,
                                     self.odom_callback, 20)
        if mode in ("perception", "both"):
            if F1stateArr is None: raise RuntimeError("input_mode requires f1_msgs/F1stateArr")
            self.create_subscription(F1stateArr, self.get_parameter("perception_topic").value,
                                     self.perception_callback, 20)
        self.create_timer(1.0 / float(self.get_parameter("publish_rate_hz").value), self.predict)
        self.get_logger().info(f"input_mode={mode}, output={self.get_parameter('output_topic').value}, dt={DT}, horizon={HORIZON}")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.history[1].append((stamp, msg.pose.pose.position.x, msg.pose.pose.position.y,
                                yaw_from_quaternion(q), speed))

    def perception_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for state in msg.f1_state_arr:
            if np.isfinite([state.x, state.y, state.yaw, state.v]).all():
                self.history[int(state.id)].append((stamp, state.x, state.y, state.yaw, state.v))

    def resampled_history(self, raw):
        data = np.asarray(raw, float)
        now = data[-1, 0]; query = now + np.arange(-HISTORY_STEPS + 1, 1) * DT
        if len(data) < 2 or query[0] < data[0, 0]: return None
        yaw = np.unwrap(data[:, 3])
        return np.column_stack((query, *[np.interp(query, data[:, 0], column)
            for column in (data[:, 1], data[:, 2], yaw, data[:, 4])]))

    def track_index(self, s):
        return int(np.searchsorted(self.track_s, s % self.track_length,
                                   side="right")-1) % len(self.track)

    def project(self, row):
        index = int(np.argmin(np.square(self.track[:, 0]-row[1]) +
                              np.square(self.track[:, 1]-row[2])))
        nx, ny = -math.sin(self.track_yaw[index]), math.cos(self.track_yaw[index])
        d = (row[1]-self.track[index, 0])*nx + (row[2]-self.track[index, 1])*ny
        return np.asarray((row[0], self.track_s[index], d,
            wrap(row[3]-self.track_yaw[index]), row[4], self.track_kappa[index],
            self.track_left[index], self.track_right[index]), float)

    def features(self, history):
        result = []
        for i, row in enumerate(history):
            if i == 0: ds = dt = 0.0
            else:
                ds = (row[1]-history[i-1][1]+.5*self.track_length)%self.track_length-.5*self.track_length
                dt = row[0]-history[i-1][0]
            result.extend((ds,row[2],row[5],row[6],row[7],dt))
        for offset in range(MAP_POINTS):
            index = self.track_index(history[-1][1]+0.5*offset)
            result.extend((self.track_kappa[index],self.track_left[index],self.track_right[index]))
        return np.asarray(result,np.float32)

    def one_prediction(self, raw):
        current = np.asarray(raw[-1], float); history = self.resampled_history(raw)
        if self.model is None or history is None:
            times = np.arange(1, HORIZON+1)*DT
            x = current[1] + current[4]*np.cos(current[3])*times
            y = current[2] + current[4]*np.sin(current[3])*times
            radius=np.full(HORIZON,self.get_parameter("opponent_radius").value)
            return x,y,np.full(HORIZON,current[3]),radius,radius
        states=[self.project(row) for row in history]
        xs=[];ys=[];yaws=[];major=[];minor=[];var_s=var_d=0.0
        for step in range(HORIZON):
            with torch.no_grad():
                logits,means,scales=self.model(torch.from_numpy(self.features(np.asarray(states[-HISTORY_STEPS:])))[None])
            probability=torch.softmax(logits[0],0).numpy();mu=means[0].numpy();sigma=scales[0].numpy()
            mean=np.sum(probability[:,None]*mu,axis=0)
            var_s += float(np.sum(probability*(sigma[:,0]**2+(mu[:,0]-mean[0])**2)))
            var_d += float(np.sum(probability*(sigma[:,1]**2+(mu[:,1]-mean[1])**2)))
            selected=mu[int(np.argmax(probability))];previous=states[-1]
            s=(previous[1]+selected[0])%self.track_length;d=previous[2]+selected[1]
            epsi=wrap(previous[3]+selected[2]);index=self.track_index(s)
            state=np.asarray((previous[0]+DT,s,d,epsi,max(0.0,selected[3]),self.track_kappa[index],self.track_left[index],self.track_right[index]))
            states.append(state);psi=self.track_yaw[index]
            xs.append(self.track[index,0]-d*math.sin(psi));ys.append(self.track[index,1]+d*math.cos(psi));yaws.append(psi+epsi)
            physical=self.get_parameter("opponent_radius").value
            major.append(min(self.get_parameter("maximum_radius").value,physical+self.get_parameter("longitudinal_ellipse_gain").value*math.sqrt(max(0,var_s))))
            minor.append(min(self.get_parameter("maximum_radius").value,physical+self.get_parameter("lateral_ellipse_gain").value*math.sqrt(max(0,var_d))))
        return np.asarray(xs),np.asarray(ys),np.asarray(yaws),np.asarray(major),np.asarray(minor)

    def predict(self):
        active = [(identifier, raw) for identifier, raw in self.history.items() if raw and self.get_clock().now().nanoseconds*1e-9-raw[-1][0] < 0.5]
        if not active: return
        msg = DynamicObstacleTrajectory(); msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = "map"
        msg.dt = DT; msg.horizon = HORIZON; markers = MarkerArray()
        for marker_id, (identifier, raw) in enumerate(active[:5]):
            x,y,yaw,major,minor=self.one_prediction(raw);radius=np.maximum(major,minor);msg.obstacle_ids.append(identifier)
            msg.x.extend(x.astype(np.float32).tolist()); msg.y.extend(y.astype(np.float32).tolist())
            msg.yaw.extend(yaw.astype(np.float32).tolist()); msg.radius.extend(np.asarray(radius, np.float32).tolist())
            for step,(px,py,angle,a,b) in enumerate(zip(x,y,yaw,major,minor)):
                marker=Marker();marker.header=msg.header;marker.ns=f"mdn_prediction_{identifier}";marker.id=marker_id*HORIZON+step
                marker.type=Marker.CYLINDER;marker.action=Marker.ADD;marker.pose.position.x=float(px);marker.pose.position.y=float(py);marker.pose.position.z=.05
                marker.pose.orientation.z=math.sin(float(angle)/2);marker.pose.orientation.w=math.cos(float(angle)/2)
                marker.scale.x=2*float(a);marker.scale.y=2*float(b);marker.scale.z=.05
                marker.color.r=1.0;marker.color.g=.15;marker.color.b=.05;marker.color.a=.12+.45*step/HORIZON;markers.markers.append(marker)
        self.publisher.publish(msg); self.marker_publisher.publish(markers)


def main():
    rclpy.init(); node = DynamicObstaclePredictor()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__": main()
