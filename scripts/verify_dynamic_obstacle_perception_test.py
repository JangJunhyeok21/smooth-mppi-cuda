#!/usr/bin/python3
"""Measure the simulator perception/predictor/MPPI integration test."""

import argparse
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

from smppi_cuda_controller.msg import DynamicObstacleTrajectory


class Monitor(Node):
    def __init__(self, static_x, static_y, static_radius):
        super().__init__("dynamic_obstacle_perception_test_monitor")
        self.static_x = static_x
        self.static_y = static_y
        self.static_radius = static_radius
        self.positions = []
        self.collision_messages = 0
        self.collision_true = 0
        self.prediction_messages = 0
        self.saw_dynamic = False
        self.saw_static = False
        self.contract_errors = 0
        self.min_static_clearance = math.inf
        self.create_subscription(Odometry, "/ego_racecar/odom", self.odom, 20)
        self.create_subscription(Bool, "/collision0", self.collision, 20)
        self.create_subscription(
            DynamicObstacleTrajectory,
            "/mppi/dynamic_obstacle_trajectory",
            self.prediction,
            20,
        )

    def odom(self, message):
        x = message.pose.pose.position.x
        y = message.pose.pose.position.y
        self.positions.append((x, y))
        clearance = math.hypot(x - self.static_x, y - self.static_y) - self.static_radius
        self.min_static_clearance = min(self.min_static_clearance, clearance)

    def collision(self, message):
        self.collision_messages += 1
        self.collision_true += int(message.data)

    def prediction(self, message):
        self.prediction_messages += 1
        self.saw_dynamic |= any(message.is_dynamic)
        self.saw_static |= any(not value for value in message.is_dynamic)
        expected = sum(message.horizon if value else 1 for value in message.is_dynamic)
        lengths = (len(message.x), len(message.y), len(message.yaw),
                   len(message.semi_major), len(message.semi_minor))
        if any(length != expected for length in lengths):
            self.contract_errors += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--static-x", type=float, default=1.2929498265693544)
    parser.add_argument("--static-y", type=float, default=-2.148326999195452)
    parser.add_argument("--static-radius", type=float, default=0.24)
    args = parser.parse_args()
    rclpy.init()
    node = Monitor(args.static_x, args.static_y, args.static_radius)
    deadline = time.monotonic() + args.duration
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    displacement = 0.0
    path_length = 0.0
    if len(node.positions) >= 2:
        displacement = math.dist(node.positions[0], node.positions[-1])
        path_length = sum(math.dist(a, b) for a, b in zip(
            node.positions[:-1], node.positions[1:]))
    print(f"duration_s: {args.duration:.1f}")
    print(f"ego_odom_samples: {len(node.positions)}")
    print(f"ego_displacement_m: {displacement:.3f}")
    print(f"ego_path_length_m: {path_length:.3f}")
    print(f"collision_true/messages: {node.collision_true}/{node.collision_messages}")
    print(f"prediction_messages: {node.prediction_messages}")
    print(f"saw_dynamic: {node.saw_dynamic}")
    print(f"saw_static: {node.saw_static}")
    print(f"packed_contract_errors: {node.contract_errors}")
    print(f"min_static_clearance_m: {node.min_static_clearance:.3f}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
