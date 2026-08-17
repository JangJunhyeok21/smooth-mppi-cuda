#!/usr/bin/env python3
"""Move the opponent onto the ego path during a high-yaw-rate turn."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

ROOT = Path("/home/a/smooth-mppi-cuda")
OUTPUT = Path(os.environ.get(
    "SUDDEN_OBSTACLE_EVENT_PATH",
    ROOT / "model_tuning/results/sudden_obstacle/event.json"))
AHEAD_DISTANCE = float(os.environ.get("SUDDEN_OBSTACLE_AHEAD_M", "2.0"))
MIN_SPEED = float(os.environ.get("SUDDEN_OBSTACLE_MIN_SPEED", "2.5"))
MIN_YAW_RATE = float(os.environ.get("SUDDEN_OBSTACLE_MIN_YAW_RATE", "0.35"))
TRIGGER_RADIUS = float(os.environ.get("SUDDEN_OBSTACLE_TRIGGER_RADIUS", "0.60"))
RANDOM_SEED = int(os.environ.get("SUDDEN_OBSTACLE_SEED", "20260817"))
MIN_TRACK_HALF_WIDTH = float(os.environ.get(
    "SUDDEN_OBSTACLE_MIN_TRACK_HALF_WIDTH", "1.00"))


class Injector(Node):
    def __init__(self):
        super().__init__("sudden_obstacle_injector")
        reference = np.genfromtxt(
            ROOT / "data/map1/map1_centerline.csv", delimiter=",", names=True)
        self.centerline = np.column_stack((reference["x_m"], reference["y_m"]))
        self.track_half_width = np.minimum(
            reference["w_tr_left_m"], reference["w_tr_right_m"])
        self.segment_length = np.linalg.norm(
            np.roll(self.centerline, -1, axis=0) - self.centerline, axis=1)
        self.trigger_index, self.target_index = self.select_random_scenario()
        self.publisher = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.subscription = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 20)
        self.injected = False
        self.first_message_time = None

    def walk_backward(self, target_index):
        index = target_index
        distance = 0.0
        while distance < AHEAD_DISTANCE:
            previous = (index - 1) % len(self.centerline)
            distance += float(self.segment_length[previous])
            index = previous
        return index

    def select_random_scenario(self):
        """Choose a wide corner whose obstacle is AHEAD_DISTANCE from reveal."""
        previous = np.roll(self.centerline, 2, axis=0)
        following = np.roll(self.centerline, -2, axis=0)
        heading_before = np.arctan2(
            self.centerline[:, 1]-previous[:, 1],
            self.centerline[:, 0]-previous[:, 0])
        heading_after = np.arctan2(
            following[:, 1]-self.centerline[:, 1],
            following[:, 0]-self.centerline[:, 0])
        heading_change = np.abs(np.arctan2(
            np.sin(heading_after-heading_before),
            np.cos(heading_after-heading_before)))
        candidates = []
        for target in range(0, len(self.centerline), 3):
            trigger = self.walk_backward(target)
            # Both the reveal point and obstacle station must be in a corner.
            # Wide stations only: do not turn an impossible bottleneck into a
            # controller failure by reducing physical collision clearance.
            if (heading_change[trigger] < 0.035 or
                    self.track_half_width[target] < MIN_TRACK_HALF_WIDTH):
                continue
            candidates.append((trigger, target))
        if not candidates:
            raise RuntimeError("no feasible random sudden-obstacle corner")
        rng = np.random.default_rng(RANDOM_SEED)
        return candidates[int(rng.integers(len(candidates)))]

    def target_ahead(self, nearest):
        target = nearest
        distance = 0.0
        while distance < AHEAD_DISTANCE:
            following = (target + 1) % len(self.centerline)
            distance += float(np.linalg.norm(
                self.centerline[following] - self.centerline[target]))
            target = following
        following = (target + 1) % len(self.centerline)
        direction = self.centerline[following] - self.centerline[target]
        yaw = math.atan2(direction[1], direction[0])
        return target, yaw, distance

    def odom_callback(self, message):
        if self.injected:
            return
        now = time.monotonic()
        if self.first_message_time is None:
            self.first_message_time = now
        # Avoid injecting during the launch transient even if initial state
        # estimation briefly reports a high angular rate.
        if now - self.first_message_time < 1.0:
            return
        speed = float(message.twist.twist.linear.x)
        yaw_rate = float(message.twist.twist.angular.z)
        if speed < MIN_SPEED or abs(yaw_rate) < MIN_YAW_RATE:
            return
        position = np.array([
            message.pose.pose.position.x, message.pose.pose.position.y])
        nearest = int(np.argmin(np.sum((self.centerline-position)**2, axis=1)))
        trigger_error = float(np.linalg.norm(
            position-self.centerline[self.trigger_index]))
        # The driven racing line can be several decimetres away from the CSV
        # centerline in a corner. Trigger by station neighbourhood, not by an
        # unrealistically tight centerline point hit.
        if trigger_error > TRIGGER_RADIUS:
            return
        target_index = self.target_index
        following = (target_index + 1) % len(self.centerline)
        direction = self.centerline[following] - self.centerline[target_index]
        target_yaw = math.atan2(direction[1], direction[0])
        # Report the actual centerline distance of this randomized pair.
        cursor, actual_ahead = self.trigger_index, 0.0
        while cursor != target_index:
            actual_ahead += float(self.segment_length[cursor])
            cursor = (cursor + 1) % len(self.centerline)
        target = self.centerline[target_index]
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(target[0])
        goal.pose.position.y = float(target[1])
        goal.pose.orientation.z = math.sin(0.5 * target_yaw)
        goal.pose.orientation.w = math.cos(0.5 * target_yaw)
        self.publisher.publish(goal)
        event = {
            "ego_x_m": float(position[0]), "ego_y_m": float(position[1]),
            "ego_speed_mps": speed, "ego_yaw_rate_radps": yaw_rate,
            "target_x_m": float(target[0]), "target_y_m": float(target[1]),
            "target_yaw_rad": target_yaw,
            "random_seed": RANDOM_SEED,
            "trigger_index": self.trigger_index,
            "target_index": self.target_index,
            "trigger_position_error_m": trigger_error,
            "requested_ahead_distance_m": AHEAD_DISTANCE,
            "centerline_ahead_distance_m": actual_ahead,
            "straight_line_detection_distance_m": float(np.linalg.norm(target-position)),
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(event, indent=2))
        self.injected = True
        self.get_logger().warn(f"Injected sudden obstacle: {event}")


def main():
    rclpy.init()
    node = Injector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
