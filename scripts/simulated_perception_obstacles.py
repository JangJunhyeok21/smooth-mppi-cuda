#!/usr/bin/python3
"""Publish simulator opponent plus one synthetic static object as F1stateArr."""

import math

import rclpy
from f1_msgs.msg import F1state, F1stateArr
from nav_msgs.msg import Odometry
from rclpy.node import Node


def quaternion_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class SimulatedPerceptionObstacles(Node):
    def __init__(self):
        super().__init__("simulated_perception_obstacles")
        self.declare_parameter("opponent_odom_topic", "/opp_racecar/odom")
        self.declare_parameter(
            "output_topic", "/f1/perception/object/obstacles/arr")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("static_x", 1.2929498265693544)
        self.declare_parameter("static_y", -2.148326999195452)
        self.declare_parameter("static_yaw", 2.140535854443902)
        self.declare_parameter("static_id", 1001.0)

        output_topic = self.get_parameter("output_topic").value
        odom_topic = self.get_parameter("opponent_odom_topic").value
        self.publisher = self.create_publisher(F1stateArr, output_topic, 10)
        self.opponent = None
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        self.create_timer(1.0 / rate, self.publish)
        self.get_logger().info(
            f"Simulator perception bridge: {odom_topic} + synthetic static "
            f"object -> {output_topic} at {rate:.1f} Hz")

    def odom_callback(self, msg):
        self.opponent = msg

    def publish(self):
        message = F1stateArr()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"

        if self.opponent is not None:
            odom = self.opponent
            moving = F1state()
            moving.header = message.header
            moving.id = 1.0
            moving.x = odom.pose.pose.position.x
            moving.y = odom.pose.pose.position.y
            moving.yaw = quaternion_yaw(odom.pose.pose.orientation)
            moving.yaw_rate = odom.twist.twist.angular.z
            moving.v_x = odom.twist.twist.linear.x
            moving.v_y = odom.twist.twist.linear.y
            moving.v = math.hypot(moving.v_x, moving.v_y)
            message.f1_state_arr.append(moving)

        static = F1state()
        static.header = message.header
        static.id = float(self.get_parameter("static_id").value)
        static.x = float(self.get_parameter("static_x").value)
        static.y = float(self.get_parameter("static_y").value)
        static.yaw = float(self.get_parameter("static_yaw").value)
        static.v = 0.0
        message.f1_state_arr.append(static)
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatedPerceptionObstacles()
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
