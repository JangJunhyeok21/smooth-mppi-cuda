#!/home/a/anaconda3/envs/RL/bin/python
"""Record one simulator MPPI episode and atomically reject collisions."""

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class EpisodeCollector(Node):
    def __init__(self):
        super().__init__('mppi_mdn_episode_collector')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('collision_topic', '/collision0')
        self.declare_parameter('output_path', '/tmp/mppi_mdn_episode.npz')
        self.declare_parameter('duration_s', 25.0)
        self.declare_parameter('minimum_samples', 400)
        self.declare_parameter('episode_metadata_json', '{}')
        self.rows = []
        self.command = (0.0, 0.0)
        self.start_stamp = None
        self.collided = False
        self.invalid_reason = ''
        self.finished = False
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value,
            self.odom_callback, 20)
        self.create_subscription(
            AckermannDriveStamped, self.get_parameter('drive_topic').value,
            self.drive_callback, 20)
        self.create_subscription(
            Bool, self.get_parameter('collision_topic').value,
            self.collision_callback, 20)
        self.create_timer(0.1, self.check_finished)

    def drive_callback(self, msg):
        self.command = (float(msg.drive.speed),
                        float(msg.drive.steering_angle))

    def collision_callback(self, msg):
        if msg.data:
            self.collided = True
            self.invalid_reason = 'simulator collision topic became true'

    def odom_callback(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        q = msg.pose.pose.orientation
        row = (stamp, float(msg.pose.pose.position.x),
               float(msg.pose.pose.position.y), yaw_from_quaternion(q),
               float(msg.twist.twist.linear.x),
               float(msg.twist.twist.linear.y),
               float(msg.twist.twist.angular.z), *self.command)
        if not np.isfinite(row).all():
            self.invalid_reason = 'non-finite odometry or command'
            self.collided = True
            return
        if self.rows:
            dt = stamp - self.rows[-1][0]
            jump = math.hypot(row[1] - self.rows[-1][1],
                              row[2] - self.rows[-1][2])
            if dt > 0.0 and jump > max(0.25, 10.0 * dt):
                elapsed = stamp - self.start_stamp
                if elapsed < 2.0:
                    # The gym publishes its default reset pose briefly before
                    # applying the randomized centerline spawn.  This is not a
                    # driving collision: discard the pre-spawn prefix.
                    self.rows = [row]
                    self.start_stamp = stamp
                    self.command = (0.0, 0.0)
                    self.get_logger().info(
                        f'resynchronized after initial spawn jump: {jump:.3f} m')
                    return
                self.invalid_reason = f'pose discontinuity {jump:.3f} m/{dt:.3f} s'
                self.collided = True
        self.rows.append(row)
        if self.start_stamp is None:
            self.start_stamp = stamp

    def check_finished(self):
        if self.finished or self.start_stamp is None:
            return
        if self.rows[-1][0] - self.start_stamp < float(
                self.get_parameter('duration_s').value) and not self.collided:
            return
        self.finished = True
        output = Path(self.get_parameter('output_path').value)
        minimum = int(self.get_parameter('minimum_samples').value)
        if self.collided or len(self.rows) < minimum:
            if output.exists():
                output.unlink()
            reason = self.invalid_reason or f'only {len(self.rows)} samples'
            self.get_logger().error(
                f'REJECTED episode: {reason}; samples={len(self.rows)}')
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            metadata = json.loads(
                self.get_parameter('episode_metadata_json').value)
            metadata.update({'collision': False, 'samples': len(self.rows),
                             'duration_s': self.rows[-1][0] - self.rows[0][0]})
            temporary = output.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(
                    stream, trajectory=np.asarray(self.rows, np.float64),
                    trajectory_columns=np.asarray(
                        ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate',
                         'speed_cmd', 'steer_cmd']),
                    collision=np.asarray(False),
                    metadata_json=np.asarray(json.dumps(metadata)))
            temporary.replace(output)
            self.get_logger().info(
                f'ACCEPTED collision-free episode: {output}; '
                f'samples={len(self.rows)}')
        # main() observes ``finished`` and exits the process.  Calling
        # rclpy.shutdown() from this timer callback can deadlock the executor.


def main():
    rclpy.init()
    node = EpisodeCollector()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
