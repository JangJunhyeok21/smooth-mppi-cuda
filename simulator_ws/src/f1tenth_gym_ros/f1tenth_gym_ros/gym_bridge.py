# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool

from sensor_msgs.msg import LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

import gymnasium as gym
import math
import numpy as np
import inspect

import pathlib
import sys
# Prefer the f1tenth_gym submodule beside this bridge. The launch file also
# prepends it to PYTHONPATH; this fallback keeps direct `ros2 run` consistent.
workspace_gym = pathlib.Path(__file__).resolve().parents[3] / 'src' / 'f1tenth_gym'
if (workspace_gym / 'f1tenth_gym').is_dir():
    sys.path.insert(0, str(workspace_gym))
from f1tenth_gym.envs.f110_env import F110Env, Track

import time


def quaternion_from_yaw(yaw):
    """Return a quaternion as (w, x, y, z) for a rotation around the Z axis."""
    half_yaw = yaw * 0.5
    return math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)


def yaw_from_quaternion(w, x, y, z):
    """Return the Z-axis rotation represented by a quaternion."""
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')
        self.get_logger().info('f1tenth_gym source: %s' % inspect.getfile(F110Env))

        self.declare_parameter('ego_namespace')
        self.declare_parameter('ego_odom_topic')
        self.declare_parameter('ego_noisy_odom_topic', 'odom_noise')
        self.declare_parameter('ego_noisy_odom_marker_topic', 'odom_noise_marker')
        self.declare_parameter('ego_noisy_odom_marker_lifetime_s', 0.2)
        self.declare_parameter('ego_opp_odom_topic')
        self.declare_parameter('ego_scan_topic')
        self.declare_parameter('ego_drive_topic')
        self.declare_parameter('opp_namespace')
        self.declare_parameter('opp_odom_topic')
        self.declare_parameter('opp_ego_odom_topic')
        self.declare_parameter('opp_scan_topic')
        self.declare_parameter('opp_drive_topic')
        self.declare_parameter('scan_distance_to_base_link')
        self.declare_parameter('scan_fov')
        self.declare_parameter('scan_beams')
        self.declare_parameter('map_path')
        self.declare_parameter('map_img_ext')
        self.declare_parameter('num_agent')
        self.declare_parameter('sx')
        self.declare_parameter('sy')
        self.declare_parameter('stheta')
        self.declare_parameter('ego_pose_noise_std_m', 0.0)
        self.declare_parameter('ego_pose_noise_max_m', 0.0)
        self.declare_parameter('ego_pose_yaw_noise_std_rad', 0.0)
        self.declare_parameter('ego_pose_yaw_noise_max_rad', 0.0)
        self.declare_parameter('ego_pose_noise_seed', 20260825)
        self.declare_parameter('initial_speed', 0.0)
        self.declare_parameter('sx1')
        self.declare_parameter('sy1')
        self.declare_parameter('stheta1')
        self.declare_parameter('static_opponent')
        self.declare_parameter('kb_teleop')
        self.declare_parameter('scale')
        self.declare_parameter('vehicle_params')
        self.declare_parameter('restart_simulation')
        self.declare_parameter('dynamics_model')
        self.declare_parameter('mlp_weights_path')
        self.declare_parameter('kinematic_noslip_noimu_weights_path')
        self.declare_parameter('dynamic_mlp_weights_path')
        self.declare_parameter('dynamic_mlp_vx_delta_weights_path')
        self.declare_parameter('dynamic_mlp_model_dt')
        self.declare_parameter('simulator_gru_model_path')
        self.declare_parameter('simulator_gru_history_steps')
        self.declare_parameter('simulator_gru_model_dt')
        self.declare_parameter('mppi_dt')
        self.declare_parameter('mppi_min_speed')
        self.declare_parameter('mppi_max_speed')
        self.declare_parameter('max_steer')
        self.declare_parameter('kinematic_steer_scale')
        self.declare_parameter('kinematic_steer_bias')
        self.declare_parameter('kinematic_no_slip')
        self.declare_parameter('speed_servo_kp')
        self.declare_parameter('mppi_min_accel')
        self.declare_parameter('mppi_max_accel')
        self.declare_parameter('speed_reference_accel_time_constant')
        self.declare_parameter('speed_reference_brake_time_constant')
        self.declare_parameter('actuator_max_speed_reference_rate')
        self.declare_parameter('steer_servo_time_constant')
        self.declare_parameter('actuator_max_steer_rate')
        self.declare_parameter('kinematic_position_speed_scale')
        for tire_parameter in ('B_f', 'C_f', 'D_f', 'E_f',
                               'B_r', 'C_r', 'D_r', 'E_r'):
            self.declare_parameter('dynamic_mlp_' + tire_parameter)
        self.declare_parameter('dynamic_mlp_I_z')
        self.declare_parameter('mlp_max_residual_ax', 0.0)
        self.declare_parameter('mlp_max_residual_ay', 8.0)
        self.declare_parameter('mlp_max_residual_yaw_accel', 12.0)
        self.declare_parameter('mass')
        self.declare_parameter('l_f')
        self.declare_parameter('l_r')

        self.declare_parameter('drive_with_accel')

        # Flag to know whether to publish the sim time or not
        # Has to be different than use_sim_time so we can still use real time to trigger timer callbacks
        self.declare_parameter('use_sim_time_bridge')

        # check num_agents
        num_agents = self.get_parameter('num_agent').value
        if num_agents < 1 or num_agents > 2:
            raise ValueError('num_agents should be either 1 or 2.')
        elif type(num_agents) != int:
            raise ValueError('num_agents should be an int.')

        self.vehicle_params = None
        if self.get_parameter('vehicle_params').value == 'f1tenth':
            self.vehicle_params = F110Env.f1tenth_vehicle_params()
        elif self.get_parameter('vehicle_params').value == 'fullscale':
            self.vehicle_params = F110Env.fullscale_vehicle_params()
        elif self.get_parameter('vehicle_params').value == 'f1fifth':
            self.vehicle_params = F110Env.f1fifth_vehicle_params()
        else:
            raise ValueError('vehicle_params should be either f1tenth, fullscale, or f1fifth.')

        scale = self.get_parameter('scale').value
        dynamics_model = self.get_parameter('dynamics_model').value
        sim_timestep = float(self.get_parameter('mppi_dt').value)
        if dynamics_model in ('kinematic', 'kinematic_mlp',
                              'kinematic_noslip_noimu_direct_speed',
                              'dynamic_mlp_residual_servo_lag',
                              'dynamic_mlp_residual_servo_lag_vx_delta_24d',
                              'dynamic_servo_lag', 'DYNAMIC_SERVO_LAG',
                              'simulator_gru'):
            self.vehicle_params = dict(self.vehicle_params)
            if dynamics_model == 'kinematic_mlp':
                self.vehicle_params['mlp_weights_path'] = self.get_parameter('mlp_weights_path').value
            elif dynamics_model == 'kinematic_noslip_noimu_direct_speed':
                self.vehicle_params['kinematic_noslip_noimu_weights_path'] = self.get_parameter('kinematic_noslip_noimu_weights_path').value
                self.vehicle_params['speed_servo_kp'] = float(self.get_parameter('speed_servo_kp').value)
                self.vehicle_params['min_accel_mppi'] = float(self.get_parameter('mppi_min_accel').value)
                self.vehicle_params['max_accel_mppi'] = float(self.get_parameter('mppi_max_accel').value)
            elif dynamics_model in ('dynamic_mlp_residual_servo_lag',
                                     'dynamic_mlp_residual_servo_lag_vx_delta_24d',
                                     'dynamic_servo_lag', 'DYNAMIC_SERVO_LAG'):
                if dynamics_model not in ('dynamic_servo_lag', 'DYNAMIC_SERVO_LAG'):
                    weight_parameter = ('dynamic_mlp_vx_delta_weights_path'
                        if dynamics_model.endswith('vx_delta_24d')
                        else 'dynamic_mlp_weights_path')
                    self.vehicle_params['dynamic_mlp_weights_path'] = self.get_parameter(weight_parameter).value
                self.vehicle_params['dynamic_mlp_model_dt'] = float(self.get_parameter('dynamic_mlp_model_dt').value)
                self.vehicle_params['speed_servo_kp'] = float(self.get_parameter('speed_servo_kp').value)
                self.vehicle_params['min_accel_mppi'] = float(self.get_parameter('mppi_min_accel').value)
                self.vehicle_params['max_accel_mppi'] = float(self.get_parameter('mppi_max_accel').value)
                self.vehicle_params['speed_reference_accel_time_constant'] = float(self.get_parameter('speed_reference_accel_time_constant').value)
                self.vehicle_params['speed_reference_brake_time_constant'] = float(self.get_parameter('speed_reference_brake_time_constant').value)
                self.vehicle_params['actuator_max_speed_reference_rate'] = float(self.get_parameter('actuator_max_speed_reference_rate').value)
                self.vehicle_params['steer_servo_time_constant'] = float(self.get_parameter('steer_servo_time_constant').value)
                self.vehicle_params['actuator_max_steer_rate'] = float(self.get_parameter('actuator_max_steer_rate').value)
                self.vehicle_params['kinematic_position_speed_scale'] = float(self.get_parameter('kinematic_position_speed_scale').value)
                for tire_parameter in ('B_f', 'C_f', 'D_f', 'E_f',
                                       'B_r', 'C_r', 'D_r', 'E_r'):
                    key = 'dynamic_mlp_' + tire_parameter
                    self.vehicle_params[key] = float(self.get_parameter(key).value)
                self.vehicle_params['dynamic_mlp_I_z'] = float(self.get_parameter('dynamic_mlp_I_z').value)
                for limit_parameter in (
                        'mlp_max_residual_ax', 'mlp_max_residual_ay',
                        'mlp_max_residual_yaw_accel'):
                    self.vehicle_params[limit_parameter] = float(
                        self.get_parameter(limit_parameter).value)
                self.vehicle_params['mass'] = float(self.get_parameter('mass').value)
                self.vehicle_params['lf'] = float(self.get_parameter('l_f').value)
                self.vehicle_params['lr'] = float(self.get_parameter('l_r').value)
            elif dynamics_model == 'simulator_gru':
                self.vehicle_params['simulator_gru_model_path'] = self.get_parameter('simulator_gru_model_path').value
                self.vehicle_params['simulator_gru_history_steps'] = int(self.get_parameter('simulator_gru_history_steps').value)
                self.vehicle_params['simulator_gru_model_dt'] = float(self.get_parameter('simulator_gru_model_dt').value)
                self.vehicle_params['speed_reference_accel_time_constant'] = float(self.get_parameter('speed_reference_accel_time_constant').value)
                self.vehicle_params['speed_reference_brake_time_constant'] = float(self.get_parameter('speed_reference_brake_time_constant').value)
                self.vehicle_params['actuator_max_speed_reference_rate'] = float(self.get_parameter('actuator_max_speed_reference_rate').value)
                self.vehicle_params['steer_servo_time_constant'] = float(self.get_parameter('steer_servo_time_constant').value)
                self.vehicle_params['actuator_max_steer_rate'] = float(self.get_parameter('actuator_max_steer_rate').value)
                self.vehicle_params['kinematic_position_speed_scale'] = float(self.get_parameter('kinematic_position_speed_scale').value)
            self.vehicle_params['v_min_mppi'] = float(self.get_parameter('mppi_min_speed').value)
            self.vehicle_params['v_max_mppi'] = float(self.get_parameter('mppi_max_speed').value)
            self.vehicle_params['s_min'] = -float(self.get_parameter('max_steer').value)
            self.vehicle_params['s_max'] = float(self.get_parameter('max_steer').value)
            self.vehicle_params['kinematic_steer_scale'] = float(self.get_parameter('kinematic_steer_scale').value)
            self.vehicle_params['kinematic_steer_bias'] = float(self.get_parameter('kinematic_steer_bias').value)
            self.vehicle_params['kinematic_no_slip'] = bool(self.get_parameter('kinematic_no_slip').value)

        # Split the path and the name
        path = self.get_parameter('map_path').value
        name = path.split('/')[-1].split('.')[0]
        path = path + '.yaml'
        self.get_logger().info('Loading map: %s from path: %s' % (name, path))
        
        

        # Load the yaml file
        path = pathlib.Path(path)
        loaded_map = Track.from_track_path(path, scale)
        self.loaded_map = loaded_map
        self._load_respawn_centerline(path.parent, name)

        self.drive_with_accel = self.get_parameter('drive_with_accel').value
        self.direct_speed_model = dynamics_model in (
            'kinematic_noslip_noimu_direct_speed',
            'dynamic_mlp_residual_servo_lag',
            'dynamic_mlp_residual_servo_lag_vx_delta_24d',
            'dynamic_servo_lag', 'DYNAMIC_SERVO_LAG',
            'simulator_gru')
        self.get_logger().info(
            'Simulator dynamics: %s, dt=%.3f s, model_dt=%.3f s, '
            'direct speed-command input=%s' % (
                dynamics_model, sim_timestep,
                float(self.vehicle_params.get('dynamic_mlp_model_dt', sim_timestep)),
                self.direct_speed_model))
        if dynamics_model in ('dynamic_servo_lag', 'DYNAMIC_SERVO_LAG'):
            self.get_logger().info(
                'DYNAMIC_SERVO_LAG parameters sourced from MPPI params.yaml: '
                'm=%.4f, lf=%.4f, lr=%.4f, Iz=%.6f, steer_tau=%.6f, '
                'speed_tau=(%.6f, %.6f)' % (
                    self.vehicle_params['mass'], self.vehicle_params['lf'],
                    self.vehicle_params['lr'], self.vehicle_params['dynamic_mlp_I_z'],
                    self.vehicle_params['steer_servo_time_constant'],
                    self.vehicle_params['speed_reference_accel_time_constant'],
                    self.vehicle_params['speed_reference_brake_time_constant']))
        # env backend
        if self.drive_with_accel and not self.direct_speed_model:
            self.env = gym.make(
                                "f1tenth_gym:f1tenth-v0",
                                config={
                                    "map": loaded_map,
                                    "num_agents": num_agents,
                                    "timestep": sim_timestep,
                                    "integrator": "rk4",
                                    "control_input": ["accl", "steering_angle"],
                                    "model": dynamics_model,
                                    "observation_config": {"type": "original"},
                                    "params": self.vehicle_params,
                                    "reset_config": {"type": "map_random_static"},
                                    "scale": scale,
                                    "lidar_dist": self.get_parameter("scan_distance_to_base_link").value
                                },
                                render_mode="rgb_array",
                            )
        else:
            self.env = gym.make(
                                "f1tenth_gym:f1tenth-v0",
                                config={
                                    "map": loaded_map,
                                    "num_agents": num_agents,
                                    "timestep": sim_timestep,
                                    "integrator": "rk4",
                                    "control_input": ["speed", "steering_angle"],
                                    "model": dynamics_model,
                                    "observation_config": {"type": "original"},
                                    "params": self.vehicle_params,
                                    "reset_config": {"type": "map_random_static"},
                                    "scale": scale,
                                    "lidar_dist": self.get_parameter("scan_distance_to_base_link").value
                                },
                                render_mode="rgb_array",
                            )

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_pose_noise_std_m = float(
            self.get_parameter('ego_pose_noise_std_m').value)
        self.ego_pose_noise_max_m = float(
            self.get_parameter('ego_pose_noise_max_m').value)
        self.ego_pose_yaw_noise_std_rad = float(
            self.get_parameter('ego_pose_yaw_noise_std_rad').value)
        self.ego_pose_yaw_noise_max_rad = float(
            self.get_parameter('ego_pose_yaw_noise_max_rad').value)
        if min(self.ego_pose_noise_std_m, self.ego_pose_noise_max_m,
               self.ego_pose_yaw_noise_std_rad,
               self.ego_pose_yaw_noise_max_rad) < 0.0:
            raise ValueError('ego pose noise parameters must be non-negative')
        self.ego_pose_noise_rng = np.random.default_rng(
            int(self.get_parameter('ego_pose_noise_seed').value))
        self.ego_observed_pose = list(self.ego_pose)
        self._sample_ego_observed_pose()
        self.get_logger().info(
            'Ego odometry pose noise: xy std=%.3f m max=%.3f m, '
            'yaw std=%.4f rad max=%.4f rad, seed=%d' % (
                self.ego_pose_noise_std_m,
                self.ego_pose_noise_max_m,
                self.ego_pose_yaw_noise_std_rad,
                self.ego_pose_yaw_noise_max_rad,
                int(self.get_parameter('ego_pose_noise_seed').value)))
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_requested_accel = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        ego_drive_topic = self.get_parameter('ego_drive_topic').value
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        self.ego_namespace = self.get_parameter('ego_namespace').value
        ego_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_odom_topic').value
        ego_noisy_odom_topic = (self.ego_namespace + '/' +
            self.get_parameter('ego_noisy_odom_topic').value)
        ego_noisy_odom_marker_topic = (self.ego_namespace + '/' +
            self.get_parameter('ego_noisy_odom_marker_topic').value)
        self.ego_noisy_odom_marker_lifetime_s = float(
            self.get_parameter('ego_noisy_odom_marker_lifetime_s').value)
        if self.ego_noisy_odom_marker_lifetime_s < 0.0:
            raise ValueError('ego_noisy_odom_marker_lifetime_s must be non-negative')
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value

        if num_agents == 2:
            self.has_opp = True
            self.opp_namespace = self.get_parameter('opp_namespace').value
            sx1 = self.get_parameter('sx1').value
            sy1 = self.get_parameter('sy1').value
            stheta1 = self.get_parameter('stheta1').value

            # Start deterministically at the poses configured in sim.yaml.
            # Random centerline poses are only used by the optional restart logic.
            options = {
                "poses": np.array([[sx, sy, stheta], [sx1, sy1, stheta1]])
            }

            # levinelobby
            # sx = 1.31628
            # sy = 1.02453
            # stheta = -2.04863
            # sx1 = 0.58487
            # sy1 = -0.526965
            # stheta1 = -1.70919
            # porto
            # sx = -1.61700
            # sy = -0.58429
            # stheta = 0.59543
            # sx1 = 0.41475
            # sy1 = 0.56980
            # stheta1 = 0.62913

            # berlin
            # sx = -1.88837
            # sy = -9.12088
            # stheta = -1.47931
            # sx1 = -0.391803
            # sy1 = -12.5262
            # stheta1 = -0.62778

            self.opp_pose = [sx1, sy1, stheta1]
            self.static_opponent = bool(self.get_parameter('static_opponent').value)
            self.static_opponent_pose = np.array([sx1, sy1, stheta1], dtype=float)
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_requested_accel = 0.0
            self.opp_steer = 0.0
            self.opp_collision = False
            self.obs, _ = self.env.reset(options=options)
            self.ego_scan = list(self.obs['scans'][0])
            self.opp_scan = list(self.obs['scans'][1])

            opp_scan_topic = self.get_parameter('opp_scan_topic').value
            opp_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_odom_topic').value
            opp_drive_topic = self.get_parameter('opp_drive_topic').value

            ego_opp_odom_topic = self.ego_namespace + '/' + self.get_parameter('ego_opp_odom_topic').value
            opp_ego_odom_topic = self.opp_namespace + '/' + self.get_parameter('opp_ego_odom_topic').value
        else:
            self.has_opp = False
            self.obs, _ = self.env.reset(options={"poses": np.array([[sx, sy, stheta]])})
            self.ego_scan = list(self.obs['scans'][0])

        initial_speed = float(self.get_parameter('initial_speed').value)
        if initial_speed < 0.0 or initial_speed > self.vehicle_params['v_max_mppi']:
            raise ValueError(
                f'initial_speed={initial_speed} is outside simulator range '
                f'[0, {self.vehicle_params["v_max_mppi"]}]')
        ego_agent = self.env.unwrapped.sim.agents[0]
        ego_agent.state[3] = initial_speed
        ego_agent.mlp_speed_reference = np.float32(initial_speed)
        self.obs['linear_vels_x'][0] = initial_speed
        self.ego_speed[0] = initial_speed
        self.get_logger().info(f'Simulator ego initial speed: {initial_speed:.3f} m/s')

        # sim physical step timer
        self.drive_timer = self.create_timer(sim_timestep, self.drive_timer_callback)
        # Odom/TF are published by drive_timer_callback immediately after the
        # physics state update.  Do not run a second asynchronous render timer.
        self.start_time = time.time()

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 1)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 1)
        self.ego_noisy_odom_pub = self.create_publisher(
            Odometry, ego_noisy_odom_topic, 1)
        self.ego_noisy_odom_marker_pub = self.create_publisher(
            Marker, ego_noisy_odom_marker_topic, 1)
        self.ego_imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.collision_pub = self.create_publisher(Bool, '/collision0', 1)
        self.ego_drive_published = False
        if num_agents == 2:
            self.opp_scan_pub = self.create_publisher(LaserScan, opp_scan_topic, 1)
            self.ego_opp_odom_pub = self.create_publisher(Odometry, ego_opp_odom_topic, 1)
            self.opp_odom_pub = self.create_publisher(Odometry, opp_odom_topic, 1)
            self.opp_imu_pub = self.create_publisher(Imu, '/opp_imu/data', 10)
            self.opp_ego_odom_pub = self.create_publisher(Odometry, opp_ego_odom_topic, 1)
            self.opp_drive_published = False


        if self.get_parameter('use_sim_time_bridge').value:
            self.get_logger().info('Using simulation time.')
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
            # Set drive timer to 0 to trigger the callback asap
            self.drive_timer.timer_period_ns = 0

        # subscribers
        self.ego_drive_sub = self.create_subscription(
            AckermannDriveStamped,
            ego_drive_topic,
            self.drive_callback,
            1)
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.ego_reset_callback,
            1)
        if num_agents == 2:
            self.opp_drive_sub = self.create_subscription(
                AckermannDriveStamped,
                opp_drive_topic,
                self.opp_drive_callback,
                1)
            self.opp_reset_sub = self.create_subscription(
                PoseStamped,
                '/goal_pose',
                self.opp_reset_callback,
                1)

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                1)

        self.sim_paused = False
        self.pause_subscriber = self.create_subscription(
            Bool,
            '/pause_sim',
            self.pause_callback,
            10)

    def _load_respawn_centerline(self, map_directory, map_name):
        """Load the same ordered centerline used by MPPI and derive yaw."""
        candidates = (
            map_directory / f'{map_name}_mppi_track_optimal.csv',
            map_directory / f'{map_name}_mppi_track.csv',
            map_directory / f'{map_name}_centerline.csv',
            map_directory / 'centerline_equal.csv',
            map_directory / 'centerline.csv',
        )
        centerline_path = next((p for p in candidates if p.is_file()), None)
        if centerline_path is None:
            raise RuntimeError(
                f'Simulator respawn centerline not found in {map_directory}')
        table = np.genfromtxt(centerline_path, delimiter=',', names=True,
                              dtype=np.float64, encoding='utf-8-sig')
        names = table.dtype.names or ()
        x_name = 'x_m' if 'x_m' in names else ('x' if 'x' in names else None)
        y_name = 'y_m' if 'y_m' in names else ('y' if 'y' in names else None)
        if x_name is None or y_name is None:
            raise RuntimeError(
                f'Respawn centerline requires x_m/y_m columns: {centerline_path}')
        self.respawn_centerline_x = np.atleast_1d(table[x_name]).astype(float)
        self.respawn_centerline_y = np.atleast_1d(table[y_name]).astype(float)
        if self.respawn_centerline_x.size < 3:
            raise RuntimeError(
                f'Respawn centerline needs at least 3 points: {centerline_path}')
        # Compute yaw from geometry. Never trust a stored wrapped/offset yaw
        # column for respawn orientation.
        next_x = np.roll(self.respawn_centerline_x, -1)
        next_y = np.roll(self.respawn_centerline_y, -1)
        prev_x = np.roll(self.respawn_centerline_x, 1)
        prev_y = np.roll(self.respawn_centerline_y, 1)
        self.respawn_centerline_yaw = np.arctan2(
            next_y - prev_y, next_x - prev_x)
        self.get_logger().info(
            f'Respawn alignment centerline: {centerline_path} '
            f'({self.respawn_centerline_x.size} points)')

    def _nearest_respawn_index(self, x, y):
        distance_squared = ((self.respawn_centerline_x - x) ** 2
                            + (self.respawn_centerline_y - y) ** 2)
        return int(np.argmin(distance_squared))

    def _centerline_aligned_yaw(self, x, y):
        index = self._nearest_respawn_index(x, y)
        return float(self.respawn_centerline_yaw[index]), index

    def select_init_poses(self):
        random_idx = np.random.randint(0, len(self.respawn_centerline_x))
        poses = np.zeros((2 if self.has_opp else 1, 3))
        poses[0, 0] = self.respawn_centerline_x[random_idx]
        poses[0, 1] = self.respawn_centerline_y[random_idx]
        poses[0, 2] = self.respawn_centerline_yaw[random_idx]
        if self.has_opp:
            opponent_idx = ((random_idx + len(self.respawn_centerline_x)//2)
                            % len(self.respawn_centerline_x))
            poses[1, 0] = self.respawn_centerline_x[opponent_idx]
            poses[1, 1] = self.respawn_centerline_y[opponent_idx]
            poses[1, 2] = self.respawn_centerline_yaw[opponent_idx]

        return poses
    def pause_callback(self, msg):
        self.sim_paused = msg.data
        self.get_logger().info(f"Simulation {'paused' if self.sim_paused else 'resumed'}")

    def drive_callback(self, drive_msg):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        self.ego_requested_speed = drive_msg.drive.speed
        self.ego_requested_accel = drive_msg.drive.acceleration
        self.ego_steer = np.clip(drive_msg.drive.steering_angle, self.vehicle_params['s_min'], self.vehicle_params['s_max'])
        self.ego_drive_published = True

    def opp_drive_callback(self, drive_msg):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_requested_accel = drive_msg.drive.acceleration
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_steer = np.clip(drive_msg.drive.steering_angle, self.vehicle_params['s_min'], self.vehicle_params['s_max'])
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        requested_yaw = yaw_from_quaternion(rqw, rqx, rqy, rqz)
        rtheta, centerline_index = self._centerline_aligned_yaw(rx, ry)
        if self.has_opp:
            opp_pose = [self.obs['poses_x'][1], self.obs['poses_y'][1], self.obs['poses_theta'][1]]
            self.obs, _ = self.env.reset(options={"poses": np.array([[rx, ry, rtheta], opp_pose])})
        else:
            self.obs, _ = self.env.reset(options={"poses": np.array([[rx, ry, rtheta]])})

        self.ego_drive_published = False
        self.get_logger().info(
            f'Respawned ego at x={rx:.3f}, y={ry:.3f}, '
            f'centerline yaw={rtheta:.3f} (requested {requested_yaw:.3f}, '
            f'index {centerline_index})')

    def opp_reset_callback(self, pose_msg):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        if self.has_opp:
            rx = pose_msg.pose.position.x
            ry = pose_msg.pose.position.y
            rqx = pose_msg.pose.orientation.x
            rqy = pose_msg.pose.orientation.y
            rqz = pose_msg.pose.orientation.z
            rqw = pose_msg.pose.orientation.w
            requested_yaw = yaw_from_quaternion(rqw, rqx, rqy, rqz)
            rtheta, centerline_index = self._centerline_aligned_yaw(rx, ry)
            new_opponent_pose = np.array([rx, ry, rtheta], dtype=float)

            # RViz's 2D Goal Pose moves only the opponent. Resetting the whole
            # environment here also resets the ego dynamics, while failing to
            # update static_opponent_pose makes the 100 Hz pinning logic move
            # the opponent straight back on the next simulation step.
            self.static_opponent_pose = new_opponent_pose.copy()
            self.env.unwrapped.sim.agents[1].reset(new_opponent_pose)
            self.obs['poses_x'][1] = rx
            self.obs['poses_y'][1] = ry
            self.obs['poses_theta'][1] = rtheta
            self.obs['linear_vels_x'][1] = 0.0
            self.obs['linear_vels_y'][1] = 0.0
            self.obs['ang_vels_z'][1] = 0.0
            self.opp_pose = [rx, ry, rtheta]
            self.opp_speed = [0.0, 0.0, 0.0]
            self.opp_requested_speed = 0.0
            self.opp_requested_accel = 0.0
            self.opp_steer = 0.0
            self._update_sim_state()
            self.get_logger().info(
                f'Moved opponent with /goal_pose to x={rx:.3f}, '
                f'y={ry:.3f}, centerline yaw={rtheta:.3f} '
                f'(requested {requested_yaw:.3f}, index {centerline_index})')

        self.opp_drive_published = False

    def teleop_callback(self, twist_msg):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def drive_timer_callback(self):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        # In a two-agent simulation either controller may be run by itself.
        # Advance the simulator when either command has arrived; the vehicle
        # without a publisher receives an explicit stationary command.
        should_step = self.ego_drive_published or (
            self.has_opp and self.opp_drive_published)

        if self.drive_with_accel and not self.direct_speed_model:
            if self.ego_drive_published and not self.has_opp:
                self.obs, _, self.done, _, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_accel]]))
            elif should_step and self.has_opp:
                # A second simulated car can be used as a stationary obstacle.
                # If no opponent controller is publishing, keep it stopped
                # instead of freezing the entire simulation while waiting for
                # /opp_drive.
                opp_steer = self.opp_steer if self.opp_drive_published else 0.0
                opp_accel = self.opp_requested_accel if self.opp_drive_published else 0.0
                ego_steer = self.ego_steer if self.ego_drive_published else 0.0
                ego_accel = self.ego_requested_accel if self.ego_drive_published else 0.0
                self.obs, _, self.done, _, _ = self.env.step(np.array([[ego_steer, ego_accel], [opp_steer, opp_accel]]))
        else:
            if self.ego_drive_published and not self.has_opp:
                self.obs, _, self.done, _, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed]]))
            elif should_step and self.has_opp:
                opp_steer = self.opp_steer if self.opp_drive_published else 0.0
                opp_speed = self.opp_requested_speed if self.opp_drive_published else 0.0
                ego_steer = self.ego_steer if self.ego_drive_published else 0.0
                ego_speed = self.ego_requested_speed if self.ego_drive_published else 0.0
                self.obs, _, self.done, _, _ = self.env.step(np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]]))

        if self.has_opp and self.static_opponent:
            # The learned vehicle dynamics can drift even for a zero speed
            # command.  A test obstacle is part of the environment, not a
            # controlled vehicle, so pin its complete state after every step.
            self.env.unwrapped.sim.agents[1].reset(self.static_opponent_pose)
            self.obs['poses_x'][1] = self.static_opponent_pose[0]
            self.obs['poses_y'][1] = self.static_opponent_pose[1]
            self.obs['poses_theta'][1] = self.static_opponent_pose[2]
            self.obs['linear_vels_x'][1] = 0.0
            self.obs['linear_vels_y'][1] = 0.0
            self.obs['ang_vels_z'][1] = 0.0

        self._update_sim_state()
        # Collision is an episode terminal signal even when automatic restart
        # is disabled. Previously this topic was only published in the restart
        # block, so recorders/controllers never observed KMLP wall impacts.
        self._publish_collision_flag(bool(self.env.collisions.any()))
        if self.get_parameter('restart_simulation').value:
            curr_time = time.time() - self.start_time
            is_close_each_agents = False
            dis_thre = 0.7
            dis = np.sqrt((self.ego_pose[0] - self.opp_pose[0])**2 + (self.ego_pose[1] - self.opp_pose[1])**2)
            if dis < dis_thre:
                is_close_each_agents = True
            if(self.env.collisions.any() == True and not is_close_each_agents or curr_time > 35):
                poses = self.select_init_poses()
                options = {"poses": poses}
                self.obs, _ = self.env.reset(options=options)
                self.ego_drive_published = False
                self.ego_requested_speed = 0.0
                self.ego_requested_accel = 0.0
                self.ego_steer = 0.0
                if self.has_opp:
                    self.opp_drive_published = False
                self._update_sim_state()
                self.get_logger().info(
                    'Respawned vehicles on the selected map centerline')
                self.start_time = time.time()
            
        if self.get_parameter('use_sim_time_bridge').value:
            clock_msg = Clock()
            clock_msg.clock.sec = int(self.env.current_time // 1.0)
            clock_msg.clock.nanosec = int((self.env.current_time % 1.0) * 1e9)
            self.clock_pub.publish(clock_msg)

        # Publish one coherent observation/TF snapshot per physics state.
        self.timer_callback()

    def timer_callback(self):
        if self.sim_paused:
            return  # Skip stepping the sim if paused

        ts = self.get_clock().now().to_msg()
        # self.get_logger().info(f'Time callback start: {time.time()}')
        # pub scans
        scan = LaserScan()
        scan.header.stamp = ts
        scan.header.frame_id = self.ego_namespace + '/laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        # convert each element to float from numpy.float32
        self.ego_scan = [float(x) for x in self.ego_scan]
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)

        if self.has_opp:
            opp_scan = LaserScan()
            opp_scan.header.stamp = ts
            opp_scan.header.frame_id = self.opp_namespace + '/laser'
            opp_scan.angle_min = self.angle_min
            opp_scan.angle_max = self.angle_max
            opp_scan.angle_increment = self.angle_inc
            opp_scan.range_min = 0.
            opp_scan.range_max = 30.
            self.opp_scan = [float(x) for x in self.opp_scan]
            opp_scan.ranges = self.opp_scan
            self.opp_scan_pub.publish(opp_scan)

        # pub tf
        self._publish_odom(ts)
        self._publish_imu(ts)
        # Publish one coherent TF snapshot per physics tick. Sending every
        # link separately produced ~700 TF messages/s and made RViz render
        # partially updated vehicle trees (visible as flicker).
        transforms = self._publish_transforms(ts)
        transforms.extend(self._publish_laser_transforms(ts))
        transforms.extend(self._publish_wheel_transforms(ts))
        self.br.sendTransform(transforms)
        # self.get_logger().info(f'Time callback end: {time.time()}')

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])
        if self.has_opp:
            self.opp_scan = list(self.obs['scans'][1])
            self.opp_pose[0]  = float(self.obs['poses_x'][1])
            self.opp_pose[1]  = float(self.obs['poses_y'][1])
            self.opp_pose[2]  = float(self.obs['poses_theta'][1])
            self.opp_speed[0] = float(self.obs['linear_vels_x'][1])
            self.opp_speed[1] = float(self.obs['linear_vels_y'][1])
            self.opp_speed[2] = float(self.obs['ang_vels_z'][1])

        self.ego_pose[0] =  float(self.obs['poses_x'][0])
        self.ego_pose[1] =  float(self.obs['poses_y'][0])
        self.ego_pose[2] =  float(self.obs['poses_theta'][0])
        self._sample_ego_observed_pose()
        self.ego_speed[0] = float(self.obs['linear_vels_x'][0])
        self.ego_speed[1] = float(self.obs['linear_vels_y'][0])
        self.ego_speed[2] = float(self.obs['ang_vels_z'][0])

    def _sample_ego_observed_pose(self):
        """Sample odometry-only localization noise without changing physics."""
        observed = list(self.ego_pose)
        if self.ego_pose_noise_std_m > 0.0 and self.ego_pose_noise_max_m > 0.0:
            xy_noise = np.clip(
                self.ego_pose_noise_rng.normal(
                    0.0, self.ego_pose_noise_std_m, size=2),
                -self.ego_pose_noise_max_m,
                self.ego_pose_noise_max_m)
            observed[0] += float(xy_noise[0])
            observed[1] += float(xy_noise[1])
        if (self.ego_pose_yaw_noise_std_rad > 0.0 and
                self.ego_pose_yaw_noise_max_rad > 0.0):
            yaw_noise = float(np.clip(
                self.ego_pose_noise_rng.normal(
                    0.0, self.ego_pose_yaw_noise_std_rad),
                -self.ego_pose_yaw_noise_max_rad,
                self.ego_pose_yaw_noise_max_rad))
            observed[2] = math.atan2(
                math.sin(observed[2] + yaw_noise),
                math.cos(observed[2] + yaw_noise))
        self.ego_observed_pose = observed

    def _publish_collision_flag(self, collision_flag):

        collision_msg = Bool()
        collision_msg.data = collision_flag
        
        self.collision_pub.publish(collision_msg)

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts
        ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = quaternion_from_yaw(self.ego_pose[2])
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)

        # Publish a separate noisy localization observation for ego MPPI.
        # The canonical simulator odometry above remains exact ground truth.
        ego_noisy_odom = Odometry()
        ego_noisy_odom.header.stamp = ts
        ego_noisy_odom.header.frame_id = 'map'
        ego_noisy_odom.child_frame_id = self.ego_namespace + '/base_link'
        ego_noisy_odom.pose.pose.position.x = self.ego_observed_pose[0]
        ego_noisy_odom.pose.pose.position.y = self.ego_observed_pose[1]
        noisy_quat = quaternion_from_yaw(self.ego_observed_pose[2])
        ego_noisy_odom.pose.pose.orientation.x = noisy_quat[1]
        ego_noisy_odom.pose.pose.orientation.y = noisy_quat[2]
        ego_noisy_odom.pose.pose.orientation.z = noisy_quat[3]
        ego_noisy_odom.pose.pose.orientation.w = noisy_quat[0]
        ego_noisy_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_noisy_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_noisy_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_noisy_odom_pub.publish(ego_noisy_odom)

        # Odometry has no lifetime field. Publish the same noisy pose as an
        # RViz marker so its persistence can be configured from sim.yaml.
        noisy_marker = Marker()
        noisy_marker.header.stamp = ts
        noisy_marker.header.frame_id = 'map'
        noisy_marker.ns = 'ego_odom_noise'
        noisy_marker.id = 0
        noisy_marker.type = Marker.ARROW
        noisy_marker.action = Marker.ADD
        noisy_marker.pose = ego_noisy_odom.pose.pose
        noisy_marker.scale.x = 0.35
        noisy_marker.scale.y = 0.08
        noisy_marker.scale.z = 0.08
        noisy_marker.color.r = 1.0
        noisy_marker.color.g = 0.2
        noisy_marker.color.b = 0.1
        noisy_marker.color.a = 0.9
        lifetime_ns = int(round(self.ego_noisy_odom_marker_lifetime_s * 1.0e9))
        noisy_marker.lifetime.sec = lifetime_ns // 1_000_000_000
        noisy_marker.lifetime.nanosec = lifetime_ns % 1_000_000_000
        self.ego_noisy_odom_marker_pub.publish(noisy_marker)

        if self.has_opp:
            opp_odom = Odometry()
            opp_odom.header.stamp = ts
            opp_odom.header.frame_id = 'map'
            opp_odom.child_frame_id = self.opp_namespace + '/base_link'
            opp_odom.pose.pose.position.x = self.opp_pose[0]
            opp_odom.pose.pose.position.y = self.opp_pose[1]
            opp_quat = quaternion_from_yaw(self.opp_pose[2])
            opp_odom.pose.pose.orientation.x = opp_quat[1]
            opp_odom.pose.pose.orientation.y = opp_quat[2]
            opp_odom.pose.pose.orientation.z = opp_quat[3]
            opp_odom.pose.pose.orientation.w = opp_quat[0]
            opp_odom.twist.twist.linear.x = self.opp_speed[0]
            opp_odom.twist.twist.linear.y = self.opp_speed[1]
            opp_odom.twist.twist.angular.z = self.opp_speed[2]
            self.opp_odom_pub.publish(opp_odom)
            self.opp_ego_odom_pub.publish(ego_odom)
            self.ego_opp_odom_pub.publish(opp_odom)

    def _publish_imu(self, ts):
        """Publish synthetic body-FLU IMU with no additional sign conversion."""
        imu = Imu()
        imu.header.stamp = ts
        imu.header.frame_id = self.ego_namespace + '/base_link'
        imu.orientation.w = 1.0
        agent = self.env.unwrapped.sim.agents[0]
        wz, ax, ay = np.asarray(agent.mlp_imu, dtype=np.float32)
        imu.angular_velocity.z = float(wz)
        imu.linear_acceleration.x = float(ax)
        imu.linear_acceleration.y = float(ay)
        imu.linear_acceleration.z = 9.81
        self.ego_imu_pub.publish(imu)
        if self.has_opp:
            opp_imu = Imu()
            opp_imu.header.stamp = ts
            opp_imu.header.frame_id = self.opp_namespace + '/base_link'
            opp_imu.orientation.w = 1.0
            opp_agent = self.env.unwrapped.sim.agents[1]
            opp_wz, opp_ax, opp_ay = np.asarray(
                opp_agent.mlp_imu, dtype=np.float32)
            opp_imu.angular_velocity.z = float(opp_wz)
            opp_imu.linear_acceleration.x = float(opp_ax)
            opp_imu.linear_acceleration.y = float(opp_ay)
            opp_imu.linear_acceleration.z = 9.81
            self.opp_imu_pub.publish(opp_imu)

    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_quat = quaternion_from_yaw(self.ego_pose[2])
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = self.ego_namespace + '/base_link'
        transforms = [ego_ts]

        if self.has_opp:
            opp_t = Transform()
            opp_t.translation.x = self.opp_pose[0]
            opp_t.translation.y = self.opp_pose[1]
            opp_t.translation.z = 0.0
            opp_quat = quaternion_from_yaw(self.opp_pose[2])
            opp_t.rotation.x = opp_quat[1]
            opp_t.rotation.y = opp_quat[2]
            opp_t.rotation.z = opp_quat[3]
            opp_t.rotation.w = opp_quat[0]

            opp_ts = TransformStamped()
            opp_ts.transform = opp_t
            opp_ts.header.stamp = ts
            opp_ts.header.frame_id = 'map'
            opp_ts.child_frame_id = self.opp_namespace + '/base_link'
            transforms.append(opp_ts)
        return transforms

    def _publish_wheel_transforms(self, ts):
        ego_wheel_ts = TransformStamped()
        ego_wheel_quat = quaternion_from_yaw(self.ego_steer)
        ego_wheel_ts.transform.rotation.x = ego_wheel_quat[1]
        ego_wheel_ts.transform.rotation.y = ego_wheel_quat[2]
        ego_wheel_ts.transform.rotation.z = ego_wheel_quat[3]
        ego_wheel_ts.transform.rotation.w = ego_wheel_quat[0]
        ego_wheel_ts.header.stamp = ts
        ego_wheel_ts.header.frame_id = self.ego_namespace + '/front_left_hinge'
        ego_wheel_ts.child_frame_id = self.ego_namespace + '/front_left_wheel'
        transforms = [ego_wheel_ts]
        ego_right_wheel_ts = TransformStamped()
        ego_right_wheel_ts.transform.rotation = ego_wheel_ts.transform.rotation
        ego_right_wheel_ts.header.stamp = ts
        ego_right_wheel_ts.header.frame_id = self.ego_namespace + '/front_right_hinge'
        ego_right_wheel_ts.child_frame_id = self.ego_namespace + '/front_right_wheel'
        transforms.append(ego_right_wheel_ts)

        if self.has_opp:
            opp_wheel_ts = TransformStamped()
            opp_wheel_quat = quaternion_from_yaw(self.opp_steer)
            opp_wheel_ts.transform.rotation.x = opp_wheel_quat[1]
            opp_wheel_ts.transform.rotation.y = opp_wheel_quat[2]
            opp_wheel_ts.transform.rotation.z = opp_wheel_quat[3]
            opp_wheel_ts.transform.rotation.w = opp_wheel_quat[0]
            opp_wheel_ts.header.stamp = ts
            opp_wheel_ts.header.frame_id = self.opp_namespace + '/front_left_hinge'
            opp_wheel_ts.child_frame_id = self.opp_namespace + '/front_left_wheel'
            transforms.append(opp_wheel_ts)
            opp_right_wheel_ts = TransformStamped()
            opp_right_wheel_ts.transform.rotation = opp_wheel_ts.transform.rotation
            opp_right_wheel_ts.header.stamp = ts
            opp_right_wheel_ts.header.frame_id = self.opp_namespace + '/front_right_hinge'
            opp_right_wheel_ts.child_frame_id = self.opp_namespace + '/front_right_wheel'
            transforms.append(opp_right_wheel_ts)
        return transforms

    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = self.scan_distance_to_base_link
        # ego_scan_ts.transform.translation.z = 0.04+0.1+0.025
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts
        ego_scan_ts.header.frame_id = self.ego_namespace + '/base_link'
        ego_scan_ts.child_frame_id = self.ego_namespace + '/laser'
        transforms = [ego_scan_ts]

        if self.has_opp:
            opp_scan_ts = TransformStamped()
            opp_scan_ts.transform.translation.x = self.scan_distance_to_base_link
            opp_scan_ts.header.stamp = ts
            opp_scan_ts.header.frame_id = self.opp_namespace + '/base_link'
            opp_scan_ts.child_frame_id = self.opp_namespace + '/laser'
            transforms.append(opp_scan_ts)
        return transforms

def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()
