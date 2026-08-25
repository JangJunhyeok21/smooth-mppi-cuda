"""Run one MPPI controller for the simulator opponent vehicle only."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('smppi_cuda_controller'),
        'config', 'opponent_params.yaml')
    # Keep this name distinct from the ego launch's ``param_file``.  Reusing
    # the generic name inside an included launch changes the parent launch
    # configuration and can make the ego node load opponent_params.yaml too.
    param_file = LaunchConfiguration('opponent_param_file')
    track_csv = LaunchConfiguration('track_csv')
    return LaunchDescription([
        DeclareLaunchArgument(
            'opponent_param_file', default_value=default_params,
            description='Opponent-only MPPI parameter file'),
        DeclareLaunchArgument(
            'track_csv', default_value='data/ifac2026/ifac2026_mppi_track_optimal.csv',
            description='Package-relative opponent track CSV'),
        Node(
            package='smppi_cuda_controller', executable='smppi_node',
            name='smppi_opponent_controller', output='screen',
            parameters=[param_file, {
                'csv_file_path': ParameterValue(track_csv, value_type=str),
                # Vehicle-role wiring belongs to this launch file so copying
                # ego model/cost parameters cannot reconnect mppi2 to ego.
                'is_simulation': True,
                'simulation_odom_topic': '/opp_racecar/odom',
                'simulation_drive_topic': '/opp_drive',
                'imu_topic': '/opp_imu/data',
                'visualization_topic': '/mppi_opponent_viz',
                'optimal_trajectory_topic': '/mppi_opponent_optimal_trajectory',
                'kf_state_topic': '/opp_kf_state',
                # The shared predictor describes this opponent for ego MPPI;
                # consuming it here would make mppi2 avoid itself.
                'obstacle_avoidance_enabled': False,
                'dynamic_obstacle_prediction_enabled': False,
            }]),
    ])
