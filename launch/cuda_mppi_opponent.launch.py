"""Run one MPPI controller for the simulator opponent vehicle only."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('smppi_cuda_controller'),
        'config', 'opponent_params.yaml')
    param_file = LaunchConfiguration('param_file')
    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file', default_value=default_params,
            description='Opponent-only MPPI parameter file'),
        Node(
            package='smppi_cuda_controller', executable='smppi_node',
            name='smppi_opponent_controller', output='screen',
            parameters=[param_file]),
    ])
