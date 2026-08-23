import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("smppi_cuda_controller")
    default_params = os.path.join(
        share, "config", "dynamic_obstacle_predictor.yaml")
    param_file = LaunchConfiguration("param_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "param_file",
            default_value=default_params,
            description="Dynamic-obstacle predictor parameter YAML",
        ),
        Node(
            package="smppi_cuda_controller",
            executable="dynamic_obstacle_predictor_node",
            name="dynamic_obstacle_predictor",
            output="screen",
            parameters=[param_file],
        ),
    ])
