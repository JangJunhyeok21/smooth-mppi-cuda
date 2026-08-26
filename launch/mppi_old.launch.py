from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_param_file = os.path.join(
        get_package_share_directory("smppi_cuda_controller"),
        "config",
        "mppi_old.yaml",
    )
    param_file = LaunchConfiguration("param_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "param_file",
            default_value=default_param_file,
            description="MPPI parameter YAML file",
        ),
        Node(
            package="smppi_cuda_controller",
            executable="smppi_node",
            name="smppi_controller",
            output="screen",
            parameters=[param_file],
        ),
        Node(
            package="smppi_cuda_controller",
            executable="path_publisher",
            name="path_publisher",
            output="screen",
            parameters=[param_file],
        ),
    ])
