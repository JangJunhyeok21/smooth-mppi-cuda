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
        "params.yaml",
    )

    param_file = LaunchConfiguration("param_file")
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to the MPPI parameters YAML file",
            ),
            Node(
                package="smppi_cuda_controller",
                executable="smppi_node",
                name="smppi_controller",
                output="screen",
                # params.yaml is the single source of all MPPI runtime values.
                parameters=[param_file],
            ),
            Node(
                package="smppi_cuda_controller",
                executable="path_publisher",
                name="path_publisher",
                output="screen",
                # Publishes csv_file_path as /mppi_target_path and its
                # left/right boundaries using the same portable YAML.
                parameters=[param_file],
            ),
        ]
    )
