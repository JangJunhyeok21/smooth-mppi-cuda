from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_param_file = os.path.join(
        get_package_share_directory("cuda_mppi_controller"),
        "config",
        "mppi_params.yaml",
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
                package="cuda_mppi_controller",
                executable="cuda_mppi_node",
                name="cuda_mppi_controller",
                output="screen",
                parameters=[param_file],
            ),
        ]
    )
