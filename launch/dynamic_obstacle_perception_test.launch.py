"""Map2 two-car MPPI test using the real F1stateArr perception contract."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("smppi_cuda_controller")
    gym_share = get_package_share_directory("f1tenth_gym_ros")
    params = LaunchConfiguration("param_file")
    opponent_params = LaunchConfiguration("opponent_param_file")
    predictor_params = LaunchConfiguration("predictor_param_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "param_file", default_value=os.path.join(share, "config", "params.yaml")),
        DeclareLaunchArgument(
            "opponent_param_file",
            default_value=os.path.join(share, "config", "opponent_params.yaml")),
        DeclareLaunchArgument(
            "predictor_param_file",
            default_value=os.path.join(
                share, "config", "dynamic_obstacle_predictor.yaml")),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(gym_share, "launch", "gym_bridge_launch.py"))),
        Node(
            package="smppi_cuda_controller",
            executable="simulated_perception_obstacles.py",
            name="simulated_perception_obstacles",
            output="screen"),
        Node(
            package="smppi_cuda_controller",
            executable="dynamic_obstacle_predictor_node",
            name="dynamic_obstacle_predictor",
            output="screen",
            parameters=[predictor_params, {"input_mode": "perception"}]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "cuda_mppi_opponent.launch.py")),
            launch_arguments={"param_file": opponent_params}.items()),
        Node(
            package="smppi_cuda_controller", executable="smppi_node",
            name="smppi_controller", output="screen", parameters=[params]),
        Node(
            package="smppi_cuda_controller", executable="path_publisher",
            name="path_publisher", output="screen", parameters=[params]),
    ])
