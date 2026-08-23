"""Map2 F1stateArr integration test with noisy dynamic-object pose."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share=get_package_share_directory("smppi_cuda_controller")
    return LaunchDescription([
        DeclareLaunchArgument("input_pose_noise_std_m",default_value="0.05"),
        DeclareLaunchArgument("input_pose_noise_max_m",default_value="0.10"),
        DeclareLaunchArgument("input_pose_noise_seed",default_value="20260824"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share,"launch","dynamic_obstacle_perception_test.launch.py")),
            launch_arguments={
                "input_pose_noise_std_m":LaunchConfiguration("input_pose_noise_std_m"),
                "input_pose_noise_max_m":LaunchConfiguration("input_pose_noise_max_m"),
                "input_pose_noise_seed":LaunchConfiguration("input_pose_noise_seed"),
            }.items()),
    ])
