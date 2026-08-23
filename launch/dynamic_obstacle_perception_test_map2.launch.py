"""Counter-clockwise map2 F1stateArr dynamic/static obstacle test."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory("smppi_cuda_controller")
    os.environ["F1TENTH_SIM_MAP_PATH"] = os.path.join(
        share, "data", "map2_reverse", "map2_reverse")
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share, "launch", "dynamic_obstacle_perception_test.launch.py")),
            launch_arguments={
                "predictor_param_file": os.path.join(
                    share, "config", "dynamic_obstacle_predictor_map2_reverse.yaml"),
                "track_csv": (
                    "data/map2_reverse/map2_reverse_mppi_track_optimal.csv"),
            }.items()),
    ])
