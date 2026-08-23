"""ICRA 2025 simulator, two MPPI controllers, and map-specific predictor."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    share = get_package_share_directory("smppi_cuda_controller")
    os.environ["F1TENTH_SIM_MAP_PATH"] = os.path.join(
        share, "data", "icra2025", "icra2025")
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share, "launch", "dynamic_obstacle_overtaking.launch.py")),
            launch_arguments={
                "predictor_param_file": os.path.join(
                    share, "config", "dynamic_obstacle_predictor_icra2025.yaml"),
                "track_csv": "data/icra2025/icra2025_mppi_track_optimal.csv",
            }.items()),
    ])
