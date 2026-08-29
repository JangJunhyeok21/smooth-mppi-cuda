"""Launch the single-car simulator and ego MPPI on the track_20260814 track.

Copied from ifac2026_sim_mppi.launch.py with paths repointed at
data/track_20260814. config/params.yaml's default csv_file_path (ifac2026) is
left untouched because the real-car launch also falls back to it.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    controller_share = get_package_share_directory("smppi_cuda_controller")
    gym_share = get_package_share_directory("f1tenth_gym_ros")
    track_csv = "data/track_20260814/track_20260814_mppi_track_optimal.csv"

    # gym_bridge_launch resolves this before it creates the simulator node and
    # derives both spawn position and yaw from the selected track directory.
    os.environ["F1TENTH_SIM_MAP_PATH"] = os.path.join(
        controller_share, "data", "track_20260814", "track_20260814")
    # This launch owns only the ego MPPI.  Do not spawn an uncontrolled second
    # vehicle that could be mistaken for a controller/track failure.
    os.environ["F1TENTH_SIM_NUM_AGENTS"] = "1"

    params = os.path.join(controller_share, "config", "params.yaml")
    max_speed = LaunchConfiguration("max_speed")
    track_override = {
        "csv_file_path": track_csv,
        "is_simulation": True,
        "max_speed": ParameterValue(max_speed, value_type=float),
    }
    return LaunchDescription([
        DeclareLaunchArgument(
            "max_speed", default_value="12.0",
            description="Maximum MPPI speed command for track_20260814 A/B tests"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                gym_share, "launch", "gym_bridge_launch.py"))),
        Node(
            package="smppi_cuda_controller",
            executable="smppi_node",
            name="smppi_controller",
            output="screen",
            parameters=[params, track_override],
        ),
        Node(
            package="smppi_cuda_controller",
            executable="path_publisher",
            name="path_publisher",
            output="screen",
            parameters=[params, track_override],
        ),
    ])
