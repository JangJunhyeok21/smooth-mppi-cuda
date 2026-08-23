"""Map2 integration test using the real F1stateArr perception contract."""

import os
import sys
from pathlib import Path

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _f1_msgs_prefix():
    """Find the F1stateArr runtime even when its overlay was not sourced."""
    configured = os.environ.get("F1_MSGS_PREFIX")
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        candidates = []
        try:
            candidates.append(Path(get_package_prefix("f1_msgs")))
        except PackageNotFoundError:
            pass
        # Local development fallback.  Deployment should either source the
        # perception workspace or set F1_MSGS_PREFIX explicitly.
        candidates.extend([
            Path.home() / "f1tenth_control/install_dynamic_servo/f1_msgs",
            Path.home() / "f1tenth_control/install/f1_msgs",
        ])
    for prefix in candidates:
        if (prefix / "share/f1_msgs/package.xml").is_file():
            return prefix.resolve()
    raise RuntimeError(
        "dynamic_obstacle_perception_test requires f1_msgs. Source its "
        "workspace or set F1_MSGS_PREFIX to the f1_msgs install prefix.")


def _prepend(prefix, current):
    return str(prefix) + (os.pathsep + current if current else "")


def generate_launch_description():
    share = get_package_share_directory("smppi_cuda_controller")
    f1_prefix = _f1_msgs_prefix()
    python_packages = f1_prefix / "local/lib" / (
        f"python{sys.version_info.major}.{sys.version_info.minor}") / "dist-packages"
    return LaunchDescription([
        SetEnvironmentVariable(
            "AMENT_PREFIX_PATH",
            _prepend(f1_prefix, os.environ.get("AMENT_PREFIX_PATH", ""))),
        SetEnvironmentVariable(
            "LD_LIBRARY_PATH",
            _prepend(f1_prefix / "lib", os.environ.get("LD_LIBRARY_PATH", ""))),
        SetEnvironmentVariable(
            "PYTHONPATH",
            _prepend(python_packages, os.environ.get("PYTHONPATH", ""))),
        DeclareLaunchArgument(
            "param_file", default_value=os.path.join(share, "config", "params.yaml")),
        DeclareLaunchArgument(
            "opponent_param_file",
            default_value=os.path.join(share, "config", "opponent_params.yaml")),
        DeclareLaunchArgument(
            "predictor_param_file",
            default_value=os.path.join(
                share, "config", "dynamic_obstacle_predictor.yaml")),
        # Use the exact same simulator + ego/opponent MPPI composition as the
        # normal overtaking launch.  Only replace predictor input with the
        # synthetic F1stateArr perception bridge below.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                share, "launch", "dynamic_obstacle_overtaking.launch.py")),
            launch_arguments={
                "param_file": LaunchConfiguration("param_file"),
                "opponent_param_file": LaunchConfiguration("opponent_param_file"),
                "predictor_param_file": LaunchConfiguration("predictor_param_file"),
                "predictor_input_mode": "perception",
            }.items()),
        Node(
            package="smppi_cuda_controller",
            executable="simulated_perception_obstacles.py",
            name="simulated_perception_obstacles",
            output="screen"),
    ])
