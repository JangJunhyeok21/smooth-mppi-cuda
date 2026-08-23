from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("smppi_cuda_controller")
    params = LaunchConfiguration("param_file")
    opponent_params = LaunchConfiguration("opponent_param_file")
    predictor_params = LaunchConfiguration("predictor_param_file")
    predictor_input_mode = LaunchConfiguration("predictor_input_mode")
    track_csv = LaunchConfiguration("track_csv")
    return LaunchDescription([
        DeclareLaunchArgument("param_file", default_value=os.path.join(share, "config", "params.yaml")),
        DeclareLaunchArgument("opponent_param_file", default_value=os.path.join(
            share, "config", "opponent_params.yaml")),
        DeclareLaunchArgument("predictor_param_file", default_value=os.path.join(
            share, "config", "dynamic_obstacle_predictor.yaml")),
        DeclareLaunchArgument(
            "predictor_input_mode", default_value="simulation",
            description="Predictor input source: simulation, perception, or both"),
        DeclareLaunchArgument(
            "track_csv",
            default_value="data/map2/map2_mppi_track_optimal.csv",
            description="Package-relative MPPI and predictor track CSV"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("f1tenth_gym_ros"), "launch", "gym_bridge_launch.py"))),
        Node(package="smppi_cuda_controller", executable="dynamic_obstacle_predictor_node",
             name="dynamic_obstacle_predictor", output="screen",
             parameters=[predictor_params, {
                 "input_mode": predictor_input_mode,
                 "track_csv": ParameterValue(track_csv, value_type=str),
             }]),
        # Opponent is controlled by its own MPPI node, exactly like the
        # standalone ``mppi2`` alias, instead of the simple waypoint driver.
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "cuda_mppi_opponent.launch.py")),
            launch_arguments={
                "opponent_param_file": opponent_params,
                "track_csv": track_csv,
            }.items()),
        Node(package="smppi_cuda_controller", executable="smppi_node",
             name="smppi_controller", output="screen", parameters=[params, {
                 "csv_file_path": ParameterValue(track_csv, value_type=str),
             }]),
        Node(package="smppi_cuda_controller", executable="path_publisher",
             name="path_publisher", output="screen", parameters=[params, {
                 "csv_file_path": ParameterValue(track_csv, value_type=str),
             }]),
    ])
