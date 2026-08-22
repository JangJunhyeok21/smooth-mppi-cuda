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
    default_mlp_param_file = os.path.join(
        get_package_share_directory("smppi_cuda_controller"),
        "config",
        "MLP_params.yaml",
    )

    param_file = LaunchConfiguration("param_file")
    mlp_param_file = LaunchConfiguration("mlp_param_file")
    dynamics_model = LaunchConfiguration("dynamics_model")
    csv_file_path = LaunchConfiguration("csv_file_path")
    is_simulation = LaunchConfiguration("is_simulation")
    obstacle_avoidance_enabled = LaunchConfiguration(
        "obstacle_avoidance_enabled")

    config_dir = os.path.join(
        get_package_share_directory("smppi_cuda_controller"), "config")

    # Only filesystem-dependent paths are resolved here. Controller tuning,
    # topics, simulation/real mode, rollout count, and visualization are all
    # owned by params.yaml.
    path_overrides = {
        "dynamic_mlp_servo_lag_weights_path": os.path.join(
            config_dir, "dynamic_40ms_residual_servo_lag.bin"),
    }
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to the MPPI parameters YAML file",
            ),
            DeclareLaunchArgument(
                "mlp_param_file",
                default_value=default_mlp_param_file,
                description="Path to MLP-only correction limits YAML file",
            ),
            DeclareLaunchArgument("dynamics_model", default_value="DYNAMIC_SERVO_LAG"),
            DeclareLaunchArgument("csv_file_path", default_value=os.path.join(
                get_package_share_directory("smppi_cuda_controller"),
                "data", "map2", "map2_mppi_track.csv")),
            DeclareLaunchArgument("is_simulation", default_value="true"),
            DeclareLaunchArgument("obstacle_avoidance_enabled", default_value="false"),
            Node(
                package="smppi_cuda_controller",
                executable="smppi_node",
                name="smppi_controller",
                output="screen",
                parameters=[param_file, mlp_param_file, path_overrides, {
                    "dynamics_model": dynamics_model,
                    "csv_file_path": csv_file_path,
                    "is_simulation": is_simulation,
                    "obstacle_avoidance_enabled": obstacle_avoidance_enabled,
                }],
            ),
        ]
    )
