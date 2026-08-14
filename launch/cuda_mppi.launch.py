from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    map_name = "map1"
    
    default_param_file = os.path.join(
        get_package_share_directory("smppi_cuda_controller"),
        "config",
        "params.yaml",
    )

    param_file = LaunchConfiguration("param_file")

    data_dir = os.path.join(
        get_package_share_directory("smppi_cuda_controller"),
        "data",
        map_name,
    )
    centerline_csv = os.path.join(data_dir, f"{map_name}_centerline.csv")
    config_dir = os.path.join(
        get_package_share_directory("smppi_cuda_controller"), "config")

    # Only filesystem-dependent paths are resolved here. Controller tuning,
    # topics, simulation/real mode, rollout count, and visualization are all
    # owned by params.yaml.
    path_overrides = {
        "residual_weights_path": os.path.join(
            config_dir, "kinematic_residual_gru.bin"),
        "mlp_weights_path": os.path.join(
            config_dir, "kinematic_mlp_residual.bin"),
        "kinematic_noslip_noimu_weights_path": os.path.join(
            config_dir, "ifac0807_strict_noslip_noimu_16d.bin"),
        "slip_kinematic_with_imu_weights_path": os.path.join(
            config_dir, "slip_kinmatic_MLP.bin"),
        "dynamic_mlp_weights_path": os.path.join(
            config_dir, "dynamic_MLP.bin"),
        "dynamic_mlp_servo_lag_weights_path": os.path.join(
            config_dir, "dynamic_40ms_residual_servo_lag.bin"),
        "effective_history_weights_path": os.path.join(
            config_dir, "effective_history_state_residual.bin"),
        "e2e_weights_path": os.path.join(
            config_dir, "E2E.bin"),
    }
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to the MPPI parameters YAML file",
            ),
            Node(
                package="smppi_cuda_controller",
                executable="path_publisher",
                name="path_publisher",
                output="screen",
                parameters=[param_file, {
                    "csv_file_path": centerline_csv,
                    "frame_id": "map",
                    "publish_rate": 1.0,
                }],
            ),
            Node(
                package="smppi_cuda_controller",
                executable="smppi_node",
                name="smppi_controller",
                output="screen",
                parameters=[param_file, path_overrides],
            ),
        ]
    )
