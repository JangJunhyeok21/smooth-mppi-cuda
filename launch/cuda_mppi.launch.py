from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    map_name = "map1"
    # True: simulator odometry contains pose + twist.
    # False: real car uses map-frame MCL pose + wheel-odometry twist.
    is_simulation = False
    
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

    common_overrides = {
        "imu_topic": "/imu/data",
        "residual_weights_path": os.path.join(
            config_dir, "kinematic_residual_gru.bin"),
        "mlp_weights_path": os.path.join(
            config_dir, "kinematic_mlp_residual.bin"),
        "kinematic_noslip_noimu_weights_path": os.path.join(
            config_dir, "ifac0807_strict_noslip_noimu_16d.bin"),
        "slip_kinematic_with_imu_weights_path": os.path.join(
            config_dir, "slip_kinematic_with_imu_direct_speed.bin"),
    }
    if is_simulation:
        controller_overrides = {
            **common_overrides,
            "use_mcl_pose": False,
            "odom_topic": "/ego_racecar/odom",
            "drive_topic": LaunchConfiguration("drive_topic"),
            "num_samples": ParameterValue(
                LaunchConfiguration("num_samples"), value_type=int),
            "visualize_candidates": ParameterValue(
                LaunchConfiguration("visualize_candidates"), value_type=bool),
        }
    else:
        controller_overrides = {
            **common_overrides,
            "use_mcl_pose": True,
            "pose_topic": "/newmcl_pose",
            "velocity_topic": "/odom",
            "drive_topic": LaunchConfiguration("drive_topic"),
            "num_samples": ParameterValue(
                LaunchConfiguration("num_samples"), value_type=int),
            "visualize_candidates": ParameterValue(
                LaunchConfiguration("visualize_candidates"), value_type=bool),
        }
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to the MPPI parameters YAML file",
            ),
            DeclareLaunchArgument(
                "drive_topic",
                default_value="/sim_drive" if is_simulation else "/drive",
                description="Ackermann command output topic",
            ),
            DeclareLaunchArgument(
                "num_samples",
                default_value="10000" if is_simulation else "4000",
                description="Number of MPPI rollouts",
            ),
            DeclareLaunchArgument(
                "visualize_candidates",
                default_value="true" if is_simulation else "false",
                description="Publish candidate and optimal trajectories to /mppi_viz",
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
                parameters=[param_file, controller_overrides],
            ),
        ]
    )
