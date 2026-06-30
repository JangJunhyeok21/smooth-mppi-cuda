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

    data_dir = os.path.join(
        get_package_share_directory("smppi_cuda_controller"),
        "data",
        map_name,
    )
    centerline_csv = os.path.join(data_dir, f"{map_name}_centerline.csv")

    param_file = LaunchConfiguration("param_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "param_file",
            default_value=default_param_file,
            description="Path to MPPI params YAML",
        ),
        DeclareLaunchArgument(
            "sim_odom_topic",
            default_value="/odom0",
            description="Simulator odometry topic",
        ),

        # 경로 발행
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

        # LiDAR 기반 상대방 추적 (별도 노드)
        Node(
            package="smppi_cuda_controller",
            executable="opponent_tracker",
            name="opponent_tracker",
            output="screen",
            parameters=[{
                "scan_topic":            "/scan",
                "odom_topic":            "/opponent_odom",
                "frame_id":              "laser",
                "cluster_dist_thresh":   0.15,
                "min_cluster_points":    3,
                "max_cluster_points":    200,
                "max_detection_range":   8.0,
                "min_detection_range":   0.3,
            }],
        ),

        # MPPI + OvertakeFSM 합성 노드 (시뮬레이터)
        Node(
            package="smppi_cuda_controller",
            executable="smppi_node_fsm",
            name="smppi_controller",
            output="screen",
            parameters=[param_file, {
                "odom_topic":            LaunchConfiguration("sim_odom_topic"),
                "drive_topic":           "/ackermann_cmd0",
                "num_samples":           8000,
                "visualize_candidates":  True,
                # FSM
                "fsm_follow_dist":       5.0,
                "fsm_prep_dist":         3.5,
                "fsm_clear_dist":        7.0,
                "fsm_prep_timeout":      2.5,
                "fsm_lateral_offset":    0.5,
                "fsm_follow_speed":      4.5,
                "fsm_overtake_speed":    6.5,
            }],
        ),
    ])
