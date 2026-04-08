from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    map_name = "icra2025"
    
    #이 변수를 False로 바꾸면 Jetson Nano 최적화 모드(실차)로 진입합니다.
    is_simulation = True

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

    if is_simulation:
        controller_overrides = {
            "use_mcl_pose": False,
            "odom_topic": "/odom0",
            "drive_topic": "/ackermann_cmd0",
            "num_samples": 10000,            # 데스크톱(시뮬)은 10000개
            "publish_debug_info": True,      
            "visualize_candidates": True,    # CPU <-> GPU 메모리 복사 발생
        }
    else:
        controller_overrides = {
            "use_mcl_pose": True,
            "pose_topic": "/mcl_pose",
            "velocity_topic": "/odom",
            "drive_topic": "/drive",
            "num_samples": 5000,             # 🚨 Jetson Nano 최적화 (5000개)
            "publish_debug_info": False,     # 🚨 디버그 토픽 발행 스킵
            "visualize_candidates": False,   # 🚨 GPU 연산 결과(h_states)의 호스트 복사 원천 차단
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
                parameters=[param_file, controller_overrides],
            ),
        ]
    )