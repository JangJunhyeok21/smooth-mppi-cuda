from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    map_name = "map1"
    is_simulation = True

    default_param_file = os.path.join(
        get_package_share_directory("smppi_cuda_controller"),
        "config",
        "params.yaml",
    )

    param_file = LaunchConfiguration("param_file")

    # 데이터 경로
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
        }
    else:
        controller_overrides = {
            "use_mcl_pose": True,
            "pose_topic": "/mcl_pose",
            "velocity_topic": "/odom",
            "drive_topic": "/drive",
        }
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to the MPPI parameters YAML file",
            ),
            # Path Publisher 노드
            Node(
                package="smppi_cuda_controller",
                executable="path_publisher",
                name="path_publisher",
                output="screen",
                # 🚨 수정: param_file을 리스트에 추가하여 yaml 설정을 읽도록 함
                parameters=[param_file, {
                    "csv_file_path": centerline_csv,
                    "frame_id": "map",
                    "publish_rate": 1.0,
                }],
            ),
            # MPPI Controller 노드
            Node(
                package="smppi_cuda_controller",
                executable="smppi_node",
                name="smppi_controller",
                output="screen",
                parameters=[param_file, controller_overrides],
            ),
        ]
    )