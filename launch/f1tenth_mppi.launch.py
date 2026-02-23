from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_param_file = os.path.join(
        get_package_share_directory("cuda_mppi_controller"),
        "config",
        "f1tenth_mppi_params.yaml",
    )

    param_file = LaunchConfiguration("param_file")

    # 데이터 경로
    data_dir = os.path.join(
        get_package_share_directory("cuda_mppi_controller"),
        "data",
        "map1"
    )
    
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=default_param_file,
                description="Path to the MPPI parameters YAML file",
            ),
            # Path Publisher 노드
            Node(
                package="cuda_mppi_controller",
                executable="path_publisher",
                name="path_publisher",
                output="screen",
                parameters=[{
                    "csv_file_path": os.path.join(data_dir, "map1_centerline.csv"),
                    "frame_id": "map",
                    "publish_rate": 10.0,
                }],
            ),
            # MPPI Controller 노드
            Node(
                package="cuda_mppi_controller",
                executable="cuda_mppi_node",
                name="cuda_mppi_controller",
                output="screen",
                parameters=[param_file],
            ),
        ]
    )
