"""Run a single-agent simulator/MPPI episode for MDN data collection."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('smppi_cuda_controller')
    sim_share = get_package_share_directory('f1tenth_gym_ros')
    output = LaunchConfiguration('output')
    duration = LaunchConfiguration('duration_s')
    max_speed = LaunchConfiguration('max_speed')
    q_v = LaunchConfiguration('q_v')
    temperature = LaunchConfiguration('lambda')
    steer_noise = LaunchConfiguration('noise_steer_std')
    accel_noise = LaunchConfiguration('noise_accel_std')
    metadata = LaunchConfiguration('episode_metadata_json')
    params = LaunchConfiguration('param_file')
    declarations = [
        DeclareLaunchArgument('param_file', default_value=os.path.join(
            share, 'config', 'params.yaml')),
        DeclareLaunchArgument('output'),
        DeclareLaunchArgument('duration_s', default_value='25.0'),
        DeclareLaunchArgument('max_speed', default_value='2.0'),
        DeclareLaunchArgument('q_v', default_value='16.0'),
        DeclareLaunchArgument('lambda', default_value='20.0'),
        DeclareLaunchArgument('noise_steer_std', default_value='0.35'),
        DeclareLaunchArgument('noise_accel_std', default_value='2.0'),
        # Consumed by the simulator through F1TENTH_EGO_INITIAL_SPEED; this
        # declaration records/accepts it as part of the episode invocation.
        DeclareLaunchArgument('initial_speed', default_value='0.0'),
        DeclareLaunchArgument('episode_metadata_json', default_value='{}'),
    ]
    simulator = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(sim_share, 'launch', 'gym_bridge_launch.py')))
    controller = Node(
        package='smppi_cuda_controller', executable='smppi_node',
        name='mppi_collection_controller', output='screen',
        parameters=[params, {
            'max_speed': ParameterValue(max_speed, value_type=float),
            'q_v': ParameterValue(q_v, value_type=float),
            'lambda': ParameterValue(temperature, value_type=float),
            'noise_steer_std': ParameterValue(steer_noise, value_type=float),
            'noise_accel_std': ParameterValue(accel_noise, value_type=float),
            'obstacle_avoidance_enabled': False,
            'dynamic_obstacle_prediction_enabled': False,
            'publish_visualization': False,
            'publish_optimal_trajectory': False,
            'boundary_publish_period_s': 0.0,
        }])
    collector = Node(
        package='smppi_cuda_controller', executable='collect_mppi_mdn_episode.py',
        name='mppi_mdn_episode_collector', output='screen', parameters=[{
            'output_path': output,
            'duration_s': ParameterValue(duration, value_type=float),
            'episode_metadata_json': ParameterValue(metadata, value_type=str),
        }])
    stop = RegisterEventHandler(OnProcessExit(
        target_action=collector,
        on_exit=[EmitEvent(event=Shutdown(reason='episode collector finished'))]))
    return LaunchDescription([*declarations, simulator, controller, collector, stop])
