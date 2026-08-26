# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.substitutions import EnvironmentVariable
from ament_index_python.packages import get_package_share_directory
import csv
import math
import os
import yaml

def simulator_paths(package_share):
    """Resolve the simulator sources belonging to this colcon workspace."""
    workspace = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(package_share))))
    gym_source = os.path.join(workspace, 'src', 'f1tenth_gym')
    project_root = os.path.dirname(workspace)
    map_path = os.environ.get('F1TENTH_SIM_MAP_PATH', os.path.join(
        project_root, 'data', 'ifac2026', 'ifac2026'))
    return gym_source, map_path


def centerline_spawn_poses(map_path):
    """Return ego/opponent poses aligned with the selected map centerline."""
    base = os.path.basename(map_path)
    directory = os.path.dirname(map_path)
    candidates = (
        os.path.join(directory, base + '_mppi_track_optimal.csv'),
        os.path.join(directory, base + '_mppi_track.csv'),
        os.path.join(directory, base + '_centerline.csv'),
        os.path.join(directory, 'centerline_equal.csv'),
        os.path.join(directory, 'centerline.csv'),
    )
    centerline_path = next((path for path in candidates if os.path.isfile(path)), None)
    if centerline_path is None:
        raise RuntimeError('No centerline CSV found for simulator map: ' + map_path)

    points = []
    with open(centerline_path, newline='') as stream:
        rows = list(csv.reader(stream))
    start = 1 if rows and rows[0] and rows[0][0].strip().lower() in ('x', 'x_m') else 0
    for row in rows[start:]:
        if len(row) >= 2:
            points.append((float(row[0]), float(row[1])))
    if len(points) < 2:
        raise RuntimeError('Centerline needs at least two points: ' + centerline_path)

    def pose(index):
        x, y = points[index]
        next_x, next_y = points[(index + 1) % len(points)]
        return x, y, math.atan2(next_y - y, next_x - x)

    # CSV sampling density is not fixed: on map2, 30 samples are only about
    # 0.30 m and overlap the two vehicle footprints. Select the opponent by
    # physical arc length instead of a sample count.
    requested_separation = float(os.environ.get(
        'F1TENTH_OPPONENT_SPAWN_DISTANCE_M', '2.5'))
    if requested_separation <= 0.0:
        raise ValueError('F1TENTH_OPPONENT_SPAWN_DISTANCE_M must be positive')
    opponent_index = 1
    arc_length = 0.0
    maximum_index = max(1, len(points) // 2)
    while opponent_index <= maximum_index:
        arc_length += math.dist(points[opponent_index - 1],
                                points[opponent_index])
        if arc_length >= requested_separation:
            break
        opponent_index += 1
    if arc_length < requested_separation:
        raise RuntimeError(
            'Centerline is too short for requested opponent separation: '
            f'{requested_separation:.3f} m (available {arc_length:.3f} m)')
    return (pose(0), pose(opponent_index), centerline_path,
            opponent_index, arc_length)

def generate_launch_description():
    ld = LaunchDescription()
    package_share = get_package_share_directory('f1tenth_gym_ros')
    config = os.path.join(
        package_share,
        'config',
        'sim.yaml'
        )
    controller_config = os.environ.get(
        'F1TENTH_SIM_PARAMS_FILE',
        os.path.join(
            get_package_share_directory('smppi_cuda_controller'),
            'config',
            'params.yaml'))
    controller_config = os.path.abspath(controller_config)
    if not os.path.isfile(controller_config):
        raise RuntimeError('Simulator params file is missing: ' + controller_config)
    controller_share = get_package_share_directory('smppi_cuda_controller')
    controller_parameters = yaml.safe_load(open(controller_config, 'r'))[
        '/**']['ros__parameters']
    # Keep simulator plant parameters sourced from the same params.yaml as
    # MPPI; sim.yaml no longer carries an independently drifting copy.
    shared_names = (
        'max_steer',
        'kinematic_steer_scale', 'kinematic_steer_bias',
        'steer_servo_time_constant', 'actuator_max_steer_rate',
        'speed_servo_kp', 'speed_reference_accel_time_constant',
        'speed_reference_brake_time_constant',
        'actuator_max_speed_reference_rate', 'kinematic_position_speed_scale',
        'dynamic_mlp_B_f', 'dynamic_mlp_C_f', 'dynamic_mlp_D_f',
        'dynamic_mlp_E_f', 'dynamic_mlp_B_r', 'dynamic_mlp_C_r',
        'dynamic_mlp_D_r', 'dynamic_mlp_E_r', 'dynamic_mlp_I_z',
        'mlp_max_residual_ax', 'mlp_max_residual_ay',
        'mlp_max_residual_yaw_accel', 'mass', 'l_f', 'l_r')
    controller_override = {
        name: controller_parameters[name] for name in shared_names}
    controller_override.update({
        'dynamics_model': controller_parameters['dynamics_model'],
        'mppi_min_speed': controller_parameters['min_speed'],
        'mppi_max_speed': controller_parameters['max_speed'],
        'mppi_min_accel': controller_parameters['min_accel'],
        'mppi_max_accel': controller_parameters['max_accel'],
        'dynamic_mlp_model_dt': controller_parameters['model_dt'],
    })
    residual_path = controller_parameters['dynamic_mlp_servo_lag_weights_path']
    if not os.path.isabs(residual_path):
        residual_path = os.path.join(controller_share, residual_path)
    controller_override['dynamic_mlp_weights_path'] = residual_path
    config_dict = yaml.safe_load(open(config, 'r'))
    requested_num_agents = int(os.environ.get(
        'F1TENTH_SIM_NUM_AGENTS',
        config_dict['bridge']['ros__parameters']['num_agent']))
    if requested_num_agents not in (1, 2):
        raise ValueError('F1TENTH_SIM_NUM_AGENTS must be 1 or 2')
    config_dict['bridge']['ros__parameters']['num_agent'] = requested_num_agents
    gym_source, local_map_path = simulator_paths(package_share)
    if not os.path.isdir(os.path.join(gym_source, 'f1tenth_gym')):
        raise RuntimeError('Pinned f1tenth_gym source is missing: ' + gym_source)
    if not os.path.isfile(local_map_path + '.yaml'):
        raise RuntimeError('Simulator map is missing: ' + local_map_path + '.yaml')
    config_dict['bridge']['ros__parameters']['map_path'] = local_map_path
    # The selected ROS map YAML is the source of truth for its image format.
    # ``sim.yaml`` historically hard-coded ``.pgm``, which makes otherwise
    # valid PNG maps fail as ``<map>.pgm`` before the simulator starts.
    with open(local_map_path + '.yaml', 'r') as file:
        map_yaml = yaml.safe_load(file)
    map_image_name = str(map_yaml.get('image', ''))
    map_image_extension = os.path.splitext(map_image_name)[1]
    if not map_image_extension:
        raise RuntimeError(
            'Map YAML image must have a file extension: ' + local_map_path + '.yaml')
    expected_map_image = os.path.join(
        os.path.dirname(local_map_path), map_image_name)
    if not os.path.isfile(expected_map_image):
        raise RuntimeError('Simulator map image is missing: ' + expected_map_image)
    # f1tenth_gym builds the image name as map_path + map_img_ext, so verify
    # that contract explicitly rather than silently loading a different file.
    if os.path.realpath(expected_map_image) != os.path.realpath(
            local_map_path + map_image_extension):
        raise RuntimeError(
            'Simulator requires the map image basename to match the YAML: '
            f'expected {local_map_path + map_image_extension}, YAML references '
            f'{expected_map_image}')
    config_dict['bridge']['ros__parameters']['map_img_ext'] = map_image_extension
    (ego_spawn, opponent_spawn, centerline_path, opponent_spawn_index,
     opponent_spawn_distance) = centerline_spawn_poses(local_map_path)
    simulator_dynamics = os.environ.get('F1TENTH_SIM_DYNAMICS_MODEL')
    simulator_override = {}
    if simulator_dynamics:
        simulator_override['dynamics_model'] = simulator_dynamics
    has_opp = requested_num_agents > 1
    teleop = config_dict['bridge']['ros__parameters']['kb_teleop']
    use_sim_time = config_dict['bridge']['ros__parameters']['use_sim_time']
    obstacle_override = dict(zip(('sx1', 'sy1', 'stheta1'), opponent_spawn))
    for environment_name, parameter_name in (
            ('F1TENTH_OBSTACLE_X', 'sx1'),
            ('F1TENTH_OBSTACLE_Y', 'sy1'),
            ('F1TENTH_OBSTACLE_YAW', 'stheta1')):
        value = os.environ.get(environment_name)
        if value is not None:
            obstacle_override[parameter_name] = float(value)
    ego_override = dict(zip(('sx', 'sy', 'stheta'), ego_spawn))
    for environment_name, parameter_name in (
        ('F1TENTH_EGO_X', 'sx'), ('F1TENTH_EGO_Y', 'sy'),
        ('F1TENTH_EGO_YAW', 'stheta')):
        value = os.environ.get(environment_name)
        if value is not None:
            ego_override[parameter_name] = float(value)
    ego_override['initial_speed'] = float(os.environ.get(
        'F1TENTH_EGO_INITIAL_SPEED', '0.0'))

    print('Simulator centerline spawn: ' + centerline_path)
    print('  ego: x={sx:.6f}, y={sy:.6f}, yaw={stheta:.6f}'.format(**ego_override))
    print('  opponent: x={sx1:.6f}, y={sy1:.6f}, yaw={stheta1:.6f} '
          '(centerline index {}, arc distance {:.3f} m)'.format(
              opponent_spawn_index, opponent_spawn_distance,
              **obstacle_override))

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config, controller_override, {
                    'map_path': local_map_path,
                    'map_img_ext': map_image_extension,
                    'num_agent': requested_num_agents}, simulator_override,
                    obstacle_override, ego_override, {
            'use_sim_time': False,  # Always use real time for the bridge node
            'use_sim_time_bridge': use_sim_time, # Whether to internally use and publish sim time
            }],
        additional_env={
            'PYTHONPATH': gym_source + os.pathsep + os.environ.get('PYTHONPATH', ''),
            'NUMBA_CACHE_DIR': os.environ.get(
                'NUMBA_CACHE_DIR', '/tmp/f1tenth_numba_cache'),
        },
    )
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'gym_bridge.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(EnvironmentVariable(
            'F1TENTH_SIM_ENABLE_RVIZ', default_value='true')),
    )

    # Create custom yaml file for map server by copying the original yaml file and scaling the resolution by the sim.yaml scale
    map_yaml['resolution'] *= config_dict['bridge']['ros__parameters']['scale']
    origin = map_yaml['origin']
    scaled_origin = [
        origin[0] * config_dict['bridge']['ros__parameters']['scale'],
        origin[1] * config_dict['bridge']['ros__parameters']['scale'],
        origin[2],
    ]
    map_yaml['origin'] = scaled_origin
    map_yaml['image'] = 'scaled_map' + config_dict['bridge']['ros__parameters']['map_img_ext']

    temp_yaml_path = None
    # Create a temporary directory to store the scaled map yaml and image
    # Create a temporary directory to store the scaled map yaml and image in the same location as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    temp_yaml_path = os.path.join(temp_dir, 'scaled_map.yaml')
    temp_img_path = os.path.join(temp_dir, 'scaled_map' + config_dict['bridge']['ros__parameters']['map_img_ext'])

    # Write the scaled map yaml to the temporary file
    with open(temp_yaml_path, 'w') as file:
        yaml.dump(map_yaml, file)

    # Copy the map image to the temporary directory
    map_image_path = os.path.join(config_dict['bridge']['ros__parameters']['map_path'] + config_dict['bridge']['ros__parameters']['map_img_ext'])
    with open(temp_img_path, 'wb') as file:
        with open(map_image_path, 'rb') as img_file:
            file.write(img_file.read())

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[{'yaml_filename': temp_yaml_path},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'},
                    {'use_sim_time': use_sim_time}],
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )


    ego_xacro = None
    if config_dict['bridge']['ros__parameters']['vehicle_params'] == 'f1tenth':
        ego_xacro = "ego_racecar.xacro"
    elif config_dict['bridge']['ros__parameters']['vehicle_params'] == 'fullscale':
        ego_xacro = "ego_racecar_fullscale.xacro"
    elif config_dict['bridge']['ros__parameters']['vehicle_params'] == 'f1fifth':
        ego_xacro = "ego_racecar_f1fifth.xacro"
    else:
        raise ValueError('vehicle_params should be either f1tenth, fullscale, or f1fifth.')

    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[
            {'robot_description': Command([
                'xacro ',
                os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', ego_xacro)
            ])},
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/robot_description', 'ego_robot_description')]
    )
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[
            {'robot_description': Command([
                'xacro ',
                os.path.join(get_package_share_directory('f1tenth_gym_ros'), 'launch', 'opp_racecar.xacro')
            ])},
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/robot_description', 'opp_robot_description')]
    )

    # finalize
    ld.add_action(rviz_node)
    ld.add_action(bridge_node)
    ld.add_action(map_server_node)
    # The lifecycle manager can otherwise issue configure before map_server's
    # lifecycle services are ready, leaving RViz without a stable fixed map.
    ld.add_action(TimerAction(period=1.0, actions=[nav_lifecycle_node]))
    ld.add_action(ego_robot_publisher)
    if has_opp:
        ld.add_action(opp_robot_publisher)

    return ld
