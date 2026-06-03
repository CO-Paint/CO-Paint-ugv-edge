import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('cap_navigation')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')

    default_map_path = os.path.join(
        pkg_share,
        'map',
        'kiss_map_1_fixed.yaml'
    )

    default_param_path = os.path.join(
        pkg_share,
        'param',
        'nav_param.yaml'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')

    nav2_launch_file_dir = os.path.join(nav2_bringup_share, 'launch')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map_path,
            description='Full path to 2D occupancy map yaml file'
        ),

        DeclareLaunchArgument(
            'params_file',
            default_value=default_param_path,
            description='Full path to Nav2 params file'
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation clock if true'
        ),

        # 1. Map server only
        # AMCL은 실행하지 않음.
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                params_file,
                {
                    'yaml_filename': map_yaml_file,
                    'use_sim_time': use_sim_time
                }
            ]
        ),

        # 2. Lifecycle manager for map_server
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[
                {
                    'use_sim_time': use_sim_time,
                    'autostart': True,
                    'node_names': ['map_server']
                }
            ]
        ),

        # 3. Navigation stack only
        # controller_server, planner_server, behavior_server, bt_navigator 등만 실행
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_launch_file_dir, 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': params_file,
            }.items(),
        ),
    ])