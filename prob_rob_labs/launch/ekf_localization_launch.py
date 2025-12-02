from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    map_file = LaunchConfiguration('map_file')

    default_map = PathJoinSubstitution([
        FindPackageShare('prob_rob_labs'),
        'maps',
        'landmarks_lab6.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock (true when using Gazebo)',
        ),
        DeclareLaunchArgument(
            'map_file',
            default_value=default_map,
            description='Path to landmark map YAML file',
        ),

        Node(
            package='prob_rob_labs',
            executable='ekf_localization',
            name='ekf_localization',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'map_file': map_file,
            }],
        ),
    ])

