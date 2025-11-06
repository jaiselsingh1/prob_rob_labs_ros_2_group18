import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),
        DeclareLaunchArgument('landmark_color', default_value='cyan',
                              description='color of the landmark to identify'),
        DeclareLaunchArgument('landmark_height', default_value='0.5',
                              description='height of the landmark (meters)'),
        Node(
            package='prob_rob_labs',
            executable='landmark_positioner',
            name='landmark_positioner',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                         'landmark_color': LaunchConfiguration('landmark_color'),
                         'landmark_height': LaunchConfiguration('landmark_height')}]
        )
    ])