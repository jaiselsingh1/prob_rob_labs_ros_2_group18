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
        Node(
            package='prob_rob_labs',
            executable='axis_identifier',
            name='axis_identifier',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                         'landmark_color': LaunchConfiguration('landmark_color')}]
        )
    ])
