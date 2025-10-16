import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),
        DeclareLaunchArgument('ref_frame', default_value='world',
                              description='reference frame to publish ground truth in'),
        Node(
            package='prob_rob_labs',
            executable='ground_truth_from_link_states',
            name='ground_truth_from_link_states',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'ref_frame': LaunchConfiguration('ref_frame')}
            ]
        )
    ])
