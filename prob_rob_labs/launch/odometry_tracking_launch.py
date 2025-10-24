import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='set to true for simulation'),
        DeclareLaunchArgument('debug', default_value='false',
                              description='enable debug logging for node'),
        Node(
            package='prob_rob_labs',
            executable='odometry_tracking',
            name='odometry_tracking',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'imu_topic': '/imu',
                'joint_topic': '/joint_states',
                'cmd_topic': '/cmd_vel',
                'sync_queue': 20,
                'sync_slop': 0.05,
                'debug': LaunchConfiguration('debug')
            }]
        )
    ])
