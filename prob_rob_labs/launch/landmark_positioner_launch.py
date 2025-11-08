from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    lc = LaunchConfiguration('landmark_color')
    lh = LaunchConfiguration('landmark_height')
    use_sim_time = LaunchConfiguration('use_sim_time')
    csv_file = LaunchConfiguration('csv_file')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='set to true for simulation'),
        DeclareLaunchArgument('landmark_color', default_value=TextSubstitution(text='cyan'), description='color of the landmark to identify'),
        DeclareLaunchArgument('landmark_height', default_value=TextSubstitution(text='0.5'), description='height of the landmark (meters)'),
        # csv_file default uses the workspace package source pictures dir to avoid
        # writing into installed package share or system paths. This can still be
        # overridden at launch time by passing csv_file:=<path>.
        DeclareLaunchArgument(
            'csv_file',
            default_value=[TextSubstitution(text='/home/hqh/ros2_ws/src/prob_rob_labs_ros_2/pictures/landmark_data_'), lc, TextSubstitution(text='.csv')],
            description='path to CSV file to write measurements'
        ),

        Node(
            package='prob_rob_labs',
            executable='landmark_positioner',
            name='landmark_positioner',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'landmark_color': lc,
                'landmark_height': lh,
                'record_data': True,
                'csv_file': csv_file,
            }]
        ),
    ])