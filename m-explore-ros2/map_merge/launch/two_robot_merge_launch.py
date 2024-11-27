import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Get the launch file for multi-robot simulation
    multirobot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('multirobot_map_merge'), 'launch'),
            '/multi_tb3_simulation_launch.py'
        ]),
        launch_arguments={'slam_gmapping': 'True'}.items()
    )

    # Get robot pose node
    get_robot_pose_node = Node(
        package='merge_map',
        executable='get_robot_pose',
        name='get_robot_pose',
        output='screen'
    )

    return LaunchDescription([
        # Start simulation first
        multirobot_launch,
        
        # Wait for 5 seconds then start get_robot_pose
        TimerAction(
            period=5.0,
            actions=[get_robot_pose_node]
        )
    ])