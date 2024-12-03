#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
    GroupAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get directories
    multirobot_map_merge_dir = get_package_share_directory('multirobot_map_merge')
    merge_map_dir = get_package_share_directory('merge_map')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    # First: Launch the simulation with SLAM
    multi_tb3_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(multirobot_map_merge_dir, 'launch', 'tb3_simulation', 'multi_tb3_simulation_launch.py')
        ),
        launch_arguments={
            'slam_gmapping': 'True'
        }.items()
    )
    
    # Second: Robot pose node
    robot_pose_node = Node(
        package='merge_map',
        executable='get_robot_pose',
        name='get_robot_pose',
        output='screen'
    )
    
    # Add delay for robot pose node
    delayed_robot_pose = TimerAction(
        period=5.0,  # 5 seconds delay
        actions=[robot_pose_node]
    )
    
    # Third: Map merge node
    map_merge_node = Node(
        package='merge_map',
        executable='merge_map',
        name='merge_map',
        output='screen'
    )
    
    # Add delay for map merge node
    delayed_map_merge = TimerAction(
        period=8.0,  # 8 seconds delay
        actions=[map_merge_node]
    )
    
    # RViz configuration
    rviz_config_file = os.path.join(
        nav2_bringup_dir,
        'rviz',
        'nav2_namespaced_view.rviz'
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()
    
    # Add actions
    ld.add_action(multi_tb3_launch)
    ld.add_action(delayed_robot_pose)
    #ld.add_action(delayed_map_merge)
    ld.add_action(rviz_node)

    return ld
