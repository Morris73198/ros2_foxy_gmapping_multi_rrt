#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(
            name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
            value='[{severity}] [{name}]: {message}'
        ),
        
        SetEnvironmentVariable(
            name='RCUTILS_LOGGING_MIN_SEVERITY',
            value='DEBUG'
        ),
        
        DeclareLaunchArgument(
            'eta',
            default_value='0.5',
            description='RRT step size'
        ),
        
        DeclareLaunchArgument(
            'map_topic',
            default_value='/merge_map',
            description='Topic name for merged map'
        ),
        
        Node(
            package='rrt_exploration_ros2',
            executable='global_rrt_detector',
            name='global_rrt_detector',
            output='screen',
            parameters=[{
                'eta': LaunchConfiguration('eta'),
                'map_topic': LaunchConfiguration('map_topic')
            }]
        )
    ])