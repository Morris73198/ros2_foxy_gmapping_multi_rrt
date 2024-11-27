#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """Generate launch description for filter node."""
    
    # Declare the launch arguments
    declare_rate = DeclareLaunchArgument(
        'rate',
        default_value='100',
        description='Filter node update rate in Hz'
    )
    
    declare_map_topic = DeclareLaunchArgument(
        'map_topic',
        default_value='/merge_map',
        description='Topic name for merged map'
    )
    
    declare_safety_threshold = DeclareLaunchArgument(
        'safety_threshold',
        default_value='70',
        description='Safety threshold for obstacle detection'
    )
    
    declare_info_radius = DeclareLaunchArgument(
        'info_radius',
        default_value='1.0',
        description='Radius for information gain calculation'
    )
    
    declare_safety_radius = DeclareLaunchArgument(
        'safety_radius',
        default_value='0.3',
        description='Radius for safety checking'
    )
    
    declare_bandwith = DeclareLaunchArgument(
        'bandwith_cluster',
        default_value='0.3',
        description='Bandwidth parameter for mean shift clustering'
    )
    
    # Create the filter node
    filter_node = Node(
        package='rrt_exploration_ros2',
        executable='filter_node',
        name='filter',
        output='screen',
        parameters=[{
            'rate': LaunchConfiguration('rate'),
            'map_topic': LaunchConfiguration('map_topic'),
            'safety_threshold': LaunchConfiguration('safety_threshold'),
            'info_radius': LaunchConfiguration('info_radius'),
            'safety_radius': LaunchConfiguration('safety_radius'),
            'bandwith_cluster': LaunchConfiguration('bandwith_cluster')
        }]
    )
    
    return LaunchDescription([
        declare_rate,
        declare_map_topic,
        declare_safety_threshold,
        declare_info_radius,
        declare_safety_radius,
        declare_bandwith,
        filter_node
    ])