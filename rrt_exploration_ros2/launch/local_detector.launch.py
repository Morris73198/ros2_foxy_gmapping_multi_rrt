#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """生成啟動多個本地RRT探索器的launch描述"""
    
    # 創建第一個探索器節點 (tb3_0)
    detector_node_0 = Node(
        package='rrt_exploration_ros2',
        executable='local_detector_node',
        name='tb3_0_detector',
        parameters=[{
            'eta': 2.0,
            'robot_frame': 'tb3_0/base_footprint',
            'robot_name': 'tb3_0',
            'update_frequency': 15.0  # 降低頻率
        }],
        remappings=[
            ('/map', '/merge_map'),
            ('/robot_markers', '/tb3_0/robot_markers'),
            ('/detected_points', '/tb3_0/detected_points')
        ],
        arguments=['--ros-args', '--log-level', 'info'],
        output='screen'
    )

    # 創建第二個探索器節點 (tb3_1)，延遲1秒啟動
    detector_node_1 = TimerAction(
        period=0.25,
        actions=[Node(
            package='rrt_exploration_ros2',
            executable='local_detector_node',
            name='tb3_1_detector',
            parameters=[{
                'eta': 2.0,
                'robot_frame': 'tb3_1/base_footprint',
                'robot_name': 'tb3_1',
                'update_frequency': 15.0
            }],
            remappings=[
                ('/map', '/merge_map'),
                ('/robot_markers', '/tb3_1/robot_markers'),
                ('/detected_points', '/tb3_1/detected_points')
            ],
            arguments=['--ros-args', '--log-level', 'info'],
            output='screen'
        )]
    )

    # 創建第三個探索器節點 (tb3_2)，延遲2秒啟動
    detector_node_2 = TimerAction(
        period=0.5,
        actions=[Node(
            package='rrt_exploration_ros2',
            executable='local_detector_node',
            name='tb3_2_detector',
            parameters=[{
                'eta': 2.0,
                'robot_frame': 'tb3_2/base_footprint',
                'robot_name': 'tb3_2',
                'update_frequency': 15.0
            }],
            remappings=[
                ('/map', '/merge_map'),
                ('/robot_markers', '/tb3_2/robot_markers'),
                ('/detected_points', '/tb3_2/detected_points')
            ],
            arguments=['--ros-args', '--log-level', 'info'],
            output='screen'
        )]
    )

    return LaunchDescription([
        detector_node_0,
        detector_node_1,
        detector_node_2
    ])