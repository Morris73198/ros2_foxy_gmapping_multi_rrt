import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Create launch description
    ld = LaunchDescription()

    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    # Path follow node for robot1
    path_follow_1 = Node(
        package='path_follow',
        executable='path_follow',
        name='path_follow_1',
        namespace='robot1',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        remappings=[
            ('path', '/robot1/path'),
            ('cmd_vel', '/robot1/cmd_vel'),
            ('odom', '/robot1/odom'),
            ('visual_path', '/robot1/visual_path'),
        ]
    )

    # Path follow node for robot2
    path_follow_2 = Node(
        package='path_follow',
        executable='path_follow',
        name='path_follow_2',
        namespace='robot2',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        remappings=[
            ('path', '/robot2/path'),
            ('cmd_vel', '/robot2/cmd_vel'),
            ('odom', '/robot2/odom'),
            ('visual_path', '/robot2/visual_path'),
        ]
    )

    # Add actions to launch description
    ld.add_action(use_sim_time_arg)
    ld.add_action(path_follow_1)
    ld.add_action(path_follow_2)

    return ld
