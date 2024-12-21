from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 使用絕對路徑
    default_model_path = os.path.join(os.path.expanduser('~'), 'try', 'src', 'rrt_exploration_ros2', 'models', 'best.global')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='Path to .global model file'
        ),
        
        Node(
            package='rrt_exploration_ros2',
            executable='global_policy_node',
            name='global_policy_node',
            parameters=[{
                'global_model_path': LaunchConfiguration('model_path'),
                'map_topic': '/merge_map',
                'robot1_pose_topic': '/robot1_pose',
                'robot2_pose_topic': '/robot2_pose',
                'map_size': 480,
                'downscaling': 4,
                'update_frequency': 1.0
            }],
            output='screen'
        )
    ])