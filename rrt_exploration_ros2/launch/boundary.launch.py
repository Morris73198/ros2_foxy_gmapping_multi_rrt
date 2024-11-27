from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 只啟動一個簡單的静態TF和邊界節點
    
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'map']
    )

    boundary_node = Node(
        package='rrt_exploration_ros2',
        executable='boundary_node',
        name='exploration_boundary',
        output='screen'
    )

    return LaunchDescription([
        static_tf,
        boundary_node
    ])