from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Robot1 RRT Detector Node
    robot1_detector = Node(
        package='rrt_exploration_ros2',
        executable='local_detector_node',
        name='robot1_detector',
        parameters=[{
            'robot_name': 'robot1'
        }]
    )
    
    # Robot2 RRT Detector Node
    robot2_detector = Node(
        package='rrt_exploration_ros2',
        executable='local_detector_node',
        name='robot2_detector',
        parameters=[{
            'robot_name': 'robot2'
        }]
    )
    
    return LaunchDescription([
        robot1_detector,
        robot2_detector
    ])