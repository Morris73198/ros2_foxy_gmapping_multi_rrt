#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
import threading

class RobotPoseNode(Node):
    def __init__(self):
        super().__init__('robot_pose_node')
        
        # 初始化 TF 数据字典
        self.tf_data = {
            'robot1': None,
            'robot2': None
        }
        self.tf_lock = threading.Lock()
        
        # 创建位置发布器
        self.pose_publishers = {
            'robot1': self.create_publisher(PoseStamped, '/robot1_pose', 10),
            'robot2': self.create_publisher(PoseStamped, '/robot2_pose', 10)
        }
        
        # 创建 TF 订阅器
        self.tf_subs = {}
        for robot_name in ['robot1', 'robot2']:
            self.get_logger().info(f'Subscribing to /{robot_name}/tf')
            self.tf_subs[robot_name] = self.create_subscription(
                TFMessage,
                f'/{robot_name}/tf',
                lambda msg, name=robot_name: self.tf_callback(msg, name),
                10
            )
        
        # 创建定时器进行定期更新
        self.timer = self.create_timer(0.1, self.publish_poses)  # 10Hz
        
        self.get_logger().info('Robot pose node initialized')

    def tf_callback(self, msg, robot_name):
        """
        处理 TF 消息
        """
        with self.tf_lock:
            for transform in msg.transforms:
                if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_footprint':
                    self.tf_data[robot_name] = transform
                    # self.get_logger().info(
                    #     f'Updated {robot_name} pose: '
                    #     f'x={transform.transform.translation.x:.2f}, '
                    #     f'y={transform.transform.translation.y:.2f}'
                    # )

    def publish_poses(self):
        """
        发布机器人位置
        """
        with self.tf_lock:
            for robot_name, transform in self.tf_data.items():
                if transform is not None:
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = "odom"
                    
                    # 设置位置
                    pose_msg.pose.position.x = transform.transform.translation.x
                    pose_msg.pose.position.y = transform.transform.translation.y
                    pose_msg.pose.position.z = 0.0
                    
                    # 设置方向
                    pose_msg.pose.orientation = transform.transform.rotation
                    
                    # 发布位置
                    self.pose_publishers[robot_name].publish(pose_msg)
                    # self.get_logger().info(
                    #     f'Published {robot_name} pose: '
                    #     f'x={pose_msg.pose.position.x:.2f}, '
                    #     f'y={pose_msg.pose.position.y:.2f}'
                    # )

def main(args=None):
    rclpy.init(args=args)
    node = RobotPoseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
