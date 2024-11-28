#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import ColorRGBA

class GreedyAssigner(Node):
    def __init__(self):
        super().__init__('greedy_assigner')
        
        # 初始化變量
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        
        # 訂閱機器人位置
        self.robot1_pose_sub = self.create_subscription(
            PoseStamped,
            '/robot1_pose',
            self.robot1_pose_callback,
            10
        )
        
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped,
            '/robot2_pose',
            self.robot2_pose_callback,
            10
        )
        
        # 訂閱過濾後的點
        self.filtered_points_sub = self.create_subscription(
            MarkerArray,
            '/filtered_points',
            self.filtered_points_callback,
            10
        )
        
        # 發布目標點
        self.robot1_target_pub = self.create_publisher(
            PoseStamped,
            '/robot1/goal_pose',
            10
        )
        
        self.robot2_target_pub = self.create_publisher(
            PoseStamped,
            '/robot2/goal_pose',
            10
        )

        # 發布目標點可視化
        self.target_viz_pub = self.create_publisher(
            MarkerArray,
            '/assigned_targets_viz',
            10
        )

        # 發布調試信息
        self.debug_pub = self.create_publisher(
            String,
            '/assigner/debug',
            10
        )
        
        # 創建定時器，定期更新目標分配和可視化
        self.create_timer(1.0, self.assign_targets)
        self.create_timer(0.1, self.publish_visualization)
        
        self.get_logger().info('Greedy assigner node started')

    def create_target_marker(self, point, robot_name, marker_id):
        """創建目標點標記"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_target"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 設置位置
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # 設置大小
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        # 設置顏色
        if robot_name == 'robot1':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)  # 紅色
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)  # 綠色
            
        return marker

    def create_path_marker(self, start_pose, end_point, robot_name, marker_id):
        """創建路徑線標記"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_path"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 添加起點和終點
        start = Point(x=start_pose.position.x, y=start_pose.position.y, z=0.0)
        end = Point(x=end_point[0], y=end_point[1], z=0.0)
        marker.points = [start, end]
        
        # 設置線的寬度
        marker.scale.x = 0.1  # 線寬
        
        # 設置顏色
        if robot_name == 'robot1':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)  # 半透明紅色
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)  # 半透明綠色
            
        return marker

    def publish_visualization(self):
        """發布目標點和路徑的可視化"""
        if not all(self.assigned_targets.values()) or \
           self.robot1_pose is None or self.robot2_pose is None:
            return

        marker_array = MarkerArray()
        
        # 添加目標點標記
        if self.assigned_targets['robot1']:
            marker_array.markers.append(
                self.create_target_marker(
                    self.assigned_targets['robot1'], 
                    'robot1', 
                    0
                )
            )
            marker_array.markers.append(
                self.create_path_marker(
                    self.robot1_pose, 
                    self.assigned_targets['robot1'],
                    'robot1',
                    1
                )
            )
            
        if self.assigned_targets['robot2']:
            marker_array.markers.append(
                self.create_target_marker(
                    self.assigned_targets['robot2'], 
                    'robot2', 
                    2
                )
            )
            marker_array.markers.append(
                self.create_path_marker(
                    self.robot2_pose, 
                    self.assigned_targets['robot2'],
                    'robot2',
                    3
                )
            )
        
        self.target_viz_pub.publish(marker_array)

    # [其他方法保持不變...]
    def robot1_pose_callback(self, msg):
        """處理robot1的位置更新"""
        self.robot1_pose = msg.pose
        self.get_logger().debug('Received robot1 pose update')

    def robot2_pose_callback(self, msg):
        """處理robot2的位置更新"""
        self.robot2_pose = msg.pose
        self.get_logger().debug('Received robot2 pose update')

    def filtered_points_callback(self, msg):
        """處理過濾後的點"""
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])
        self.get_logger().debug(f'Received {len(self.available_points)} filtered points')

    def get_distance(self, pose, point):
        """計算姿態到點的距離"""
        return np.sqrt(
            (pose.position.x - point[0])**2 + 
            (pose.position.y - point[1])**2
        )

    def assign_targets(self):
        """使用貪婪算法分配目標"""
        if not self.available_points or self.robot1_pose is None or self.robot2_pose is None:
            return

        # 創建距離矩陣
        points = np.array(self.available_points)
        robots = {
            'robot1': self.robot1_pose,
            'robot2': self.robot2_pose
        }

        # 為每個機器人找最近的點
        assigned_points = set()
        for robot_name, robot_pose in robots.items():
            if not set(self.available_points) - assigned_points:
                break

            # 計算到所有未分配點的距離
            distances = [
                (point, self.get_distance(robot_pose, point))
                for point in self.available_points
                if tuple(point) not in assigned_points
            ]

            if distances:
                # 選擇最近的點
                closest_point = min(distances, key=lambda x: x[1])[0]
                assigned_points.add(tuple(closest_point))
                self.assigned_targets[robot_name] = closest_point

                # 發布目標點
                target_pose = PoseStamped()
                target_pose.header.frame_id = 'merge_map'
                target_pose.header.stamp = self.get_clock().now().to_msg()
                target_pose.pose.position.x = closest_point[0]
                target_pose.pose.position.y = closest_point[1]
                target_pose.pose.orientation.w = 1.0

                if robot_name == 'robot1':
                    self.robot1_target_pub.publish(target_pose)
                else:
                    self.robot2_target_pub.publish(target_pose)

                # 發布調試信息
                debug_msg = String()
                debug_msg.data = f'Assigned target {closest_point} to {robot_name}'
                self.debug_pub.publish(debug_msg)
                self.get_logger().info(debug_msg.data)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = GreedyAssigner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()