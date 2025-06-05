#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, ColorRGBA
import numpy as np
import cv2
import socket
import json

def send_state_and_get_target(state, host='127.0.0.1', port=9000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(json.dumps(state).encode())
    data = s.recv(4096)
    target = json.loads(data.decode())
    s.close()
    return target

class SocketAssigner(Node):
    def __init__(self):
        super().__init__('socket_assigner')

        # 基本狀態
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}

        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None
        self.max_frontiers = 50

        # ROS2 通訊
        self.setup_subscribers()
        self.setup_publishers()

        self.create_timer(1.0, self.assign_targets)
        self.create_timer(0.1, self.publish_visualization)

        self.get_logger().info('Socket Assigner node started (using robot_rl server for target assignment)')

    def setup_subscribers(self):
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/merge_map', self.map_callback, 10)

        self.robot1_pose_sub = self.create_subscription(
            PoseStamped, '/robot1_pose', self.robot1_pose_callback, 10)

        self.robot2_pose_sub = self.create_subscription(
            PoseStamped, '/robot2_pose', self.robot2_pose_callback, 10)

        self.filtered_points_sub = self.create_subscription(
            MarkerArray, '/filtered_points', self.filtered_points_callback, 10)

    def setup_publishers(self):
        self.robot1_target_pub = self.create_publisher(
            PoseStamped, '/robot1/goal_pose', 10)

        self.robot2_target_pub = self.create_publisher(
            PoseStamped, '/robot2/goal_pose', 10)

        self.target_viz_pub = self.create_publisher(
            MarkerArray, '/assigned_targets_viz', 10)

        self.debug_pub = self.create_publisher(
            String, '/assigner/debug', 10)

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin

        # 轉成模型格式（假設要 84x84）
        try:
            map_array = self.map_data
            map_binary = np.zeros_like(map_array, dtype=np.uint8)
            map_binary[map_array == 0] = 255    # Free
            map_binary[map_array == 100] = 0    # Obstacle
            map_binary[map_array == -1] = 127   # Unknown

            resized_map = cv2.resize(map_binary, (84, 84), interpolation=cv2.INTER_LINEAR)
            normalized_map = resized_map.astype(np.float32) / 255.0
            processed_map = np.expand_dims(normalized_map, axis=-1)
            self.processed_map = processed_map
        except Exception as e:
            self.get_logger().warn(f'地圖處理錯誤: {e}')

    def robot1_pose_callback(self, msg):
        self.robot1_pose = msg.pose

    def robot2_pose_callback(self, msg):
        self.robot2_pose = msg.pose

    def filtered_points_callback(self, msg):
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])

    def assign_targets(self):
        if (not self.available_points or
            self.robot1_pose is None or
            self.robot2_pose is None):
            return

        assigned_points = set()
        for robot, target in self.assigned_targets.items():
            if target is not None:
                assigned_points.add(tuple(target))

        # 組成 state dict
        state = {
            "map": self.processed_map.tolist() if self.processed_map is not None else None,
            "frontiers": self.available_points,
            "robot1_pose": [
                self.robot1_pose.position.x, self.robot1_pose.position.y
            ] if self.robot1_pose else None,
            "robot2_pose": [
                self.robot2_pose.position.x, self.robot2_pose.position.y
            ] if self.robot2_pose else None
        }

        for robot_name in ['robot1', 'robot2']:
            if not self.robot_status[robot_name] or self.assigned_targets[robot_name] is not None:
                continue

            try:
                target_result = send_state_and_get_target(state)
                best_point = target_result.get('target_point')
                if best_point and tuple(best_point) not in assigned_points:
                    assigned_points.add(tuple(best_point))
                    self.assigned_targets[robot_name] = best_point

                    # 發布目標點
                    target_pose = PoseStamped()
                    target_pose.header.frame_id = 'merge_map'
                    target_pose.header.stamp = self.get_clock().now().to_msg()
                    target_pose.pose.position.x = best_point[0]
                    target_pose.pose.position.y = best_point[1]
                    target_pose.pose.orientation.w = 1.0

                    if robot_name == 'robot1':
                        self.robot1_target_pub.publish(target_pose)
                    else:
                        self.robot2_target_pub.publish(target_pose)

                    debug_msg = String()
                    debug_msg.data = f'已將目標點 {best_point} 分配給 {robot_name} (使用RL server)'
                    self.debug_pub.publish(debug_msg)
                    self.get_logger().info(debug_msg.data)
            except Exception as e:
                self.get_logger().warn(f'與 RL server 通訊失敗: {e}')

    def create_target_marker(self, point, robot_name, marker_id):
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_target"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.5
        if robot_name == 'robot1':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        return marker

    def publish_visualization(self):
        marker_array = MarkerArray()
        for robot_name in ['robot1', 'robot2']:
            if self.assigned_targets[robot_name]:
                marker_array.markers.append(
                    self.create_target_marker(
                        self.assigned_targets[robot_name],
                        robot_name,
                        len(marker_array.markers)
                    )
                )
        self.target_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = SocketAssigner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
    except Exception as e:
        print(f'錯誤: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            try:
                node.destroy_node()
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()