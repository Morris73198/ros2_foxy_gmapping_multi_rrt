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
    """發送狀態並接收目標，增強錯誤處理"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)  # 設定超時
        s.connect((host, port))
        
        # 發送資料
        data_to_send = json.dumps(state, ensure_ascii=False).encode('utf-8')
        s.sendall(data_to_send)
        
        # 接收資料 - 處理大型回應
        all_data = b''
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            all_data += chunk
            # 嘗試解析 JSON，如果完整就跳出
            try:
                target = json.loads(all_data.decode('utf-8'))
                break
            except json.JSONDecodeError:
                continue  # 繼續接收更多資料
                
        s.close()
        return target
    except Exception as e:
        print(f"Socket通訊錯誤: {e}")
        if 's' in locals():
            s.close()
        return {"target_point": None, "error": str(e)}

class SocketAssigner(Node):
    def __init__(self):
        super().__init__('socket_assigner')

        # 基本狀態
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}

        # 地圖相關
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None
        self.max_frontiers = 50
        
        # 目標到達檢查
        self.target_threshold = 0.5  # 機器人距離目標小於此值視為到達

        # ROS2 通訊
        self.setup_subscribers()
        self.setup_publishers()

        # 定時器 - 增加頻率以便更好的除錯
        self.create_timer(2.0, self.assign_targets)  # 每2秒嘗試分配
        self.create_timer(0.5, self.check_target_reached)  # 每0.5秒檢查到達
        self.create_timer(0.1, self.publish_visualization)
        self.create_timer(1.0, self.publish_debug_info)  # 新增：除錯資訊

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

    def publish_debug_info(self):
        """發布除錯資訊"""
        debug_msg = String()
        debug_info = {
            "robot1_pose": "OK" if self.robot1_pose else "MISSING",
            "robot2_pose": "OK" if self.robot2_pose else "MISSING", 
            "map_data": "OK" if self.map_data is not None else "MISSING",
            "processed_map": "OK" if self.processed_map is not None else "MISSING",
            "available_points": len(self.available_points),
            "robot1_status": self.robot_status['robot1'],
            "robot2_status": self.robot_status['robot2'],
            "robot1_target": self.assigned_targets['robot1'],
            "robot2_target": self.assigned_targets['robot2']
        }
        debug_msg.data = f"SocketAssigner狀態: {json.dumps(debug_info, ensure_ascii=False)}"
        self.debug_pub.publish(debug_msg)

    def map_callback(self, msg):
        """地圖回調，增強錯誤處理"""
        try:
            self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.map_resolution = msg.info.resolution
            self.map_width = msg.info.width
            self.map_height = msg.info.height
            self.map_origin = msg.info.origin

            # 轉成模型格式（84x84）
            map_array = self.map_data.copy()
            map_binary = np.zeros_like(map_array, dtype=np.uint8)
            map_binary[map_array == 0] = 255    # Free space -> 255
            map_binary[map_array == 100] = 0    # Obstacle -> 0
            map_binary[map_array == -1] = 127   # Unknown -> 127

            # 調整大小並正規化
            resized_map = cv2.resize(map_binary, (84, 84), interpolation=cv2.INTER_LINEAR)
            normalized_map = resized_map.astype(np.float32) / 255.0
            self.processed_map = np.expand_dims(normalized_map, axis=-1)
            
            self.get_logger().debug(f'地圖處理成功: {self.processed_map.shape}')
            
        except Exception as e:
            self.get_logger().error(f'地圖處理錯誤: {e}')
            self.processed_map = None

    def robot1_pose_callback(self, msg):
        self.robot1_pose = msg.pose
        self.get_logger().debug(f'收到robot1位置: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def robot2_pose_callback(self, msg):
        self.robot2_pose = msg.pose
        self.get_logger().debug(f'收到robot2位置: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def filtered_points_callback(self, msg):
        """處理過濾後的點"""
        old_count = len(self.available_points)
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])
        
        if len(self.available_points) != old_count:
            self.get_logger().info(f'更新frontier點: {old_count} -> {len(self.available_points)}')

    def check_target_reached(self):
        """檢查機器人是否到達目標點"""
        robots = {
            'robot1': self.robot1_pose,
            'robot2': self.robot2_pose
        }

        for robot_name, robot_pose in robots.items():
            if not robot_pose or not self.assigned_targets[robot_name]:
                continue

            target = self.assigned_targets[robot_name]
            current_pos = (robot_pose.position.x, robot_pose.position.y)
            target_pos = target

            # 計算距離
            distance = np.sqrt(
                (current_pos[0] - target_pos[0])**2 + 
                (current_pos[1] - target_pos[1])**2
            )

            # 如果到達目標
            if distance < self.target_threshold:
                if not self.robot_status[robot_name]:  # 之前是忙碌狀態
                    self.get_logger().info(f'{robot_name} 已到達目標點 {target_pos}，設為可用狀態')
                
                self.robot_status[robot_name] = True
                self.assigned_targets[robot_name] = None
            else:
                self.robot_status[robot_name] = False  # 標記為忙碌

    def assign_targets(self):
        """分配目標給機器人"""
        # 檢查前置條件
        if not self.available_points:
            self.get_logger().debug('沒有可用的frontier點')
            return
            
        if self.robot1_pose is None or self.robot2_pose is None:
            self.get_logger().debug('機器人位置資訊不完整')
            return
            
        if self.processed_map is None:
            self.get_logger().debug('地圖資料未處理完成')
            return

        self.get_logger().info(f'開始分配目標 - 可用點數: {len(self.available_points)}')

        # 紀錄已分配的點
        assigned_points = set()
        for robot, target in self.assigned_targets.items():
            if target is not None:
                assigned_points.add(tuple(target))

        # 組成狀態字典
        state = {
            "map": self.processed_map.tolist(),
            "frontiers": self.available_points,
            "robot1_pose": [
                self.robot1_pose.position.x, 
                self.robot1_pose.position.y
            ],
            "robot2_pose": [
                self.robot2_pose.position.x, 
                self.robot2_pose.position.y
            ]
        }

        # 為每個可用機器人分配目標
        for robot_name in ['robot1', 'robot2']:
            if not self.robot_status[robot_name]:
                self.get_logger().debug(f'{robot_name} 忙碌中，跳過分配')
                continue
                
            if self.assigned_targets[robot_name] is not None:
                self.get_logger().debug(f'{robot_name} 已有目標，跳過分配')
                continue

            self.get_logger().info(f'為 {robot_name} 請求RL目標分配...')

            try:
                # 發送請求到RL服務器
                target_result = send_state_and_get_target(state)
                
                if "error" in target_result:
                    self.get_logger().error(f'RL服務器錯誤: {target_result["error"]}')
                    continue
                
                best_point = target_result.get('target_point')
                
                if best_point is None:
                    self.get_logger().warn(f'RL服務器沒有回傳有效目標給 {robot_name}')
                    continue
                    
                if tuple(best_point) in assigned_points:
                    self.get_logger().warn(f'目標點 {best_point} 已被分配，跳過')
                    continue

                # 分配目標
                assigned_points.add(tuple(best_point))
                self.assigned_targets[robot_name] = best_point
                self.robot_status[robot_name] = False  # 設為忙碌

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

                # 發布除錯訊息
                debug_msg = String()
                debug_msg.data = f'RL分配: {robot_name} -> {best_point}'
                self.debug_pub.publish(debug_msg)
                self.get_logger().info(debug_msg.data)
                
            except Exception as e:
                self.get_logger().error(f'分配目標給 {robot_name} 時發生錯誤: {e}')

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
        if marker_array.markers:
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