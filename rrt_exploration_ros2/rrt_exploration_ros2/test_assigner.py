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
        
        # 機器人運動狀態追蹤
        self.robot_last_pose = {'robot1': None, 'robot2': None}
        self.robot_static_time = {'robot1': 0.0, 'robot2': 0.0}
        self.robot_last_move_time = {'robot1': self.get_clock().now(), 'robot2': self.get_clock().now()}
        
        # 參數設定
        self.static_threshold = 5.0  # 靜止超過5秒重新分配
        self.movement_threshold = 0.1  # 移動距離閾值
        self.target_threshold = 0.5  # 到達目標距離閾值
        
        # 地圖相關
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

        # 定時器
        self.create_timer(2.0, self.assign_targets)  # 每2秒檢查分配
        self.create_timer(0.5, self.check_robot_status)  # 每0.5秒檢查機器人狀態
        self.create_timer(0.1, self.publish_visualization)
        self.create_timer(1.0, self.publish_debug_info)

        self.get_logger().info('Smart Socket Assigner started - 監控機器人運動狀態')

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

    def check_robot_status(self):
        """檢查機器人狀態：是否到達目標、是否靜止太久"""
        current_time = self.get_clock().now()
        
        robots = {
            'robot1': self.robot1_pose,
            'robot2': self.robot2_pose
        }
        
        for robot_name, current_pose in robots.items():
            if current_pose is None:
                continue
                
            current_pos = [current_pose.position.x, current_pose.position.y]
            
            # 檢查1：是否到達目標
            if self.assigned_targets[robot_name] is not None:
                target_pos = self.assigned_targets[robot_name]
                distance_to_target = np.sqrt(
                    (current_pos[0] - target_pos[0])**2 + 
                    (current_pos[1] - target_pos[1])**2
                )
                
                if distance_to_target < self.target_threshold:
                    self.get_logger().info(f'{robot_name} 已到達目標點，清除目標')
                    self.assigned_targets[robot_name] = None
                    self.robot_static_time[robot_name] = 0.0
                    self.robot_last_move_time[robot_name] = current_time
                    continue
            
            # 檢查2：是否移動（靜止檢測）
            if self.robot_last_pose[robot_name] is not None:
                last_pos = [
                    self.robot_last_pose[robot_name].position.x,
                    self.robot_last_pose[robot_name].position.y
                ]
                
                movement_distance = np.sqrt(
                    (current_pos[0] - last_pos[0])**2 + 
                    (current_pos[1] - last_pos[1])**2
                )
                
                if movement_distance > self.movement_threshold:
                    # 機器人有移動，重置靜止時間
                    self.robot_static_time[robot_name] = 0.0
                    self.robot_last_move_time[robot_name] = current_time
                else:
                    # 機器人沒有移動，累積靜止時間
                    time_diff = (current_time - self.robot_last_move_time[robot_name]).nanoseconds / 1e9
                    self.robot_static_time[robot_name] = time_diff
                    
                    # 如果靜止太久且有目標，清除目標重新分配
                    if (self.robot_static_time[robot_name] > self.static_threshold and 
                        self.assigned_targets[robot_name] is not None):
                        self.get_logger().info(
                            f'{robot_name} 靜止 {self.robot_static_time[robot_name]:.1f}秒，清除目標重新分配'
                        )
                        self.assigned_targets[robot_name] = None
                        self.robot_static_time[robot_name] = 0.0
                        self.robot_last_move_time[robot_name] = current_time
            
            # 更新上次位置
            self.robot_last_pose[robot_name] = current_pose

    def assign_targets(self):
        """智能分配目標 - 確保不分配相同點"""
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

        # 檢查哪些機器人需要新目標
        need_assignment = []
        for robot_name in ['robot1', 'robot2']:
            if self.assigned_targets[robot_name] is None:
                need_assignment.append(robot_name)
        
        if not need_assignment:
            return

        self.get_logger().info(f'需要分配目標: {need_assignment}, 可用frontier: {len(self.available_points)}')

        # 如果可用frontier數量少於需要分配的機器人數量
        if len(self.available_points) < len(need_assignment):
            self.get_logger().warn(f'frontier數量({len(self.available_points)})少於需要分配的機器人數量({len(need_assignment)})')
            # 按距離排序，優先分配給較近的機器人
            need_assignment = self.sort_robots_by_distance_to_frontiers(need_assignment)

        # 組成狀態字典
        state = {
            "map": self.processed_map.tolist(),
            "frontiers": self.available_points,
            "robot1_pose": [self.robot1_pose.position.x, self.robot1_pose.position.y],
            "robot2_pose": [self.robot2_pose.position.x, self.robot2_pose.position.y]
        }

        try:
            self.get_logger().info('向RL服務器請求目標分配...')
            target_result = send_state_and_get_target(state)
            
            if "error" in target_result:
                self.get_logger().error(f'RL服務器錯誤: {target_result["error"]}')
                return
            
            # 智能分配目標，確保不重複
            self.smart_assign_targets(target_result, need_assignment)
                
        except Exception as e:
            self.get_logger().error(f'分配目標時發生錯誤: {e}')

    def smart_assign_targets(self, target_result, need_assignment):
        """智能分配目標，確保不重複"""
        assigned_points = set()
        
        # 檢查回傳格式
        if "robot1_target" in target_result and "robot2_target" in target_result:
            # 多機器人格式
            potential_targets = {
                "robot1": target_result["robot1_target"],
                "robot2": target_result["robot2_target"]
            }
        else:
            # 單目標格式，需要分別請求
            potential_targets = {}
            for robot_name in need_assignment:
                single_state = {
                    "map": self.processed_map.tolist(),
                    "frontiers": self.available_points,
                    "robot1_pose": [self.robot1_pose.position.x, self.robot1_pose.position.y],
                    "robot2_pose": [self.robot2_pose.position.x, self.robot2_pose.position.y],
                    "request_robot": robot_name
                }
                single_result = send_state_and_get_target(single_state)
                potential_targets[robot_name] = single_result.get('target_point')

        # 智能分配邏輯：避免衝突
        final_assignments = {}
        
        # 第一輪：分配不衝突的目標
        for robot_name in need_assignment:
            if robot_name in potential_targets:
                target = potential_targets[robot_name]
                if target and tuple(target) not in assigned_points:
                    final_assignments[robot_name] = target
                    assigned_points.add(tuple(target))

        # 第二輪：為沒有分配到目標的機器人找替代目標
        unassigned_robots = [r for r in need_assignment if r not in final_assignments]
        available_frontiers = [p for p in self.available_points if tuple(p) not in assigned_points]
        
        for i, robot_name in enumerate(unassigned_robots):
            if i < len(available_frontiers):
                alternative_target = available_frontiers[i]
                final_assignments[robot_name] = alternative_target
                assigned_points.add(tuple(alternative_target))
                self.get_logger().info(f'為 {robot_name} 分配替代目標: {alternative_target}')

        # 發布分配結果
        for robot_name, target in final_assignments.items():
            self.publish_target_to_robot(robot_name, target)

    def sort_robots_by_distance_to_frontiers(self, robot_list):
        """按機器人到frontier的平均距離排序"""
        robot_distances = []
        
        for robot_name in robot_list:
            robot_pose = getattr(self, f'{robot_name}_pose')
            robot_pos = [robot_pose.position.x, robot_pose.position.y]
            
            # 計算到所有frontier的平均距離
            distances = []
            for frontier in self.available_points:
                dist = np.sqrt(
                    (robot_pos[0] - frontier[0])**2 + 
                    (robot_pos[1] - frontier[1])**2
                )
                distances.append(dist)
            
            avg_distance = np.mean(distances) if distances else float('inf')
            robot_distances.append((robot_name, avg_distance))
        
        # 按距離排序（距離近的優先）
        robot_distances.sort(key=lambda x: x[1])
        return [robot_name for robot_name, _ in robot_distances]

    def publish_target_to_robot(self, robot_name, target):
        """發布目標點給機器人"""
        self.assigned_targets[robot_name] = target
        
        # 創建目標訊息
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'merge_map'
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = target[0]
        target_pose.pose.position.y = target[1]
        target_pose.pose.orientation.w = 1.0

        # 發布到對應的topic
        if robot_name == 'robot1':
            self.robot1_target_pub.publish(target_pose)
        else:
            self.robot2_target_pub.publish(target_pose)

        # 發布除錯訊息
        debug_msg = String()
        debug_msg.data = f'分配目標: {robot_name} -> {target}'
        self.debug_pub.publish(debug_msg)
        self.get_logger().info(debug_msg.data)

    def publish_debug_info(self):
        """發布詳細除錯資訊"""
        debug_msg = String()
        debug_info = {
            "robot1_pose": "OK" if self.robot1_pose else "MISSING",
            "robot2_pose": "OK" if self.robot2_pose else "MISSING", 
            "map_data": "OK" if self.map_data is not None else "MISSING",
            "processed_map": "OK" if self.processed_map is not None else "MISSING",
            "available_points": len(self.available_points),
            "robot1_target": self.assigned_targets['robot1'],
            "robot2_target": self.assigned_targets['robot2'],
            "robot1_static_time": f"{self.robot_static_time['robot1']:.1f}s",
            "robot2_static_time": f"{self.robot_static_time['robot2']:.1f}s"
        }
        debug_msg.data = f"智能分配器狀態: {json.dumps(debug_info, ensure_ascii=False)}"
        self.debug_pub.publish(debug_msg)

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