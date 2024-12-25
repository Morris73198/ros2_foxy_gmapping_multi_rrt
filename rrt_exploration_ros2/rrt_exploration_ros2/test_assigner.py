#!/usr/bin/env python3

# CNN + DQN one robot

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, ColorRGBA
import numpy as np
import tensorflow as tf
import cv2
import os
from rrt_exploration_ros2.network import FrontierNetworkModel

class DLAssigner(Node):
    def __init__(self):
        super().__init__('dl_assigner')
        
        # 初始化神經網路模型
        self.model = FrontierNetworkModel(
            input_shape=(84, 84, 1),
            max_frontiers=50
        )
        
        # 獲取package的安装路径
        import ament_index_python
        package_path = ament_index_python.get_package_share_directory('rrt_exploration_ros2')
        
        # 建構模型文件的完整路徑
        default_model_path = os.path.join(package_path, 'saved_models', 'frontier_model_ep000740.h5')
        model_path = self.declare_parameter('model_path', default_model_path).value
        
        self.get_logger().info(f'Loading model from: {model_path}')
        
        if not os.path.exists(model_path):
            error_msg = f'Model file not found at {model_path}'
            self.get_logger().error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            self.model.load(model_path)
            self.get_logger().info('Model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise
        
        # 初始化變量
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}
        
        # 機器人速度相關變量
        self.robot_velocities = {'robot1': None, 'robot2': None}
        self.velocity_check_threshold = 0.01  # 速度閾值，用於判斷機器人是否靜止
        self.static_duration = {'robot1': 0.0, 'robot2': 0.0}  # 記錄機器人靜止的持續時間
        self.static_threshold = 2.0  # 靜止超過此時間（秒）就重新分配目標
        
        # 地圖相關變量
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None
        
        # 目標到達閾值
        self.target_threshold = 0.3
        
        # 設置訂閱者
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/merge_map',
            self.map_callback,
            10
        )
        
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
        
        self.filtered_points_sub = self.create_subscription(
            MarkerArray,
            '/filtered_points',
            self.filtered_points_callback,
            10
        )
        
        # 訂閱各機器人的速度命令
        self.robot1_cmd_vel_sub = self.create_subscription(
            Twist,
            '/robot1/cmd_vel',
            lambda msg: self.cmd_vel_callback(msg, 'robot1'),
            10
        )
        
        self.robot2_cmd_vel_sub = self.create_subscription(
            Twist,
            '/robot2/cmd_vel',
            lambda msg: self.cmd_vel_callback(msg, 'robot2'),
            10
        )
            
        # 設置發布者
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
        
        self.target_viz_pub = self.create_publisher(
            MarkerArray,
            '/assigned_targets_viz',
            10
        )
        
        self.debug_pub = self.create_publisher(
            String,
            '/assigner/debug',
            10
        )
            
        # 創建定時器
        self.create_timer(1.0, self.assign_targets)
        self.create_timer(0.1, self.publish_visualization)
        self.create_timer(0.1, self.check_target_reached)
        self.create_timer(0.1, self.check_robot_motion)  # 檢查機器人運動狀態
        
        self.get_logger().info('深度學習frontier分配節點已啟動')

    def process_map(self, occupancy_grid):
        """處理地圖數據為模型輸入格式"""
        map_array = np.array(occupancy_grid)
        map_binary = np.zeros_like(map_array, dtype=np.uint8)
        map_binary[map_array == 0] = 255    # 已知空間
        map_binary[map_array == 100] = 0    # 障礙物
        map_binary[map_array == -1] = 127   # 未知空間
        
        resized_map = cv2.resize(map_binary, (84, 84), interpolation=cv2.INTER_LINEAR)
        resized_map = resized_map.astype(np.float32) / 255.0
        return np.expand_dims(resized_map, axis=-1)

    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度"""
        padded = np.zeros((50, 2))  # max_frontiers = 50
        if len(frontiers) > 0:
            normalized_frontiers = np.array(frontiers).copy()
            normalized_frontiers[:, 0] = normalized_frontiers[:, 0] / float(self.map_width)
            normalized_frontiers[:, 1] = normalized_frontiers[:, 1] / float(self.map_height)
            
            n_frontiers = min(len(frontiers), 50)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        return padded

    def get_normalized_position(self, pose):
        """獲取正規化後的機器人位置"""
        return np.array([
            pose.position.x / float(self.map_width * self.map_resolution),
            pose.position.y / float(self.map_height * self.map_resolution)
        ])

    def map_callback(self, msg):
        """處理地圖數據"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin
        self.processed_map = self.process_map(self.map_data)
        self.get_logger().debug('收到地圖更新')

    def robot1_pose_callback(self, msg):
        """處理機器人1的位置更新"""
        self.robot1_pose = msg.pose
        self.get_logger().debug('收到機器人1位置更新')

    def robot2_pose_callback(self, msg):
        """處理機器人2的位置更新"""
        self.robot2_pose = msg.pose
        self.get_logger().debug('收到機器人2位置更新')

    def filtered_points_callback(self, msg):
        """處理過濾後的frontier點"""
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])
        self.get_logger().debug(f'收到 {len(self.available_points)} 個frontier點')

    def cmd_vel_callback(self, msg: Twist, robot_name: str):
        """處理速度命令消息，更新機器人速度狀態"""
        # 計算線速度和角速度的總和
        total_velocity = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        self.robot_velocities[robot_name] = total_velocity

    def check_robot_motion(self):
        """檢查機器人是否靜止"""
        for robot_name in ['robot1', 'robot2']:
            if self.robot_velocities[robot_name] is None:
                continue
                
            # 檢查速度是否低於閾值
            if self.robot_velocities[robot_name] < self.velocity_check_threshold:
                self.static_duration[robot_name] += 0.1  # 增加靜止時間計數
                
                # 如果靜止時間超過閾值且當前有目標，強制設置為可用狀態
                if (self.static_duration[robot_name] >= self.static_threshold and 
                    not self.robot_status[robot_name] and 
                    self.assigned_targets[robot_name] is not None):
                    self.get_logger().info(f'{robot_name} 已靜止 {self.static_threshold} 秒，強制重新分配目標')
                    self.robot_status[robot_name] = True
                    self.assigned_targets[robot_name] = None
                    self.static_duration[robot_name] = 0.0  # 重置靜止時間
            else:
                # 如果有運動，重置靜止時間計數
                self.static_duration[robot_name] = 0.0

    def predict_best_frontier(self, robot_name):
        """使用神經網路預測最佳frontier點"""
        if not self.available_points:
            return None
            
        state = np.expand_dims(self.processed_map, 0)
        frontiers = np.expand_dims(self.pad_frontiers(self.available_points), 0)
        
        robot_pose = self.robot1_pose if robot_name == 'robot1' else self.robot2_pose
        robot_pos = np.expand_dims(self.get_normalized_position(robot_pose), 0)
        
        q_values = self.model.predict(state, frontiers, robot_pos)[0]
        valid_q = q_values[:len(self.available_points)]
        best_idx = np.argmax(valid_q)
        
        return self.available_points[best_idx]

    def check_target_reached(self):
        """检查機器人是否到達目標點"""
        for robot_name, robot_pose in [('robot1', self.robot1_pose), ('robot2', self.robot2_pose)]:
            if not robot_pose or not self.assigned_targets[robot_name]:
                continue

            current_pos = (robot_pose.position.x, robot_pose.position.y)
            target_pos = self.assigned_targets[robot_name]
            distance = np.sqrt(
                (current_pos[0] - target_pos[0])**2 + 
                (current_pos[1] - target_pos[1])**2
            )

            # 如果距離小於值閥值，認為已經達到目標
            if distance < self.target_threshold:
                if not self.robot_status[robot_name]:
                    self.get_logger().info(f'{robot_name} 已到達目標點 {target_pos}')
                self.robot_status[robot_name] = True
                self.assigned_targets[robot_name] = None
                
                # 重新分配
                self.assign_targets()
            else:
                self.robot_status[robot_name] = False

    def create_target_marker(self, point, robot_name, marker_id):
        """創建目標點標記"""
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
        
        marker.color = ColorRGBA(
            r=1.0 if robot_name == 'robot1' else 0.0,
            g=0.0 if robot_name == 'robot1' else 1.0,
            b=0.0,
            a=0.8
        )
        
        return marker

    def publish_visualization(self):
        """發布目標點的可視化"""
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

    def assign_targets(self):
        """分配目標給機器人"""
        if (len(self.available_points) == 0 or 
            self.processed_map is None or
            self.robot1_pose is None or 
            self.robot2_pose is None):
            return

        # 獲取已分配的點
        assigned_points = set()
        for robot_name in ['robot1', 'robot2']:
            if self.assigned_targets[robot_name] is not None:
                assigned_points.add(tuple(self.assigned_targets[robot_name]))

        # 移除與已分配的點太近的點
        available_points = []
        for point in self.available_points:
            too_close = False
            for assigned_point in assigned_points:
                dist = np.sqrt(
                    (point[0] - assigned_point[0])**2 + 
                    (point[1] - assigned_point[1])**2
                )
                if dist < 0.5:  
                    too_close = True
                    break
            if not too_close:
                available_points.append(point)

        if not available_points:
            return

        # 對每一台機器人進行目標分配
        for robot_name in ['robot1', 'robot2']:
            
            if not self.robot_status[robot_name] or self.assigned_targets[robot_name] is not None:
                continue

            # 對剩餘的點進行評估
            valid_points = []
            for point in available_points:
                if tuple(point) in assigned_points:
                    continue

                # 計算距離
                robot_pose = self.robot1_pose if robot_name == 'robot1' else self.robot2_pose
                dist = np.sqrt(
                    (point[0] - robot_pose.position.x)**2 + 
                    (point[1] - robot_pose.position.y)**2
                )
                
                # 如果距離太近就跳過
                if dist < 1.0:  
                    continue

                # 使用訓練的模型
                state = np.expand_dims(self.processed_map, 0)
                frontiers = np.expand_dims(self.pad_frontiers(available_points), 0)
                robot_pos = np.expand_dims(self.get_normalized_position(robot_pose), 0)
                q_values = self.model.predict(state, frontiers, robot_pos)[0]
                valid_points.append((point, q_values[available_points.index(point)]))

            if valid_points:
                # Q值最高的點
                best_point = max(valid_points, key=lambda x: x[1])[0]
                self.assigned_targets[robot_name] = best_point
                assigned_points.add(tuple(best_point))

                # 發布目標點
                target_pose = PoseStamped()
                target_pose.header.frame_id = 'merge_map'
                target_pose.header.stamp = self.get_clock().now().to_msg()
                target_pose.pose.position.x = best_point[0]
                target_pose.pose.position.y = best_point[1]
                target_pose.pose.orientation.w = 1.0

                # 根據機器人選擇發布給誰
                if robot_name == 'robot1':
                    self.robot1_target_pub.publish(target_pose)
                else:
                    self.robot2_target_pub.publish(target_pose)

            
                debug_msg = String()
                debug_msg.data = f'已將frontier點 {best_point} 分配给 {robot_name}'
                self.debug_pub.publish(debug_msg)
                self.get_logger().info(debug_msg.data)

def main(args=None):
    rclpy.init(args=args)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f'GPU配置錯誤: {e}')
    
    try:
        node = DLAssigner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()