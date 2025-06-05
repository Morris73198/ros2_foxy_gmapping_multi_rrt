#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, ColorRGBA, Float32MultiArray
import numpy as np
import heapq
from typing import List, Tuple, Set
import cv2
import os
import sys

# 配置 TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 警告信息

# 嘗試導入 TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # 配置 GPU（如果可用）
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'GPU configured: {len(gpus)} devices found')
        except RuntimeError as e:
            print(f'GPU configuration error: {e}')
    else:
        print("No GPU detected, using CPU")
        
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    tf = None

# 嘗試導入 DRL 模型，使用多種可能的路徑
MultiRobotNetworkModel = None

def try_import_model():
    """嘗試從多個可能的路徑導入模型類"""
    global MultiRobotNetworkModel
    
    import_attempts = [
        # 嘗試從當前包導入
        'rrt_exploration_ros2.multi_robot_network',
        # 嘗試從 two_robot_dueling_dqn_attention 包導入
        'two_robot_dueling_dqn_attention.models.multi_robot_network',
        # 嘗試從當前目錄導入
        'multi_robot_network',
    ]
    
    for module_path in import_attempts:
        try:
            module = __import__(module_path, fromlist=['MultiRobotNetworkModel'])
            MultiRobotNetworkModel = getattr(module, 'MultiRobotNetworkModel')
            print(f"Successfully imported MultiRobotNetworkModel from {module_path}")
            return True
        except (ImportError, AttributeError) as e:
            print(f"Failed to import from {module_path}: {e}")
            continue
    
    print("Warning: Unable to import MultiRobotNetworkModel from any location")
    print("Will use distance-based assignment only")
    return False

# 嘗試導入模型
try_import_model()

class DRLAssigner(Node):
    def __init__(self):
        super().__init__('drl_assigner')
        
        # 檢查依賴項
        self.tf_available = tf is not None
        self.model_class_available = MultiRobotNetworkModel is not None
        
        # DRL模型相關參數
        self.map_size_for_model = (84, 84)  # 模型輸入的地圖尺寸
        self.max_frontiers = 50  # 最大frontier點數量
        
        # 聲明參數
        self.declare_parameter('model_path', '')
        self.declare_parameter('use_drl', True)
        self.declare_parameter('model_dir', '/home/airlab/rrt_ws/src/ros2_foxy_gmapping_multi_rrt/rrt_exploration_ros2/rrt_exploration_ros2/saved_models/')
        
        # 獲取參數
        self.model_path = self.get_parameter('model_path').value
        self.use_drl = self.get_parameter('use_drl').value and self.tf_available and self.model_class_available
        self.model_dir = self.get_parameter('model_dir').value
        
        # 如果沒有指定模型路徑，嘗試自動找到
        if not self.model_path:
            self.model_path = self.find_model_file()
        
        # 初始化變量
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}
        
        # 機器人速度相關變量
        self.robot_velocities = {'robot1': None, 'robot2': None}
        self.velocity_check_threshold = 0.01
        self.static_duration = {'robot1': 0.0, 'robot2': 0.0}
        self.static_threshold = 2.0
        
        # 地圖相關變量
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None
        
        # 目標到達閾值
        self.target_threshold = 0.3
        
        # 協調參數
        self.MIN_TARGET_DISTANCE = 1.5  # 機器人之間的最小目標距離
        
        # 初始化DRL模型
        self.drl_model = None
        if self.use_drl:
            self.load_drl_model()
        
        # 設置訂閱者和發布者
        self.setup_subscribers()
        self.setup_publishers()
        
        # 創建定時器
        self.create_timer(1.0, self.assign_targets)
        self.create_timer(0.1, self.publish_visualization)
        self.create_timer(0.1, self.check_target_reached)
        self.create_timer(0.1, self.check_robot_motion)
        
        # 輸出狀態信息
        mode = "DRL增強" if self.use_drl and self.drl_model else "距離基礎"
        self.get_logger().info(f'{mode} 分配節點已啟動')
        
        if not self.tf_available:
            self.get_logger().warn('TensorFlow不可用，僅使用距離基礎分配')
        elif not self.model_class_available:
            self.get_logger().warn('MultiRobotNetworkModel類不可用，僅使用距離基礎分配')
        elif self.use_drl and self.drl_model:
            self.get_logger().info(f'DRL模型已載入: {self.model_path}')
        elif self.use_drl:
            self.get_logger().warn('DRL模型載入失敗，使用距離基礎分配')

    def find_model_file(self):
        """自動尋找可用的模型文件"""
        possible_files = [
            'multi_robot_model_attention.h5',
            'robot_rl_model.h5',
            'best_model.h5',
            'final_model.h5'
        ]
        
        # 檢查模型目錄
        if os.path.exists(self.model_dir):
            # 首先查找特定名稱的文件
            for filename in possible_files:
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    self.get_logger().info(f'找到模型文件: {filepath}')
                    return filepath
            
            # 如果沒找到，列出所有.h5文件
            h5_files = [f for f in os.listdir(self.model_dir) if f.endswith('.h5')]
            if h5_files:
                # 按修改時間排序，選擇最新的
                h5_files_with_time = []
                for f in h5_files:
                    filepath = os.path.join(self.model_dir, f)
                    mtime = os.path.getmtime(filepath)
                    h5_files_with_time.append((filepath, mtime))
                
                h5_files_with_time.sort(key=lambda x: x[1], reverse=True)
                latest_file = h5_files_with_time[0][0]
                
                self.get_logger().info(f'使用最新的模型文件: {latest_file}')
                self.get_logger().info(f'可用的模型文件: {h5_files}')
                return latest_file
            else:
                self.get_logger().warn(f'在 {self.model_dir} 中未找到.h5模型文件')
        else:
            self.get_logger().warn(f'模型目錄不存在: {self.model_dir}')
        
        return ''

    def load_drl_model(self):
        """載入預訓練的DRL模型"""
        if not self.tf_available:
            self.get_logger().error('TensorFlow不可用，無法載入DRL模型')
            self.use_drl = False
            return
            
        if not self.model_class_available:
            self.get_logger().error('MultiRobotNetworkModel類不可用，無法載入DRL模型')
            self.use_drl = False
            return
        
        if not self.model_path or not os.path.exists(self.model_path):
            self.get_logger().error(f'模型文件不存在: {self.model_path}')
            self.use_drl = False
            return
        
        try:
            # 初始化模型
            self.drl_model = MultiRobotNetworkModel(
                input_shape=(84, 84, 1),
                max_frontiers=self.max_frontiers
            )
            
            # 載入預訓練權重
            self.drl_model.load(self.model_path)
            
            self.get_logger().info(f'成功載入DRL模型: {self.model_path}')
            
        except Exception as e:
            self.get_logger().error(f'載入DRL模型失敗: {str(e)}')
            self.get_logger().info('將使用距離基礎分配方法')
            self.drl_model = None
            self.use_drl = False

    def setup_subscribers(self):
        """設置所有訂閱者"""
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/merge_map', self.map_callback, 10)
        
        self.robot1_pose_sub = self.create_subscription(
            PoseStamped, '/robot1_pose', self.robot1_pose_callback, 10)
        
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped, '/robot2_pose', self.robot2_pose_callback, 10)
        
        self.filtered_points_sub = self.create_subscription(
            MarkerArray, '/filtered_points', self.filtered_points_callback, 10)
        
        # 訂閱速度命令
        self.robot1_cmd_vel_sub = self.create_subscription(
            Twist, '/robot1/cmd_vel', 
            lambda msg: self.cmd_vel_callback(msg, 'robot1'), 10)
        
        self.robot2_cmd_vel_sub = self.create_subscription(
            Twist, '/robot2/cmd_vel', 
            lambda msg: self.cmd_vel_callback(msg, 'robot2'), 10)

    def setup_publishers(self):
        """設置所有發布者"""
        self.robot1_target_pub = self.create_publisher(
            PoseStamped, '/robot1/goal_pose', 10)
        
        self.robot2_target_pub = self.create_publisher(
            PoseStamped, '/robot2/goal_pose', 10)

        self.target_viz_pub = self.create_publisher(
            MarkerArray, '/assigned_targets_viz', 10)

        self.debug_pub = self.create_publisher(
            String, '/assigner/debug', 10)

    def process_map_for_model(self, occupancy_grid):
        """處理地圖數據為DRL模型輸入格式"""
        if occupancy_grid is None:
            return None
            
        try:
            map_array = np.array(occupancy_grid)
            map_binary = np.zeros_like(map_array, dtype=np.uint8)
            
            # 地圖值映射
            map_binary[map_array == 0] = 255    # 自由空間
            map_binary[map_array == 100] = 0    # 障礙物
            map_binary[map_array == -1] = 127   # 未知空間
            
            # 調整尺寸
            resized_map = cv2.resize(
                map_binary, self.map_size_for_model, 
                interpolation=cv2.INTER_LINEAR)
            
            # 正規化並添加通道維度
            normalized_map = resized_map.astype(np.float32) / 255.0
            processed_map = np.expand_dims(normalized_map, axis=-1)
            
            return processed_map
            
        except Exception as e:
            self.get_logger().error(f'地圖處理錯誤: {str(e)}')
            return None

    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度並進行標準化"""
        padded = np.zeros((self.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            # 標準化座標
            if self.map_width and self.map_height and self.map_resolution and self.map_origin:
                map_width_m = self.map_width * self.map_resolution
                map_height_m = self.map_height * self.map_resolution
                
                normalized_frontiers = frontiers.copy()
                normalized_frontiers[:, 0] = (frontiers[:, 0] - self.map_origin.position.x) / map_width_m
                normalized_frontiers[:, 1] = (frontiers[:, 1] - self.map_origin.position.y) / map_height_m
                
                # 確保在[0,1]範圍內
                normalized_frontiers = np.clip(normalized_frontiers, 0.0, 1.0)
            else:
                normalized_frontiers = frontiers
            
            n_frontiers = min(len(frontiers), self.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded

    def get_normalized_position(self, pose):
        """獲取正規化後的機器人位置"""
        if not pose or not all([self.map_width, self.map_height, self.map_resolution, self.map_origin]):
            return np.array([0.0, 0.0])
            
        map_width_m = self.map_width * self.map_resolution
        map_height_m = self.map_height * self.map_resolution
        
        normalized_x = (pose.position.x - self.map_origin.position.x) / map_width_m
        normalized_y = (pose.position.y - self.map_origin.position.y) / map_height_m
        
        return np.array([
            np.clip(normalized_x, 0.0, 1.0),
            np.clip(normalized_y, 0.0, 1.0)
        ])

    def get_normalized_target(self, target):
        """標準化目標位置"""
        if target is None or not all([self.map_width, self.map_height, self.map_resolution, self.map_origin]):
            return np.array([0.0, 0.0])
            
        map_width_m = self.map_width * self.map_resolution
        map_height_m = self.map_height * self.map_resolution
        
        normalized_x = (target[0] - self.map_origin.position.x) / map_width_m
        normalized_y = (target[1] - self.map_origin.position.y) / map_height_m
        
        return np.array([
            np.clip(normalized_x, 0.0, 1.0),
            np.clip(normalized_y, 0.0, 1.0)
        ])

    def map_callback(self, msg):
        """處理地圖數據"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin
        
        # 處理地圖供DRL模型使用
        if self.use_drl:
            self.processed_map = self.process_map_for_model(self.map_data)
        
        self.get_logger().debug('地圖數據已更新')

    def robot1_pose_callback(self, msg):
        """處理機器人1的位置更新"""
        self.robot1_pose = msg.pose

    def robot2_pose_callback(self, msg):
        """處理機器人2的位置更新"""
        self.robot2_pose = msg.pose

    def filtered_points_callback(self, msg):
        """處理過濾後的點"""
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])

    def cmd_vel_callback(self, msg: Twist, robot_name: str):
        """處理速度命令消息"""
        total_velocity = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        self.robot_velocities[robot_name] = total_velocity

    def check_robot_motion(self):
        """檢查機器人是否靜止"""
        for robot_name in ['robot1', 'robot2']:
            if self.robot_velocities[robot_name] is None:
                continue
                
            if self.robot_velocities[robot_name] < self.velocity_check_threshold:
                self.static_duration[robot_name] += 0.1
                
                if (self.static_duration[robot_name] >= self.static_threshold and 
                    not self.robot_status[robot_name]):
                    self.get_logger().info(f'{robot_name} 已靜止 {self.static_threshold} 秒，重新分配')
                    self.robot_status[robot_name] = True
                    self.assigned_targets[robot_name] = None
            else:
                self.static_duration[robot_name] = 0.0

    def check_target_reached(self):
        """檢查機器人是否到達目標點"""
        for robot_name, robot_pose in [('robot1', self.robot1_pose), ('robot2', self.robot2_pose)]:
            if not robot_pose or not self.assigned_targets[robot_name]:
                continue

            target = self.assigned_targets[robot_name]
            current_pos = (robot_pose.position.x, robot_pose.position.y)

            distance = np.sqrt(
                (current_pos[0] - target[0])**2 + 
                (current_pos[1] - target[1])**2
            )

            if distance < self.target_threshold:
                if not self.robot_status[robot_name]:
                    self.get_logger().info(f'{robot_name} 已到達目標點')
                self.robot_status[robot_name] = True
                self.assigned_targets[robot_name] = None
            else:
                self.robot_status[robot_name] = False

    def calculate_distance(self, point1, point2):
        """計算兩點間距離"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_utility_score(self, robot_pose, target_point, other_robot_pose=None, other_target=None):
        """計算目標點的效用分數"""
        if not robot_pose:
            return float('-inf')
        
        robot_pos = (robot_pose.position.x, robot_pose.position.y)
        
        # 距離分數（距離越近分數越高）
        distance = self.calculate_distance(robot_pos, target_point)
        distance_score = 1.0 / (1.0 + distance)
        
        # 避免衝突分數
        conflict_penalty = 0.0
        
        if other_robot_pose and other_target:
            other_pos = (other_robot_pose.position.x, other_robot_pose.position.y)
            
            # 如果另一個機器人更接近這個目標，給予懲罰
            other_distance = self.calculate_distance(other_pos, target_point)
            if other_distance < distance:
                conflict_penalty += 0.3
            
            # 如果目標太接近另一個機器人的目標，給予懲罰
            target_distance = self.calculate_distance(target_point, other_target)
            if target_distance < self.MIN_TARGET_DISTANCE:
                conflict_penalty += 0.5
        
        return distance_score - conflict_penalty

    def distance_based_assignment(self, robot_name, available_points, assigned_points):
        """基於距離和效用的智能分配方法"""
        robot_pose = self.robot1_pose if robot_name == 'robot1' else self.robot2_pose
        other_robot_pose = self.robot2_pose if robot_name == 'robot1' else self.robot1_pose
        other_robot_name = 'robot2' if robot_name == 'robot1' else 'robot1'
        other_target = self.assigned_targets.get(other_robot_name)
        
        if not robot_pose:
            return None
            
        valid_targets = []
        MIN_DISTANCE = 0.8  # 最小距離要求
        
        for point in available_points:
            if tuple(point) in assigned_points:
                continue
                
            distance = self.calculate_distance(
                (robot_pose.position.x, robot_pose.position.y), point)
            
            if distance >= MIN_DISTANCE:
                # 計算效用分數
                utility_score = self.calculate_utility_score(
                    robot_pose, point, other_robot_pose, other_target)
                valid_targets.append((point, utility_score, distance))
        
        if valid_targets:
            # 選擇效用分數最高的點
            best_target = max(valid_targets, key=lambda x: x[1])
            return best_target[0]
            
        return None

    def predict_best_frontier_with_drl(self, robot_name):
        """使用DRL模型預測最佳frontier點"""
        if (not self.use_drl or 
            self.drl_model is None or 
            self.processed_map is None or 
            len(self.available_points) == 0):
            return None
            
        try:
            # 準備輸入數據
            state = np.expand_dims(self.processed_map, 0)
            frontiers = np.expand_dims(self.pad_frontiers(self.available_points), 0)
            
            # 獲取機器人位置
            robot1_pos = self.get_normalized_position(self.robot1_pose)
            robot2_pos = self.get_normalized_position(self.robot2_pose)
            robot1_pos_batch = np.expand_dims(robot1_pos, 0)
            robot2_pos_batch = np.expand_dims(robot2_pos, 0)
            
            # 獲取當前目標位置
            robot1_target = self.get_normalized_target(self.assigned_targets.get('robot1'))
            robot2_target = self.get_normalized_target(self.assigned_targets.get('robot2'))
            robot1_target_batch = np.expand_dims(robot1_target, 0)
            robot2_target_batch = np.expand_dims(robot2_target, 0)
            
            # 使用DRL模型進行預測
            predictions = self.drl_model.predict(
                state, frontiers,
                robot1_pos_batch, robot2_pos_batch,
                robot1_target_batch, robot2_target_batch
            )
            
            # 提取對應機器人的Q值
            valid_frontiers = min(self.max_frontiers, len(self.available_points))
            if robot_name == 'robot1':
                q_values = predictions['robot1'][0, :valid_frontiers]
            else:
                q_values = predictions['robot2'][0, :valid_frontiers]
            
            # 選擇Q值最高的動作
            best_action = np.argmax(q_values)
            
            # 確保索引有效
            if best_action < len(self.available_points):
                best_frontier = self.available_points[best_action]
                return best_frontier
            else:
                self.get_logger().warn(f'DRL模型返回無效索引: {best_action}')
                return None
            
        except Exception as e:
            self.get_logger().error(f'DRL預測錯誤: {str(e)}')
            return None

    def create_target_marker(self, point: Tuple[float, float], robot_name: str, marker_id: int) -> Marker:
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
        
        if robot_name == 'robot1':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            
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
        """主要分配邏輯"""
        if (not self.available_points or 
            self.robot1_pose is None or 
            self.robot2_pose is None):
            return

        # 記錄已分配的點
        assigned_points = set()
        for robot, target in self.assigned_targets.items():
            if target is not None:
                assigned_points.add(tuple(target))

        for robot_name in ['robot1', 'robot2']:
            # 只在機器人可用且沒有當前目標時分配新目標
            if not self.robot_status[robot_name] or self.assigned_targets[robot_name] is not None:
                continue

            # 過濾掉已分配的點
            available_points = [
                point for point in self.available_points 
                if tuple(point) not in assigned_points
            ]
            
            if not available_points:
                continue

            # 嘗試使用DRL模型
            best_point = None
            method_used = "距離基礎"
            
            if self.use_drl and self.drl_model:
                best_point = self.predict_best_frontier_with_drl(robot_name)
                if best_point and tuple(best_point) not in assigned_points:
                    method_used = "DRL模型"
                else:
                    best_point = None
            
            # 如果DRL失敗，使用距離基礎方法
            if best_point is None:
                best_point = self.distance_based_assignment(robot_name, available_points, assigned_points)
            
            if best_point is not None:
                # 確保選擇的點還沒有被分配
                if tuple(best_point) not in assigned_points:
                    assigned_points.add(tuple(best_point))
                    self.assigned_targets[robot_name] = best_point

                    # 創建並發布目標點消息
                    target_pose = PoseStamped()
                    target_pose.header.frame_id = 'merge_map'
                    target_pose.header.stamp = self.get_clock().now().to_msg()
                    target_pose.pose.position.x = best_point[0]
                    target_pose.pose.position.y = best_point[1]
                    target_pose.pose.orientation.w = 1.0

                    # 發布目標點
                    if robot_name == 'robot1':
                        self.robot1_target_pub.publish(target_pose)
                    else:
                        self.robot2_target_pub.publish(target_pose)

                    # 發布調試信息
                    debug_msg = String()
                    debug_msg.data = f'已將目標點 {best_point} 分配給 {robot_name} (使用{method_used})'
                    self.debug_pub.publish(debug_msg)
                    self.get_logger().info(debug_msg.data)
            else:
                self.get_logger().warn(f'未找到 {robot_name} 的有效目標')


def main(args=None):
    """主函數"""
    rclpy.init(args=args)
    
    try:
        node = DRLAssigner()
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