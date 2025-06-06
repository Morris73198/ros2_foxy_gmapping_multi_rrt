#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from sklearn.cluster import MeanShift

class FilterNode(Node):
    def __init__(self):
        super().__init__('filter')
        
        # 聲明參數
        self.declare_parameter('map_topic', '/merge_map')
        self.declare_parameter('safety_threshold', 90)  # 提高到90，只避免確定的障礙物
        self.declare_parameter('info_radius', 0.5)
        self.declare_parameter('safety_radius', 0.002)  # 大幅減少安全半徑，允許窄縫
        self.declare_parameter('bandwith_cluster', 0.3)
        self.declare_parameter('rate', 2.0)
        self.declare_parameter('process_interval', 1.0)
        self.declare_parameter('narrow_passage_mode', True)  # 新增：窄縫模式
        
        # 獲取參數值
        self.map_topic = self.get_parameter('map_topic').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.info_radius = self.get_parameter('info_radius').value
        self.safety_radius = self.get_parameter('safety_radius').value
        self.bandwith = self.get_parameter('bandwith_cluster').value
        self.rate = self.get_parameter('rate').value
        self.process_interval = self.get_parameter('process_interval').value
        self.narrow_passage_mode = self.get_parameter('narrow_passage_mode').value
        
        # 初始化變量
        self.mapData = None
        self.frontiers = []
        self.frame_id = 'merge_map'
        self.last_process_time = self.get_clock().now()
        self.assigned_points = set()
        
        # 訂閱者
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )
        
        self.markers_sub = self.create_subscription(
            MarkerArray,
            '/found',
            self.markers_callback,
            10
        )
        
        self.assigned_targets_sub = self.create_subscription(
            MarkerArray,
            '/assigned_targets_viz',
            self.assigned_targets_callback,
            10
        )
        
        # 發布者
        self.filtered_points_pub = self.create_publisher(
            MarkerArray,
            'filtered_points',
            10
        )
        
        self.raw_points_pub = self.create_publisher(
            MarkerArray,
            'raw_frontiers',
            10
        )
        
        self.cluster_centers_pub = self.create_publisher(
            MarkerArray,
            'cluster_centers',
            10
        )
        
        # 創建定時器
        self.create_timer(1.0/self.rate, self.filter_points)
        
        self.get_logger().info('Narrow Passage Friendly Filter node started')
        self.get_logger().info(f'Narrow passage mode: {self.narrow_passage_mode}')
        self.get_logger().info(f'Safety radius: {self.safety_radius}, Safety threshold: {self.safety_threshold}')

    def map_callback(self, msg):
        """地圖數據回調"""
        self.mapData = msg
        self.frame_id = msg.header.frame_id
        self.get_logger().debug('Received map update')

    def markers_callback(self, msg):
        """處理前沿點標記"""
        try:
            for marker in msg.markers:
                for point in marker.points:
                    point_arr = [point.x, point.y]
                    # 檢查是否已存在該點（考慮一定的容差）
                    is_new = True
                    for existing_point in self.frontiers:
                        if np.linalg.norm(np.array(point_arr) - np.array(existing_point)) < 0.2:  # 減少容差
                            is_new = False
                            break
                    if is_new:
                        self.frontiers.append(point_arr)
            
            self.get_logger().debug(f'Current frontiers count: {len(self.frontiers)}')
                
        except Exception as e:
            self.get_logger().error(f'Error in markers_callback: {str(e)}')

    def assigned_targets_callback(self, msg):
        """處理已分配的目標點"""
        try:
            for marker in msg.markers:
                assigned_point = (
                    marker.pose.position.x,
                    marker.pose.position.y
                )
                self.assigned_points.add(assigned_point)
                
                self.frontiers = [
                    point for point in self.frontiers 
                    if not self.is_point_near_assigned(point, assigned_point)
                ]
                
            self.get_logger().debug(f'Updated assigned points: {len(self.assigned_points)}')
                    
        except Exception as e:
            self.get_logger().error(f'Error in assigned_targets_callback: {str(e)}')

    def is_point_near_assigned(self, point, assigned_point, threshold=0.5):
        """檢查點是否接近已分配的點"""
        return np.linalg.norm(
            np.array(point) - np.array(assigned_point)
        ) < threshold

    def check_safety_narrow_passage_friendly(self, point):
        """
        窄縫友好的安全性檢查
        - 對窄縫更寬容
        - 只檢查很小的區域
        - 允許通過窄通道
        """
        if not self.mapData:
            return False
            
        resolution = self.mapData.info.resolution
        x = int((point[0] - self.mapData.info.origin.position.x) / resolution)
        y = int((point[1] - self.mapData.info.origin.position.y) / resolution)
        width = self.mapData.info.width
        height = self.mapData.info.height
        
        # 邊界檢查
        if not (0 <= x < width and 0 <= y < height):
            return False
            
        # 檢查點本身必須在自由空間
        center_value = self.mapData.data[y * width + x]
        if center_value != 0:  # 0表示自由空間
            return False
        
        if not self.narrow_passage_mode:
            # 原始的安全檢查
            safety_cells = int(self.safety_radius / resolution)
            for dx in range(-safety_cells, safety_cells + 1):
                for dy in range(-safety_cells, safety_cells + 1):
                    nx = x + dx
                    ny = y + dy
                    if (0 <= nx < width and 0 <= ny < height):
                        cell_value = self.mapData.data[ny * width + nx]
                        if cell_value >= self.safety_threshold:
                            return False
        else:
            # 窄縫友好檢查：只檢查緊鄰的4個點（上下左右）
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
            obstacle_count = 0
            total_checked = 0
            
            for dx, dy in directions:
                nx = x + dx
                ny = y + dy
                if (0 <= nx < width and 0 <= ny < height):
                    cell_value = self.mapData.data[ny * width + nx]
                    total_checked += 1
                    if cell_value >= self.safety_threshold:
                        obstacle_count += 1
            
            # 如果4個方向都被障礙物包圍，才認為不安全
            # 這允許窄縫中的frontier保留下來
            if obstacle_count >= total_checked:
                return False
                
        return True

    def check_narrow_passage_accessibility(self, point):
        """
        檢查窄縫中的點是否可達
        使用更寬鬆的標準，特別適合窄通道
        """
        if not self.mapData:
            return False
            
        resolution = self.mapData.info.resolution
        x = int((point[0] - self.mapData.info.origin.position.x) / resolution)
        y = int((point[1] - self.mapData.info.origin.position.y) / resolution)
        width = self.mapData.info.width
        height = self.mapData.info.height
        
        # 檢查8個方向，看是否至少有2個方向可以通行
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        free_directions = 0
        
        for dx, dy in directions:
            path_clear = True
            # 檢查這個方向上的幾個點
            for step in range(1, 4):  # 檢查3步距離
                nx = x + dx * step
                ny = y + dy * step
                
                if not (0 <= nx < width and 0 <= ny < height):
                    path_clear = False
                    break
                    
                cell_value = self.mapData.data[ny * width + nx]
                if cell_value >= self.safety_threshold:
                    path_clear = False
                    break
                    
            if path_clear:
                free_directions += 1
                
        # 如果至少有2個方向可以通行，認為是可達的
        return free_directions >= 2

    def calculate_info_gain(self, point):
        """計算信息增益"""
        if not self.mapData:
            return 0
            
        info_gain = 0
        resolution = self.mapData.info.resolution
        info_cells = int(self.info_radius / resolution)
        
        x = int((point[0] - self.mapData.info.origin.position.x) / resolution)
        y = int((point[1] - self.mapData.info.origin.position.y) / resolution)
        width = self.mapData.info.width
        
        for dx in range(-info_cells, info_cells + 1):
            for dy in range(-info_cells, info_cells + 1):
                nx = x + dx
                ny = y + dy
                if (0 <= nx < width and 
                    0 <= ny < self.mapData.info.height):
                    index = ny * width + nx
                    if index < len(self.mapData.data):
                        if self.mapData.data[index] == -1:
                            info_gain += 1
                            
        return info_gain * (resolution ** 2)

    def filter_points(self):
        """過濾和聚類前沿點 - 窄縫友好版本"""
        current_time = self.get_clock().now()
        if (current_time - self.last_process_time).nanoseconds / 1e9 < self.process_interval:
            return

        if len(self.frontiers) < 1 or not self.mapData:
            return
                
        try:
            self.last_process_time = current_time
            initial_points = len(self.frontiers)
            
            # 1. 移除已分配的點
            filtered_frontiers = []
            for point in self.frontiers:
                is_assigned = False
                for assigned_point in self.assigned_points:
                    if self.is_point_near_assigned(point, assigned_point):
                        is_assigned = True
                        break
                if not is_assigned:
                    filtered_frontiers.append(point)
                        
            self.frontiers = filtered_frontiers
            
            # 2. 發布原始前沿點
            self.publish_raw_points()

            # 3. 窄縫友好的安全性檢查
            safe_frontiers = []
            narrow_passage_frontiers = []
            
            for point in self.frontiers:
                if self.check_safety_narrow_passage_friendly(point):
                    # 進一步檢查窄縫可達性
                    if self.narrow_passage_mode and self.check_narrow_passage_accessibility(point):
                        narrow_passage_frontiers.append(point)
                    safe_frontiers.append(point)
                    
            if not safe_frontiers:
                self.get_logger().info('No safe frontiers found')
                return

            self.get_logger().info(f'Found {len(narrow_passage_frontiers)} narrow passage friendly frontiers out of {len(safe_frontiers)} safe frontiers')

            # 4. 執行聚類（使用較小的bandwidth以保留更多細節）
            points_array = np.array(safe_frontiers)
            cluster_bandwidth = self.bandwith * 0.7 if self.narrow_passage_mode else self.bandwith  # 窄縫模式下減少聚類半徑
            ms = MeanShift(bandwidth=cluster_bandwidth)
            ms.fit(points_array)
            centroids = ms.cluster_centers_
            
            self.get_logger().info(f'Clustering {len(points_array)} points into {len(centroids)} centroids')
            
            # 5. 發布聚類中心
            self.publish_cluster_centers(centroids)

            # 6. 最終過濾
            filtered_centroids = []
            for point in centroids:
                # 更寬鬆的信息增益要求
                min_info_gain = 0.1 if self.narrow_passage_mode else 0.2
                
                if (self.check_safety_narrow_passage_friendly(point) and 
                    self.calculate_info_gain(point) > min_info_gain):
                    
                    # 檢查是否接近已分配的點
                    is_near_assigned = False
                    for assigned_point in self.assigned_points:
                        if self.is_point_near_assigned(point, assigned_point):
                            is_near_assigned = True
                            break
                    
                    if not is_near_assigned:
                        filtered_centroids.append(point)
            
            # 7. 發布最終結果
            self.publish_filtered_points(filtered_centroids)
            
            self.get_logger().info(
                f'Processed {initial_points} points: '
                f'{len(centroids)} clusters, '
                f'{len(filtered_centroids)} final points '
                f'(narrow passage mode: {self.narrow_passage_mode})'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in filter_points: {str(e)}')

    def publish_raw_points(self):
        """發布原始前沿點"""
        raw_marker_array = MarkerArray()
        raw_marker = Marker()
        raw_marker.header.frame_id = self.frame_id
        raw_marker.header.stamp = self.get_clock().now().to_msg()
        raw_marker.ns = "raw_frontiers"
        raw_marker.id = 0
        raw_marker.type = Marker.POINTS
        raw_marker.action = Marker.ADD
        raw_marker.pose.orientation.w = 1.0
        raw_marker.scale.x = 0.08
        raw_marker.scale.y = 0.08
        raw_marker.color.r = 1.0
        raw_marker.color.g = 1.0
        raw_marker.color.b = 0.0
        raw_marker.color.a = 0.5

        for point in self.frontiers:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            raw_marker.points.append(p)
        
        raw_marker_array.markers.append(raw_marker)
        self.raw_points_pub.publish(raw_marker_array)

    def publish_cluster_centers(self, centroids):
        """發布聚類中心"""
        cluster_marker_array = MarkerArray()
        cluster_marker = Marker()
        cluster_marker.header.frame_id = self.frame_id
        cluster_marker.header.stamp = self.get_clock().now().to_msg()
        cluster_marker.ns = "cluster_centers"
        cluster_marker.id = 0
        cluster_marker.type = Marker.SPHERE_LIST
        cluster_marker.action = Marker.ADD
        cluster_marker.pose.orientation.w = 1.0
        cluster_marker.scale.x = 0.15
        cluster_marker.scale.y = 0.15
        cluster_marker.scale.z = 0.15
        cluster_marker.color.r = 1.0
        cluster_marker.color.g = 0.0
        cluster_marker.color.b = 1.0
        cluster_marker.color.a = 0.7

        for point in centroids:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            cluster_marker.points.append(p)
        
        cluster_marker_array.markers.append(cluster_marker)
        self.cluster_centers_pub.publish(cluster_marker_array)

    def publish_filtered_points(self, filtered_centroids):
        """發布過濾後的點"""
        filtered_marker_array = MarkerArray()
        filtered_marker = Marker()
        filtered_marker.header.frame_id = self.frame_id
        filtered_marker.header.stamp = self.get_clock().now().to_msg()
        filtered_marker.ns = "filtered_frontiers"
        filtered_marker.id = 0
        filtered_marker.type = Marker.CUBE_LIST
        filtered_marker.action = Marker.ADD
        filtered_marker.pose.orientation.w = 1.0
        filtered_marker.scale.x = 0.25
        filtered_marker.scale.y = 0.25
        filtered_marker.scale.z = 0.25
        
        # 窄縫模式下使用不同顏色
        if self.narrow_passage_mode:
            filtered_marker.color.r = 0.0
            filtered_marker.color.g = 1.0
            filtered_marker.color.b = 1.0  # 青色表示窄縫友好模式
        else:
            filtered_marker.color.r = 1.0
            filtered_marker.color.g = 1.0
            filtered_marker.color.b = 0.0  # 黃色表示標準模式
            
        filtered_marker.color.a = 0.9

        for point in filtered_centroids:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            filtered_marker.points.append(p)
        
        filtered_marker_array.markers.append(filtered_marker)
        self.filtered_points_pub.publish(filtered_marker_array)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = FilterNode()
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