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
        self.declare_parameter('safety_threshold', 70)
        self.declare_parameter('info_radius', 0.5)
        self.declare_parameter('safety_radius', 0.005)
        self.declare_parameter('bandwith_cluster', 0.3)
        self.declare_parameter('rate', 2.0)  # 降低處理頻率
        self.declare_parameter('process_interval', 1.0)  # 處理間隔(秒)
        
        # 獲取參數值
        self.map_topic = self.get_parameter('map_topic').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.info_radius = self.get_parameter('info_radius').value
        self.safety_radius = self.get_parameter('safety_radius').value
        self.bandwith = self.get_parameter('bandwith_cluster').value
        self.rate = self.get_parameter('rate').value
        self.process_interval = self.get_parameter('process_interval').value
        
        # 初始化變量
        self.mapData = None
        self.frontiers = []  # 存儲所有前沿點
        self.frame_id = 'merge_map'
        self.last_process_time = self.get_clock().now()
        self.assigned_points = set()  # 追蹤已分配的點
        
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
        
        # 添加訂閱已分配目標的話題
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
        
        self.get_logger().info('Filter node started')
        self.get_logger().info(f'Processing interval: {self.process_interval} seconds')

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
                        if np.linalg.norm(np.array(point_arr) - np.array(existing_point)) < 0.3:
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
                # 將已分配的點添加到集合中
                assigned_point = (
                    marker.pose.position.x,
                    marker.pose.position.y
                )
                self.assigned_points.add(assigned_point)
                
                # 從 frontiers 中移除已分配的點
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

    def check_safety(self, point):
        """檢查點的安全性"""
        if not self.mapData:
            return False
            
        resolution = self.mapData.info.resolution
        x = int((point[0] - self.mapData.info.origin.position.x) / resolution)
        y = int((point[1] - self.mapData.info.origin.position.y) / resolution)
        
        # 檢查範圍
        safety_cells = int(self.safety_radius / resolution)
        width = self.mapData.info.width
        height = self.mapData.info.height
        
        # 檢查周圍區域
        for dx in range(-safety_cells, safety_cells + 1):
            for dy in range(-safety_cells, safety_cells + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    index = ny * width + nx
                    if 0 <= index < len(self.mapData.data):
                        if self.mapData.data[index] >= self.safety_threshold:
                            return False
        return True

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
    
    
    
    
    
    def check_safety(self, point):
        """
        檢查點的安全性和可達性
        1. 確保點本身在安全距離內沒有障礙物
        2. 確保有一條全部經過已知自由空間的路徑可以到達未知區域
        
        Args:
            point: 要檢查的點 [x, y]
        Returns:
            bool: 如果點安全且可達則返回True，否則返回False
        """
        if not self.mapData:
            return False
            
        resolution = self.mapData.info.resolution
        x = int((point[0] - self.mapData.info.origin.position.x) / resolution)
        y = int((point[1] - self.mapData.info.origin.position.y) / resolution)
        width = self.mapData.info.width
        
        # 1. 基本邊界檢查
        if not (0 <= x < width and 0 <= y < self.mapData.info.height):
            return False
            
        # 2. 檢查點本身必須在已知的自由空間
        if self.mapData.data[y * width + x] != 0:  # 0表示自由空間
            return False
        
        # 3. 檢查安全距離內是否有障礙物
        safety_cells = int(self.safety_radius / resolution)
        for dx in range(-safety_cells, safety_cells + 1):
            for dy in range(-safety_cells, safety_cells + 1):
                check_x = x + dx
                check_y = y + dy
                
                if not (0 <= check_x < width and 0 <= check_y < self.mapData.info.height):
                    continue
                    
                cell_value = self.mapData.data[check_y * width + check_x]
                if cell_value > 50:  # 如果是障礙物
                    return False
        
        # 4. 檢查是否有一條全部經過已知自由空間的路徑可以到達未知區域
        found_valid_path = False
        for angle in np.linspace(0, 2*np.pi, 16):  # 檢查16個方向
            path_length = int(self.safety_radius * 1.5 / resolution)  # 稍微延長檢查距離
            path_valid = True
            reached_unknown = False
            
            # 沿著這個方向檢查每個點
            for dist in range(1, path_length + 1):
                check_x = int(x + dist * np.cos(angle))
                check_y = int(y + dist * np.sin(angle))
                
                # 檢查邊界
                if not (0 <= check_x < width and 0 <= check_y < self.mapData.info.height):
                    path_valid = False
                    break
                    
                cell_value = self.mapData.data[check_y * width + check_x]
                
                # 如果遇到未知區域，標記找到了未知區域並停止搜索
                if cell_value == -1:
                    reached_unknown = True
                    break
                    
                # 如果遇到障礙物或未知區域，這條路徑無效
                if cell_value != 0:  # 不是自由空間
                    path_valid = False
                    break
            
            # 如果這條路徑有效且到達了未知區域
            if path_valid and reached_unknown:
                found_valid_path = True
                break
        
        return found_valid_path

    def check_path_to_point(self, start_point, end_point):
        """
        檢查從起點到終點的路徑是否安全（只經過已知的自由空間）
        
        Args:
            start_point: 起點 [x, y]
            end_point: 終點 [x, y]
        Returns:
            bool: 如果路徑安全則返回True，否則返回False
        """
        if not self.mapData:
            return False
            
        resolution = self.mapData.info.resolution
        
        # 轉換為地圖坐標
        start_x = int((start_point[0] - self.mapData.info.origin.position.x) / resolution)
        start_y = int((start_point[1] - self.mapData.info.origin.position.y) / resolution)
        end_x = int((end_point[0] - self.mapData.info.origin.position.x) / resolution)
        end_y = int((end_point[1] - self.mapData.info.origin.position.y) / resolution)
        
        # 使用Bresenham算法檢查路徑上的每個點
        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)
        x = start_x
        y = start_y
        
        n = 1 + dx + dy
        x_inc = 1 if end_x > start_x else -1
        y_inc = 1 if end_y > start_y else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        
        for _ in range(n):
            # 檢查當前點是否在地圖範圍內
            if not (0 <= x < self.mapData.info.width and 0 <= y < self.mapData.info.height):
                return False
                
            # 檢查當前點是否是已知的自由空間
            cell_value = self.mapData.data[y * self.mapData.info.width + x]
            if cell_value != 0:  # 不是自由空間
                return False
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
                
        return True
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def filter_points(self):
        """過濾和聚類前沿點"""
        # 檢查是否達到處理間隔
        current_time = self.get_clock().now()
        if (current_time - self.last_process_time).nanoseconds / 1e9 < self.process_interval:
            return

        if len(self.frontiers) < 1 or not self.mapData:
            return
                
        try:
            # 更新處理時間戳
            self.last_process_time = current_time
            
            # 記錄開始處理的點數
            initial_points = len(self.frontiers)
            
            # 1. 首先移除已分配的點
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
            
            # 2. 發布原始前沿點（用於可視化）
            raw_marker_array = MarkerArray()
            raw_marker = Marker()
            raw_marker.header.frame_id = self.frame_id
            raw_marker.header.stamp = self.get_clock().now().to_msg()
            raw_marker.ns = "raw_frontiers"
            raw_marker.id = 0
            raw_marker.type = Marker.POINTS
            raw_marker.action = Marker.ADD
            raw_marker.pose.orientation.w = 1.0
            raw_marker.scale.x = 0.1
            raw_marker.scale.y = 0.1
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

            # 3. 安全性檢查 - 在聚類之前先進行初步篩選
            safe_frontiers = []
            for point in self.frontiers:
                if self.check_safety([point[0], point[1]]):
                    safe_frontiers.append(point)
                    
            if not safe_frontiers:
                self.get_logger().info('No safe frontiers found')
                return

            # 4. 執行聚類
            points_array = np.array(safe_frontiers)
            ms = MeanShift(bandwidth=self.bandwith)
            ms.fit(points_array)
            centroids = ms.cluster_centers_
            
            self.get_logger().info(f'Clustering {len(points_array)} points into {len(centroids)} centroids')
            
            # 5. 發布聚類中心（用於可視化）
            cluster_marker_array = MarkerArray()
            cluster_marker = Marker()
            cluster_marker.header.frame_id = self.frame_id
            cluster_marker.header.stamp = self.get_clock().now().to_msg()
            cluster_marker.ns = "cluster_centers"
            cluster_marker.id = 0
            cluster_marker.type = Marker.SPHERE_LIST
            cluster_marker.action = Marker.ADD
            cluster_marker.pose.orientation.w = 1.0
            cluster_marker.scale.x = 0.2
            cluster_marker.scale.y = 0.2
            cluster_marker.scale.z = 0.2
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

            # 6. 對聚類中心進行最終的安全性和信息增益檢查
            filtered_marker_array = MarkerArray()
            filtered_marker = Marker()
            filtered_marker.header.frame_id = self.frame_id
            filtered_marker.header.stamp = self.get_clock().now().to_msg()
            filtered_marker.ns = "filtered_frontiers"
            filtered_marker.id = 0
            filtered_marker.type = Marker.CUBE_LIST
            filtered_marker.action = Marker.ADD
            filtered_marker.pose.orientation.w = 1.0
            filtered_marker.scale.x = 0.3
            filtered_marker.scale.y = 0.3
            filtered_marker.scale.z = 0.3
            filtered_marker.color.r = 1.0
            filtered_marker.color.g = 1.0
            filtered_marker.color.b = 0.0
            filtered_marker.color.a = 0.8

            filtered_centroids = []
            for point in centroids:
                # 確保聚類中心也是安全的並且有足夠的信息增益
                if (self.check_safety(point) and 
                    self.calculate_info_gain(point) > 0.2):
                    # 檢查是否接近已分配的點
                    is_near_assigned = False
                    for assigned_point in self.assigned_points:
                        if self.is_point_near_assigned(point, assigned_point):
                            is_near_assigned = True
                            break
                    
                    if not is_near_assigned:
                        filtered_centroids.append(point)
                        p = Point()
                        p.x = float(point[0])
                        p.y = float(point[1])
                        p.z = 0.0
                        filtered_marker.points.append(p)
            
            filtered_marker_array.markers.append(filtered_marker)
            self.filtered_points_pub.publish(filtered_marker_array)
            
            self.get_logger().info(
                f'Processed {initial_points} points:'
                f' {len(centroids)} clusters,'
                f' {len(filtered_centroids)} filtered points'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in filter_points: {str(e)}')

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