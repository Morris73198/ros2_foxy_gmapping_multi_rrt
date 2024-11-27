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
        self.declare_parameter('info_radius', 1.0)
        self.declare_parameter('safety_radius', 0.3)
        self.declare_parameter('bandwith_cluster', 0.3)
        self.declare_parameter('rate', 2.0)  # 降低處理頻率
        self.declare_parameter('process_interval', 2.0)  # 處理間隔(秒)
        
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
            
            # 發布原始前沿點
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

            # 執行聚類
            points_array = np.array(self.frontiers)
            ms = MeanShift(bandwidth=self.bandwith)
            ms.fit(points_array)
            centroids = ms.cluster_centers_
            
            self.get_logger().info(f'Clustering {len(points_array)} points into {len(centroids)} centroids')
            
            # 發布聚類中心
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

            # 過濾和發布最終結果
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
            filtered_marker.color.a = 0.878787

            filtered_centroids = []
            for point in centroids:
                if self.check_safety(point) and self.calculate_info_gain(point) > 0.2:
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
            
            # 保留未過濾的點以供下次處理
            # self.frontiers = []  # 如果你想清除所有點，取消註釋此行
            
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