#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PointStamped, PolygonStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
import cv2
import traceback

class GridFrontierDetector(Node):
    def __init__(self):
        super().__init__('grid_frontier_detector')
        
        # 參數聲明
        self.declare_parameter('map_topic', '/merge_map')
        self.declare_parameter('detection_frequency', 2.0)
        # 移除不需要的參數
        # self.declare_parameter('min_frontier_size', 3)
        # self.declare_parameter('frontier_detection_radius', 2)
        # self.declare_parameter('safety_margin', 1)
        # self.declare_parameter('min_distance_threshold', 0.4)
        
        # 獲取參數值
        self.map_topic = self.get_parameter('map_topic').value
        self.detection_frequency = self.get_parameter('detection_frequency').value
        
        # 初始化變量
        self.mapData = None
        self.boundary = None
        self.boundary_received = False
        self.frontiers = []
        self.frontier_count = 0
        
        # 訂閱者
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10
        )
        
        self.boundary_sub = self.create_subscription(
            PolygonStamped,
            '/exploration_boundary',
            self.boundary_callback,
            10
        )
        
        # 發布者
        self.frontier_pub = self.create_publisher(
            PointStamped,
            '/detected_points',
            10
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/visualization_marker_array',
            10
        )
        
        self.found_frontiers_pub = self.create_publisher(
            MarkerArray,
            '/found',
            10
        )
        
        # 初始化可視化標記
        self.init_markers()
        
        # 創建定時器
        self.create_timer(1.0 / self.detection_frequency, self.detect_frontiers)
        
        self.get_logger().info('Grid Frontier Detector initialized - OUTPUT ALL FRONTIERS')
        self.get_logger().info('Waiting for map and boundary...')

    def init_markers(self):
        """初始化可視化標記"""
        self.marker_array = MarkerArray()
        
        # Frontier標記
        self.frontier_marker = Marker()
        self.frontier_marker.header.frame_id = 'map'
        self.frontier_marker.ns = "grid_frontiers"
        self.frontier_marker.id = 0
        self.frontier_marker.type = Marker.SPHERE_LIST
        self.frontier_marker.action = Marker.ADD
        self.frontier_marker.pose.orientation.w = 1.0
        self.frontier_marker.scale.x = 0.2
        self.frontier_marker.scale.y = 0.2
        self.frontier_marker.scale.z = 0.2
        self.frontier_marker.color.r = 1.0
        self.frontier_marker.color.g = 0.0
        self.frontier_marker.color.b = 0.0
        self.frontier_marker.color.a = 0.8
        self.frontier_marker.points = []
        self.frontier_marker.lifetime = Duration(seconds=0).to_msg()
        
        # Found frontiers標記
        self.found_marker = Marker()
        self.found_marker.header.frame_id = 'merge_map'
        self.found_marker.ns = "found_frontiers"
        self.found_marker.id = 0
        self.found_marker.type = Marker.SPHERE_LIST
        self.found_marker.action = Marker.ADD
        self.found_marker.pose.orientation.w = 1.0
        self.found_marker.scale.x = 0.2
        self.found_marker.scale.y = 0.2
        self.found_marker.scale.z = 0.2
        self.found_marker.color.r = 1.0
        self.found_marker.color.g = 0.5
        self.found_marker.color.b = 0.0
        self.found_marker.color.a = 0.8
        self.found_marker.points = []

    def map_callback(self, msg):
        """處理地圖數據"""
        if self.mapData is None:
            self.get_logger().info('First map data received')
            self.get_logger().info(f'Map size: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}')
        
        self.mapData = msg
        
        # 更新標記的frame_id
        self.frontier_marker.header.frame_id = msg.header.frame_id
        self.found_marker.header.frame_id = msg.header.frame_id

    def boundary_callback(self, msg):
        """處理邊界數據"""
        if not self.boundary_received:
            self.boundary = msg.polygon.points
            self.boundary_received = True
            self.get_logger().info('Received exploration boundary')
            self.get_logger().info('Starting grid-based frontier detection - NO FILTERING...')

    def is_point_in_boundary(self, point):
        """檢查點是否在探索邊界內"""
        if not self.boundary:
            return True
        
        inside = False
        j = len(self.boundary) - 1
        
        for i in range(len(self.boundary)):
            if ((self.boundary[i].y > point[1]) != (self.boundary[j].y > point[1]) and
                point[0] < (self.boundary[j].x - self.boundary[i].x) * 
                (point[1] - self.boundary[i].y) / 
                (self.boundary[j].y - self.boundary[i].y) + 
                self.boundary[i].x):
                inside = not inside
            j = i
        
        return inside

    def is_frontier_cell(self, x, y, map_data, width, height):
        """檢查格子是否為frontier點"""
        if not (0 <= x < width and 0 <= y < height):
            return False
        
        cell_value = map_data[y * width + x]
        
        # 必須是已知的自由空間 (occupancy value = 0)
        if cell_value != 0:
            return False
        
        # 檢查8個相鄰格子是否有未知區域
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx = x + dx
                ny = y + dy
                
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_value = map_data[ny * width + nx]
                    if neighbor_value == -1:  # 未知區域
                        return True
        
        return False

    def grid_to_world(self, x, y):
        """將網格座標轉換為世界座標"""
        if not self.mapData:
            return None
        
        world_x = x * self.mapData.info.resolution + self.mapData.info.origin.position.x
        world_y = y * self.mapData.info.resolution + self.mapData.info.origin.position.y
        
        return [world_x, world_y]

    def detect_frontiers(self):
        """執行網格搜尋frontier檢測 - 輸出所有找到的frontier"""
        if not self.mapData:
            self.get_logger().debug('No map data available')
            return
            
        if not self.boundary_received:
            self.get_logger().debug('No boundary received yet')
            return
        
        try:
            map_data = self.mapData.data
            width = self.mapData.info.width
            height = self.mapData.info.height
            
            frontier_points = []
            free_cells = 0
            unknown_cells = 0
            obstacle_cells = 0
            
            # 統計地圖信息並遍歷尋找frontier
            for y in range(height):
                for x in range(width):
                    cell_value = map_data[y * width + x]
                    
                    if cell_value == 0:
                        free_cells += 1
                    elif cell_value == -1:
                        unknown_cells += 1
                    elif cell_value > 50:
                        obstacle_cells += 1
                    
                    # 檢查是否為frontier點
                    if self.is_frontier_cell(x, y, map_data, width, height):
                        world_point = self.grid_to_world(x, y)
                        if world_point and self.is_point_in_boundary(world_point):
                            frontier_points.append(world_point)
            
            # 每10次檢測輸出一次統計信息
            if hasattr(self, 'detection_count'):
                self.detection_count += 1
            else:
                self.detection_count = 1
                
            if self.detection_count % 10 == 0:
                total_cells = width * height
                self.get_logger().info(f'Map stats - Free: {free_cells}, Unknown: {unknown_cells}, Obstacles: {obstacle_cells}, Total: {total_cells}')
                self.get_logger().info(f'Found {len(frontier_points)} frontier points (NO FILTERING)')
            
            # 直接發布所有找到的frontier點，不進行過濾或聚類
            new_frontiers = []
            for frontier in frontier_points:
                # 可以選擇是否要去除重複點（保留最基本的重複檢查）
                is_new = True
                MIN_DISTANCE = 0.1  # 非常小的距離閾值，只去除完全重複的點
                
                for existing_frontier in self.frontiers:
                    distance = np.sqrt(
                        (frontier[0] - existing_frontier[0])**2 + 
                        (frontier[1] - existing_frontier[1])**2
                    )
                    if distance < MIN_DISTANCE:
                        is_new = False
                        break
                
                if is_new:
                    new_frontiers.append(frontier)
            
            # 發布新的frontier點
            for frontier in new_frontiers:
                self.publish_frontier(frontier)
            
            # 更新frontier列表(保留最近的200個)
            self.frontiers.extend(new_frontiers)
            if len(self.frontiers) > 200:  # 增加保留數量
                self.frontiers = self.frontiers[-200:]
            
            # 更新可視化
            self.update_visualization()
            
            if new_frontiers:
                self.get_logger().info(f'Detected {len(new_frontiers)} new frontiers, total: {len(self.frontiers)} (ALL OUTPUT)')
        
        except Exception as e:
            self.get_logger().error(f'Error in frontier detection: {str(e)}')
            traceback.print_exc()

    def publish_frontier(self, point):
        """發布單個frontier點"""
        # 發布到 /detected_points
        msg = PointStamped()
        msg.header.frame_id = self.mapData.header.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = 0.0
        self.frontier_pub.publish(msg)
        
        self.frontier_count += 1
        if self.frontier_count % 10 == 0:  # 減少日誌輸出頻率
            self.get_logger().info(f'Published frontier {self.frontier_count}: ({point[0]:.2f}, {point[1]:.2f})')

    def update_visualization(self):
        """更新可視化標記"""
        if not self.frontiers:
            return
        
        # 更新frontier標記
        self.frontier_marker.points = []
        self.frontier_marker.header.stamp = self.get_clock().now().to_msg()
        
        for frontier in self.frontiers:
            p = Point()
            p.x = float(frontier[0])
            p.y = float(frontier[1])
            p.z = 0.0
            self.frontier_marker.points.append(p)
        
        # 發布到 /visualization_marker_array
        marker_array = MarkerArray()
        marker_array.markers = [self.frontier_marker]
        self.marker_pub.publish(marker_array)
        
        # 發布到 /found
        self.found_marker.points = self.frontier_marker.points.copy()
        self.found_marker.header.stamp = self.get_clock().now().to_msg()
        
        found_marker_array = MarkerArray()
        found_marker_array.markers = [self.found_marker]
        self.found_frontiers_pub.publish(found_marker_array)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = GridFrontierDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Caught exception: {str(e)}')
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()