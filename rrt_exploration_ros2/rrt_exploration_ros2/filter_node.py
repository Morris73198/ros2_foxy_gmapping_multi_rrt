#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from sklearn.cluster import MeanShift
import math

class ImprovedFilterNode(Node):
    def __init__(self):
        super().__init__('improved_filter')
        
        # 聲明參數
        self.declare_parameter('map_topic', '/merge_map')
        self.declare_parameter('safety_threshold', 90)
        self.declare_parameter('info_radius', 0.5)
        self.declare_parameter('safety_radius', 0.002)
        self.declare_parameter('bandwith_cluster', 0.8)  # 增加聚類半徑
        self.declare_parameter('rate', 2.0)
        self.declare_parameter('process_interval', 1.0)
        self.declare_parameter('narrow_passage_mode', True)
        self.declare_parameter('max_frontiers', 50)  # 新增：最大frontier數量
        self.declare_parameter('min_frontier_distance', 1.0)  # 新增：frontier間最小距離
        self.declare_parameter('line_of_sight_check', True)  # 新增：視線檢查開關
        
        # 獲取參數值
        self.map_topic = self.get_parameter('map_topic').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.info_radius = self.get_parameter('info_radius').value
        self.safety_radius = self.get_parameter('safety_radius').value
        self.bandwith = self.get_parameter('bandwith_cluster').value
        self.rate = self.get_parameter('rate').value
        self.process_interval = self.get_parameter('process_interval').value
        self.narrow_passage_mode = self.get_parameter('narrow_passage_mode').value
        self.max_frontiers = self.get_parameter('max_frontiers').value
        self.min_frontier_distance = self.get_parameter('min_frontier_distance').value
        self.line_of_sight_check = self.get_parameter('line_of_sight_check').value
        
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
        
        self.line_of_sight_pub = self.create_publisher(
            MarkerArray,
            'line_of_sight_check',
            10
        )
        
        # 創建定時器
        self.create_timer(1.0/self.rate, self.filter_points)
        
        self.get_logger().info('Improved Filter node started with Line-of-Sight checking')
        self.get_logger().info(f'Max frontiers: {self.max_frontiers}, Min distance: {self.min_frontier_distance}m')
        self.get_logger().info(f'Line of sight check: {self.line_of_sight_check}')

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
                        if np.linalg.norm(np.array(point_arr) - np.array(existing_point)) < 0.2:
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

    def world_to_map(self, x, y):
        """將世界坐標轉換為地圖座標"""
        if not self.mapData:
            return None, None
        
        mx = int((x - self.mapData.info.origin.position.x) / self.mapData.info.resolution)
        my = int((y - self.mapData.info.origin.position.y) / self.mapData.info.resolution)
        return mx, my

    def map_to_world(self, mx, my):
        """將地圖座標轉換為世界坐標"""
        if not self.mapData:
            return None, None
        
        x = mx * self.mapData.info.resolution + self.mapData.info.origin.position.x
        y = my * self.mapData.info.resolution + self.mapData.info.origin.position.y
        return x, y

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham直線算法 - 獲取兩點間所有格子"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x, y))
        return points

    def has_line_of_sight(self, point1, point2):
        """檢查兩點間是否有視線 - 即沒有牆壁阻擋"""
        if not self.mapData:
            return True
        
        # 轉換為地圖座標
        x1, y1 = self.world_to_map(point1[0], point1[1])
        x2, y2 = self.world_to_map(point2[0], point2[1])
        
        if x1 is None or x2 is None:
            return False
        
        # 獲取直線上的所有點
        line_points = self.bresenham_line(x1, y1, x2, y2)
        
        width = self.mapData.info.width
        height = self.mapData.info.height
        
        # 檢查直線上的每個點
        for x, y in line_points:
            # 邊界檢查
            if not (0 <= x < width and 0 <= y < height):
                return False
            
            # 檢查是否是障礙物
            cell_value = self.mapData.data[y * width + x]
            if cell_value >= self.safety_threshold:  # 障礙物
                return False
        
        return True

    def calculate_info_gain_with_line_of_sight(self, point):
        """計算考慮視線遮擋的信息增益"""
        if not self.mapData:
            return 0
            
        info_gain = 0
        resolution = self.mapData.info.resolution
        info_cells = int(self.info_radius / resolution)
        
        x, y = self.world_to_map(point[0], point[1])
        if x is None:
            return 0
        
        width = self.mapData.info.width
        height = self.mapData.info.height
        
        # 檢查信息半徑內的每個格子
        for dx in range(-info_cells, info_cells + 1):
            for dy in range(-info_cells, info_cells + 1):
                nx = x + dx
                ny = y + dy
                
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                
                # 計算實際距離
                world_nx, world_ny = self.map_to_world(nx, ny)
                if world_nx is None:
                    continue
                
                distance = np.sqrt((point[0] - world_nx)**2 + (point[1] - world_ny)**2)
                if distance > self.info_radius:
                    continue
                
                # 檢查是否是未知區域
                index = ny * width + nx
                if index < len(self.mapData.data) and self.mapData.data[index] == -1:
                    # 如果啟用視線檢查，確保該未知區域可見
                    if not self.line_of_sight_check or self.has_line_of_sight(point, [world_nx, world_ny]):
                        info_gain += 1
                    # else:
                    #     self.get_logger().debug(f'Unknown cell at ({world_nx:.2f}, {world_ny:.2f}) blocked by obstacle')
        
        return info_gain * (resolution ** 2)

    def check_safety_narrow_passage_friendly(self, point):
        """窄縫友好的安全性檢查"""
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
            # 窄縫友好檢查：只檢查緊鄰的4個點
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
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
            if obstacle_count >= total_checked:
                return False
                
        return True

    def check_narrow_passage_accessibility(self, point):
        """檢查窄縫中的點是否可達"""
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
            for step in range(1, 4):
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
                
        return free_directions >= 2

    def intelligent_frontier_selection(self, candidates, max_count):
        """智能選擇frontier點 - 平衡覆蓋範圍和數量限制"""
        if len(candidates) <= max_count:
            return candidates
        
        # 策略1：優先選擇信息增益高的點
        candidates_with_gain = []
        for point in candidates:
            info_gain = self.calculate_info_gain_with_line_of_sight(point)
            candidates_with_gain.append((point, info_gain))
        
        # 按信息增益排序
        candidates_with_gain.sort(key=lambda x: x[1], reverse=True)
        
        # 策略2：從高信息增益的點中選擇，確保空間分布
        selected = []
        for point, gain in candidates_with_gain:
            if len(selected) >= max_count:
                break
            
            # 檢查與已選點的距離
            too_close = False
            for selected_point in selected:
                distance = np.linalg.norm(np.array(point) - np.array(selected_point))
                if distance < self.min_frontier_distance:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(point)
        
        # 如果選擇的點還不夠，放寬距離要求
        if len(selected) < max_count * 0.8:  # 如果少於目標的80%
            relaxed_distance = self.min_frontier_distance * 0.7
            for point, gain in candidates_with_gain:
                if len(selected) >= max_count:
                    break
                
                if point in selected:
                    continue
                
                too_close = False
                for selected_point in selected:
                    distance = np.linalg.norm(np.array(point) - np.array(selected_point))
                    if distance < relaxed_distance:
                        too_close = True
                        break
                
                if not too_close:
                    selected.append(point)
        
        self.get_logger().info(f'智能選擇：從 {len(candidates)} 個候選點中選擇了 {len(selected)} 個')
        return selected

    def filter_points(self):
        """過濾和聚類前沿點 - 改進版本"""
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

            # 3. 安全性檢查
            safe_frontiers = []
            line_of_sight_passed = []
            
            for point in self.frontiers:
                if self.check_safety_narrow_passage_friendly(point):
                    safe_frontiers.append(point)
                    
                    # 檢查信息增益（包含視線檢查）
                    info_gain = self.calculate_info_gain_with_line_of_sight(point)
                    if info_gain > 0.1:  # 只保留有實際信息增益的點
                        line_of_sight_passed.append(point)
                    
            if not safe_frontiers:
                self.get_logger().info('No safe frontiers found')
                return

            self.get_logger().info(f'視線檢查：{len(safe_frontiers)} -> {len(line_of_sight_passed)} 個點通過')

            # 4. 執行聚類（如果點數太多）
            points_to_cluster = line_of_sight_passed
            if len(points_to_cluster) > self.max_frontiers * 1.5:  # 如果點數是目標的1.5倍以上才聚類
                points_array = np.array(points_to_cluster)
                ms = MeanShift(bandwidth=self.bandwith)
                ms.fit(points_array)
                centroids = ms.cluster_centers_
                
                self.get_logger().info(f'聚類：{len(points_array)} 點 -> {len(centroids)} 個聚類中心')
                
                # 發布聚類中心
                self.publish_cluster_centers(centroids)
                final_candidates = centroids.tolist()
            else:
                final_candidates = points_to_cluster

            # 5. 智能選擇最終的frontier點
            if len(final_candidates) > self.max_frontiers:
                selected_frontiers = self.intelligent_frontier_selection(final_candidates, self.max_frontiers)
            else:
                selected_frontiers = final_candidates

            # 6. 最終過濾（再次確認安全性和信息增益）
            final_frontiers = []
            for point in selected_frontiers:
                if (self.check_safety_narrow_passage_friendly(point) and 
                    self.calculate_info_gain_with_line_of_sight(point) > 0.1):
                    
                    # 檢查是否接近已分配的點
                    is_near_assigned = False
                    for assigned_point in self.assigned_points:
                        if self.is_point_near_assigned(point, assigned_point):
                            is_near_assigned = True
                            break
                    
                    if not is_near_assigned:
                        final_frontiers.append(point)
            
            # 7. 發布最終結果
            self.publish_filtered_points(final_frontiers)
            
            self.get_logger().info(
                f'處理完成：{initial_points} -> {len(final_frontiers)} 個frontier點 '
                f'(目標最大: {self.max_frontiers}, 視線檢查: {self.line_of_sight_check})'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error in filter_points: {str(e)}')
            import traceback
            traceback.print_exc()

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

    def publish_filtered_points(self, filtered_frontiers):
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
        
        # 根據點數量改變顏色
        if len(filtered_frontiers) <= self.max_frontiers:
            filtered_marker.color.r = 0.0
            filtered_marker.color.g = 1.0
            filtered_marker.color.b = 0.0  # 綠色表示數量合適
        else:
            filtered_marker.color.r = 1.0
            filtered_marker.color.g = 0.5
            filtered_marker.color.b = 0.0  # 橙色表示超出限制
            
        filtered_marker.color.a = 0.9

        for point in filtered_frontiers:
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
        node = ImprovedFilterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()