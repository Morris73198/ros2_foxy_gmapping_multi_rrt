#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import math
import traceback

class LocalRRTDetector(Node):
    def __init__(self):
        super().__init__('local_rrt_detector')
        
        # 參數聲明
        self.declare_parameter('eta', 1.0)
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('robot_name', 'robot1')
        self.declare_parameter('update_frequency', 30.0)
        
        # 獲取參數
        self.eta = self.get_parameter('eta').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.robot_name = self.get_parameter('robot_name').value
        self.update_frequency = self.get_parameter('update_frequency').value
        
        if self.robot_name not in ['robot1', 'robot2']:
            self.get_logger().error('Invalid robot name. Must be robot1 or robot2')
            return
        
        # 常數設置
        self.MAX_VERTICES = 500
        self.MAX_FRONTIERS = 100
        
        # 初始化變量
        self.mapData = None
        self.V = []
        self.parents = {}
        self.init_map_x = 0.0
        self.init_map_y = 0.0
        self.init_x = 0.0
        self.init_y = 0.0
        self.robot_pose = None
        self.frontiers = []
        
        # 訂閱機器人位置
        self.pose_sub = self.create_subscription(
            PoseStamped,
            f'/{self.robot_name}_pose',
            self.pose_callback,
            10
        )
        
        # 發布器
        self.frontier_pub = self.create_publisher(
            PointStamped,
            '/detected_points',
            10
        )
        
        self.marker_pub = self.create_publisher(
            Marker,
            f'/{self.robot_name}/local_rrt_markers',
            10
        )
        
        self.frontier_markers_pub = self.create_publisher(
            MarkerArray,
            f'/{self.robot_name}/frontier_markers',
            10
        )

        self.unified_frontier_pub = self.create_publisher(
            MarkerArray,
            '/found',
            10
        )
        
        self.debug_publisher = self.create_publisher(
            String,
            f'/{self.robot_name}/debug',
            10
        )
        
        # 訂閱地圖
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/merge_map',
            self.map_callback,
            10
        )

        # 初始化 unified marker
        self.unified_marker = Marker()
        self.unified_marker.header.frame_id = "merge_map"
        self.unified_marker.ns = f'{self.robot_name}_frontier'
        self.unified_marker.id = 0
        self.unified_marker.type = Marker.SPHERE_LIST
        self.unified_marker.action = Marker.ADD
        self.unified_marker.pose.orientation.w = 1.0
        self.unified_marker.scale.x = 0.2
        self.unified_marker.scale.y = 0.2
        self.unified_marker.scale.z = 0.2
        
        if self.robot_name == 'robot1':
            self.unified_marker.color.r = 1.0
            self.unified_marker.color.g = 0.0
            self.unified_marker.color.b = 0.0
        else:  # robot2
            self.unified_marker.color.r = 0.0
            self.unified_marker.color.g = 1.0
            self.unified_marker.color.b = 0.0
        
        self.unified_marker.color.a = 0.8
        self.unified_marker.points = []
        
        # 初始化可視化標記
        self.points_marker = self.create_points_marker()
        self.line_marker = self.create_line_marker()
        self.frontier_marker_array = MarkerArray()
        
        # 創建定時器
        self.create_timer(1.0 / self.update_frequency, self.rrt_iteration)
        self.create_timer(0.1, self.publish_markers)
        self.create_timer(0.1, self.publish_frontier_markers)
        
        self.get_logger().info('Local RRT detector initialized with frontier visualization')

    def create_points_marker(self):
        """創建點的可視化標記"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.ns = f'{self.robot_name}_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        if self.robot_name == 'robot1':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:  # robot2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
        marker.color.a = 1.0
        return marker

    def create_line_marker(self):
        """創建線的可視化標記"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.ns = f'{self.robot_name}_lines'
        marker.id = 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        if self.robot_name == 'robot1':
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
        else:  # robot2
            marker.color.r = 0.2
            marker.color.g = 1.0
            marker.color.b = 0.2
            
        marker.color.a = 0.6
        return marker

    def create_frontier_marker(self, point, marker_id):
        """創建單個 frontier 的標記"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.ns = f'{self.robot_name}_frontier'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        if self.robot_name == 'robot1':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        else:  # robot2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        
        marker.color.a = 0.8
        marker.lifetime = rclpy.duration.Duration(seconds=5.0).to_msg()
        
        return marker

    def publish_markers(self):
        """發布RRT樹的可視化標記"""
        if self.V:
            # 發布點標記
            self.points_marker.points = []
            for vertex in self.V:
                p = Point()
                p.x = vertex[0]
                p.y = vertex[1]
                p.z = 0.0
                self.points_marker.points.append(p)
            self.points_marker.header.stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(self.points_marker)
            
            # 發布線標記
            if len(self.V) > 1:
                self.line_marker.points = []
                for child_str, parent in self.parents.items():
                    child = eval(child_str)
                    
                    p1 = Point()
                    p1.x = parent[0]
                    p1.y = parent[1]
                    p1.z = 0.0
                    
                    p2 = Point()
                    p2.x = child[0]
                    p2.y = child[1]
                    p2.z = 0.0
                    
                    self.line_marker.points.extend([p1, p2])
                
                self.line_marker.header.stamp = self.get_clock().now().to_msg()
                self.marker_pub.publish(self.line_marker)

    def publish_frontier_markers(self):
        """發布所有frontier標記"""
        if not self.frontiers:
            return
        
        marker_array = MarkerArray()
        
        for i, frontier in enumerate(self.frontiers):
            marker = self.create_frontier_marker(frontier, i)
            marker.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers.append(marker)
        
        self.frontier_markers_pub.publish(marker_array)

    def pose_callback(self, msg):
        """處理機器人位置更新"""
        self.robot_pose = [msg.pose.position.x, msg.pose.position.y]
        self.get_logger().debug(f'Updated {self.robot_name} pose: x={self.robot_pose[0]:.2f}, y={self.robot_pose[1]:.2f}')

    def add_frontier(self, point):
        """添加新的frontier點"""
        MIN_DISTANCE = 0.5
        
        for existing_point in self.frontiers:
            distance = np.linalg.norm(np.array(point) - np.array(existing_point))
            if distance < MIN_DISTANCE:
                return False
        
        self.frontiers.append(point)
        
        if len(self.frontiers) > self.MAX_FRONTIERS:
            self.frontiers.pop(0)
        
        return True

    def publish_frontier(self, point):
        """發布frontier點"""
        if self.add_frontier(point):
            msg = PointStamped()
            msg.header.frame_id = "merge_map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = point[0]
            msg.point.y = point[1]
            msg.point.z = 0.0
            self.frontier_pub.publish(msg)

            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            
            for existing_point in self.unified_marker.points:
                dist = np.sqrt(
                    (existing_point.x - p.x)**2 + 
                    (existing_point.y - p.y)**2
                )
                if dist < 0.5:
                    return
            
            if len(self.unified_marker.points) > self.MAX_FRONTIERS:
                self.unified_marker.points.pop(0)
            self.unified_marker.points.append(p)
            
            marker_array = MarkerArray()
            self.unified_marker.header.stamp = self.get_clock().now().to_msg()
            marker_array.markers = [self.unified_marker]
            self.unified_frontier_pub.publish(marker_array)

            debug_msg = String()
            debug_msg.data = f'Found new frontier at: ({point[0]:.2f}, {point[1]:.2f})'
            self.debug_publisher.publish(debug_msg)

    def map_callback(self, msg):
        """處理地圖數據"""
        if self.mapData is None:
            self.get_logger().info('First map data received')
            self.get_logger().info(f'Map size: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}')
            
            self.init_map_x = msg.info.width * msg.info.resolution
            self.init_map_y = msg.info.height * msg.info.resolution
            self.init_x = msg.info.origin.position.x + self.init_map_x/2
            self.init_y = msg.info.origin.position.y + self.init_map_y/2
            
        self.mapData = msg

    def get_robot_position(self):
        """獲取機器人位置"""
        return self.robot_pose if self.robot_pose is not None else None

    def is_valid_point(self, point):
        """檢查點是否有效"""
        if not self.mapData:
            return False
            
        resolution = self.mapData.info.resolution
        origin_x = self.mapData.info.origin.position.x
        origin_y = self.mapData.info.origin.position.y
        width = self.mapData.info.width
        
        x = int((point[0] - origin_x) / resolution)
        y = int((point[1] - origin_y) / resolution)
        
        if not (0 <= x < width and 0 <= y < self.mapData.info.height):
            return False
            
        cell_value = self.mapData.data[y * width + x]
        
        return cell_value == 0 or (50 <= cell_value <= 80)

    def check_path(self, start, end):
        """
        檢查路徑狀態
        返回:
            -1: 找到frontier
            0: 無效路徑
            1: 有效路徑
        """
        if not self.mapData:
            return 0
            
        resolution = self.mapData.info.resolution
        origin_x = self.mapData.info.origin.position.x
        origin_y = self.mapData.info.origin.position.y
        width = self.mapData.info.width
        
        # 檢查終點周圍8個格子
        end_x = int((end[0] - origin_x) / resolution)
        end_y = int((end[1] - origin_y) / resolution)
        
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                    
                check_x = end_x + dx
                check_y = end_y + dy
                
                # 確保檢查點在地圖範圍內
                if not (0 <= check_x < width and 0 <= check_y < self.mapData.info.height):
                    continue
                    
                # 如果周圍有未知區域，就是frontier
                if self.mapData.data[check_y * width + check_x] == -1:
                    return -1
    
        # 檢查路徑是否可行
        steps = int(np.ceil(np.linalg.norm(np.array(end) - np.array(start)) / resolution))
        obstacle_count = 0
        
        for i in range(steps + 1):
            t = i / steps
            point = [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1])
            ]
            
            x = int((point[0] - origin_x) / resolution)
            y = int((point[1] - origin_y) / resolution)
            
            if not (0 <= x < width and 0 <= y < self.mapData.info.height):
                return 0
                
            cell = self.mapData.data[y * width + x]
            
            if cell > 0 and not (50 <= cell <= 80):
                obstacle_count += 1
                
        if obstacle_count > steps * 0.1:
            return 0
            
        return 1

    def rrt_iteration(self):
        """執行一次RRT迭代"""
        if not self.mapData or self.robot_pose is None:
            return
            
        try:
            robot_pos = self.robot_pose
            
            if not self.V:
                self.V = [robot_pos]
                self.parents = {}
                self.get_logger().info(f'Tree initialized at {robot_pos}')
            
            # 在整個地圖範圍內隨機採樣
            for _ in range(10):
                x_rand = [
                    np.random.uniform(
                        self.mapData.info.origin.position.x,
                        self.mapData.info.origin.position.x + self.mapData.info.width * self.mapData.info.resolution
                    ),
                    np.random.uniform(
                        self.mapData.info.origin.position.y,
                        self.mapData.info.origin.position.y + self.mapData.info.height * self.mapData.info.resolution
                    )
                ]
                
                if not self.is_valid_point(x_rand):
                    continue
                
                V_array = np.array(self.V)
                dist = np.linalg.norm(V_array - np.array(x_rand), axis=1)
                nearest_idx = np.argmin(dist)
                x_nearest = self.V[nearest_idx]
                
                dist = np.linalg.norm(np.array(x_rand) - np.array(x_nearest))
                if dist <= self.eta:
                    x_new = x_rand
                else:
                    dir_vector = np.array(x_rand) - np.array(x_nearest)
                    x_new = (x_nearest + (dir_vector / dist) * self.eta).tolist()
                
                if not self.is_valid_point(x_new):
                    continue
                    
                path_status = self.check_path(x_nearest, x_new)
                
                if path_status == -1:  # 找到frontier
                    self.get_logger().info(f'Found frontier point at {x_new}')
                    self.publish_frontier(x_new)
                    self.V = [robot_pos]  # 重置樹
                    self.parents = {}  # 重置父節點記錄
                    break
                    
                elif path_status == 1:  # 有效路徑
                    self.V.append(x_new)
                    self.parents[str(x_new)] = x_nearest
                    break
                    
            # 控制樹的大小
            if len(self.V) > self.MAX_VERTICES:
                self.V = [robot_pos] + self.V[-(self.MAX_VERTICES-1):]
                new_parents = {}
                for child_str, parent in self.parents.items():
                    child = eval(child_str)
                    if child in self.V and parent in self.V:
                        new_parents[child_str] = parent
                self.parents = new_parents

        except Exception as e:
            self.get_logger().error(f'RRT iteration error: {str(e)}')
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = LocalRRTDetector()
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