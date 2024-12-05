#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, PointStamped, PolygonStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
import math
import traceback

class GlobalRRTDetector(Node):
    def __init__(self):
        super().__init__('global_rrt_detector')
        
       
        self.declare_parameter('eta', 0.5)
        self.declare_parameter('map_topic', '/merge_map')
        
       
        self.eta = self.get_parameter('eta').value
        self.map_topic = self.get_parameter('map_topic').value
        
      
        self.MAX_VERTICES = 1000  # RRT 最大node數量
        self.MIN_FRONTIER_DIST = 0.2  # 尋找到的frontier之間最小距離
        
        # 初始化
        self.mapData = None
        self.boundary_received = False
        self.start_point_received = False
        self.boundary = None
        self.V = []  # RRT node
        self.init_x = 0.0
        self.init_y = 0.0
        self.init_map_x = 0.0
        self.init_map_y = 0.0
        self.frontier_count = 0

        
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

        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )

       
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

        # 可視化
        self.init_markers()

        # RRT迭代定時器
        self.create_timer(0.005, self.rrt_iteration)
        
        self.get_logger().info('Global RRT detector initialized')
        self.get_logger().info('Waiting for map and boundary...')

    def init_markers(self):
        """初始化所有可视化标记"""
        self.marker_array = MarkerArray()

        # RRT
        self.tree_marker = Marker()
        self.tree_marker.header.frame_id = 'map'
        self.tree_marker.ns = "rrt_tree"
        self.tree_marker.id = 0
        self.tree_marker.type = Marker.LINE_LIST
        self.tree_marker.action = Marker.ADD
        self.tree_marker.pose.orientation.w = 1.0
        self.tree_marker.scale.x = 0.05
        self.tree_marker.color.r = 0.0
        self.tree_marker.color.g = 0.8
        self.tree_marker.color.b = 0.8
        self.tree_marker.color.a = 0.8
        self.tree_marker.points = []
        self.tree_marker.lifetime = Duration(seconds=0).to_msg()  # 0表示永久
        self.marker_array.markers.append(self.tree_marker)

        # Frontier edge
        self.frontier_edge_marker = Marker()
        self.frontier_edge_marker.header.frame_id = 'map'
        self.frontier_edge_marker.ns = "frontier_edges"
        self.frontier_edge_marker.id = 1
        self.frontier_edge_marker.type = Marker.LINE_LIST
        self.frontier_edge_marker.action = Marker.ADD
        self.frontier_edge_marker.pose.orientation.w = 1.0
        self.frontier_edge_marker.scale.x = 0.05
        self.frontier_edge_marker.color.r = 1.0
        self.frontier_edge_marker.color.g = 0.4
        self.frontier_edge_marker.color.b = 0.0
        self.frontier_edge_marker.color.a = 0.8
        self.frontier_edge_marker.points = []
        self.marker_array.markers.append(self.frontier_edge_marker)

        # Frontier node
        self.frontier_marker = Marker()
        self.frontier_marker.header.frame_id = 'map'
        self.frontier_marker.ns = "frontiers"
        self.frontier_marker.id = 2
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
        self.marker_array.markers.append(self.frontier_marker)

        # start point (root of rrt)
        self.start_marker = Marker()
        self.start_marker.header.frame_id = 'map'
        self.start_marker.ns = "start_point"
        self.start_marker.id = 3
        self.start_marker.type = Marker.CUBE
        self.start_marker.action = Marker.ADD
        self.start_marker.pose.orientation.w = 1.0
        self.start_marker.scale.x = 0.3
        self.start_marker.scale.y = 0.3
        self.start_marker.scale.z = 0.3
        self.start_marker.color.r = 0.0
        self.start_marker.color.g = 1.0
        self.start_marker.color.b = 0.0
        self.start_marker.color.a = 1.0
        self.marker_array.markers.append(self.start_marker)




        # 初始化 found frontiers marker array
        self.found_frontiers_markers = MarkerArray()
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
        self.found_frontiers_markers.markers.append(self.found_marker)






    def map_callback(self, msg):
      
        if self.mapData is None:
            self.get_logger().info('First map data received')
            self.get_logger().info(f'Map size: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}')
        self.mapData = msg
        
       
        for marker in self.marker_array.markers:
            marker.header.frame_id = msg.header.frame_id

    def boundary_callback(self, msg):
   
        if not self.boundary_received:
            self.boundary = msg.polygon.points
            points = msg.polygon.points
            
            x_coords = [p.x for p in points]
            y_coords = [p.y for p in points]
            
            self.init_map_x = max(x_coords) - min(x_coords)
            self.init_map_y = max(y_coords) - min(y_coords)
            self.init_x = (max(x_coords) + min(x_coords)) / 2
            self.init_y = (max(y_coords) + min(y_coords)) / 2
            
            self.boundary_received = True
            self.get_logger().info('Received exploration boundary')
            self.get_logger().info(f'Area size: {self.init_map_x:.2f} x {self.init_map_y:.2f}')
            self.get_logger().info(f'Center: ({self.init_x:.2f}, {self.init_y:.2f})')
            self.get_logger().info('Please click a point in RViz to set RRT start position')

    def is_new_frontier(self, point):
   
        if not self.frontier_marker.points:
            return True
            
        for p in self.frontier_marker.points:
            dist = np.sqrt((p.x - point[0])**2 + (p.y - point[1])**2)
            if dist < self.MIN_FRONTIER_DIST:
                return False
        return True

    def clicked_point_callback(self, msg):
      
        if not self.boundary_received:
            self.get_logger().warn('Waiting for boundary to be set first')
            return

        if self.start_point_received:
            self.get_logger().warn('Start point already set. Ignoring new click.')
            return

        point = [msg.point.x, msg.point.y]
        if not self.is_point_in_boundary(point):
            self.get_logger().warn('Clicked point is outside boundary. Please click inside the boundary.')
            return

        if self.check_point(point) != 1:
            self.get_logger().warn('Clicked point is not in free space. Please choose another point.')
            return

        self.V = [point]
        self.start_point_received = True
        
        self.start_marker.pose.position.x = point[0]
        self.start_marker.pose.position.y = point[1]
        self.start_marker.pose.position.z = 0.0
        self.start_marker.header.stamp = self.get_clock().now().to_msg()
        
        self.marker_array.markers[3] = self.start_marker
        self.marker_pub.publish(self.marker_array)

        self.get_logger().info(f'Set RRT start point: ({point[0]:.2f}, {point[1]:.2f})')
        self.get_logger().info('Starting RRT exploration...')
            
    def is_point_in_boundary(self, point):
     
        if not self.boundary:
            return False
        
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

    def check_point(self, point):
    
        if not self.mapData or not self.is_point_in_boundary(point):
            return 0

        resolution = self.mapData.info.resolution
        origin_x = self.mapData.info.origin.position.x
        origin_y = self.mapData.info.origin.position.y
        width = self.mapData.info.width
        
        x = int((point[0] - origin_x) / resolution)
        y = int((point[1] - origin_y) / resolution)
        
        if not (0 <= x < width and 0 <= y < self.mapData.info.height):
            return 0

        cell_value = self.mapData.data[y * width + x]
        
        # 檢查是否在自由空間
        if cell_value != 0:  
            return 0
            
        # 檢查周圍8個點
        has_unknown = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx = x + dx
                ny = y + dy
                if (0 <= nx < width and 0 <= ny < self.mapData.info.height):
                    neighbor_value = self.mapData.data[ny * width + nx]
                    if neighbor_value == -1:  # 未知區域
                        has_unknown = True
                        
        return -1 if has_unknown else 1

    


    def check_path(self, p1, p2):
        
        if not self.mapData:
            return 0

        if not (self.is_point_in_boundary(p1) and self.is_point_in_boundary(p2)):
            return 0

        resolution = self.mapData.info.resolution
       
        steps = int(np.ceil(np.linalg.norm(np.array(p2) - np.array(p1)) / (resolution * 0.1)))  # 提高采样密度
    
        
        for i in range(steps + 1):
            t = i / steps
            point = [
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1])
            ]
        
            if not self.is_valid_point(point):  
                return 0
                
        return self.check_point(p2)  








    def is_valid_point(self, point):
        
        if not self.mapData or not self.is_point_in_boundary(point):
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
    
        safety_margin = 2  
        for dx in range(-safety_margin, safety_margin + 1):
            for dy in range(-safety_margin, safety_margin + 1):
                nx = x + dx
                ny = y + dy
                if (0 <= nx < width and 0 <= ny < self.mapData.info.height):
                    neighbor_value = self.mapData.data[ny * width + nx]
                   
                    if neighbor_value > 0:  
                        return False
                
        return cell_value == 0 









    def publish_tree(self, p1, p2, is_frontier=False):
       
        point1 = Point()
        point1.x = float(p1[0])
        point1.y = float(p1[1])
        point1.z = 0.0
        
        point2 = Point()
        point2.x = float(p2[0])
        point2.y = float(p2[1])
        point2.z = 0.0

        if is_frontier:
            # frontier edge 使用其他顏色
            
            # if len(self.frontier_edge_marker.points) > 100:
            #     self.frontier_edge_marker.points = self.frontier_edge_marker.points[2:]
            self.frontier_edge_marker.points.extend([point1, point2])
            self.marker_array.markers[1] = self.frontier_edge_marker
        else:
            # 普通RRT
            # if len(self.tree_marker.points) > 200:
            #     self.tree_marker.points = self.tree_marker.points[2:]
            self.tree_marker.points.extend([point1, point2])
            self.marker_array.markers[0] = self.tree_marker

        self.marker_pub.publish(self.marker_array)

    def publish_frontier(self, point):
        
        if not self.is_new_frontier(point):
            return

        msg = PointStamped()
        msg.header.frame_id = self.mapData.header.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = 0.0
        self.frontier_pub.publish(msg)

        p = Point()
        p.x = float(point[0])
        p.y = float(point[1])
        p.z = 0.0
        
        if len(self.frontier_marker.points) > 50:
            self.frontier_marker.points = self.frontier_marker.points[1:]
        self.frontier_marker.points.append(p)
        
        self.marker_array.markers[2] = self.frontier_marker
        self.marker_pub.publish(self.marker_array)


        # 新增: 發布到 /found topic
        self.found_marker.header.stamp = self.get_clock().now().to_msg()
        self.found_marker.points.append(p)
        self.found_frontiers_markers.markers[0] = self.found_marker
        self.found_frontiers_pub.publish(self.found_frontiers_markers)

        self.frontier_count += 1
        self.get_logger().info(f'Found frontier {self.frontier_count}: ({point[0]:.2f}, {point[1]:.2f})')

    def rrt_iteration(self):
       
        if not all([self.boundary_received, self.start_point_received, self.mapData]):
            return

        try:
           
            attempts = 0
            x_rand = None
            while attempts < 100:
                x_rand = [
                    np.random.uniform(self.init_x - self.init_map_x/2, 
                                    self.init_x + self.init_map_x/2),
                    np.random.uniform(self.init_y - self.init_map_y/2, 
                                    self.init_y + self.init_map_y/2)
                ]
               
                if self.is_point_in_boundary(x_rand) and self.is_valid_point(x_rand):
                    break
                attempts += 1
        
            if x_rand is None or attempts >= 100:
                return

            
            V_array = np.array(self.V)
            dist = np.linalg.norm(V_array - np.array(x_rand).reshape(1, 2), axis=1)
            nearest_idx = np.argmin(dist)
            x_nearest = self.V[nearest_idx]
        
            
            dist = np.linalg.norm(np.array(x_rand) - np.array(x_nearest))
            if dist <= self.eta:
                x_new = x_rand
            else:
                dir_vector = np.array(x_rand) - np.array(x_nearest)
                x_new = (x_nearest + (dir_vector / dist) * self.eta).tolist()

            
            if not self.is_valid_point(x_new):
                return

            
            path_status = self.check_path(x_nearest, x_new)
            if path_status == 0:  
                return

            
            self.V.append(x_new)

            if path_status == -1: 
                if self.is_new_frontier(x_new):
                    self.publish_frontier(x_new)
                self.publish_tree(x_nearest, x_new, is_frontier=True)
            elif path_status == 1:  
                self.publish_tree(x_nearest, x_new, is_frontier=False)

            
            if len(self.V) > self.MAX_VERTICES:
                self.V = self.V[:1] + self.V[-(self.MAX_VERTICES-1):]

        except Exception as e:
            self.get_logger().error(f'Error in RRT iteration: {str(e)}')
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = GlobalRRTDetector()
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