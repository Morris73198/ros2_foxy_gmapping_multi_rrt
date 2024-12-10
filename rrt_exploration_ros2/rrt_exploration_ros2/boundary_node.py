#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PolygonStamped, Polygon, PointStamped, Point32
from visualization_msgs.msg import Marker, MarkerArray
import math

class ExplorationBoundaryNode(Node):
    def __init__(self):
        super().__init__('exploration_boundary')
        
        
        self.recorded_points = []
        self.frame_id = 'merge_map'
        self.boundary_completed = False
        
        
        self.point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.point_callback,
            10
        )
        
        
        self.boundary_pub = self.create_publisher(
            PolygonStamped,
            '/exploration_boundary',
            10
        )
        
        self.circles_pub = self.create_publisher(
            MarkerArray,
            '/boundary_circles',
            10
        )
        
        self.line_marker_pub = self.create_publisher(
            Marker,
            '/boundary_line',
            10
        )
        
        self.create_timer(0.1, self.timer_callback)
        
        self.get_logger().info('Boundary node started')
        self.get_logger().info('Please click 4 points to define the exploration boundary')

    def create_circle_marker(self, point, index):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'boundary_circles'
        marker.id = index
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        num_points = 32
        radius = 0.3
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            p = Point()
            p.x = point.x + radius * math.cos(angle)
            p.y = point.y + radius * math.sin(angle)
            p.z = 0.0
            marker.points.append(p)
        
        return marker

    def point_callback(self, msg):
        if len(self.recorded_points) < 4:  # 只接收4點
            point = Point32()
            point.x = float(msg.point.x)
            point.y = float(msg.point.y)
            point.z = 0.0
            self.recorded_points.append(point)
            
            self.get_logger().info(f'Boundary point {len(self.recorded_points)} recorded: ({point.x}, {point.y})')
            self.publish_markers()
            
            if len(self.recorded_points) == 4:
                self.boundary_completed = True
                self.publish_boundary()
                self.publish_line()
                self.get_logger().info('Boundary completed. Global RRT detector will handle start point selection.')

    def publish_markers(self):
        
        marker_array = MarkerArray()
        for i, point in enumerate(self.recorded_points):
            circle_marker = self.create_circle_marker(point, i)
            marker_array.markers.append(circle_marker)
        self.circles_pub.publish(marker_array)

    def publish_line(self):
        
        if len(self.recorded_points) >= 4:
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'boundary_line'
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            
            for point in self.recorded_points:
                p = Point()
                p.x = point.x
                p.y = point.y
                p.z = 0.0
                marker.points.append(p)
            
            
            if len(marker.points) > 0:
                p = Point()
                p.x = self.recorded_points[0].x
                p.y = self.recorded_points[0].y
                p.z = 0.0
                marker.points.append(p)
            
            self.line_marker_pub.publish(marker)

    def publish_boundary(self):
        
        if len(self.recorded_points) >= 4:
            msg = PolygonStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            
            points = self.recorded_points.copy()
            points.append(points[0])  # 闭合多边形
            msg.polygon.points = points
            
            self.boundary_pub.publish(msg)

    def timer_callback(self):
        
        if len(self.recorded_points) > 0:
            self.publish_markers()
            
        if self.boundary_completed:
            self.publish_boundary()
            self.publish_line()

def main(args=None):
    rclpy.init(args=args)
    node = ExplorationBoundaryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()