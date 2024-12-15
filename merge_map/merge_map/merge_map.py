#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy import ndimage
from std_msgs.msg import ColorRGBA
import threading

class MergeMapNode(Node):
    def __init__(self):
        super().__init__('merge_map_node')
        
        # Declare parameters
        self.declare_parameter('map_size', 40.0)
        self.declare_parameter('origin_offset', -20.0)
        self.declare_parameter('dilation_size', 2)
        self.declare_parameter('robot_marker_size', 0.3)
        self.declare_parameter('marker_update_rate', 1.0)  # Hz
        self.declare_parameter('debug_mode', True)
        
        # Get parameters
        self.map_size = self.get_parameter('map_size').value
        self.origin_offset = self.get_parameter('origin_offset').value
        self.dilation_size = self.get_parameter('dilation_size').value
        self.robot_marker_size = self.get_parameter('robot_marker_size').value
        self.marker_update_rate = self.get_parameter('marker_update_rate').value
        self.debug_mode = self.get_parameter('debug_mode').value
        
        # Initialize publishers
        self.map_publisher = self.create_publisher(
            OccupancyGrid, 
            '/merge_map', 
            10
        )
        
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/robot_markers',
            10
        )
        
        # Initialize subscribers with threading lock
        self.map_lock = threading.Lock()
        self.subscription1 = self.create_subscription(
            OccupancyGrid,
            '/robot1/map',
            lambda msg: self.map_callback(msg, 'robot1'),
            10
        )
        self.subscription2 = self.create_subscription(
            OccupancyGrid,
            '/robot2/map',
            lambda msg: self.map_callback(msg, 'robot2'),
            10
        )
        
        # Initialize map and pose data
        self.maps = {
            'robot1': None,
            'robot2': None
        }
        self.robot_poses = {
            'robot1': None,
            'robot2': None
        }
        self.pose_lock = threading.Lock()
        
        # Add pose subscribers
        self.pose_subs = {}
        for robot_name in ['robot1', 'robot2']:
            self.pose_subs[robot_name] = self.create_subscription(
                PoseStamped,
                f'/{robot_name}_pose',
                lambda msg, name=robot_name: self.pose_callback(msg, name),
                10
            )
        
        # Set robot colors
        self.robot_colors = {
            'robot1': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
            'robot2': ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)   # Green
        }
        
        # Create timer for map merging and marker updates
        self.timer = self.create_timer(1.0 / self.marker_update_rate, self.merge_and_publish)
        
        self.get_logger().info('Map merge node initialized')

    def map_callback(self, msg, robot_name):
        """
        Callback for map messages with thread safety
        """
        with self.map_lock:
            self.maps[robot_name] = msg
            if self.debug_mode:
                self.get_logger().debug(f'Received map from {robot_name}')

    def pose_callback(self, msg, robot_name):
        """
        Callback for pose messages
        """
        with self.pose_lock:
            self.robot_poses[robot_name] = msg
            if self.debug_mode:
                self.get_logger().info(
                    f'Updated {robot_name} pose: '
                    f'x={msg.pose.position.x:.2f}, '
                    f'y={msg.pose.position.y:.2f}'
                )

    def create_robot_marker(self, robot_name, position):
        """
        Create a marker for robot visualization
        """
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = robot_name
        marker.id = 0 if robot_name == 'robot1' else 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position = position
        
        # Set orientation (upright)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set size
        marker.scale.x = self.robot_marker_size
        marker.scale.y = self.robot_marker_size
        marker.scale.z = self.robot_marker_size / 3  # Height
        
        # Set color
        marker.color = self.robot_colors[robot_name]
        
        return marker

    def get_robot_position(self, robot_name):
        """
        Get robot position from pose message
        """
        try:
            with self.pose_lock:
                pose_msg = self.robot_poses[robot_name]
                if pose_msg is None:
                    raise ValueError(f"No pose data available for {robot_name}")
                
                # 直接使用原始位置
                pos = Point(
                    x=pose_msg.pose.position.x,
                    y=pose_msg.pose.position.y,
                    z=0.0
                )
                
                if self.debug_mode:
                    self.get_logger().info(
                        f'{robot_name} position: x={pos.x:.2f}, y={pos.y:.2f}'
                    )
                
                return pos
                    
        except Exception as e:
            self.get_logger().warning(f'Failed to get position for {robot_name}: {str(e)}')
            return None

    def merge_maps(self):
        """
        Merge maps with thread safety
        """
        with self.map_lock:
            if any(map_data is None for map_data in self.maps.values()):
                return None
            
            # Create merged map
            merged_map = OccupancyGrid()
            first_map = next(iter(self.maps.values()))
            merged_map.header = first_map.header
            merged_map.header.frame_id = 'merge_map'
            
            # Use minimum resolution
            merged_map.info.resolution = min(
                map_data.info.resolution for map_data in self.maps.values()
            )
            
            # Set origin and size
            merged_map.info.origin.position.x = self.origin_offset
            merged_map.info.origin.position.y = self.origin_offset
            merged_map.info.width = int(self.map_size / merged_map.info.resolution)
            merged_map.info.height = int(self.map_size / merged_map.info.resolution)
            
            # Initialize as unknown
            merged_data = np.full(
                (merged_map.info.height, merged_map.info.width),
                -1,
                dtype=np.int8
            )
            
            # Merge maps
            for map_data in self.maps.values():
                map_array = np.array(map_data.data).reshape(
                    map_data.info.height,
                    map_data.info.width
                )
                
                # Calculate coordinates in merged map
                for y in range(map_data.info.height):
                    for x in range(map_data.info.width):
                        merged_x = int((map_data.info.origin.position.x +
                                      x * map_data.info.resolution -
                                      self.origin_offset) /
                                     merged_map.info.resolution)
                        merged_y = int((map_data.info.origin.position.y +
                                      y * map_data.info.resolution -
                                      self.origin_offset) /
                                     merged_map.info.resolution)
                        
                        if (0 <= merged_x < merged_map.info.width and
                            0 <= merged_y < merged_map.info.height):
                            current_value = map_array[y, x]
                            existing_value = merged_data[merged_y, merged_x]
                            
                            # Merge rules
                            if existing_value == -1:
                                merged_data[merged_y, merged_x] = current_value
                            elif current_value == 100 or existing_value == 100:
                                merged_data[merged_y, merged_x] = 100
                            elif current_value >= 0:
                                merged_data[merged_y, merged_x] = current_value
            
            # Apply dilation to obstacles
            obstacle_mask = (merged_data == 100)
            kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
            dilated_obstacles = ndimage.binary_dilation(obstacle_mask, kernel)
            merged_data[dilated_obstacles] = 100
            
            # Convert back to 1D list
            merged_map.data = merged_data.flatten().tolist()
            
            return merged_map
        
    def merge_and_publish(self):
        """
        Periodically merge maps and publish robot markers
        """
        # Merge maps
        merged_map = self.merge_maps()
        if merged_map is None:
            if self.debug_mode:
                self.get_logger().debug('Not all maps available for merging')
            return
            
        # Publish merged map
        merged_map.header.stamp = self.get_clock().now().to_msg()
        self.map_publisher.publish(merged_map)
        
        # Create and publish robot markers
        marker_array = MarkerArray()
        for robot_name in self.maps.keys():
            position = self.get_robot_position(robot_name)
            if position is not None:
                marker = self.create_robot_marker(robot_name, position)
                marker_array.markers.append(marker)
        
        if marker_array.markers:
            if self.debug_mode:
                self.get_logger().info(
                    f'Publishing {len(marker_array.markers)} robot markers'
                )
            self.marker_publisher.publish(marker_array)
        else:
            self.get_logger().warning('No robot positions available for visualization')

def main(args=None):
    rclpy.init(args=args)
    merge_map_node = MergeMapNode()
    
    try:
        rclpy.spin(merge_map_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        merge_map_node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        merge_map_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()