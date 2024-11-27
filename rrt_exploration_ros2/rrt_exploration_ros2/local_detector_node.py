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
        
        # 原有的參數聲明
        self.declare_parameter('eta', 1.0)
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('robot_name', 'robot1')  # 改成robot1或robot2
        self.declare_parameter('update_frequency', 10.0)
        
        # 獲取參數
        self.eta = self.get_parameter('eta').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.robot_name = self.get_parameter('robot_name').value
        self.update_frequency = self.get_parameter('update_frequency').value
        
        # 常數設置
        self.MAX_VERTICES = 500
        self.MAX_FRONTIERS = 100  # 最大儲存的 frontier 數量
        
        # 初始化變量
        self.mapData = None
        self.V = []
        self.parents = {}
        self.init_map_x = 0.0
        self.init_map_y = 0.0
        self.init_x = 0.0
        self.init_y = 0.0
        self.frontiers = []  # 儲存所有找到的 frontiers
        
        # 機器人位置
        self.robot1_pose = None
        self.robot2_pose = None
        
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
        
        # 調試發布器
        self.debug_publisher = self.create_publisher(
            String,
            f'/{self.robot_name}/debug',
            10
        )
        
        # 訂閱合併地圖
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/merge_map',
            self.map_callback,
            10
        )
        
        # 訂閱機器人位置
        self.robot1_pose_sub = self.create_subscription(
            PoseStamped,
            '/robot1_pose',
            self.robot1_pose_callback,
            10
        )
        
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped,
            '/robot2_pose',
            self.robot2_pose_callback,
            10
        )

        # 初始化unified marker
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
        
        # 根據機器人設置顏色 (只有兩個機器人)
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
        
        # 修改為兩個機器人的顏色
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
        
        # 修改為兩個機器人的顏色
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
        
        # 修改為兩個機器人的顏色
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

    def robot1_pose_callback(self, msg):
        """處理機器人1的位置"""
        self.robot1_pose = [msg.pose.position.x, msg.pose.position.y]

    def robot2_pose_callback(self, msg):
        """處理機器人2的位置"""
        self.robot2_pose = [msg.pose.position.x, msg.pose.position.y]

    def get_robot_position(self):
        """根據robot_name獲取對應的機器人位置"""
        if self.robot_name == 'robot1':
            return self.robot1_pose
        else:  # robot2
            return self.robot2_pose

    # [其餘方法保持不變...]

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
