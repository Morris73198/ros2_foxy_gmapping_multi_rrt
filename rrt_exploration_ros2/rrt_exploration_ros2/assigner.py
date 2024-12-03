#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, ColorRGBA, Float32MultiArray
import numpy as np
import heapq
from typing import List, Tuple, Set
import cv2

class GreedyAssigner(Node):
    def __init__(self):
        super().__init__('greedy_assigner')
        
        # 初始化變量
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}  # True means robot is available for new target
        
        # 地圖相關變量
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        
        # 目標到達閾值
        self.target_threshold = 0.3  # 機器人距離目標點小於此值視為到達
        
        # 訂閱和發布
        self.setup_subscribers()
        self.setup_publishers()
        
        # 創建定時器
        self.create_timer(1.0, self.assign_targets)
        self.create_timer(0.1, self.publish_visualization)
        self.create_timer(0.1, self.check_target_reached)
        
        self.get_logger().info('Greedy assigner node with A* pathfinding started')

    def setup_subscribers(self):
        """設置所有訂閱者"""
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/merge_map',
            self.map_callback,
            10
        )
        
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
        
        self.filtered_points_sub = self.create_subscription(
            MarkerArray,
            '/filtered_points',
            self.filtered_points_callback,
            10
        )

    def setup_publishers(self):
        """設置所有發布者"""
        # 目標點發布
        self.robot1_target_pub = self.create_publisher(
            PoseStamped,
            '/robot1/goal_pose',
            10
        )
        
        self.robot2_target_pub = self.create_publisher(
            PoseStamped,
            '/robot2/goal_pose',
            10
        )

        # 可視化發布
        self.target_viz_pub = self.create_publisher(
            MarkerArray,
            '/assigned_targets_viz',
            10
        )

        # 調試信息發布
        self.debug_pub = self.create_publisher(
            String,
            '/assigner/debug',
            10
        )

    def map_callback(self, msg):
        """處理地圖數據"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin
        self.get_logger().debug('Received map update')

    def robot1_pose_callback(self, msg):
        """處理robot1的位置更新"""
        self.robot1_pose = msg.pose
        self.get_logger().debug('Received robot1 pose update')

    def robot2_pose_callback(self, msg):
        """處理robot2的位置更新"""
        self.robot2_pose = msg.pose
        self.get_logger().debug('Received robot2 pose update')

    def filtered_points_callback(self, msg):
        """處理過濾後的點"""
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])
        self.get_logger().debug(f'Received {len(self.available_points)} filtered points')

    def check_target_reached(self):
        """檢查機器人是否到達目標點"""
        robots = {
            'robot1': (self.robot1_pose, self.robot1_pose_callback),
            'robot2': (self.robot2_pose, self.robot2_pose_callback)
        }

        for robot_name, (robot_pose, _) in robots.items():
            if not robot_pose or not self.assigned_targets[robot_name]:
                continue

            target = self.assigned_targets[robot_name]
            current_pos = (robot_pose.position.x, robot_pose.position.y)
            target_pos = target

            # 計算當前位置與目標點的距離
            distance = np.sqrt(
                (current_pos[0] - target_pos[0])**2 + 
                (current_pos[1] - target_pos[1])**2
            )

            # 如果距離小於閾值，認為已到達目標
            if distance < self.target_threshold:
                if not self.robot_status[robot_name]:
                    self.get_logger().info(f'{robot_name} has reached target {target_pos}')
                self.robot_status[robot_name] = True
                self.assigned_targets[robot_name] = None
            else:
                self.robot_status[robot_name] = False

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        """將世界坐標轉換為地圖坐標"""
        mx = int((wx - self.map_origin.position.x) / self.map_resolution)
        my = int((wy - self.map_origin.position.y) / self.map_resolution)
        return (mx, my)
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """將地圖坐標轉換為世界坐標"""
        wx = mx * self.map_resolution + self.map_origin.position.x
        wy = my * self.map_resolution + self.map_origin.position.y
        return (wx, wy)

    def is_valid_point(self, x: int, y: int) -> bool:
        """檢查點是否在地圖範圍內且可通行"""
        if x < 0 or x >= self.map_width or y < 0 or y >= self.map_height:
            return False
        return self.map_data[y, x] < 50  # 假設0-49是可通行的

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """獲取相鄰的可通行點"""
        x, y = pos
        neighbors = []
        # 8方向移動
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_point(new_x, new_y):
                # 檢查對角線移動是否會碰到障礙物
                if abs(dx) + abs(dy) == 2:  # 對角線移動
                    if self.is_valid_point(x + dx, y) and self.is_valid_point(x, y + dy):
                        neighbors.append((new_x, new_y))
                else:  # 直線移動
                    neighbors.append((new_x, new_y))
        return neighbors

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """計算啟發式值（使用歐幾里得距離）"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """使用A*檢查目標點是否可達"""
        if not self.is_valid_point(*start) or not self.is_valid_point(*goal):
            return False

        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                return True

            for next_pos in self.get_neighbors(current):
                movement_cost = 1.414 if abs(next_pos[0] - current[0]) + abs(next_pos[1] - current[1]) == 2 else 1
                new_cost = cost_so_far[current] + movement_cost

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        return False

    def create_target_marker(self, point: Tuple[float, float], robot_name: str, marker_id: int) -> Marker:
        """創建目標點標記"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_target"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        if robot_name == 'robot1':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            
        return marker

    def publish_visualization(self):
        """發布目標點的可視化"""
        if not all(self.assigned_targets.values()) or \
           self.robot1_pose is None or self.robot2_pose is None:
            return

        marker_array = MarkerArray()
        
        for robot_name in ['robot1', 'robot2']:
            if self.assigned_targets[robot_name]:
                marker_array.markers.append(
                    self.create_target_marker(
                        self.assigned_targets[robot_name],
                        robot_name,
                        len(marker_array.markers)
                    )
                )
        
        self.target_viz_pub.publish(marker_array)

    def assign_targets(self):
        """分配目標給機器人"""
        if not self.available_points or self.robot1_pose is None or \
            self.robot2_pose is None or self.map_data is None:
            return

        MIN_DISTANCE = 1.0
        robots = {
            'robot1': self.robot1_pose,
            'robot2': self.robot2_pose
        }

        assigned_points = set()
        for robot, target in self.assigned_targets.items():
            if target is not None:
                assigned_points.add(tuple(target))

        for robot_name, robot_pose in robots.items():
            # 只在機器人可用且沒有當前目標時分配新目標
            if not self.robot_status[robot_name] or self.assigned_targets[robot_name] is not None:
                continue

            if not set(self.available_points) - assigned_points:
                break

            robot_map_pos = self.world_to_map(
                robot_pose.position.x,
                robot_pose.position.y
            )

            valid_targets = []
            for point in self.available_points:
                if tuple(point) in assigned_points:
                    continue
                    
                direct_dist = np.sqrt(
                    (point[0] - robot_pose.position.x)**2 + 
                    (point[1] - robot_pose.position.y)**2
                )
                
                if direct_dist < MIN_DISTANCE:
                    continue

                target_map_pos = self.world_to_map(point[0], point[1])
                
                if self.a_star(robot_map_pos, target_map_pos):
                    valid_targets.append((point, direct_dist))

            if valid_targets:
                closest_point = min(valid_targets, key=lambda x: x[1])[0]
                assigned_points.add(tuple(closest_point))
                self.assigned_targets[robot_name] = closest_point

                target_pose = PoseStamped()
                target_pose.header.frame_id = 'merge_map'
                target_pose.header.stamp = self.get_clock().now().to_msg()
                target_pose.pose.position.x = closest_point[0]
                target_pose.pose.position.y = closest_point[1]
                target_pose.pose.orientation.w = 1.0

                if robot_name == 'robot1':
                    self.robot1_target_pub.publish(target_pose)
                else:
                    self.robot2_target_pub.publish(target_pose)

                debug_msg = String()
                debug_msg.data = f'Assigned target {closest_point} to {robot_name}'
                self.debug_pub.publish(debug_msg)
                self.get_logger().info(debug_msg.data)
            else:
                self.get_logger().warn(f'No valid path found for {robot_name}')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = GreedyAssigner()
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