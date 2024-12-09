import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import numpy as np
import heapq
import math

class DualRobotPathPlanner(Node):
    def __init__(self):
        super().__init__('dual_robot_path_planner')
        
        # 初始化變量
        self.has_map = False
        self.current_paths = {'robot1': None, 'robot2': None}
        self.current_goals = {'robot1': None, 'robot2': None}
        self.robot_positions = {'robot1': None, 'robot2': None}
        self.map_data = None
        self.resolution = None
        self.origin = None
        self.width = None
        self.height = None
        
        # 發布者
        self.path_publishers = {
            'robot1': self.create_publisher(Float32MultiArray, 'robot1/path', 10),
            'robot2': self.create_publisher(Float32MultiArray, 'robot2/path', 10)
        }
        
        self.viz_path_publishers = {
            'robot1': self.create_publisher(Path, 'robot1/viz_path', 10),
            'robot2': self.create_publisher(Path, 'robot2/viz_path', 10)
        }
        
        # 訂閱者
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            'merge_map',
            self.map_callback,
            10
        )
        
        # Robot 1 訂閱
        self.goal_subscription_1 = self.create_subscription(
            PoseStamped,
            'robot1/goal_pose',
            lambda msg: self.goal_callback(msg, 'robot1'),
            10
        )
        
        self.odom_subscription_1 = self.create_subscription(
            Odometry,
            'robot1/odom',
            lambda msg: self.odom_callback(msg, 'robot1'),
            10
        )
        
        # Robot 2 訂閱
        self.goal_subscription_2 = self.create_subscription(
            PoseStamped,
            'robot2/goal_pose',
            lambda msg: self.goal_callback(msg, 'robot2'),
            10
        )
        
        self.odom_subscription_2 = self.create_subscription(
            Odometry,
            'robot2/odom',
            lambda msg: self.odom_callback(msg, 'robot2'),
            10
        )
        
        # 創建定時器進行路徑檢查 (0.5秒)
        self.create_timer(0.5, self.check_paths_validity)
        
        self.get_logger().info('Dual robot path planner has started')

    def map_callback(self, msg):
        """處理地圖數據"""
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.width = msg.info.width
        self.height = msg.info.height
        
        # 轉換為numpy數組並二值化
        self.map_data = np.array(msg.data).reshape((self.height, self.width))
        self.map_data = np.where(self.map_data > 50, 1, 0)
        
        self.has_map = True
        self.get_logger().info('Map received')

    def odom_callback(self, msg, robot_id):
        """更新機器人位置"""
        self.robot_positions[robot_id] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def goal_callback(self, msg, robot_id):
        """處理新的目標點"""
        if not self.has_map or self.robot_positions[robot_id] is None:
            self.get_logger().warn(f'Cannot plan path for {robot_id}: waiting for map or robot position')
            return
            
        goal = (msg.pose.position.x, msg.pose.position.y)
        self.current_goals[robot_id] = goal
        
        # 規劃新路徑
        path = self.plan_path(self.robot_positions[robot_id], goal)
        if path:
            self.current_paths[robot_id] = path
            self.publish_path(path, robot_id)
            self.get_logger().info(f'New path planned for {robot_id}')
        else:
            self.get_logger().warn(f'No valid path found for {robot_id}')

    def world_to_map(self, x, y):
        """將世界坐標轉換為地圖坐標"""
        mx = int((x - self.origin.position.x) / self.resolution)
        my = int((y - self.origin.position.y) / self.resolution)
        return (my, mx)  # 注意：返回 (row, col)

    def map_to_world(self, row, col):
        """將地圖坐標轉換為世界坐標"""
        x = col * self.resolution + self.origin.position.x
        y = row * self.resolution + self.origin.position.y
        return (x, y)

    def heuristic(self, a, b):
        """計算啟發式值（歐幾里得距離）"""
        return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def is_point_safe(self, pos):
        """檢查一個點是否滿足安全距離要求"""
        row, col = pos
        safety_distance = int(0.2 / self.resolution)  # 0.2米的安全距離
        
        # 檢查安全距離內是否有障礙物
        for dr in range(-safety_distance, safety_distance + 1):
            for dc in range(-safety_distance, safety_distance + 1):
                check_row = row + dr
                check_col = col + dc
                
                # 檢查點是否在地圖範圍內
                if not (0 <= check_row < self.height and 0 <= check_col < self.width):
                    continue
                
                # 如果在安全距離內發現障礙物
                if self.map_data[check_row][check_col] == 1:
                    # 計算到障礙物的實際距離
                    distance = math.sqrt(dr**2 + dc**2) * self.resolution
                    if distance < 0.2:  # 安全距離為0.3米
                        return False
        return True

    def get_neighbors(self, pos):
        """獲取相鄰的可行點，考慮安全距離"""
        row, col = pos
        neighbors = [
            (row-1, col), (row+1, col),
            (row, col-1), (row, col+1),
            (row-1, col-1), (row-1, col+1),
            (row+1, col-1), (row+1, col+1)
        ]
        
        valid_neighbors = []
        for n in neighbors:
            if (0 <= n[0] < self.height and 
                0 <= n[1] < self.width and 
                self.is_point_safe(n)):
                valid_neighbors.append(n)
        return valid_neighbors

    def a_star(self, start, goal):
        """A*路徑規劃"""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current):
                # 對角線移動的成本為√2，直線移動成本為1
                movement_cost = (math.sqrt(2) if abs(next_pos[0] - current[0]) + 
                               abs(next_pos[1] - current[1]) == 2 else 1)
                new_cost = cost_so_far[current] + movement_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # 重建路徑
        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path

    def plan_path(self, start, goal):
        """規劃路徑（從世界坐標到世界坐標），考慮安全距離"""
        if not self.has_map:
            return None
            
        # 轉換為地圖坐標
        start_map = self.world_to_map(*start)
        goal_map = self.world_to_map(*goal)
        
        # 檢查起點和終點是否有效且安全
        if not (0 <= start_map[0] < self.height and 
                0 <= start_map[1] < self.width and
                0 <= goal_map[0] < self.height and 
                0 <= goal_map[1] < self.width):
            return None
            
        if not self.is_point_safe(start_map) or not self.is_point_safe(goal_map):
            return None
            
        # 使用A*尋找路徑
        path_map = self.a_star(start_map, goal_map)
        if not path_map:
            return None
            
        # 轉換回世界坐標
        return [self.map_to_world(p[0], p[1]) for p in path_map]

    def check_paths_validity(self):
        """檢查所有機器人的當前路徑是否有效"""
        if not self.has_map:
            return
            
        for robot_id in ['robot1', 'robot2']:
            if self.current_paths[robot_id] and self.current_goals[robot_id]:
                if self.check_single_path_validity(robot_id):
                    self.get_logger().info(f'Path blocked for {robot_id}, replanning...')
                    # 重新規劃路徑
                    new_path = self.plan_path(
                        self.robot_positions[robot_id], 
                        self.current_goals[robot_id]
                    )
                    if new_path:
                        self.current_paths[robot_id] = new_path
                        self.publish_path(new_path, robot_id)
                        self.get_logger().info(f'New path planned for {robot_id}')

    def check_single_path_validity(self, robot_id):
        """檢查單個機器人的路徑是否被阻擋"""
        path = self.current_paths[robot_id]
        for i in range(len(path) - 1):
            if self.is_segment_blocked(path[i], path[i + 1]):
                return True
        return False

    def is_segment_blocked(self, start, end):
        """檢查兩點之間的線段是否被阻擋，考慮安全距離"""
        start_map = self.world_to_map(*start)
        end_map = self.world_to_map(*end)
        
        # 使用Bresenham算法檢查線段上的點
        points = self.bresenham_line(start_map[0], start_map[1], end_map[0], end_map[1])
        
        for point in points:
            if not self.is_point_safe(point):
                return True
        return False

    def bresenham_line(self, x0, y0, x1, y1):
        """Bresenham直線算法"""
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

    def publish_path(self, path, robot_id):
        """發布路徑"""
        # 發布 Float32MultiArray 用於路徑跟隨
        float_msg = Float32MultiArray()
        float_msg.data = [coord for point in path for coord in point]
        self.path_publishers[robot_id].publish(float_msg)
        
        # 發布 Path 用於視覺化
        path_msg = Path()
        path_msg.header.frame_id = 'merge_map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            
        self.viz_path_publishers[robot_id].publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DualRobotPathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()