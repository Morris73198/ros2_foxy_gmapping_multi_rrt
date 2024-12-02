import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from std_msgs.msg import Float32MultiArray
import numpy as np
import heapq, math, time, threading
import scipy.interpolate as si
import datetime

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data = data[::-1]
            return data
            
        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return False

def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i], ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path

class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control')
        
        # Speed settings
        self.speed = 0.18
        
        # Initialize robot states
        self.robot1_path = []
        self.robot2_path = []
        self.visited_points = []
        self.has_map = False
        
        # Add path monitoring parameters
        self.replan_distance = 0.5
        self.active_goals = {'robot1': None, 'robot2': None}
        
        # Publishers for robot paths
        self.publisher_robot1_path = self.create_publisher(Float32MultiArray, 'robot1/path', 10)
        self.publisher_robot2_path = self.create_publisher(Float32MultiArray, 'robot2/path', 10)
        
        # Publishers for visualization
        self.viz_publisher_robot1 = self.create_publisher(Path, 'robot1/viz_path', 10)
        self.viz_publisher_robot2 = self.create_publisher(Path, 'robot2/viz_path', 10)
        
        # Setup subscribers
        self.setup_subscribers()
        
        # Initialize robot positions
        self.robot1_x = 0.0
        self.robot1_y = 0.0
        self.robot2_x = 0.0
        self.robot2_y = 0.0
        
        # Initialize robot status
        self.robot1_status = True
        self.robot2_status = True
        
        # Create path checking timer
        self.check_path_timer = self.create_timer(0.5, self.check_path_validity)
        
        print("[INFO] Robot Control System Active")

    def setup_subscribers(self):
        self.subscription_map = self.create_subscription(
            OccupancyGrid,
            'merge_map',
            self.map_callback,
            10)
            
        self.subscription_robot1_odom = self.create_subscription(
            Odometry,
            'robot1/odom',
            self.robot1_odom_callback,
            10)
            
        self.subscription_robot2_odom = self.create_subscription(
            Odometry,
            'robot2/odom',
            self.robot2_odom_callback,
            10)
            
        self.subscription_robot1_cmd_vel = self.create_subscription(
            Twist,
            'robot1/cmd_vel',
            self.robot1_status_control,
            4)
            
        self.subscription_robot2_cmd_vel = self.create_subscription(
            Twist,
            'robot2/cmd_vel',
            self.robot2_status_control,
            4)
            
        self.subscription_robot1_goal = self.create_subscription(
            PoseStamped,
            'robot1/goal_pose',
            self.robot1_goal_callback,
            10)
            
        self.subscription_robot2_goal = self.create_subscription(
            PoseStamped,
            'robot2/goal_pose',
            self.robot2_goal_callback,
            10)

    def check_path_validity(self):
        if not self.has_map:
            return
            
        for robot_num in [1, 2]:
            self._check_robot_path(robot_num)
    
    def _check_robot_path(self, robot_num):
        path = self.robot1_path if robot_num == 1 else self.robot2_path
        current_x = self.robot1_x if robot_num == 1 else self.robot2_x
        current_y = self.robot1_y if robot_num == 2 else self.robot2_y
        goal = self.active_goals[f'robot{robot_num}']
        
        if not path or not goal:
            return
            
        current_idx = self._find_closest_point_index(path, current_x, current_y)
        
        if self._check_path_collision(path[current_idx:]):
            self.get_logger().info(f'Obstacle detected in robot{robot_num} path. Replanning...')
            new_path = self.plan_path(
                current_x, current_y,
                goal.pose.position.x, goal.pose.position.y
            )
            if new_path:
                self.publish_path(new_path, robot_num)
            else:
                self.get_logger().warn(f'No valid path found for robot{robot_num}')

    def _find_closest_point_index(self, path, x, y):
        distances = [np.hypot(p[0] - x, p[1] - y) for p in path]
        return np.argmin(distances)
    
    def _check_path_collision(self, path):
        for i in range(len(path) - 1):
            if self._check_line_collision(path[i], path[i + 1]):
                return True
        return False
    
    def _check_line_collision(self, start, end):
        points = self._bresenham_line(
            self.world_to_map(*start),
            self.world_to_map(*end)
        )
        
        for x, y in points:
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            if self.data[y][x] == 1:  # Obstacle
                return True
        return False
    
    def _bresenham_line(self, start, end):
        x1, y1 = start
        x2, y2 = end
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        
        if dx > dy:
            err = dx / 2
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        points.append((x, y))
        return points

    def map_callback(self, msg):
        self.has_map = True
        self.map_data = msg
        self.resolution = msg.info.resolution
        self.originX = msg.info.origin.position.x
        self.originY = msg.info.origin.position.y
        self.width = msg.info.width
        self.height = msg.info.height
        self.data = np.array(msg.data).reshape(self.height, self.width)
        self.data = np.where(self.data > 50, 1, 0)
        print("[INFO] Map received")

    def robot1_odom_callback(self, msg):
        self.robot1_x = msg.pose.pose.position.x
        self.robot1_y = msg.pose.pose.position.y

    def robot2_odom_callback(self, msg):
        self.robot2_x = msg.pose.pose.position.x
        self.robot2_y = msg.pose.pose.position.y

    def robot1_status_control(self, msg):
        if msg.linear.x == 0 and msg.angular.z == 0:
            self.robot1_status = True
        else:
            self.robot1_status = False

    def robot2_status_control(self, msg):
        if msg.linear.x == 0 and msg.angular.z == 0:
            self.robot2_status = True
        else:
            self.robot2_status = False

    def world_to_map(self, x, y):
        map_x = int((x - self.originX) / self.resolution)
        map_y = int((y - self.originY) / self.resolution)
        return (map_y, map_x)

    def map_to_world(self, row, col):
        world_x = col * self.resolution + self.originX
        world_y = row * self.resolution + self.originY
        return (world_x, world_y)

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        if not self.has_map:
            print("[WARNING] No map data available for path planning")
            return [(start_x, start_y), (goal_x, goal_y)]

        start_map = self.world_to_map(start_x, start_y)
        goal_map = self.world_to_map(goal_x, goal_y)

        if (not (0 <= start_map[0] < self.height and 0 <= start_map[1] < self.width) or
            not (0 <= goal_map[0] < self.height and 0 <= goal_map[1] < self.width)):
            print("[WARNING] Start or goal position out of map bounds")
            return [(start_x, start_y), (goal_x, goal_y)]

        path_map = astar(self.data, start_map, goal_map)
        
        if not path_map:
            print("[WARNING] No path found")
            return None

        path_world = [self.map_to_world(p[0], p[1]) for p in path_map]
        
        if len(path_world) > 2:
            path_world = bspline_planning(path_world, len(path_world) * 3)

        return path_world

    def publish_path(self, path, robot_number):
        if robot_number == 1:
            self.robot1_path = path
        else:
            self.robot2_path = path
            
        message = Float32MultiArray()
        message.data = [elem for point in path for elem in point]
        
        viz_path = Path()
        viz_path.header.stamp = self.get_clock().now().to_msg()
        viz_path.header.frame_id = 'merge_map'
        
        for x, y in path:
            pose = PoseStamped()
            pose.header = viz_path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            viz_path.poses.append(pose)
        
        path_length = sum(np.sqrt(np.sum(np.diff(np.array(path), axis=0)**2, axis=1)))
        t = path_length/self.speed - 0.2
        if t < 0:
            t = 0
            
        if robot_number == 1:
            self.publisher_robot1_path.publish(message)
            self.viz_publisher_robot1.publish(viz_path)
            self.visited_points.append(path[-1])
            self.timer1 = threading.Timer(t, self.robot1_goal_reached)
            self.timer1.start()
        else:
            self.publisher_robot2_path.publish(message)
            self.viz_publisher_robot2.publish(viz_path)
            self.visited_points.append(path[-1])
            self.timer2 = threading.Timer(t, self.robot2_goal_reached)
            self.timer2.start()

    def robot1_goal_callback(self, msg):
        if self.robot1_status:
            now = datetime.datetime.now()
            print(f"[INFO] {now.strftime('%H:%M:%S')}: Planning path for Robot 1")
            
            self.active_goals['robot1'] = msg
            path = self.plan_path(
                self.robot1_x, self.robot1_y,
                msg.pose.position.x, msg.pose.position.y
            )
            if path:
                self.publish_path(path, 1)
                print(f"[INFO] {now.strftime('%H:%M:%S')}: Path published for Robot 1")
            else:
                print(f"[WARNING] No valid path found for Robot 1")

    def robot2_goal_callback(self, msg):
        if self.robot2_status:
            now = datetime.datetime.now()
            print(f"[INFO] {now.strftime('%H:%M:%S')}: Planning path for Robot 2")
            
            self.active_goals['robot2'] = msg
            path = self.plan_path(
                self.robot2_x, self.robot2_y,
                msg.pose.position.x, msg.pose.position.y
            )
            
            if path:
                self.publish_path(path, 2)
                print(f"[INFO] {now.strftime('%H:%M:%S')}: Path published for Robot 2")
            else:
                print(f"[WARNING] No valid path found for Robot 2")

    def robot1_goal_reached(self):
        self.get_logger().info('Robot 1 reached goal')
        self.robot1_status = True

    def robot2_goal_reached(self):
        self.get_logger().info('Robot 2 reached goal')
        self.robot2_status = True

def main(args=None):
    rclpy.init(args=args)
    robot_control = RobotControl()
    rclpy.spin(robot_control)
    robot_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()