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
        self.speed = 0.18  # Robot speed in m/s
        
        # Initialize robot states
        self.robot1_path = []
        self.robot2_path = []
        self.visited_points = []
        self.has_map = False
        
        # Publishers for robot paths
        self.publisher_robot1_path = self.create_publisher(Float32MultiArray, 'robot1/path', 10)
        self.publisher_robot2_path = self.create_publisher(Float32MultiArray, 'robot2/path', 10)
        
        # Publishers for visualization
        self.viz_publisher_robot1 = self.create_publisher(Path, 'robot1/viz_path', 10)
        self.viz_publisher_robot2 = self.create_publisher(Path, 'robot2/viz_path', 10)
        
        # Subscriber for map
        self.subscription_map = self.create_subscription(
            OccupancyGrid,
            'merge_map',
            self.map_callback,
            10)
        
        # Subscribers for robot odometry
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
        
        # Subscribers for robot status
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
        
        # Subscribers for goal poses
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

        # Initialize robot positions
        self.robot1_x = 0.0
        self.robot1_y = 0.0
        self.robot2_x = 0.0
        self.robot2_y = 0.0
        
        # Initialize robot status
        self.robot1_status = True  # True means robot is ready for new goal
        self.robot2_status = True
        
        print("[INFO] Robot Control System Active")

    def map_callback(self, msg):
        """Store map data for path planning"""
        self.has_map = True
        self.map_data = msg
        self.resolution = msg.info.resolution
        self.originX = msg.info.origin.position.x
        self.originY = msg.info.origin.position.y
        self.width = msg.info.width
        self.height = msg.info.height
        self.data = np.array(msg.data).reshape(self.height, self.width)
        # Convert to binary map (0 for free space, 1 for obstacles)
        self.data = np.where(self.data > 50, 1, 0)
        print("[INFO] Map received")

    def robot1_odom_callback(self, msg):
        """Update robot1 position from odometry"""
        self.robot1_x = msg.pose.pose.position.x
        self.robot1_y = msg.pose.pose.position.y

    def robot2_odom_callback(self, msg):
        """Update robot2 position from odometry"""
        self.robot2_x = msg.pose.pose.position.x
        self.robot2_y = msg.pose.pose.position.y

    def robot1_status_control(self, msg):
        """Monitor robot1 movement status"""
        if msg.linear.x == 0 and msg.angular.z == 0:
            self.robot1_status = True
        else:
            self.robot1_status = False

    def robot2_status_control(self, msg):
        """Monitor robot2 movement status"""
        if msg.linear.x == 0 and msg.angular.z == 0:
            self.robot2_status = True
        else:
            self.robot2_status = False

    def world_to_map(self, x, y):
        """Convert world coordinates to map coordinates"""
        map_x = int((x - self.originX) / self.resolution)
        map_y = int((y - self.originY) / self.resolution)
        return (map_y, map_x)  # Note: map coordinates are (row, col)

    def map_to_world(self, row, col):
        """Convert map coordinates to world coordinates"""
        world_x = col * self.resolution + self.originX
        world_y = row * self.resolution + self.originY
        return (world_x, world_y)

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """Plan path from start to goal"""
        if not self.has_map:
            print("[WARNING] No map data available for path planning")
            return [(start_x, start_y), (goal_x, goal_y)]

        # Convert world coordinates to map coordinates
        start_map = self.world_to_map(start_x, start_y)
        goal_map = self.world_to_map(goal_x, goal_y)

        # Check if coordinates are within map bounds
        if (not (0 <= start_map[0] < self.height and 0 <= start_map[1] < self.width) or
            not (0 <= goal_map[0] < self.height and 0 <= goal_map[1] < self.width)):
            print("[WARNING] Start or goal position out of map bounds")
            return [(start_x, start_y), (goal_x, goal_y)]

        # Plan path using A*
        path_map = astar(self.data, start_map, goal_map)
        
        if not path_map:
            print("[WARNING] No path found")
            return [(start_x, start_y), (goal_x, goal_y)]

        # Convert path to world coordinates
        path_world = [self.map_to_world(p[0], p[1]) for p in path_map]
        
        # Smooth path using B-spline
        if len(path_world) > 2:
            path_world = bspline_planning(path_world, len(path_world) * 3)

        return path_world

    def publish_path(self, path, robot_number):
        """Publish path for specified robot and visualize it"""
        # Store current path
        if robot_number == 1:
            self.robot1_path = path
        else:
            self.robot2_path = path
            
        # Publish Float32MultiArray message
        message = Float32MultiArray()
        message.data = [elem for point in path for elem in point]
        
        # Create visualization message
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
        
        # Calculate path execution time
        path_length = sum(np.sqrt(np.sum(np.diff(np.array(path), axis=0)**2, axis=1)))
        t = path_length/self.speed - 0.2
        if t < 0:
            t = 0
            
        # Publish messages
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
        """Handle new goal for robot1"""
        if self.robot1_status:
            now = datetime.datetime.now()
            print(f"[INFO] {now.strftime('%H:%M:%S')}: Planning path for Robot 1")
            
            path = self.plan_path(
                self.robot1_x, self.robot1_y,
                msg.pose.position.x, msg.pose.position.y
            )
            
            self.publish_path(path, 1)
            print(f"[INFO] {now.strftime('%H:%M:%S')}: Path published for Robot 1")

    def robot2_goal_callback(self, msg):
        """Handle new goal for robot2"""
        if self.robot2_status:
            now = datetime.datetime.now()
            print(f"[INFO] {now.strftime('%H:%M:%S')}: Planning path for Robot 2")
            
            path = self.plan_path(
                self.robot2_x, self.robot2_y,
                msg.pose.position.x, msg.pose.position.y
            )
            
            self.publish_path(path, 2)
            print(f"[INFO] {now.strftime('%H:%M:%S')}: Path published for Robot 2")

    def robot1_goal_reached(self):
        """Called when robot1 is expected to reach its goal"""
        self.get_logger().info('Robot 1 reached goal')
        self.robot1_status = True

    def robot2_goal_reached(self):
        """Called when robot2 is expected to reach its goal"""
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
