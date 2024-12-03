import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import threading
import math
import time

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        
        # Publishers
        self.publisher_visual_path = self.create_publisher(Path, 'visual_path', 10)
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.subscription_path = self.create_subscription(
            Float32MultiArray,
            'path',
            self.path_callback,
            10)
            
        self.subscription_odom = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
            
        # Improved PID Parameters
        self.Kp = 0.4  # Reduced from 0.6
        self.Ki = 0.0005  # Reduced from 0.001
        self.Kd = 0.4  # Increased derivative term
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_integral = 0.3  # Reduced integral limit
        self.dt = 0.1
        
        # Improved Motion Parameters
        self.lookahead_distance = 0.15  # Increased lookahead
        self.speed = 0.15
        self.goal_tolerance = 0.15
        self.max_angular_velocity = 0.6  # Reduced max angular velocity
        self.min_speed = 0.05
        
        # Turn Control Parameters
        self.angular_velocity_filter = 0.8  # Smoothing factor
        self.turn_detection_threshold = math.pi/3
        self.turn_deceleration_factor = 0.6
        self.min_turn_radius = 0.3
        
        # State Variables
        self.x = None
        self.y = None
        self.yaw = None
        self.path = None
        self.following_path = False
        self.prev_angular_velocity = 0.0
        self.current_path_index = 0
        
        self.get_logger().info("Enhanced PID Path follower node started")

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def calculate_pid(self, error):
        # Anti-windup integral with improved limits
        self.integral = max(min(self.integral + error * self.dt, self.max_integral), -self.max_integral)
        
        # Calculate PID terms
        p_term = self.Kp * error
        i_term = self.Ki * self.integral
        d_term = self.Kd * (error - self.prev_error) / self.dt
        
        # Enhanced derivative filtering
        d_term = 0.8 * d_term + 0.2 * (self.prev_error / self.dt)
        self.prev_error = error
        
        # Calculate total control output
        output = p_term + i_term + d_term
        
        # Apply smoothing filter
        output = self.angular_velocity_filter * output + (1 - self.angular_velocity_filter) * self.prev_angular_velocity
        self.prev_angular_velocity = output
        
        return output

    def detect_sharp_turn(self, current_x, current_y, path, index):
        if index + 2 >= len(path):
            return False, 0.0
            
        # Calculate angles between consecutive path segments
        v1 = (path[index+1][0] - path[index][0], path[index+1][1] - path[index][1])
        v2 = (path[index+2][0] - path[index+1][0], path[index+2][1] - path[index+1][1])
        
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if v1_mag * v2_mag == 0:
            return False, 0.0
            
        cos_angle = dot_product / (v1_mag * v2_mag)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle = math.acos(cos_angle)
        
        return angle > self.turn_detection_threshold, angle

    def pure_pursuit(self, current_x, current_y, current_heading, path, index):
        closest_point = None
        v = self.speed
        
        # Enhanced turn detection
        is_sharp_turn, turn_angle = self.detect_sharp_turn(current_x, current_y, path, index)
        
        # Look ahead point selection with dynamic distance
        current_lookahead = self.lookahead_distance
        if is_sharp_turn:
            current_lookahead *= (1.0 + turn_angle / math.pi)
            v *= self.turn_deceleration_factor
        
        # Find target point with improved lookahead
        for i in range(index, len(path)):
            x = path[i][0]
            y = path[i][1]
            distance = math.hypot(current_x - x, current_y - y)
            
            if current_lookahead < distance:
                # Calculate perpendicular distance to path
                if i > 0:
                    path_vector = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                    to_robot = (current_x - path[i-1][0], current_y - path[i-1][1])
                    path_len = math.sqrt(path_vector[0]**2 + path_vector[1]**2)
                    if path_len > 0:
                        # Normalize path vector
                        path_vector = (path_vector[0]/path_len, path_vector[1]/path_len)
                        # Calculate cross track error
                        cross_track = abs(to_robot[0]*path_vector[1] - to_robot[1]*path_vector[0])
                        # Adjust speed based on cross track error
                        v *= max(0.3, 1.0 - cross_track/self.min_turn_radius)
                
                closest_point = (x, y)
                index = i
                break
        
        if closest_point is None:
            closest_point = path[-1]
            index = len(path)-1
        
        # Calculate steering control with improved angle handling
        target_heading = math.atan2(closest_point[1] - current_y,
                                  closest_point[0] - current_x)
        error = target_heading - current_heading
        while error > math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi
        
        # Enhanced speed control for turns
        turn_factor = abs(error) / math.pi
        v *= max(0.3, 1.0 - turn_factor)
        
        # Calculate angular velocity with improved PID
        angular_velocity = self.calculate_pid(error)
        
        # Dynamic angular velocity limits
        current_max_angular_velocity = self.max_angular_velocity
        if is_sharp_turn:
            current_max_angular_velocity *= 0.7
        
        # Apply limits
        angular_velocity = max(min(angular_velocity, current_max_angular_velocity),
                             -current_max_angular_velocity)
        
        return v, angular_velocity, index

    def path_callback(self, msg):
        self.get_logger().info("New path received")
        data_list = list(msg.data)
        self.path = [(data_list[i], data_list[i+1]) 
                    for i in range(0, len(data_list), 2)]
        
        if not self.following_path:
            self.integral = 0.0
            self.prev_error = 0.0
            self.prev_angular_velocity = 0.0
            self.current_path_index = 0
            threading.Thread(target=self.follow_path).start()

    def follow_path(self):
        self.following_path = True
        
        twist = Twist()
        path_msg = Path()
        path_msg.header.frame_id = "map"
        
        while self.x is None:
            time.sleep(0.1)
            
        for x, y in self.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)
        
        while self.following_path:
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Get control commands
            v, w, self.current_path_index = self.pure_pursuit(
                self.x, self.y, self.yaw,
                self.path, self.current_path_index
            )
            
            # Goal check with improved tolerance
            distance_to_goal = math.hypot(
                self.x - self.path[-1][0],
                self.y - self.path[-1][1]
            )
            
            if distance_to_goal < self.goal_tolerance:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.publisher_cmd_vel.publish(twist)
                self.get_logger().info("Goal reached")
                break
            
            # Publish commands
            twist.linear.x = max(self.min_speed, v)
            twist.angular.z = w
            self.publisher_visual_path.publish(path_msg)
            self.publisher_cmd_vel.publish(twist)
            
            time.sleep(self.dt)
            
        self.following_path = False

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = self.euler_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()