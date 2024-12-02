import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import threading
import math, time

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
            
        # Parameters
        self.lookahead_distance = 0.25  # 前視距離
        self.speed = 0.15  # 最大速度
        self.goal_tolerance = 0.15  # 目標容許誤差
        
        # Initialize states
        self.x = None
        self.y = None
        self.yaw = None
        self.path = None
        self.following_path = False
        
        self.get_logger().info("Path follower node has been started")

    def euler_from_quaternion(self, x, y, z, w):
        """Convert quaternion to yaw angle"""
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def pure_pursuit(self, current_x, current_y, current_heading, path, index):
        """改進的Pure Pursuit算法,實現更平滑的轉向"""
        closest_point = None
        v = self.speed
        
        # 尋找前視點
        for i in range(index, len(path)):
            x = path[i][0]
            y = path[i][1]
            distance = math.hypot(current_x - x, current_y - y)
            if self.lookahead_distance < distance:
                closest_point = (x, y)
                index = i
                break
                
        # 計算轉向角
        if closest_point is not None:
            target_heading = math.atan2(closest_point[1] - current_y, 
                                    closest_point[0] - current_x)
        else:
            target_heading = math.atan2(path[-1][1] - current_y, 
                                    path[-1][0] - current_x)
            index = len(path)-1
            
        desired_steering_angle = target_heading - current_heading
        
        # 角度歸一化到[-pi, pi]
        if desired_steering_angle > math.pi:
            desired_steering_angle -= 2 * math.pi
        elif desired_steering_angle < -math.pi:
            desired_steering_angle += 2 * math.pi
            
        # 轉彎時漸進式降速
        angle_threshold = math.pi/6
        if abs(desired_steering_angle) > angle_threshold:
            # 根據轉向角度按比例降速
            speed_factor = 1.0 - (abs(desired_steering_angle) - angle_threshold) / (math.pi/2 - angle_threshold)
            speed_factor = max(0.3, speed_factor)  # 保持最低30%速度
            v *= speed_factor
            
            # 限制最大轉向角
            if abs(desired_steering_angle) > math.pi/3:
                sign = 1 if desired_steering_angle > 0 else -1
                desired_steering_angle = sign * math.pi/3
                
        return v, desired_steering_angle, index

    def path_callback(self, msg):
        """Handle new path data"""
        self.get_logger().info("New path received")
        
        # Convert path data format
        data_list = list(msg.data)
        self.path = [(data_list[i], data_list[i+1]) 
                    for i in range(0, len(data_list), 2)]
        
        # Start new path following if not already following
        if not self.following_path:
            threading.Thread(target=self.follow_path).start()

    def follow_path(self):
        """Path following control loop"""
        self.following_path = True
        
        twist = Twist()
        path_msg = Path()
        path_msg.header.frame_id = "map"
        
        # Wait for initial position
        while self.x is None:
            time.sleep(0.1)
            
        # Create visualization path message
        for x, y in self.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)
            
        index = 0
        while self.following_path:
            # Update timestamp
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Calculate control commands
            v, w, index = self.pure_pursuit(
                self.x, self.y, self.yaw,
                self.path, index
            )
            
            # Check if goal reached
            if (abs(self.x - self.path[-1][0]) < self.goal_tolerance and 
                abs(self.y - self.path[-1][1]) < self.goal_tolerance):
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.publisher_cmd_vel.publish(twist)
                self.get_logger().info("Goal reached")
                break
                
            # Publish control commands and visualization path
            twist.linear.x = v
            twist.angular.z = w
            self.publisher_visual_path.publish(path_msg)
            self.publisher_cmd_vel.publish(twist)
            
            time.sleep(0.1)
            
        self.following_path = False

    def odom_callback(self, msg):
        """Handle odometry data"""
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
