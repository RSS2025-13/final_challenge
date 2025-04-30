import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from enum import Enum, auto
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Header, String
from sensor_msgs.msg import Image
from color_segmentation import cd_color_segmentation
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

class HeistState(Enum):
    IDLE = auto()              # Initial state, waiting for start
    PLANNING = auto()          # Planning path to next location
    NAVIGATING = auto()        # Following path to next location
    STOPLIGHT_DETECTED = auto()# Detected a stoplight
    WAITING_AT_LIGHT = auto()  # Waiting at stoplight
    BANANA_DETECTED = auto()   # Detected a banana, need to verify if it's the correct part
    VERIFYING = auto()         # Verifying if the detected banana is the correct part
    COLLECTING = auto()        # Stopped at correct banana, collecting part
    WAITING = auto()           # Waiting after collecting a part
    FINISHED = auto()          # All parts collected, mission complete
    ERROR = auto()             # Error state for handling issues

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')
        
        # State variables
        self.state = HeistState.IDLE
        self.bridge = CvBridge()
        self.parts_collected = 0
        self.total_parts = 2  # We need to collect 2 parts
        self.current_goal = None
        self.wait_timer = None
        self.wait_duration = 5.0  # seconds to wait at banana
        self.stoplight_wait_duration = 3.0  # seconds to wait at stoplight
        self.shrinkray_points = []  # List of points to visit
        self.current_path = None
        self.path_index = 0
        
        # Pure pursuit parameters
        self.lookahead = 2.0  # meters
        self.speed = 2.0  # meters/second
        self.wheelbase_length = 0.34  # meters
        
        # Subscribers
        self.image_subscriber = self.create_subscription(
            Image, '/zed/zed_node/rgb/image_rect_color', 
            self.image_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', 
            self.odom_callback, 10)
        self.banana_detector_subscriber = self.create_subscription(
            Bool, '/banana_detected', 
            self.banana_detector_callback, 10)
        self.shrinkray_part_subscriber = self.create_subscription(
            PoseArray, '/shrinkray_part', 
            self.shrinkray_part_callback, 10)
        self.path_subscriber = self.create_subscription(
            Path, '/path', 
            self.path_callback, 10)
        
        # Publishers
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, '/ackermann_cmd_mux/input/navigation', 10)
        self.trajectory_publisher = self.create_publisher(
            PoseArray, '/trajectory/current', 10)
        self.state_publisher = self.create_publisher(
            Bool, '/heist_state', 10)
        self.goal_publisher = self.create_publisher(
            PoseStamped, '/goal_pose', 10)
        
        self.get_logger().info('Shrink Ray Heist State Machine initialized!')

    def image_callback(self, image_msg):
        try:
            # Convert ROS Image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            # Check for stoplight
            if self.detect_stoplight(image) and self.state == HeistState.NAVIGATING:
                self.state = HeistState.STOPLIGHT_DETECTED
                self.get_logger().info('Stoplight detected! Stopping...')
                self.stop_car()
                self.enter_stoplight_wait_state()
                return
            
            # Check for bananas
            banana_detected = cd_color_segmentation(image)
            if banana_detected and self.state == HeistState.NAVIGATING:
                self.state = HeistState.BANANA_DETECTED
                self.get_logger().info('Banana detected! Entering verification state.')
                
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            self.state = HeistState.ERROR

    def detect_stoplight(self, image):
        # TODO: Implement stoplight detection
        # This could use color segmentation or other computer vision techniques
        # Return True if stoplight is detected
        return False  # Placeholder

    def path_callback(self, msg):
        if self.state == HeistState.PLANNING:
            self.current_path = msg.poses
            self.path_index = 0
            self.state = HeistState.NAVIGATING
            self.get_logger().info('Path received, starting navigation')

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
        if self.state == HeistState.NAVIGATING:
            if self.is_at_goal():
                if self.path_index < len(self.current_path) - 1:
                    self.path_index += 1
                    self.publish_next_goal()
                else:
                    self.state = HeistState.WAITING
                    self.enter_wait_state()
            else:
                # Use pure pursuit to follow the path
                self.follow_path()

    def follow_path(self):
        if not self.current_path or self.path_index >= len(self.current_path):
            return

        # Get current position and orientation
        x_car = self.current_pose.position.x
        y_car = self.current_pose.position.y
        orientation = self.current_pose.orientation
        sin = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cos = 1 - 2 * (orientation.y**2 + orientation.z**2)
        yaw = np.arctan2(sin, cos)

        # Find the closest point on the path
        path_pts = np.array([[p.position.x, p.position.y] for p in self.current_path])
        dists = np.linalg.norm(path_pts - np.array([x_car, y_car]), axis=1)
        closest_idx = np.argmin(dists)

        # Find lookahead point
        lookahead_pt = None
        for i in range(closest_idx, len(path_pts) - 1):
            p1 = path_pts[i]
            p2 = path_pts[i + 1]

            Q = np.array([x_car, y_car])
            r = self.lookahead
            V = p2 - p1           
            F = p1 - Q

            a = np.dot(V, V)
            b = 2 * np.dot(V, F)
            c = np.dot(F, F) - r**2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0: 
                continue

            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            if 0 <= t1 <= 1:
                lookahead_pt = p1 + t1 * V
                break
            elif 0 <= t2 <= 1:
                lookahead_pt = p1 + t2 * V
                break

        if lookahead_pt is None:
            return

        # Calculate lookahead point in car's frame
        dx = lookahead_pt[0] - x_car
        dy = lookahead_pt[1] - y_car
        local_x = np.cos(-yaw) * dx - np.sin(-yaw) * dy
        local_y = np.sin(-yaw) * dx + np.cos(-yaw) * dy

        if local_x == 0:
            return

        # Calculate steering angle using pure pursuit
        curvature = (2 * local_y) / (self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase_length * curvature)

        # Publish drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(drive_msg)

    def publish_next_goal(self):
        if self.current_path and self.path_index < len(self.current_path):
            goal = PoseStamped()
            goal.header = Header()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = "map"
            goal.pose = self.current_path[self.path_index]
            self.goal_publisher.publish(goal)

    def enter_stoplight_wait_state(self):
        self.state = HeistState.WAITING_AT_LIGHT
        self.get_logger().info(f'Waiting {self.stoplight_wait_duration} seconds at stoplight...')
        
        if self.wait_timer:
            self.wait_timer.cancel()
        self.wait_timer = self.create_timer(self.stoplight_wait_duration, self.stoplight_wait_done)

    def stoplight_wait_done(self):
        self.state = HeistState.NAVIGATING
        self.get_logger().info('Stoplight wait complete, resuming navigation')
        
        if self.wait_timer:
            self.wait_timer.cancel()
            self.wait_timer = None

    def banana_detector_callback(self, msg):
        if msg.data and self.state == HeistState.BANANA_DETECTED:
            self.state = HeistState.VERIFYING
            self.get_logger().info('Verifying banana...')

    def shrinkray_part_callback(self, msg):
        if self.state == HeistState.VERIFYING:
            # Verify if this is the correct part
            if self.is_correct_part(msg):
                self.state = HeistState.COLLECTING
                self.get_logger().info('Correct part found! Collecting...')
                self.stop_car()
                self.enter_wait_state()

    def is_at_goal(self):
        if not self.current_path or self.path_index >= len(self.current_path):
            return False
            
        # Calculate distance to current path point
        current_goal = self.current_path[self.path_index]
        dx = self.current_pose.position.x - current_goal.position.x
        dy = self.current_pose.position.y - current_goal.position.y
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance < 0.5  # Within 0.5 meters of goal

    def is_correct_part(self, part_msg):
        # TODO: Implement logic to verify if this is the correct part
        # This could involve checking specific features or properties of the part
        return True  # Placeholder

    def enter_wait_state(self):
        self.state = HeistState.WAITING
        self.get_logger().info(f'Waiting {self.wait_duration} seconds...')
        
        if self.wait_timer:
            self.wait_timer.cancel()
        self.wait_timer = self.create_timer(self.wait_duration, self.wait_done)

    def wait_done(self):
        self.parts_collected += 1
        self.get_logger().info(f'Part {self.parts_collected}/{self.total_parts} collected!')
        
        if self.parts_collected >= self.total_parts:
            self.state = HeistState.FINISHED
            self.get_logger().info('All parts collected! Mission complete!')
        else:
            self.state = HeistState.PLANNING
            self.get_logger().info('Planning path to next location...')
        
        if self.wait_timer:
            self.wait_timer.cancel()
            self.wait_timer = None

    def stop_car(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(drive_msg)

    def publish_state(self):
        state_msg = Bool()
        state_msg.data = self.state != HeistState.FINISHED
        self.state_publisher.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
