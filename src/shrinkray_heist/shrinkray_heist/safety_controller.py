import rclpy
from rclpy.node import Node

import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_controller")
        self.publisher = self.create_publisher(Bool, "/safety/stop", 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/vesc/low_level/input/safety", 10)
        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("Safety Initialized")
        self.MIN_RANGE = 0.5  # 0.5m
        self.FOV_ANGLE = 15*np.pi/180
        self.stopped = False
        self.prev_time = 0
        self.last_drive_cmd = None

    def laser_callback(self, msg):
        self.scan = np.array(msg.ranges)
        self.angle_range = [msg.angle_min, msg.angle_max]

    def timer_callback(self):
        stop_msg = Bool()
        stop_msg.data = False
        
        current_time = self.get_clock().now().nanoseconds/1e9
        
        if self.stopped:
            if (current_time - self.prev_time) < 3.0:
                stop_msg.data = True
                return
            else:
                self.stopped = False
                # Resume previous drive command if available
                if self.last_drive_cmd is not None:
                    self.drive_publisher.publish(self.last_drive_cmd)
                    self.last_drive_cmd = None
                return

        # Check for obstacles
        all_angles = np.linspace(self.angle_range[0], self.angle_range[1], len(self.scan))
        desired = np.where(np.abs(all_angles) <= self.FOV_ANGLE)
        front = self.scan[desired]

        if min(front) < self.MIN_RANGE:
            stop_msg.data = True
            self.stopped = True
            self.prev_time = current_time
            self.get_logger().info('Stopping due to obstacle')
            
        self.publisher.publish(stop_msg)

    def drive_callback(self, msg):
        # Store the last drive command for resuming after stop
        if not self.stopped:
            self.last_drive_cmd = msg

def main(args=None):
    rclpy.init(args=args)
    safety_controller = SafetyNode()
    rclpy.spin(safety_controller)
    rclpy.shutdown()

if __name__=="__main__":
    main()