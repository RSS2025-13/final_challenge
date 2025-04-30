import rclpy
from rclpy.node import Node

import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan

class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_controller")
        self.publisher = self.create_publisher(Bool, "/safety/stop", 10)
        #self.debug_publisher = self.create_publisher(Image, "/safety/debug", 1)
        self.timer = self.create_subscription(0.1,self.timer_callback)
        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        self.get_logger().info("Safety Initialized")
        self.MIN_RANGE = 0.5 #0.5m
        self.FOV_ANGLE = 15*np.pi/180
        self.stopped = False
        self.prev_time = 0

    def laser_callback(self, msg):
        self.scan = np.array(msg.ranges)
        self.angle_range = [msg.angle_min, msg.angle_max]

    def timer_callback(self):
        # Alerted to maybe stop if in range (from central control loop)
        stop_msg = Bool()
        stop_msg.data = False
        if self.stopped and (self.get_clock().now().nanoseconds/1e9 - self.prev_time) < 3:
            return
        else:
            self.stopped = False #default
            #TODO: check LIDAR data for incoming collision
            all_angles = np.linspace(self.angle_range[0],self.angle_range[1],len(self.scan))
            desired = np.where(np.abs(all_angles) <= self.FOV_ANGLE)
            front = self.scan[desired]

            if min(front) < self.MIN_RANGE: #0.5m
                stop_msg.data = True
                self.stoped = True
                self.prev_time = self.get_clock().now().nanoseconds/1e9
                self.get_logger().info('Stopping')
        self.publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    safety_controller = SafetyNode()
    rclpy.spin(safety_controller)
    rclpy.shutdown()

if __name__=="__main__":
    main()