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
        self.alert_subscriber = self.create_subscription(Bool, "safety/alert", self.alert_callback, 10)
        self.laser_subscriber = self.create_subscription(LaserScan, self.laser_callback, 10)

        self.get_logger().info("Safety Initialized")
        self.alert = False

    def alert_callback(self, msg):
        self.alert = msg.data #determined by central node which reads in pose, sends alert calls

    def laser_callback(self, msg):
        # Alerted to maybe stop if in range (from central control loop)
        stop_msg = Bool()
        stop_msg.data = False
        if self.alert:
            #TODO: check LIDAR data for incoming collision
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            incr = msg.angle_increment
            all_angles = np.linspace(angle_min,angle_max,incr)
            start = -15*np.pi/180
            stop = 15*np.pi/180
            desired = all_angles >= start and all_angles <= stop       
            front = np.array(msg.ranges)[desired]

            if min(front) < 0.5: #0.5m
                stop_msg.data = True
                self.get_logger().info('Stopping')
        # else:
        #     self.get_logger().info('Not alerted - safety')
        self.publisher.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    safety_controller = SafetyNode()
    rclpy.spin(safety_controller)
    rclpy.shutdown()

if __name__=="__main__":
    main()