import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

class StoplightNode(Node):
    def __init__(self):
        super().__init__("stoplight_controller")
        self.publisher = self.create_publisher(Bool, "/stoplight/red", 10)
        self.debug_publisher = self.create_publisher(Image, "/stoplight/marked_img", 1)
        self.alert_subscriber = self.create_subscription(Bool, "stoplight/alert", self.alert_callback, 10)
        self.image_subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 1)
        self.bridge = CvBridge()

        self.get_logger().info("Stoplight Initialized")
        self.alert = False

    def alert_callback(self, msg):
        self.alert = msg.data #determined by central node which reads in pose, sends alert calls

    def image_callback(self, img_msg):
        # Process image with CV Bridge
        red_msg = Bool()
        red_msg.data = False
        if self.alert:
            #TODO: run image through color segmentation to detect redlight
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            #optimization on pictures?
            lower_red_bound = np.array([0, int(255*0.8), int(255*0.8)])
            upper_red_bound = np.array([8, 255, 255])
            mask = cv2.inRange(hsv_img, lower_red_bound, upper_red_bound)

            # Find the largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)  # Find the contour with the largest area
                largest_blob_area = cv2.contourArea(largest_contour)  # Get the area of the largest blob
                if largest_blob_area > 40:
                    red_msg.data = True
                else:
                    self.get_logger().info(f"Contour is only {largest_blob_area}")
            else:
                self.get_logger().info("No contours found")

            out = cv2.bitwise_and(img,mask)
            img_out = self.bridge.cv2_to_imgmsg(out)
            self.debug_publisher.publish(img_out)
        # else:
        #     self.get_logger().info('Not alerted - stoplight')
        self.publisher.publish(red_msg)

def main(args=None):
    rclpy.init(args=args)
    stoplight_controller = StoplightNode()
    rclpy.spin(stoplight_controller)
    rclpy.shutdown()

if __name__=="__main__":
    main()