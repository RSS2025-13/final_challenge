#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from custom_msgs.msg import LineLocationPixels
          
from computer_vision.color_hough_segmentation import cd_color_hough


class LineDetector(Node):
    """
    A class for applying your line detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image).
    Publishes to: /relative_line_pxs (LineLocationPixels).
    """
    def __init__(self):
        super().__init__("line_detector")

        # Publishers
        self.line_pub = self.create_publisher(LineLocationPixels, "/relative_line_pxs", 10)
        self.debug_pub = self.create_publisher(Image, "/line_debug_img", 10)
        
        # Subscriber
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        
        # Bridge for OpenCV-ROS conversion
        self.bridge = CvBridge()

        self.get_logger().info("Line Detector Initialized")

    def image_callback(self, image_msg):
        try:
            # Convert ROS image message to OpenCV image
            image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # Get lines from cd_color_hough
        lines = cd_color_hough(np.asarray(image), None)

        if not lines or len(lines) != 2:
            self.get_logger().warn("Did not detect both left and right lane lines.")
            return

        left_line, right_line = lines
        (lx1, ly1), (lx2, ly2) = left_line
        (rx1, ry1), (rx2, ry2) = right_line

        # Fit lines in y = m*x + b form to allow interpolation
        left_m = (ly2 - ly1) / (lx2 - lx1 + 1e-6)  # add small value to avoid div by zero
        left_b = ly1 - left_m * lx1

        right_m = (ry2 - ry1) / (rx2 - rx1 + 1e-6)
        right_b = ry1 - right_m * rx1

        # Invert to x = (y - b)/m because we sample by y
        num_points = 10
        min_y = int(max(min(ly1, ly2), min(ry1, ry2)))  # ROI min y
        max_y = int(min(max(ly1, ly2), max(ry1, ry2)))  # ROI max y

        ys = np.linspace(min_y, max_y, num_points)

        center_points = []

        for y in ys:
            lx = (y - left_b) / (left_m + 1e-6)
            rx = (y - right_b) / (right_m + 1e-6)
            cx = (lx + rx) / 2.0
            pt = Point()
            pt.x = float(cx)
            pt.y = float(y)
            pt.z = 0.0
            center_points.append(pt)

        # Publish the path
        line_msg = LineLocationPixels()
        line_msg.pixels = center_points
        self.line_pub.publish(line_msg)

        # Debug draw
        debug_img = image.copy()
        cv2.line(debug_img, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
        cv2.line(debug_img, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        for pt in center_points:
            cv2.circle(debug_img, (int(pt.x), int(pt.y)), 3, (255, 0, 0), -1)

        try:
            debug_img_msg = self.bridge.cv2_to_imgmsg(debug_img, "bgr8")
            self.debug_pub.publish(debug_img_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Debug image publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    line_detector = LineDetector()
    rclpy.spin(line_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
