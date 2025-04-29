import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

class StoplightNode(Node):
    def __init__(self):
        super().__init__("stoplight_control")
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
        if self.alert:
            image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            #TODO: run image through color segmentation to detect redlight
            
            red_msg = Bool()
            red_msg.data = False
            self.publisher.publish(red_msg)

            out = 3
            img_out = self.bridge.cv2_to_imgmsg(out)
            self.image_publisher.publish(img_out)
        # else:
        #     self.get_logger().info('Not alerted - stoplight')

def main(args=None):
    rclpy.init(args=args)
    stoplight_control = StoplightNode()
    rclpy.spin(stoplight_control)
    rclpy.shutdown()

if __name__=="__main__":
    main()