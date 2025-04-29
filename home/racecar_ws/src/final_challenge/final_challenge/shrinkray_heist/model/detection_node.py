import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from custom_msg.msg import Pixel
from std_msgs.msg import Bool
from .detector import Detector

class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")
        self.detector = Detector(yolo_dir='./shrinkray_heist/model',from_tensor_rt=False, threshold=0.5)
        self.publisher = self.create_publisher(Pixel, "detector/banana_pix_loc", 1)
        self.debug_publisher = self.create_publisher(Image, "detector/marked_img", 1)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.alert_subscriber = self.create_subscription(Bool, "detector/alert", self.alert_callback, 10)
        self.image_subscriber = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 1)
        self.bridge = CvBridge()

        self.get_logger().info("Detector Initialized")
        self.alert = False

    def alert_callback(self, msg):
        self.alert = msg.data
    
    def timer_callback(self):
        if self.alert:
            img_path = "./src/final_challenge/media/minion.png" 
            image = Image.open(img_path)
            #TODO: run image through self.detector and publish bounding box image
            results = self.detector.predict(image)
            predictions = results["predictions"]

            #ideally, there should onyl be 1 banana detected
            x1, y1, x2, y2 = predictions[0]
            pix_loc = Pixel()
            pix_loc.u = round((x1+x2)/2)
            pix_loc.v = round((y1+y2)/2)

            self.publisher.publish(pix_loc)

            original_image = results["original_image"] #in rgb
            out = self.detector.draw_point(original_image, (pix_loc.u,pix_loc.v))
            img_out = self.bridge.cv2_to_imgmsg(out)
            self.image_publisher.publish(img_out)
            self.alert = False

    def image_callback(self, img_msg):
        # Process image with CV Bridge
        if self.alert:
            image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

            #TODO: run image through self.detector and publish bounding box image
            results = self.detector.predict(image)
            predictions = results["predictions"]
            # original_image = results["original_image"] #in rgb
            # out = self.detector.draw_box(original_image, predictions, draw_all=True)
            # img_out = self.bridge.cv2_to_imgmsg(out)

            #ideally, there should onyl be 1 banana detected
            x1, y1, x2, y2 = predictions[0]
            pix_loc = Pixel()
            pix_loc.u = round((x1+x2)/2)
            pix_loc.v = round((y1+y2)/2)

            self.publisher.publish(pix_loc)

            original_image = results["original_image"] #in rgb
            out = self.detector.draw_point(original_image, (pix_loc.u,pix_loc.v))
            img_out = self.bridge.cv2_to_imgmsg(out)
            self.image_publisher.publish(img_out)
            self.alert = False
        # else:
        #     self.get_logger().info('Not alerted - banana')

def main(args=None):
    rclpy.init(args=args)
    detector = DetectorNode()
    rclpy.spin(detector)
    rclpy.shutdown()

if __name__=="__main__":
    main()
