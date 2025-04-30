import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray

class BasementPointPublisher(Node):
    '''
    Node that publishes a list of "shell" points
    Subscribes to the "Publish Point" topic when you click on the map in RViz
    After 2 points have been chosen, it publishes the 2 points as a PoseArray and resets the array
    '''

    def __init__(self):
        super().__init__("BasementPointPub")
        self.shrinkray_publisher = self.create_publisher(PoseArray, "/shrinkray_loc", 1)
        self.stoplight_publisher = self.create_publisher(PoseArray, "/stoplight_loc", 1)
        self.subscriber = self.create_subscription(PointStamped, "/clicked_point", self.callback, 1)

        self.array = []

        self.get_logger().info("Point Publisher Initialized")

    def callback(self, point_msg: PointStamped):
        x,y = point_msg.point.x, point_msg.point.y
        self.get_logger().info(f"Received point: {x}, {y}")
        self.array.append(Pose(position=Point(x=x, y=y, z=0.0)))
        
        if len(self.array) == 1: #stoplight
            self.publish(first=True)
        if len(self.array) == 3: #three points published (then two bananas)
            self.publish()

    def publish(self, first=False):
        # Publish PoseArrays
        if first:
            stoplight = PoseArray()
            stoplight.header.frame_id = "map"
            stoplight.poses = self.array[0]
            self.publisher.publish(stoplight)

            point_str = '\n'+'\n'.join([f"({p.position.x},{p.position.y})" for p in self.array])
            self.get_logger().info(f"Published stoplight point: {point_str}")
        else:
            shrinkray = PoseArray()
            shrinkray.header.frame_id = "map"
            shrinkray.poses = self.array[1:]
            self.publisher.publish(shrinkray)
            
            points_str = '\n'+'\n'.join([f"({p.position.x},{p.position.y})" for p in self.array])
            self.get_logger().info(f"Published 2 points: {points_str}")

            # Reset Array
            self.array = []
    

def main(args=None):
    rclpy.init(args=args)
    node = BasementPointPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=="__main__":
    main()