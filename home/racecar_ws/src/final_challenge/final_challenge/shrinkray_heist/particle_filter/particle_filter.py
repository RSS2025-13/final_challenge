from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray

from rclpy.node import Node
import rclpy

assert rclpy

import numpy as np
import time
import tf2_ros
from std_msgs.msg import Float32, Float32MultiArray
import geometry_msgs
from sensor_msgs.msg import LaserScan

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('deterministic', False)
        self.deterministic = self.get_parameter('deterministic').get_parameter_value().bool_value

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('pose_truth_topic', "/pose_truth")
        self.declare_parameter('initial_xyangle_topic', "/initial_xyangle")

        self.declare_parameter('rosbag_input', False)
        self.rosbag_input = self.get_parameter('rosbag_input').get_parameter_value().bool_value

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        pose_truth_topic = self.get_parameter("pose_truth_topic").get_parameter_value().string_value
        initial_xyangle_topic = self.get_parameter("initial_xyangle_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)
        if self.rosbag_input:
            self.pose_truth_sub = self.create_subscription(Pose, pose_truth_topic,
                                                 self.pose_truth_callback,
                                                 1)
            self.initial_xyangle_sub = self.create_subscription(Float32MultiArray, initial_xyangle_topic,
                                                                self.initial_xyangle_callback,
                                                                1)

            self.x_expected = None
            self.y_expected = None
            self.angle_expected = None
            self.xyangle0 = None

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.debug_pub = self.create_publisher(Float32MultiArray, "/debug", 10)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")
        #Notes-----------
        #particles in world frame?
        #probabilities of particles
        #pose_truth published as expected?

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.declare_parameter('num_particles', 0)
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.num_beams_per_particle = self.get_parameter("num_beams_per_particle").get_parameter_value().integer_value

        self.dist_error_pub = self.create_publisher(Float32, '/distance_error', 10)
        self.angle_error_pub = self.create_publisher(Float32, '/angle_error', 10)
        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)
        if not self.rosbag_input:
            self.pose_truth_pub = self.create_publisher(Pose, pose_truth_topic, 10)
            self.initial_xyangle_pub = self.create_publisher(Float32MultiArray, initial_xyangle_topic, 1)

        self.ranges_laser = np.array([])
        self.particle_poses = np.array([])

        self.previous_time = time.perf_counter()
        self.broadcaster = tf2_ros.TransformBroadcaster(self)

        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

    def get_particle_poses(self):
        poses = []
        for i in range(len(self.particle_poses)):
            particle = self.particle_poses[i, :]
            msg = Pose()
            
            msg.position.x = particle[0]
            msg.position.y = particle[1]

            angle = particle[2]
            msg.orientation.x = 0.0
            msg.orientation.y = 0.0
            msg.orientation.z = np.sin(angle / 2)
            msg.orientation.w = np.cos(angle / 2)

            poses.append(msg)

        msg = PoseArray()
        msg.header.frame_id = '/map'
        msg.poses = poses
        self.particles_pub.publish(msg)

    def get_average_pose(self):

        self.get_particle_poses()
        positions = self.particle_poses

        xys = positions[:, :2]
        angles = positions[:, 2]

        average_position = np.mean(xys, axis=0)
        average_angle = np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))
        # self.get_logger().info(f"avg_angle {average_angle}")
        # self.get_logger().info(f"angle {angles}")
        
        average_pose = np.hstack((average_position, average_angle))
        return average_pose, average_angle
    
    def publish_average_pose(self):
        
        avg_pose, avg_angle = self.get_average_pose()

        msg = Odometry()
        msg.header.frame_id = '/map'
        msg.pose.pose.position.x = avg_pose[0]
        msg.pose.pose.position.y = avg_pose[1]

        # z axis rotation
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = np.sin(avg_angle / 2)
        msg.pose.pose.orientation.w = np.cos(avg_angle / 2)

        msg.child_frame_id = '/base_link'
        #self.odom_pub.publish(msg)

        obj = geometry_msgs.msg.TransformStamped()
        obj.header.frame_id = '/map'
        obj.child_frame_id = '/base_link_pf'

        obj.transform.translation.x = avg_pose[0]
        obj.transform.translation.y = avg_pose[1]
        obj.transform.translation.z = 0.0

        obj.transform.rotation.x = 0.0
        obj.transform.rotation.y = 0.0
        obj.transform.rotation.z = np.sin(avg_angle / 2)
        obj.transform.rotation.w = np.cos(avg_angle / 2)

        self.broadcaster.sendTransform(obj)
        self.odom_pub.publish(msg)

        if not self.rosbag_input:
            try:
                tf_world_to_car: geometry_msgs.msg.TransformStamped = self.buffer.lookup_transform('map', 'base_link',
                                                                                    rclpy.time.Time())
                
                x_expected = tf_world_to_car.transform.translation.x
                y_expected = tf_world_to_car.transform.translation.y
                angle_expected = 2 * np.arctan2(tf_world_to_car.transform.rotation.z, tf_world_to_car.transform.rotation.w)

                distance_error_msg = Float32()
                distance_error_msg.data = np.sqrt((x_expected - avg_pose[0])**2 + (y_expected - avg_pose[1])**2)
                
                angle_error_msg = Float32()
                angle_error_msg.data = angle_expected - avg_angle
                #-------------
                pose_truth_msg = Pose()
                
                pose_truth_msg.position.x = x_expected
                pose_truth_msg.position.y = y_expected

                pose_truth_msg.orientation.x = 0.0
                pose_truth_msg.orientation.y = 0.0
                pose_truth_msg.orientation.z = np.sin(angle_expected / 2)
                pose_truth_msg.orientation.w = np.cos(angle_expected / 2)

                self.pose_truth_pub.publish(pose_truth_msg)
                #-------------

            except tf2_ros.TransformException:
                self.get_logger().info('no transform from world to base_link_gt found')
                return
        else:
            if self.x_expected is None:
                self.get_logger().info("self.x_expected was not given a value")
                return
            distance_error_msg = Float32()
            distance_error_msg.data = np.sqrt((self.x_expected - avg_pose[0])**2 + (self.y_expected - avg_pose[1])**2)
            
            angle_error_msg = Float32()
            angle_error_msg.data = self.angle_expected - avg_angle
        
        #regardless of input type (rosbag or null)
        self.dist_error_pub.publish(distance_error_msg)
        self.angle_error_pub.publish(angle_error_msg)
        
    def odom_callback(self, msg):
        if len(self.particle_poses) == 0:
            if self.rosbag_input:
                if not self.pose_callback():
                    return
            else:
                return
        curr_time = time.perf_counter()
        dT = curr_time - self.previous_time

        x_vel = msg.twist.twist.linear.x
        y_vel = msg.twist.twist.linear.y
        angle_vel = msg.twist.twist.angular.z
        
        odom = np.array([x_vel*dT, y_vel*dT, angle_vel*dT])
        self.get_logger().info(f'odom: {odom}')
        self.particle_poses = self.motion_model.evaluate(self.particle_poses, odom)

        #self.get_logger().info(f'n1: {min(noise[0]), max(noise[0])}\nn2: {min(noise[1]), max(noise[1])}\nn3: {min(noise[2]), max(noise[2])}')
        self.publish_average_pose()
        self.previous_time = curr_time

    def laser_callback(self, msg):
        if len(self.particle_poses) == 0:
            if self.rosbag_input:
                if not self.pose_callback(): #if this returns false, in which case the particles were not defined
                    return
            else:
                return

        downsampled_indices = np.linspace(0, len(msg.ranges)-1, self.num_beams_per_particle).astype(int)
        downsampled_laser_ranges = np.array(msg.ranges)[downsampled_indices]
        #Wonky three lines? KEEP IT LIKE THIS IT WORKS REALLY WELL IN SIM
        # downsampled_laser_ranges = np.random.choice(np.array(msg.ranges), self.num_beams_per_particle) 
        weights = self.sensor_model.evaluate(self.particle_poses, downsampled_laser_ranges)
        if weights is None or np.sum(weights==0):
            return

        weights /= np.sum(weights)
        particles_to_maintain = int(self.num_particles * 0.975)

        if np.count_nonzero(weights) < particles_to_maintain: 
            return

        particle_samples_idxs = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        self.particle_poses = self.particle_poses[particle_samples_idxs,:]
        if not self.deterministic:
            self.particle_poses += np.random.normal(0, 0.1, self.particle_poses.shape)

        self.publish_average_pose()
    
    def pose_truth_callback(self, msg):
        self.x_expected = msg.position.x
        self.y_expected = msg.position.y
        self.angle_expected = 2*np.arcsin(msg.orientation.z)
        
    def initial_xyangle_callback(self, msg):
        self.xyangle0 = [msg.data[0],msg.data[1],msg.data[2]]

    def pose_callback(self, msg=None):
        if self.rosbag_input:
            if msg is not None: #you clicked but we are using only std initialization, getting data from rosbag
                return True
            #clickable pose setting inactive when running rosbag playback
            if self.xyangle0 is None:
                self.get_logger().info("self.xyangle0 was not given a value")
                return False
            [x, y, angle] = self.xyangle0
        else:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            angle = 2 * np.arctan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
            initial_xyangle_msg = Float32MultiArray()
            initial_xyangle_msg.data = [x,y,angle]
            self.initial_xyangle_pub.publish(initial_xyangle_msg)
        theta = angle

        def normalize_angle(angle):
            # Normalize the angle to be within the range [-π, π]
            normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
            return normalized_angle
        
        if not self.deterministic:
            x = np.random.normal(loc=x, scale=1.0, size=(self.num_particles,1))
            y = np.random.normal(loc=y, scale=1.0, size=(self.num_particles,1))
            theta = np.random.normal(loc=angle, scale=1.0, size=(self.num_particles,1))
        else:
            x = np.full((self.num_particles, 1), x)
            y = np.full((self.num_particles, 1), y)
            theta = np.full((self.num_particles, 1), theta)

        self.particle_poses = np.hstack((x, y, theta))
            
        self.get_particle_poses()
        return True


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()