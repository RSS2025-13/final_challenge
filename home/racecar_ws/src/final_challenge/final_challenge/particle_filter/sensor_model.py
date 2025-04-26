import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# # Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# # if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201 #= zmax
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        z, d = np.indices((self.table_width, self.table_width))
        z_max = self.table_width - 1

        p_hit = np.zeros((self.table_width, self.table_width))
        p_short = np.zeros((self.table_width, self.table_width))
        p_max = np.zeros((self.table_width, self.table_width))
        p_rand = np.full((self.table_width, self.table_width), 1 / z_max)

        #-------p_hit----------
        eta = 1
        p_hit = (np.exp(-(z-d)**2 / (2*self.sigma_hit**2))) * (eta/np.sqrt(2*np.pi*self.sigma_hit**2))
        p_hit /= np.sum(p_hit, axis = 0)

        #-------p_short---------
        p_short_idxs = np.where(np.logical_and(z <= d, d != 0))
        p_short[p_short_idxs] = 2/d[p_short_idxs] * (1 -z[p_short_idxs]/d[p_short_idxs])
           
        #------p_max-----------
        p_max[np.where(z==z_max)] = 1
           
        #-------p_rand----------
        p_rand = 1/z_max

        self.sensor_model_table = self.alpha_hit * p_hit + self.alpha_short * p_short + self.alpha_max * p_max + self.alpha_rand * p_rand
        self.sensor_model_table /= np.sum(self.sensor_model_table, axis=0)

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """
        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle

        def clip_to_zmax(arr):
            return np.clip(arr,0,z_max).astype(int)
            return np.clip(arr,0,z_max).astype(int)

        scans = self.scan_sim.scan(particles) #NxM, where M is num_beams_per_particle
        z_max = self.table_width - 1
        scale_factor = self.resolution  * self.lidar_scale_to_map_scale

        scans /= scale_factor
        scans = clip_to_zmax(scans)
        observation /= scale_factor
        observation = clip_to_zmax(observation)

        #????
        # idxs = (scans, observation)
        # all_probs = self.sensor_model_table[idxs]
        # # multiply probs for cumulative likelihood for each particle
        # probs = np.prod(all_probs, axis=1)
        # return probs

        probabilities = []
        for scan in scans:
            probability = np.prod(self.sensor_model_table[observation, scan])
            probabilities.append(probability)
            
        return np.array(probabilities)

        ####################################

    def map_callback(self, map_msg):
        self.node.get_logger().info(f'--entered map callback-----------------------------')
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")

if __name__ == "__main__":
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(0, 200, 201)
    y = np.linspace(0, 200, 201)
    X, Y = np.meshgrid(x, y)

    sm = SensorModel(None)
    Z = sm.sensor_model_table

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()