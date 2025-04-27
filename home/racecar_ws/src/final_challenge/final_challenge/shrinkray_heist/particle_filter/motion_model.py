import numpy as np
import matplotlib.pyplot as plt

class MotionModel:

    def __init__(self, node, std_dev = (0.1, 0.1, np.pi/15)):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.std_dev = std_dev
        self.deterministic = node.deterministic
        self.node = node

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO

        def get_rot_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        
        def get_transform_matrix(pose): #actual transformation
            mat = np.eye(3)

            xy = pose[:2]
            angle = pose[2]

            mat[:2, -1] = np.array(xy)
            mat[:2, :2] = get_rot_matrix(angle)
            return mat
        
        if not self.deterministic:
            odometry = np.array(odometry)
            #odom_noise = np.random.normal(0, self.std_dev, odometry.shape)

            odom_noise_x = np.random.normal(0, self.std_dev[0])
            odom_noise_y = np.random.normal(0, self.std_dev[1])
            odom_noise_t = np.random.normal(0, self.std_dev[2])

            #odometry = odometry + odom_nois
            odometry[0] += odom_noise_x
            odometry[1] += odom_noise_y
            odometry[2] += odom_noise_t


        # 3N x 3 Matrix (every particles row converted to transform matrix)
        particle_txns = np.apply_along_axis(get_transform_matrix, axis=1, arr=particles)
        particle_txns = np.vstack(particle_txns)

        
        # apply transform from the odometry
        transform_delta = get_transform_matrix(odometry)
        particle_txns = particle_txns @ transform_delta
        # particle_txns = np.einsum("ij, tj -> ti", transform_delta, particle_txns)
        #print(particle_txns)
        
        # convert back to original form (N x 3 matrix)
        poses = np.zeros(particles.shape)

        # each row is the first column of the rotation matrices for each particle
        rotations = particle_txns[:, 0].reshape(particles.shape[0], 3) 
        poses[:, 2] = np.arctan2(rotations[:, 1], rotations[:, 0])

        # each row is the last column of the transformation matrix for each particle
        positions = particle_txns[:, 2].reshape((particles.shape[0], 3))
        poses[:, :2] = positions[:, :2]
        
        return poses