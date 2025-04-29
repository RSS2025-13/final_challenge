from setuptools import find_packages, setup

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='racecar@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #path following and planning
            'trajectory_builder = final_challenge.trajectory_builder:main',
            'trajectory_loader = final_challenge.trajectory_loader:main',
            'trajectory_follower = final_challenge.trajectory_follower:main',
            'trajectory_planner = final_challenge.trajectory_planner:main',
            #localization
            'particle_filter = final_challenge.particle_filter:main',
            #banana finder model
            'detection_node = final_challenge.detection_node:main',
            #Point publisher
            'basement_point_publisher = final_challenge.basement_point_publisher:main'
        ],
    },
)
