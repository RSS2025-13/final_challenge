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
            'trajectory_builder = final_challenge.shrinkray_heist.trajectory_follower.trajectory_builder:main',
            'trajectory_loader = final_challenge.shrinkray_heist.trajectory_follower.trajectory_loader:main',
            'trajectory_follower = final_challenge.shrinkray_heist.trajectory_follower.trajectory_follower:main',
            'trajectory_planner = final_challenge.shrinkray_heist.trajectory_planner.trajectory_planner:main',
            #localization
            'particle_filter = final_challenge.shrinkray_heist.particle_filter.particle_filter:main',
            #banana finder model
            'detection_node = final_challenge.shrinkray_heist.model.detection_node:main',
            #Point publisher
            'basement_point_publisher = final_challenge.shrinkray_heist.basement_point_publisher:main',
            #Stop light
            'stoplight_controller = final_challenge.shrinkray_heist.stoplight_controller:main'
        ],
    },
)
