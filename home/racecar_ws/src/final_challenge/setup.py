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
            'trajectory_builder = path_planning.trajectory_builder:main',
            'trajectory_loader = path_planning.trajectory_loader:main',
            'trajectory_follower = path_planning.trajectory_follower:main',
            'trajectory_planner = path_planning.trajectory_planner:main',
        ],
    },
)
