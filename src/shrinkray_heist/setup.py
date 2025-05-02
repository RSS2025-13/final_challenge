from setuptools import find_packages, setup

package_name = 'shrinkray_heist'

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
            #banana finder model
            'detection_node = shrinkray_heist.model.detection_node:main',
            #Point publisher
            'basement_point_publisher = shrinkray_heist.basement_point_publisher:main',
            #Stop light
            'stoplight_controller = shrinkray_heist.stoplight_controller:main',
            #Safety controller
            'safety_controller = shrinkray_heist.safety_controller:main',
            #State Machine
            'state_machine = shrinkray_heist.state_machine:main'
        ],
    },
)
