import os
import glob
from setuptools import find_packages, setup

package_name = 'race_to_moon'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/'+package_name+"/computer_vision", glob.glob(os.path.join('race_to_moon/computer_vision', '*.py'))),
        ('share/race_to_moon/launch', glob.glob(os.path.join('launch', '*launch.xml'))),
        ('share/race_to_moon/launch', glob.glob(os.path.join('launch', '*launch.py')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='russellperez2004@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_detector = race_to_moon.line_detector:main',
            'homography_transformer = race_to_moon.homography_transformer:main',
            'pure_pursuit = race_to_moon.pure_pursuit:main',
        ],
    },
)
