from setuptools import setup
import os
from glob import glob

package_name = 'rrt_exploration_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='RRT exploration for ROS 2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'boundary_node = rrt_exploration_ros2.boundary_node:main',
            'global_rrt_detector = rrt_exploration_ros2.global_detector_node:main',
            'local_detector_node = rrt_exploration_ros2.local_detector_node:main',
            'filter_node = rrt_exploration_ros2.filter_node:main',
        ],
    },
)