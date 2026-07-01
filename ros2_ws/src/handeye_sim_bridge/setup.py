from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'handeye_sim_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),
         glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*')),
        # Install main script to lib/<pkg>/ for ros2 launch compatibility
        (os.path.join('lib', package_name),
         [os.path.join(package_name, 'bridge_combined_runner.py'),
          os.path.join(package_name, 'scene_publisher_node.py'),
          os.path.join(package_name, 'srdf_publisher_node.py'),
          os.path.join(package_name, 'auto_calib_node.py'),
          os.path.join(package_name, 'auto_servo_collect.py'),
          os.path.join(package_name, 'replay_calib_poses.py')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zhang HaHa',
    maintainer_email='z@z.com',
    description='ROS2 桥接 — 手眼标定仿真可视化',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'bridge_combined_runner = handeye_sim_bridge.bridge_combined_runner:main',
            'manual_control_node = handeye_sim_bridge.manual_control_node:main',
            'scene_publisher_node = handeye_sim_bridge.scene_publisher_node:main',
            'srdf_publisher_node = handeye_sim_bridge.srdf_publisher_node:main',
            'auto_calib_node = handeye_sim_bridge.auto_calib_node:main',
            'auto_servo_collect = handeye_sim_bridge.auto_servo_collect:main',
        ],
    },
)
