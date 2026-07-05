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
        (os.path.join('lib', package_name),
         [os.path.join(package_name, 'scene_publisher_node.py'),
          os.path.join(package_name, 'srdf_publisher_node.py'),
          os.path.join(package_name, 'auto_calib_v2_node.py'),
          os.path.join(package_name, 'profile_viz_node.py')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zhang HaHa',
    maintainer_email='z@z.com',
    description='ROS2 — hand-eye calibration simulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'scene_publisher_node = handeye_sim_bridge.scene_publisher_node:main',
            'srdf_publisher_node = handeye_sim_bridge.srdf_publisher_node:main',
            'auto_calib_v2 = handeye_sim_bridge.auto_calib_v2_node:main',
            'profile_viz = handeye_sim_bridge.profile_viz_node:main',
        ],
    },
)
