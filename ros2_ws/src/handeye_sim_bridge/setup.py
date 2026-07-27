import os
from glob import glob

from setuptools import find_packages, setup

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
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'PyYAML'],
    zip_safe=True,
    maintainer='Zhang HaHa',
    maintainer_email='z@z.com',
    description='ROS2 — hand-eye calibration simulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'scene_publisher_node = handeye_sim_bridge.scene_publisher_node:main',
            'srdf_publisher_node = handeye_sim_bridge.srdf_publisher_node:main',
            'profile_viz = handeye_sim_bridge.profile_viz_node:main',
            'seed_collection = handeye_sim_bridge.seed_collection_node:main',
            'active_calibration_sim = handeye_sim_bridge.active_calibration_sim_node:main',
            'calibration_demo = calibration_pipeline.cli:main',
        ],
    },
)
