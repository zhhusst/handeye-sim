#!/usr/bin/env python3
"""
handeye_sim_manual.launch.py — 手动控制模式

启动:
  1. robot_state_publisher (URDF)
  2. manual_control_node (键盘控关节 + 扫描线)
  3. rviz2
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # 预生成 URDF
    urdf_path = '/workspace/urdf/calib_robot.urdf'
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    pkg_share = get_package_share_directory('handeye_sim_bridge')
    install_dir = os.path.dirname(os.path.dirname(pkg_share))
    manual_exe = os.path.join(install_dir, 'bin', 'manual_control_node')

    return LaunchDescription([

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
        ),

        ExecuteProcess(
            cmd=[manual_exe, '--ros-args'],
            name='manual_control_node',
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim.rviz'],
            output='screen',
        ),
    ])
