#!/usr/bin/env python3
"""
handeye_sim.launch.py — 手眼标定仿真 (完整机械臂+GoCator)

启动:
  1. robot_state_publisher (从静态 URDF 发布 TF)
  2. bridge_combined_runner (场景 + 轨迹 + IK + 关节角 + Marker)
  3. rviz2 (3D 可视化)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    # 预生成的静态 URDF (无需 xacro)
    urdf_path = '/workspace/urdf/calib_robot.urdf'

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # console_scripts 入口
    pkg_share = get_package_share_directory('handeye_sim_bridge')
    install_dir = os.path.dirname(os.path.dirname(pkg_share))
    runner_exe = os.path.join(install_dir, 'bin', 'bridge_combined_runner')

    return LaunchDescription([

        # 1. 机器人模型发布
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
        ),

        # 2. 标定仿真主节点 (含 IK + 关节角 + Marker)
        ExecuteProcess(
            cmd=[
                runner_exe,
                '--ros-args',
                '-p', 'n_frames:=40',
                '-p', 'frame_rate:=5.0',
                '-p', 'half_fov_deg:=15.0',
            ],
            name='calib_sim_runner',
            output='screen',
        ),

        # 3. RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/workspace/ros2_ws/src/handeye_sim_bridge/rviz/handeye_sim.rviz'],
            output='screen',
        ),
    ])
