"""handeye_sim_moveit.launch.py — MoveIt2 + 标定场景
参数藏在 bash -c 里，避免 launch 系统拦截 --ros-args
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

ros_lib = '/opt/ros/jazzy/lib'
pkg_share = '/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge'
ws = '/workspace'
scene_exe = os.path.join(ws, 'ros2_ws', 'install', 'handeye_sim_bridge',
                         'lib', 'handeye_sim_bridge', 'scene_publisher_node.py')
rsp_exe = os.path.join(ros_lib, 'robot_state_publisher', 'robot_state_publisher')
jsp_exe = os.path.join(ros_lib, 'joint_state_publisher_gui', 'joint_state_publisher_gui')
mg_exe = os.path.join(ros_lib, 'moveit_ros_move_group', 'move_group')
rviz_exe = os.path.join(ros_lib, 'rviz2', 'rviz2')
rviz_cfg = os.path.join(pkg_share, 'rviz', 'handeye_sim_moveit.rviz')
urdf_path = os.path.join(ws, 'urdf', 'calib_robot.urdf')
srdf_path = os.path.join(pkg_share, 'config', 'fanuc.srdf')
srdf_path_ws = os.path.join(pkg_share, 'config')
kin_path = os.path.join(srdf_path_ws, 'kinematics.yaml')
ompl_path = os.path.join(srdf_path_ws, 'ompl_planning.yaml')
jl_path = os.path.join(srdf_path_ws, 'joint_limits.yaml')

# 预先加载参数到文件（避免 shell 引用问题）
import yaml
urdf_text = open(urdf_path).read()
srdf_text = open(srdf_path).read()

# 注意：这些文件在容器内
# 写临时 bash 包装脚本
script = f'''#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash

# 启动 robot_state_publisher (直接读文件传参)
{rsp_exe} --ros-args \\
  -p robot_description:="$(cat {urdf_path})" \\
  -p publish_frequency:=30.0 &
sleep 0.5

# 启动 joint_state_publisher_gui
{jsp_exe} &
sleep 0.5

# 启动 move_group
{mg_exe} --ros-args \\
  -p robot_description:="$(cat {urdf_path})" \\
  -p robot_description_semantic:="$(cat {srdf_path})" \\
  -p moveit_manage_controllers:=false \\
  -p planning_pipelines:="['ompl']" \\
  -p default_planning_pipeline:=ompl &
sleep 0.5

# 启动 scene_publisher
python3 "{scene_exe}" &
sleep 0.5

# 启动 rviz2
{rviz_exe} -d {rviz_cfg} &

# 等待所有后台进程
wait
'''

with open('/workspace/run_moveit.sh', 'w') as f:
    f.write(script)
os.chmod('/workspace/run_moveit.sh', 0o755)


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['bash', '/workspace/run_moveit.sh'],
            name='moveit_launcher', output='screen'),
    ])
