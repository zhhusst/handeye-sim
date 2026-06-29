#!/usr/bin/env python3
"""Quick test: start RSP + controller_manager + spawn controllers"""
import subprocess, time, sys, os
from pathlib import Path

# Generate YAML
urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

rsp_yaml = (
    '/**:\n  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n'
    '    publish_frequency: 30.0\n'
)
Path('/tmp/rsp_test.yaml').write_text(rsp_yaml)

ctrl_yaml = (
    'controller_manager:\n'
    '  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n'
    '    update_rate: 100\n'
    '    joint_state_broadcaster:\n'
    '      type: joint_state_broadcaster/JointStateBroadcaster\n'
    '    joint_trajectory_controller:\n'
    '      type: joint_trajectory_controller/JointTrajectoryController\n'
    '      joints: ["J1_joint","J2_joint","J3_joint",'
    '"J4_joint","J5_joint","J6_joint"]\n'
    '      command_interfaces: ["position"]\n'
    '      state_interfaces: ["position","velocity"]\n'
    '      state_publish_rate: 50\n'
    '      action_monitor_rate: 20\n'
    '      allow_partial_joints_goal: false\n'
    '      constraints:\n'
    '        stopped_velocity_tolerance: 0.01\n'
    '        goal_time: 0.0\n'
)
Path('/tmp/ctrl_test.yaml').write_text(ctrl_yaml)

# Start RSP
rsp = subprocess.Popen([
    '/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher',
    '--ros-args', '--params-file', '/tmp/rsp_test.yaml'
])
print(f"RSP started PID={rsp.pid}")

time.sleep(1)

# Start controller_manager
cm = subprocess.Popen([
    '/opt/ros/jazzy/lib/controller_manager/ros2_control_node',
    '--ros-args', '--params-file', '/tmp/ctrl_test.yaml'
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(f"CM started PID={cm.pid}")

# Wait and check output
time.sleep(3)
try:
    out, _ = cm.communicate(timeout=2)
    print("CM output:")
    print(out)
except subprocess.TimeoutExpired:
    print("CM still running (good sign!)")
    cm.kill()

rsp.kill()
print("Done")
