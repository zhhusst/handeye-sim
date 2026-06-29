#!/usr/bin/env python3
"""Test: spawner with -p PARAM_FILE for each controller"""
import subprocess, time
from pathlib import Path

urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

# RSP params
Path('/tmp/rsp.yaml').write_text(
    '/**:\n  ros__parameters:\n    robot_description: |\n' + indent
    + '\n    publish_frequency: 30.0\n')

# CM params (only for controller_manager itself)
Path('/tmp/cm.yaml').write_text(
    'controller_manager:\n  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n'
    '    update_rate: 100\n')

# JSB params (separate file)
Path('/tmp/jsb.yaml').write_text(
    'joint_state_broadcaster:\n  ros__parameters:\n'
    '    type: joint_state_broadcaster/JointStateBroadcaster\n')

# JTC params (separate file)
Path('/tmp/jtc.yaml').write_text(
    'joint_trajectory_controller:\n  ros__parameters:\n'
    '    type: joint_trajectory_controller/JointTrajectoryController\n'
    '    joints: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint]\n'
    '    command_interfaces: [position]\n'
    '    state_interfaces: [position, velocity]\n'
    '    state_publish_rate: 50\n'
    '    action_monitor_rate: 20\n'
    '    allow_partial_joints_goal: false\n'
    '    constraints:\n'
    '      stopped_velocity_tolerance: 0.01\n      goal_time: 0.0\n')

# Start RSP
subprocess.Popen(['/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher',
    '--ros-args', '--params-file', '/tmp/rsp.yaml'])
time.sleep(1)

# Start CM
cm = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/ros2_control_node',
    '--ros-args', '--params-file', '/tmp/cm.yaml'])
time.sleep(2)

# Spawn JSB with its own params file
sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner',
    'joint_state_broadcaster', '--controller-manager', '/controller_manager',
    '-p', '/tmp/jsb.yaml'])
sp.wait()
print(f"JSB spawn: exit={sp.returncode}")

# Spawn JTC with its own params file
sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner',
    'joint_trajectory_controller', '--controller-manager', '/controller_manager',
    '-p', '/tmp/jtc.yaml'])
sp.wait()
print(f"JTC spawn: exit={sp.returncode}")

time.sleep(2)

# Check /joint_states
r = subprocess.run(['/opt/ros/jazzy/bin/ros2', 'topic', 'echo', '/joint_states',
    '--once', '--no-arr', '--field', 'name'],
    capture_output=True, text=True, timeout=5)
if r.stdout:
    print(f"joint_states names: {r.stdout.strip()}")
else:
    print(f"No joint_states. stderr: {r.stderr[:200]}")
