#!/usr/bin/env python3
"""Full test: controller_manager with nested params + spawner with -p args"""
import subprocess, time
from pathlib import Path

urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

# RSP params
Path('/tmp/rsp.yaml').write_text(
    '/**:\n  ros__parameters:\n    robot_description: |\n' + indent
    + '\n    publish_frequency: 30.0\n')

# CM params: controller configs NESTED inside controller_manager
Path('/tmp/cm.yaml').write_text(
    'controller_manager:\n  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n'
    '    update_rate: 100\n'
    '    joint_state_broadcaster:\n'
    '      type: joint_state_broadcaster/JointStateBroadcaster\n'
    '    joint_trajectory_controller:\n'
    '      type: joint_trajectory_controller/JointTrajectoryController\n'
    '      joints: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint]\n'
    '      command_interfaces: [position]\n'
    '      state_interfaces: [position, velocity]\n'
    '      state_publish_rate: 50\n'
    '      action_monitor_rate: 20\n'
    '      allow_partial_joints_goal: false\n'
    '      constraints:\n'
    '        stopped_velocity_tolerance: 0.01\n        goal_time: 0.0\n')

# Start RSP
subprocess.Popen(['/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher',
    '--ros-args', '--params-file', '/tmp/rsp.yaml'])
time.sleep(1)

# Start CM
cm = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/ros2_control_node',
    '--ros-args', '--params-file', '/tmp/cm.yaml'])
time.sleep(2)

# Spawn JSB with -p type (simple, no other params needed)
sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner',
    'joint_state_broadcaster', '--controller-manager', '/controller_manager',
    '-p', 'type:=joint_state_broadcaster/JointStateBroadcaster'])
sp.wait()
print(f"JSB spawn: exit={sp.returncode}")

# Spawn JTC with -p params
sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner',
    'joint_trajectory_controller', '--controller-manager', '/controller_manager',
    '-p', 'type:=joint_trajectory_controller/JointTrajectoryController',
    '-p', 'joints:=[\"J1_joint\",\"J2_joint\",\"J3_joint\",\"J4_joint\",\"J5_joint\",\"J6_joint\"]',
    '-p', 'command_interfaces:=[\"position\"]',
    '-p', 'state_interfaces:=[\"position\",\"velocity\"]',
    '-p', 'state_publish_rate:=50',
    '-p', 'action_monitor_rate:=20',
    '-p', 'allow_partial_joints_goal:=false'])
sp.wait()
print(f"JTC spawn: exit={sp.returncode}")

time.sleep(2)

# Check /joint_states
r = subprocess.run(['/opt/ros/jazzy/bin/ros2', 'topic', 'echo', '/joint_states',
    '--once', '--no-arr', '--field', 'name'],
    capture_output=True, text=True, timeout=5)
print(f"joint_states: {r.stdout.strip() if r.stdout else 'TIMEOUT'}")
