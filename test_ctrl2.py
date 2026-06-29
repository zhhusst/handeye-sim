#!/usr/bin/env python3
"""Full test: start RSP + CM + spawn controllers, verify joint_states"""
import subprocess, time, sys
from pathlib import Path

urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

rsp_yaml = '/**:\n  ros__parameters:\n    robot_description: |\n' + indent + '\n    publish_frequency: 30.0\n'
Path('/tmp/rsp_t.yaml').write_text(rsp_yaml)

ctrl_yaml = (
    'controller_manager:\n  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n'
    '    update_rate: 100\n'
    '    joint_state_broadcaster:\n      type: joint_state_broadcaster/JointStateBroadcaster\n'
    '    joint_trajectory_controller:\n'
    '      type: joint_trajectory_controller/JointTrajectoryController\n'
    '      joints: ["J1_joint","J2_joint","J3_joint","J4_joint","J5_joint","J6_joint"]\n'
    '      command_interfaces: ["position"]\n'
    '      state_interfaces: ["position","velocity"]\n'
    '      state_publish_rate: 50\n'
    '      action_monitor_rate: 20\n'
    '      allow_partial_joints_goal: false\n'
    '      constraints:\n'
    '        stopped_velocity_tolerance: 0.01\n        goal_time: 0.0\n'
)
Path('/tmp/ctrl_t.yaml').write_text(ctrl_yaml)

# Start RSP
subprocess.Popen(['/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher',
    '--ros-args', '--params-file', '/tmp/rsp_t.yaml'])
time.sleep(1)

# Start CM
subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/ros2_control_node',
    '--ros-args', '--params-file', '/tmp/ctrl_t.yaml'])
time.sleep(2)

# Spawn controllers
for ctrl in ['joint_state_broadcaster', 'joint_trajectory_controller']:
    sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner', ctrl,
        '--controller-manager', '/controller_manager'])
    sp.wait()
    print(f"spawn {ctrl}: exit={sp.returncode}")

time.sleep(2)

# Check joint_states
result = subprocess.run(['/opt/ros/jazzy/bin/ros2', 'topic', 'echo', '/joint_states', '--once', '--no-arr'],
    capture_output=True, text=True, timeout=5)
print("--- joint_states ---")
print(result.stdout[:500] if result.stdout else "NO JOINT STATES")
print(result.stderr[:200] if result.stderr else "")
