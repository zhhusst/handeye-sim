#!/usr/bin/env python3
"""Full test with correct YAML format"""
import subprocess, time, sys
from pathlib import Path

urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

yaml_text = ('controller_manager:\n  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n    update_rate: 100\n'
    '\n'
    'joint_state_broadcaster:\n  ros__parameters:\n'
    '    type: joint_state_broadcaster/JointStateBroadcaster\n'
    '\n'
    'joint_trajectory_controller:\n  ros__parameters:\n'
    '    type: joint_trajectory_controller/JointTrajectoryController\n'
    '    joints: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint]\n'
    '    command_interfaces: [position]\n'
    '    state_interfaces: [position, velocity]\n'
    '    state_publish_rate: 50\n    action_monitor_rate: 20\n'
    '    allow_partial_joints_goal: false\n'
    '    constraints:\n'
    '      stopped_velocity_tolerance: 0.01\n      goal_time: 0.0\n')

rsp_yaml = ('/**:\n  ros__parameters:\n    robot_description: |\n' + indent
    + '\n    publish_frequency: 30.0\n')

for p, c in [('/tmp/rsp_f.yaml', rsp_yaml), ('/tmp/ctrl_f.yaml', yaml_text)]:
    Path(p).write_text(c)

# Start RSP
subprocess.Popen(['/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher',
    '--ros-args', '--params-file', '/tmp/rsp_f.yaml'])
time.sleep(1)

# Start CM
subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/ros2_control_node',
    '--ros-args', '--params-file', '/tmp/ctrl_f.yaml'])
time.sleep(2)

# Spawn
for ctrl in ['joint_state_broadcaster', 'joint_trajectory_controller']:
    sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner', ctrl,
        '--controller-manager', '/controller_manager'])
    sp.wait()
    print(f"spawn {ctrl}: exit={sp.returncode}")

time.sleep(2)

r = subprocess.run(['/opt/ros/jazzy/bin/ros2', 'topic', 'echo', '/joint_states',
    '--once', '--no-arr', '--field', 'name'], capture_output=True, text=True, timeout=5)
print("joint_states names:", r.stdout.strip() if r.stdout else "NO DATA")
