#!/usr/bin/env python3
"""Final test - fix param types"""
import subprocess, time
from pathlib import Path

urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

for name, content in [
    ('/tmp/rsp.yaml',
     '/**:\n  ros__parameters:\n    robot_description: |\n' + indent
     + '\n    publish_frequency: 30.0\n'),
    ('/tmp/cm.yaml',
     'controller_manager:\n  ros__parameters:\n'
     '    robot_description: |\n' + indent + '\n'
     '    update_rate: 100\n'),
    ('/tmp/jsb.yaml',
     'joint_state_broadcaster:\n  ros__parameters:\n'
     '    type: joint_state_broadcaster/JointStateBroadcaster\n'),
    ('/tmp/jtc.yaml',
     'joint_trajectory_controller:\n  ros__parameters:\n'
     '    type: joint_trajectory_controller/JointTrajectoryController\n'
     '    joints: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint]\n'
     '    command_interfaces: [position]\n'
     '    state_interfaces: [position, velocity]\n'
     '    state_publish_rate: 50.0\n'
     '    action_monitor_rate: 20.0\n'
     '    allow_partial_joints_goal: false\n'
     '    constraints:\n'
     '      stopped_velocity_tolerance: 0.01\n      goal_time: 0.0\n'),
]:
    Path(name).write_text(content)

subprocess.Popen(['/opt/ros/jazzy/lib/robot_state_publisher/robot_state_publisher',
    '--ros-args', '--params-file', '/tmp/rsp.yaml'])
time.sleep(1)

subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/ros2_control_node',
    '--ros-args', '--params-file', '/tmp/cm.yaml'])
time.sleep(2)

for name, params in [
    ('joint_state_broadcaster', '/tmp/jsb.yaml'),
    ('joint_trajectory_controller', '/tmp/jtc.yaml'),
]:
    sp = subprocess.Popen(['/opt/ros/jazzy/lib/controller_manager/spawner',
        name, '--controller-manager', '/controller_manager', '-p', params])
    sp.wait()
    print(f"spawn {name}: exit={sp.returncode}")

time.sleep(2)

r = subprocess.run(['/opt/ros/jazzy/bin/ros2', 'topic', 'echo', '/joint_states',
    '--once', '--no-arr'], capture_output=True, text=True, timeout=5)
if r.stdout:
    import json
    data = json.loads(r.stdout)
    print(f"joint_states: {data.get('name', [])}")
else:
    print(f"No joint_states. Error: {'timeout' if not r.stderr else r.stderr[:200]}")
