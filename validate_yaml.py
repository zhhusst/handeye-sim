#!/usr/bin/env python3
"""Validate controller_manager YAML"""
import yaml

with open('/workspace/urdf/calib_robot.urdf') as f:
    u = f.read()

ui = '\n'.join('      ' + l for l in u.split('\n'))

ctrl_all = (
    'controller_manager:\n'
    '  ros__parameters:\n'
    '    robot_description: |\n' + ui + '\n'
    '    update_rate: 100\n'
    '    joint_state_broadcaster:\n'
    '      type: joint_state_broadcaster/JointStateBroadcaster\n'
    '    joint_trajectory_controller:\n'
    '      type: joint_trajectory_controller/JointTrajectoryController\n'
    '      joints: ["J1_joint", "J2_joint", "J3_joint", '
    '"J4_joint", "J5_joint", "J6_joint"]\n'
    '      command_interfaces: ["position"]\n'
    '      state_interfaces: ["position", "velocity"]\n'
    '      state_publish_rate: 50\n'
    '      action_monitor_rate: 20\n'
    '      allow_partial_joints_goal: false\n'
    '      constraints:\n'
    '        stopped_velocity_tolerance: 0.01\n'
    '        goal_time: 0.0\n'
)

data = yaml.safe_load(ctrl_all)
cm = data.get('controller_manager', {})
rp = cm.get('ros__parameters', {})
print(f"Keys: {list(rp.keys())}")
print(f"desc len: {len(rp.get('robot_description', ''))}")
print(f"jsb: {rp.get('joint_state_broadcaster', {}).get('type')}")
jtc = rp.get('joint_trajectory_controller', {})
print(f"jtc type: {jtc.get('type')}")
print(f"jtc joints: {jtc.get('joints')}")
print("---OK---")
