#!/usr/bin/env python3
"""Generate correct ctrl_all.yaml with ros__parameters for each controller"""
from pathlib import Path

urdf = Path('/workspace/urdf/calib_robot.urdf').read_text()
indent = '\n'.join('      ' + l for l in urdf.split('\n'))

yaml_text = (
    'controller_manager:\n'
    '  ros__parameters:\n'
    '    robot_description: |\n' + indent + '\n'
    '    update_rate: 100\n'
    '\n'
    'joint_state_broadcaster:\n'
    '  ros__parameters:\n'
    '    type: joint_state_broadcaster/JointStateBroadcaster\n'
    '\n'
    'joint_trajectory_controller:\n'
    '  ros__parameters:\n'
    '    type: joint_trajectory_controller/JointTrajectoryController\n'
    '    joints:\n'
    '      - J1_joint\n'
    '      - J2_joint\n'
    '      - J3_joint\n'
    '      - J4_joint\n'
    '      - J5_joint\n'
    '      - J6_joint\n'
    '    command_interfaces:\n'
    '      - position\n'
    '    state_interfaces:\n'
    '      - position\n'
    '      - velocity\n'
    '    state_publish_rate: 50\n'
    '    action_monitor_rate: 20\n'
    '    allow_partial_joints_goal: false\n'
    '    constraints:\n'
    '      stopped_velocity_tolerance: 0.01\n'
    '      goal_time: 0.0\n'
)

Path('/tmp/ctrl_final.yaml').write_text(yaml_text)
print(f"Written {len(yaml_text)} bytes, {yaml_text.count(chr(10))+1} lines")

# Validate YAML
import yaml
data = yaml.safe_load(yaml_text)
for k, v in data.items():
    print(f"  {k}: params keys={list(v.get('ros__parameters', {}).keys())}")
