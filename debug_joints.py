#!/usr/bin/env python3
"""Debug: check YAML joints parsing"""
import yaml

text = (
    'controller_manager:\n'
    '  ros__parameters:\n'
    '    joint_trajectory_controller:\n'
    '      joints: ["J1_joint","J2_joint","J3_joint",'
    '"J4_joint","J5_joint","J6_joint"]\n'
)
data = yaml.safe_load(text)
jtc = data['controller_manager']['ros__parameters']['joint_trajectory_controller']
print(f"joints type: {type(jtc['joints'])}")
print(f"joints value: {jtc['joints']}")
print(f"joints len: {len(jtc['joints'])}")
