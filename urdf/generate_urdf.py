#!/usr/bin/env python3
"""
generate_urdf.py — 从原始宏定义生成 FANUC M-20iD/25 + GoCator URDF

关节定义直接从焊接工作区的 m20_25-18d_urdf_macro.xacro 复制。
输出: /workspace/urdf/calib_robot.urdf
"""

import os

MESH = 'file:///workspace/meshes'

LINKS = [
    ('base_link',   f'{MESH}/base.dae'),
    ('J1_link',     f'{MESH}/j1.dae'),
    ('J2_link',     f'{MESH}/j2.dae'),
    ('J3_link',     f'{MESH}/j3.dae'),
    ('J4_link',     f'{MESH}/j4.dae'),
    ('J5_link',     f'{MESH}/j5.dae'),
    ('J6_link',     f'{MESH}/j6.dae'),
]

# 关节定义: (name, type, xyz, rpy, axis, parent, child, limits)
# limits = (lower_deg, upper_deg, effort, velocity)
JOINTS = [
    ('J1', 'revolute', '0 0 .425', '0 0 0', '0 0 1',
     'base_link', 'J1_link', (-185, 185, 7000, 210)),
    ('J2', 'revolute', '.075 0 0', '0 0 0', '0 1 0',
     'J1_link', 'J2_link', (-100, 160, 8000, 210)),
    ('J3', 'revolute', '0 0 .84',  '0 0 0', '0 -1 0',
     'J2_link', 'J3_link', (-90, 220, 4000, 265)),
    ('J4', 'revolute', '0 0 .215', '0 0 0', '-1 0 0',
     'J3_link', 'J4_link', (-200, 200, 300, 420)),
    ('J5', 'revolute', '.89 0 0',  '0 0 0', '0 -1 0',
     'J4_link', 'J5_link', (-179.999, 179.999, 300, 420)),
    ('J6', 'revolute', '0 0 0',    '0 0 0', '-1 0 0',
     'J5_link', 'J6_link', (-450, 450, 200, 720)),
    ('J6-flange', 'fixed', '.09 0 0', '0 0 0', None,
     'J6_link', 'flange', None),
    ('flange-fanuc_flange', 'fixed', '0 0 0', '3.14159 -1.5708 0', None,
     'flange', 'fanuc_flange', None),
    ('fanuc_flange-gocator_sensor', 'fixed',
     '-0.011579 -0.004621 0.359284', '0.485145 0.160648 -1.509479', None,
     'fanuc_flange', 'gocator_sensor', None),
    ('child_joint', 'fixed', '0 0 0', '0 0 0', None,
     'flange', 'ee_link', None),
]


def gen():
    lines = []
    w = lambda s: lines.append(s)

    w('<?xml version="1.0" ?>')
    w('<robot name="m20id25_calib">')

    # world
    w('  <link name="world" />')

    # linkages
    for name, mesh in LINKS:
        w(f'  <link name="{name}">')
        w(f'    <visual>')
        w(f'      <origin xyz="0 0 0" rpy="0 0 0" />')
        w(f'      <geometry><mesh filename="{mesh}" /></geometry>')
        w(f'    </visual>')
        w(f'  </link>')

    # flanges & end
    for name in ('flange', 'fanuc_flange', 'ee_link'):
        w(f'  <link name="{name}" />')

    # GoCator
    w('  <link name="gocator_sensor">')
    w('    <visual>')
    w('      <origin xyz="0 0 -0.270" rpy="0 0 0" />')
    w(f'      <geometry><mesh filename="{MESH}/Gocator_2450.dae" /></geometry>')
    w('    </visual>')
    w('  </link>')

    # base_joint: world → base_link
    w('  <joint name="base_joint" type="fixed">')
    w('    <origin xyz="0 0 0" rpy="0 0 0" />')
    w('    <parent link="world" />')
    w('    <child link="base_link" />')
    w('  </joint>')

    # joints
    for name, jtype, xyz, rpy, axis, parent, child, limits in JOINTS:
        w(f'  <joint name="{name}_joint" type="{jtype}">')
        w(f'    <origin xyz="{xyz}" rpy="{rpy}" />')
        w(f'    <parent link="{parent}" />')
        w(f'    <child link="{child}" />')
        if axis:
            w(f'    <axis xyz="{axis}" />')
        if limits:
            lo, hi, effort, vel = limits
            w(f'    <limit lower="{lo}" upper="{hi}" effort="{effort}" velocity="{vel}" />')
        w('  </joint>')

    w('</robot>')
    return '\n'.join(lines)


if __name__ == '__main__':
    urdf = gen()
    out = '/workspace/urdf/calib_robot.urdf'
    with open(out, 'w') as f:
        f.write(urdf)
    print(f'URDF generated: {out}  ({len(urdf)} chars)')
