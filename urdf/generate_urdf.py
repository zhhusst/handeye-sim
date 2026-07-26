#!/usr/bin/env python3
"""Generate the self-contained FANUC M-20iD/25 + Gocator URDF."""

import math
from pathlib import Path

URDF_DIR = Path(__file__).resolve().parent
MESH = f'file://{URDF_DIR.parent / "meshes"}'

# name, mesh, mass, diagonal inertia.  These inertial values are intentionally
# kept in the generator: Gazebo removes movable URDF links without inertia.
LINKS = [
    ('base_link', f'{MESH}/base.dae', 200.0, (10.0, 10.0, 10.0)),
    ('J1_link', f'{MESH}/j1.dae', 80.0, (5.0, 5.0, 5.0)),
    ('J2_link', f'{MESH}/j2.dae', 100.0, (6.0, 6.0, 6.0)),
    ('J3_link', f'{MESH}/j3.dae', 60.0, (4.0, 4.0, 4.0)),
    ('J4_link', f'{MESH}/j4.dae', 20.0, (1.0, 1.0, 1.0)),
    ('J5_link', f'{MESH}/j5.dae', 15.0, (0.5, 0.5, 0.5)),
    ('J6_link', f'{MESH}/j6.dae', 8.0, (0.3, 0.3, 0.3)),
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
    for name, mesh, mass, inertia in LINKS:
        ixx, iyy, izz = inertia
        w(f'  <link name="{name}">')
        w('    <visual>')
        w('      <origin xyz="0 0 0" rpy="0 0 0" />')
        w(f'      <geometry><mesh filename="{mesh}" /></geometry>')
        w('    </visual>')
        w('    <collision>')
        w('      <origin xyz="0 0 0" rpy="0 0 0" />')
        w(f'      <geometry><mesh filename="{mesh}" /></geometry>')
        w('    </collision>')
        w('    <inertial>')
        w(f'      <mass value="{mass}"/>')
        w(
            f'      <inertia ixx="{ixx}" ixy="0.0" ixz="0.0" '
            f'iyy="{iyy}" iyz="0.0" izz="{izz}"/>'
        )
        w('    </inertial>')
        w(f'  </link>')

    # flanges & end
    w('  <link name="flange">')
    w('    <inertial>')
    w('      <mass value="1.0"/>')
    w('      <inertia ixx="0.1" ixy="0.0" ixz="0.0" '
      'iyy="0.1" iyz="0.0" izz="0.1"/>')
    w('    </inertial>')
    w('  </link>')
    for name in ('fanuc_flange', 'ee_link'):
        w(f'  <link name="{name}" />')

    # GoCator
    w('  <link name="gocator_sensor">')
    w('    <visual>')
    w('      <origin xyz="0 0 -0.270" rpy="0 0 0" />')
    w(f'      <geometry><mesh filename="{MESH}/Gocator_2450.dae" /></geometry>')
    w('    </visual>')
    w('    <collision>')
    w('      <origin xyz="0 0 -0.270" rpy="0 0 0" />')
    w('      <geometry><box size="0.065 0.110 0.132" /></geometry>')
    w('    </collision>')
    w('    <inertial>')
    w('      <mass value="1.5"/>')
    w('      <inertia ixx="0.1" ixy="0.0" ixz="0.0" '
      'iyy="0.1" iyz="0.0" izz="0.1"/>')
    w('    </inertial>')
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
            lo_deg, hi_deg, effort, velocity_deg = limits
            lo = math.radians(lo_deg)
            hi = math.radians(hi_deg)
            velocity = math.radians(velocity_deg)
            w(
                f'    <limit lower="{lo:.9f}" upper="{hi:.9f}" '
                f'effort="{effort}" velocity="{velocity:.9f}" />'
            )
        w('  </joint>')

    w('')
    w('  <!-- ros2_control: Gazebo Sim hardware interface -->')
    w('  <ros2_control name="GazeboSystem" type="system">')
    w('    <hardware>')
    w('      <plugin>gz_ros2_control/GazeboSimSystem</plugin>')
    w('    </hardware>')
    for name in ('J1', 'J2', 'J3', 'J4', 'J5', 'J6'):
        w(f'    <joint name="{name}_joint">')
        w('      <command_interface name="position"/>')
        w('      <state_interface name="position"/>')
        w('      <state_interface name="velocity"/>')
        w('    </joint>')
    w('  </ros2_control>')
    w('')
    controller_config = (
        URDF_DIR.parent
        / 'ros2_ws'
        / 'src'
        / 'handeye_sim_bridge'
        / 'config'
        / 'gz_controllers.yaml'
    )
    w('  <!-- Gazebo Sim plugin loader for ros2_control -->')
    w('  <gazebo>')
    w(
        '    <plugin filename="gz_ros2_control-system" '
        'name="gz_ros2_control::GazeboSimROS2ControlPlugin">'
    )
    w(f'      <parameters>{controller_config}</parameters>')
    w('    </plugin>')
    w('  </gazebo>')
    w('</robot>')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    urdf = gen()
    out = URDF_DIR / 'calib_robot.urdf'
    out.write_text(urdf, encoding='utf-8')
    print(f'URDF generated: {out}  ({len(urdf)} chars)')
