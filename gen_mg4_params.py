#!/usr/bin/env python3
"""Generate move_group parameters YAML - correct planning_plugins name."""
import os

with open('/workspace/urdf/calib_robot.urdf') as f:
    urdf = f.read()
with open('/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf') as f:
    srdf = f.read()

urdf_i = '\n'.join('      ' + l for l in urdf.split('\n'))
srdf_i = '\n'.join('      ' + l for l in srdf.split('\n'))

yaml = """/**:
  ros__parameters:
    robot_description: |
""" + urdf_i + """
    robot_description_semantic: |
""" + srdf_i + """
    moveit_manage_controllers: false
    planning_pipelines: ["ompl"]
    default_planning_pipeline: ompl
    arm:
      kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
      kinematics_solver_search_resolution: 0.005
      kinematics_solver_timeout: 0.05
      kinematics_solver_attempts: 3
    ompl:
      planning_plugins: ["ompl_interface/OMPLPlanner"]
"""

with open('/tmp/mg4.yaml', 'w') as f:
    f.write(yaml)

import yaml
with open('/tmp/mg4.yaml') as f:
    data = yaml.safe_load(f)
params = data['/**']['ros__parameters']
print('ompl.planning_plugins:', params.get('ompl', {}).get('planning_plugins'))
print(f'Size: {os.path.getsize("/tmp/mg4.yaml")} bytes')
