#!/usr/bin/env python3
"""Generate move_group parameters YAML file with literal block scalars."""
import os

# Read URDF and SRDF
with open('/workspace/urdf/calib_robot.urdf') as f:
    urdf = f.read()
with open('/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf') as f:
    srdf = f.read()

# Indent URDF/SRDF lines by 6 spaces for YAML literal block
urdf_indented = '\n'.join('      ' + line for line in urdf.split('\n'))
srdf_indented = '\n'.join('      ' + line for line in srdf.split('\n'))

yaml_content = f"""/**:
  ros__parameters:
    robot_description: |
{urdf_indented}
    robot_description_semantic: |
{srdf_indented}
    moveit_manage_controllers: false
    planning_pipelines: ["ompl"]
    default_planning_pipeline: ompl
    arm:
      kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
      kinematics_solver_search_resolution: 0.005
      kinematics_solver_timeout: 0.05
      kinematics_solver_attempts: 3
"""

with open('/tmp/mg2.yaml', 'w') as f:
    f.write(yaml_content)

print(f"Written: {os.path.getsize('/tmp/mg2.yaml')} bytes")
