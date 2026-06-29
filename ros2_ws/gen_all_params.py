#!/usr/bin/env python3
"""Generate ALL parameter files for the handeye simulation"""
import os

# Read URDF and SRDF
U = open("/workspace/urdf/calib_robot.urdf").read()
S = open("/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf").read()

# Indent for YAML
UI = "\n".join("      " + l for l in U.split("\n"))
SI = "\n".join("      " + l for l in S.split("\n"))

# 1. RSP params
with open("/tmp/rsp.yaml", "w") as f:
    f.write(f"""/**:
  ros__parameters:
    robot_description: |
{UI}
    publish_frequency: 30.0
""")

# 2. Controller manager params
with open("/tmp/cm.yaml", "w") as f:
    f.write(f"""controller_manager:
  ros__parameters:
    robot_description: |
{UI}
    update_rate: 100
""")

# 3. Joint state broadcaster
with open("/tmp/jsb.yaml", "w") as f:
    f.write("""joint_state_broadcaster:
  ros__parameters:
    type: joint_state_broadcaster/JointStateBroadcaster
""")

# 4. Joint trajectory controller
with open("/tmp/jtc.yaml", "w") as f:
    f.write("""joint_trajectory_controller:
  ros__parameters:
    type: joint_trajectory_controller/JointTrajectoryController
    joints: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint]
    command_interfaces: [position]
    state_interfaces: [position, velocity]
    state_publish_rate: 50.0
    action_monitor_rate: 20.0
    allow_partial_joints_goal: false
    constraints:
      stopped_velocity_tolerance: 0.01
      goal_time: 0.0
""")

# 5. MoveGroup main params
with open("/tmp/move_group_main.yaml", "w") as f:
    f.write(f"""move_group:
  ros__parameters:
    robot_description: |
{UI}
    robot_description_semantic: |
{SI}
    moveit_manage_controllers: false
    default_velocity_scaling_factor: 0.1
    default_acceleration_scaling_factor: 0.1
    default_planning_pipeline: ompl
    controller_names: [joint_trajectory_controller]
    moveit_simple_controller_manager:
      controller_names: [joint_trajectory_controller]
      joint_trajectory_controller:
        type: FollowJointTrajectory
        action_ns: follow_joint_trajectory
        default: true
        joints: [J1_joint, J2_joint, J3_joint, J4_joint, J5_joint, J6_joint]
""")

# 6. OMPL pipeline params (separate file to avoid key conflict)
with open("/tmp/ompl_pipeline.yaml", "w") as f:
    f.write("""move_group:
  ros__parameters:
    planning_pipelines:
      ompl:
        planning_plugins: [ompl_interface/OMPLPlanner]
        request_adapters:
          - default_planning_request_adapters/ResolveConstraintFrames
          - default_planning_request_adapters/FixWorkspaceBounds
          - default_planning_request_adapters/FixStartStateBounds
          - default_planning_request_adapters/FixStartStateCollision
          - default_planning_request_adapters/FixStartStatePathConstraints
        response_adapters:
          - default_planning_response_adapters/AddTimeOptimalParameterization
          - default_planning_response_adapters/ValidateSolution
          - default_planning_response_adapters/DisplayMotionPath
""")

# 7. RViz params
with open("/tmp/rviz.yaml", "w") as f:
    f.write(f"""/**:
  ros__parameters:
    robot_description_semantic: |
{SI}
    robot_description_kinematics:
      arm:
        kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
        kinematics_solver_search_resolution: 0.005
        kinematics_solver_timeout: 0.05
        kinematics_solver_attempts: 3
""")

print("All param files generated")
