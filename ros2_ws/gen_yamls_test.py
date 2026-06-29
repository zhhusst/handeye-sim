#!/usr/bin/env python3
"""Test: does ROS2 handle planning_pipelines with both list value and children?"""
U = open("/workspace/urdf/calib_robot.urdf").read()
S = open("/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf").read()

# File 1: Everything except nested planning_pipelines
with open("/tmp/main.yaml", "w") as f:
    f.write("""move_group:
  ros__parameters:
    robot_description: |
""" + "\n".join("      " + l for l in U.split("\n")) + """
    robot_description_semantic: |
""" + "\n".join("      " + l for l in S.split("\n")) + """
    moveit_manage_controllers: false
    planning_pipelines: ["ompl"]
    default_planning_pipeline: ompl
    controller_names: ["joint_trajectory_controller"]
    moveit_simple_controller_manager:
      controller_names: ["joint_trajectory_controller"]
      joint_trajectory_controller:
        type: FollowJointTrajectory
        action_ns: follow_joint_trajectory
        default: true
        joints: ["J1_joint", "J2_joint", "J3_joint", "J4_joint", "J5_joint", "J6_joint"]
""")

# File 2: Pipeline-specific params under planning_pipelines.ompl.*
with open("/tmp/pipeline.yaml", "w") as f:
    f.write("""move_group:
  ros__parameters:
    planning_pipelines:
      ompl:
        planning_plugins: ["ompl_interface/OMPLPlanner"]
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

print("Files generated")
