#!/usr/bin/env python3
"""Generate correct move_group YAML with parameters in the right namespace"""
import yaml
import sys

U = open("/workspace/urdf/calib_robot.urdf").read()
S = open("/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf").read()

# Build the YAML structure correctly
# planning_pipelines is both a list param AND a parent namespace
# In ROS2 YAML, use flat naming to avoid key collision
params = {
    "move_group": {
        "ros__parameters": {
            "robot_description": U,
            "robot_description_semantic": S,
            "moveit_manage_controllers": False,
            "default_velocity_scaling_factor": 0.1,
            "default_acceleration_scaling_factor": 0.1,
            "planning_pipelines": ["ompl"],
            "default_planning_pipeline": "ompl",
            "controller_names": ["joint_trajectory_controller"],
            "moveit_simple_controller_manager": {
                "controller_names": ["joint_trajectory_controller"],
                "joint_trajectory_controller": {
                    "type": "FollowJointTrajectory",
                    "action_ns": "follow_joint_trajectory",
                    "default": True,
                    "joints": list("J1_joint J2_joint J3_joint J4_joint J5_joint J6_joint".split()),
                }
            },
            # Pipeline-specific params under planning_pipelines.ompl.*
            "planning_pipelines.ompl.planning_plugins": [
                "ompl_interface/OMPLPlanner"
            ],
            "planning_pipelines.ompl.request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/FixWorkspaceBounds",
                "default_planning_request_adapters/FixStartStateBounds",
                "default_planning_request_adapters/FixStartStateCollision",
                "default_planning_request_adapters/FixStartStatePathConstraints",
            ],
            "planning_pipelines.ompl.response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
                "default_planning_response_adapters/DisplayMotionPath",
            ],
        }
    }
}

with open("/tmp/move_group.yaml", "w") as f:
    yaml.dump(params, f, default_flow_style=False, allow_unicode=True)

print("Generated /tmp/move_group.yaml")
print("Keys:", list(params["move_group"]["ros__parameters"].keys()))
