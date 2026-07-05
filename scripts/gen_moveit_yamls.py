#!/usr/bin/env python3
"""Generate correct move_group YAML files for two-file loading"""
import yaml

U = open("/workspace/urdf/calib_robot.urdf").read()
S = open("/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf").read()

# File 1: Main params (NO planning_pipelines - set via -p to avoid conflict)
main = {
    "move_group": {
        "ros__parameters": {
            "robot_description": U,
            "robot_description_semantic": S,
            "moveit_manage_controllers": False,
            "default_velocity_scaling_factor": 0.1,
            "default_acceleration_scaling_factor": 0.1,
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
        }
    }
}

# File 2: Pipeline-specific params under planning_pipelines.ompl.*
# ROS2 merges this as children of the planning_pipelines parameter
ompl = {
    "move_group": {
        "ros__parameters": {
            "planning_pipelines": {
                "ompl": {
                    "planning_plugins": ["ompl_interface/OMPLPlanner"],
                    "request_adapters": [
                        "default_planning_request_adapters/ResolveConstraintFrames",
                        "default_planning_request_adapters/FixWorkspaceBounds",
                        "default_planning_request_adapters/FixStartStateBounds",
                        "default_planning_request_adapters/FixStartStateCollision",
                        "default_planning_request_adapters/FixStartStatePathConstraints",
                    ],
                    "response_adapters": [
                        "default_planning_response_adapters/AddTimeOptimalParameterization",
                        "default_planning_response_adapters/ValidateSolution",
                        "default_planning_response_adapters/DisplayMotionPath",
                    ],
                }
            }
        }
    }
}

with open("/tmp/move_group_main.yaml", "w") as f:
    yaml.dump(main, f, default_flow_style=False, allow_unicode=True)
    print("Written /tmp/move_group_main.yaml")

with open("/tmp/ompl_pipeline.yaml", "w") as f:
    yaml.dump(ompl, f, default_flow_style=False, allow_unicode=True)
    print("Written /tmp/ompl_pipeline.yaml")

print("DONE")
