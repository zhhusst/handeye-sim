#!/usr/bin/env python3
"""Validate the YAML param file for move_group"""
import yaml

U = open("/workspace/urdf/calib_robot.urdf").read()
S = open("/workspace/ros2_ws/install/handeye_sim_bridge/share/handeye_sim_bridge/config/fanuc.srdf").read()

UI = "\n".join("      " + l for l in U.split("\n"))
SI = "\n".join("      " + l for l in S.split("\n"))

y = """/**:
  ros__parameters:
    robot_description: |
""" + UI + """
    robot_description_semantic: |
""" + SI + """
    planning_response_adapters:
      - default_planning_response_adapters/AddTimeOptimalParameterization
    ompl:
      planning_plugins: ["ompl_interface/OMPLPlanner"]
"""

with open("/tmp/mg_test.yaml", "w") as f:
    f.write(y)

try:
    data = yaml.safe_load(y)
    print("YAML VALID")
    params = data.get("/**", {}).get("ros__parameters", {})
    for k in ["robot_description", "robot_description_semantic", "planning_response_adapters"]:
        if k in params:
            v = params[k]
            if isinstance(v, list):
                print("  %s: %s" % (k, v))
            elif isinstance(v, str):
                print("  %s: (%d chars)" % (k, len(v)))
        else:
            print("  %s: MISSING!" % k)
except yaml.YAMLError as e:
    print("YAML ERROR: %s" % e)
    if hasattr(e, "problem_mark"):
        mark = e.problem_mark
        print("  Line %d, Col %d" % (mark.line+1, mark.column+1))
