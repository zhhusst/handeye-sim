#!/usr/bin/env python3
"""Select a collision-valid simulation reference candidate before moving.

Reads candidates produced from simulation truth.  MoveIt validates the robot
state along a joint-space approach, including the configured board obstacle.
Only the selected trajectory goal is printed on stdout for start_simulation.sh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPlanningScene, GetStateValidity
from moveit_msgs.msg import PlanningSceneComponents

JOINT_NAMES = [f"J{i}_joint" for i in range(1, 7)]
NOMINAL_JOINTS = [-0.236623, -0.049108, -0.636688, -0.402198, -1.071110, 0.878050]


def goal_for(joints: list[float]) -> str:
    return (
        "{trajectory: {joint_names: [" + ", ".join(JOINT_NAMES) + "], "
        "points: [{positions: [" + ", ".join(f"{q:.12g}" for q in joints) + "], "
        "time_from_start: {sec: 3, nanosec: 0}}]}}"
    )


class MoveItCheck(Node):
    def __init__(self) -> None:
        super().__init__("simulation_reference_pose_check")
        self.joints = None
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.validity = self.create_client(GetStateValidity, "/check_state_validity")
        self.scene = self.create_client(GetPlanningScene, "/get_planning_scene")

    def on_joints(self, message: JointState) -> None:
        try:
            self.joints = np.array([message.position[message.name.index(name)] for name in JOINT_NAMES])
        except (ValueError, IndexError):
            pass

    def call(self, client, request, timeout_s=5.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"MoveIt service {client.srv_name} did not respond")
        return future.result()

    def wait_ready(self, timeout_s=20.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.joints is None or not self.validity.service_is_ready() or not self.scene.service_is_ready():
                continue
            request = GetPlanningScene.Request()
            request.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
            response = self.call(self.scene, request)
            if any(obj.id == "calibration_plate" for obj in response.scene.world.collision_objects):
                return
        raise RuntimeError("MoveIt or joint states unavailable, or configured plate collision was not loaded")

    def state_valid(self, joints: np.ndarray) -> bool:
        request = GetStateValidity.Request()
        request.group_name = "arm"
        request.robot_state.joint_state.name = JOINT_NAMES
        request.robot_state.joint_state.position = joints.tolist()
        return bool(self.call(self.validity, request).valid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not data["enabled"]:
        args.selected.write_text(json.dumps({"enabled": False, "joints_rad": NOMINAL_JOINTS}), encoding="utf-8")
        print(goal_for(NOMINAL_JOINTS))
        return

    rclpy.init()
    checker = MoveItCheck()
    try:
        checker.wait_ready()
        for candidate in data["candidates"]:
            target = np.asarray(candidate["joints_rad"], dtype=float)
            # State validity at finite intervals is a conservative startup
            # screen, not a continuous collision-free trajectory proof.
            approach = [checker.joints + alpha * (target - checker.joints) for alpha in np.linspace(0.0, 1.0, 25)]
            if all(checker.state_valid(joints) for joints in approach):
                args.selected.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
                print(goal_for(target.tolist()))
                return
        raise RuntimeError("initialization infeasible: no candidate passed MoveIt board/robot collision and approach checks")
    finally:
        checker.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Simulation automatic reference pose failed: {error}", file=sys.stderr)
        sys.exit(1)
