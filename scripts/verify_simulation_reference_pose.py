#!/usr/bin/env python3
"""Verify that the automatic simulation reference is actually detected."""

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
from std_srvs.srv import Trigger
import yaml

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "ros2_ws/src/handeye_calibration_core"))
from calibration_pipeline.models import SensorROI, TrapezoidDomain  # noqa: E402
from calibration_pipeline.geometry import make_transform  # noqa: E402
from calibration_pipeline.seed_collection.features import evaluate_bilateral_feature  # noqa: E402
from calibration_pipeline.seed_collection.initial_pose import InitialPoseCriteria, assess_initial_pose  # noqa: E402
from calibration_pipeline.simulation.scene_truth import load_handeye_truth  # noqa: E402
from fanuc_m20id25_support.fanuc_kinematic import (  # noqa: E402
    JOINT_LIMITS_DEG, forward_kinematics_urdf,
)

JOINT_NAMES = [f"J{i}_joint" for i in range(1, 7)]


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("verify_simulation_reference_pose")
        self.client = self.create_client(Trigger, "/profile_endpoint_detector/status")
        self.joints = None
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)

    def on_joints(self, message: JointState) -> None:
        try:
            self.joints = np.array([message.position[message.name.index(name)] for name in JOINT_NAMES])
        except (ValueError, IndexError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    if not selected.get("enabled", True):
        return
    params = yaml.safe_load(args.config.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    seed, initial = params["seed"], params["seed"]["initial"]
    roi = SensorROI(
        hard_domain=TrapezoidDomain(*params["sensor"]["hard_trapezoid"]),
        safe_domain=TrapezoidDomain(*params["sensor"]["safe_trapezoid"]),
    )
    rclpy.init()
    probe = Probe()
    last = "no detector status"
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            rclpy.spin_once(probe, timeout_sec=0.2)
            if probe.joints is None or not probe.client.service_is_ready():
                continue
            future = probe.client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(probe, future, timeout_sec=2.0)
            if not future.done() or future.result() is None:
                continue
            response = future.result()
            status = json.loads(response.message)
            last = status.get("reason", status.get("state", "no reason"))
            if not response.success or status.get("state") != "VALID" or status.get("alignment_stable_frames", 0) < 5:
                continue
            first = np.asarray(status["endpoint_first_mm"], dtype=float) / 1000.0
            second = np.asarray(status["endpoint_second_mm"], dtype=float) / 1000.0
            feature = evaluate_bilateral_feature(first, second, roi)
            criteria = InitialPoseCriteria(
                maximum_abs_x_mid_m=initial["maximum_abs_x_mid_m"],
                minimum_z_mid_m=initial["minimum_z_mid_m"],
                maximum_z_mid_m=initial["maximum_z_mid_m"],
                minimum_domain_margin_m=initial["minimum_domain_margin_m"],
                minimum_profile_length_m=initial["minimum_profile_length_m"],
                maximum_profile_length_m=initial["maximum_profile_length_m"],
                minimum_absolute_endpoint_depth_delta_m=initial["minimum_absolute_endpoint_depth_delta_m"],
                minimum_normalized_joint_margin=initial["minimum_normalized_joint_margin"],
                minimum_local_ik_directions=initial["minimum_local_ik_directions"],
            )
            assessment = assess_initial_pose(
                feature, probe.joints, np.deg2rad(JOINT_LIMITS_DEG),
                local_ik_directions=selected["local_ik_directions"], criteria=criteria,
            )
            rotation_fs, translation_fs = load_handeye_truth(args.config)
            actual_sensor = forward_kinematics_urdf(probe.joints) @ make_transform(
                rotation_fs, translation_fs
            )
            sensor_xy = actual_sensor[:2, 3]
            outward_cosine = (
                float(np.dot(actual_sensor[:2, 1], sensor_xy / np.linalg.norm(sensor_xy)))
                if np.linalg.norm(sensor_xy) > 1e-6 else -1.0
            )
            minimum_outward_cosine = float(params.get("simulation_auto_reference", {}).get(
                "minimum_outward_sensor_y_cosine", -1.0
            ))
            if (
                assessment.accepted
                and seed["servo"]["length_lower_m"] <= feature.profile_length <= seed["servo"]["length_upper_m"]
                and abs(feature.x_mid) <= seed["x_mid_tolerance_m"]
                and outward_cosine >= minimum_outward_cosine
            ):
                print(
                    "Simulation reference verified: "
                    f"x_mid={feature.x_mid * 1000:.2f} mm, "
                    f"spacing={feature.profile_length * 1000:.2f} mm, "
                    f"domain_margin={feature.domain_margin * 1000:.2f} mm, "
                    f"sensor_+Y_outward_cosine={outward_cosine:.2f}"
                )
                return
            last = (f"initial envelope={assessment.reasons}, spacing={feature.profile_length:.4f} m, "
                    f"sensor_+Y_outward_cosine={outward_cosine:.2f}")
        raise RuntimeError(f"observed reference did not pass detection and seed work band: {last}")
    finally:
        probe.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Simulation automatic reference verification failed: {error}", file=sys.stderr)
        sys.exit(1)
