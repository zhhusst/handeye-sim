#!/usr/bin/env python3
"""ROS 2 adapter for Phase 0b calibration-free bilateral seed collection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as point_cloud2
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from calibration_pipeline.geometry import make_transform, so3_exp
from calibration_pipeline.models import SensorROI
from calibration_pipeline.seed_collection import (
    TranslationServo,
    evaluate_bilateral_feature,
    rotation_diversity,
    star_rotation_plan,
)
from handeye_sim_bridge.fanuc_kinematic import (
    forward_kinematics_urdf,
    inverse_kinematics_numeric,
)


JOINT_NAMES = (
    "J1_joint",
    "J2_joint",
    "J3_joint",
    "J4_joint",
    "J5_joint",
    "J6_joint",
)


class SeedCollectionNode(Node):
    """Execute the star plan while preserving bilateral visibility."""

    def __init__(self) -> None:
        super().__init__("bilateral_seed_collection")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("seed.rotation_target_deg", 15.0)
        self.declare_parameter("seed.rotation_step_deg", 2.0)
        self.declare_parameter("seed.minimum_rotation_step_deg", 0.5)
        self.declare_parameter("seed.probe_step_m", 0.003)
        self.declare_parameter("seed.x_mid_tolerance_m", 0.003)
        self.declare_parameter("seed.maximum_servo_iterations", 8)
        self.declare_parameter("settling_time_s", 0.5)
        self.declare_parameter("output_file", "data/seed_measurements.json")
        self.declare_parameter("sensor.min_range_m", 0.27)
        self.declare_parameter("sensor.max_range_m", 0.82)
        self.declare_parameter("sensor.half_fov_deg", 15.0)
        self.declare_parameter("sensor.roi_safe_margin_m", 0.01)

        self.roi = SensorROI(
            min_range=float(self.get_parameter("sensor.min_range_m").value),
            max_range=float(self.get_parameter("sensor.max_range_m").value),
            half_fov_deg=float(self.get_parameter("sensor.half_fov_deg").value),
            safe_margin=float(self.get_parameter("sensor.roi_safe_margin_m").value),
        )
        self.rotation_target = np.deg2rad(
            float(self.get_parameter("seed.rotation_target_deg").value)
        )
        self.rotation_step_default = np.deg2rad(
            float(self.get_parameter("seed.rotation_step_deg").value)
        )
        self.rotation_step_minimum = np.deg2rad(
            float(self.get_parameter("seed.minimum_rotation_step_deg").value)
        )
        self.rotation_step = self.rotation_step_default
        self.probe_step = float(self.get_parameter("seed.probe_step_m").value)
        self.x_tolerance = float(
            self.get_parameter("seed.x_mid_tolerance_m").value
        )
        self.maximum_servo_iterations = int(
            self.get_parameter("seed.maximum_servo_iterations").value
        )
        self.settling_time = float(self.get_parameter("settling_time_s").value)
        self.output_file = Path(str(self.get_parameter("output_file").value))

        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.create_subscription(PointCloud2, "/gocator/profile", self._profile_callback, 10)
        self.create_subscription(
            Float64MultiArray, "/gocator/endpoints", self._endpoint_callback, 10
        )
        self._trajectory = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.create_service(Trigger, "~/start", self._start_callback)
        self.create_service(Trigger, "~/status", self._status_callback)
        self.create_timer(0.05, self._tick)

        self.latest_joints: np.ndarray | None = None
        self.latest_profile: np.ndarray | None = None
        self.latest_endpoints: tuple[np.ndarray, np.ndarray] | None = None
        self.reference_joints: np.ndarray | None = None
        self.reference_transform: np.ndarray | None = None
        self.last_valid_joints: np.ndarray | None = None
        self.records: list[dict] = []
        self.seed_rotations: list[np.ndarray] = []
        self.plan = star_rotation_plan()
        self.target_index = 0
        self.stage_index = 0
        self.accumulated_angle = 0.0
        self.failure_count = 0
        self.started = bool(self.get_parameter("auto_start").value)
        self.state = "WAIT_MANUAL_INIT"
        self.settle_until_ns = 0
        self.after_settle = ""
        self.pending_rotation = 0.0
        self.probe_axis = 0
        self.probe_base_transform: np.ndarray | None = None
        self.probe_base_feature = None
        self.probe_sensitivities: dict[int, float] = {}
        self.servo = TranslationServo()
        self.servo_previous_x = 0.0
        self.servo_previous_step = 0.0
        self.servo_iterations = 0
        self.get_logger().info(
            "waiting for one manually positioned, stable bilateral profile; "
            "call ~/start when ready"
        )

    def _joint_callback(self, message: JointState) -> None:
        try:
            self.latest_joints = np.array(
                [message.position[message.name.index(name)] for name in JOINT_NAMES],
                dtype=float,
            )
        except (ValueError, IndexError):
            return

    def _profile_callback(self, message: PointCloud2) -> None:
        points = list(
            point_cloud2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True
            )
        )
        self.latest_profile = np.asarray(points, dtype=float) if points else None

    def _endpoint_callback(self, message: Float64MultiArray) -> None:
        data = list(message.data)
        if len(data) < 9 or not bool(data[4]) or not bool(data[8]):
            self.latest_endpoints = None
            return
        self.latest_endpoints = (
            np.array(data[1:4], dtype=float),
            np.array(data[5:8], dtype=float),
        )

    def _start_callback(self, _request, response):
        self.started = True
        response.success = True
        response.message = "seed collection armed; waiting for stable bilateral data"
        return response

    def _status_callback(self, _request, response):
        response.success = self.state != "FAILED"
        response.message = (
            f"state={self.state}, seeds={len(self.records)}, "
            f"target={self.target_index}/{len(self.plan)}"
        )
        return response

    def _feature(self):
        if self.latest_endpoints is None:
            return None
        try:
            return evaluate_bilateral_feature(*self.latest_endpoints, self.roi)
        except ValueError:
            return None

    def _current_transform(self) -> np.ndarray | None:
        return (
            forward_kinematics_urdf(self.latest_joints)
            if self.latest_joints is not None
            else None
        )

    def _command_transform(self, transform: np.ndarray, after_settle: str) -> bool:
        if self.latest_joints is None:
            return False
        solutions = inverse_kinematics_numeric(transform, q_init=self.latest_joints)
        if len(solutions) == 0:
            self.get_logger().warning("IK rejected the requested flange pose")
            return False
        return self._command_joints(solutions[0], after_settle)

    def _command_joints(self, joints: np.ndarray, after_settle: str) -> bool:
        if not self._trajectory.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("joint trajectory action server is unavailable")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in joints]
        point.time_from_start = Duration(sec=2)
        goal.trajectory.points = [point]
        self.after_settle = after_settle
        self.state = "MOVING"
        self._trajectory.send_goal_async(goal).add_done_callback(self._goal_response)
        return True

    def _goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._fail("trajectory goal was rejected")
            return
        handle.get_result_async().add_done_callback(self._motion_result)

    def _motion_result(self, _future) -> None:
        self.settle_until_ns = (
            self.get_clock().now().nanoseconds + int(self.settling_time * 1e9)
        )
        self.state = "SETTLING"

    def _tick(self) -> None:
        if self.state == "SETTLING":
            if self.get_clock().now().nanoseconds >= self.settle_until_ns:
                next_state = self.after_settle
                self.after_settle = ""
                getattr(self, f"_after_{next_state.lower()}")()
            return
        if not self.started or self.state != "WAIT_MANUAL_INIT":
            return
        feature = self._feature()
        transform = self._current_transform()
        if (
            feature is None
            or not feature.safe
            or transform is None
            or self.latest_profile is None
        ):
            return
        self.reference_joints = self.latest_joints.copy()
        self.reference_transform = transform.copy()
        self.last_valid_joints = self.latest_joints.copy()
        self._save_seed("reference", feature)
        self.get_logger().info("reference seed accepted; starting star rotation plan")
        self._return_reference()

    def _return_reference(self) -> None:
        if self.reference_joints is None:
            self._fail("reference joints are unavailable")
            return
        self.rotation_step = self.rotation_step_default
        self.accumulated_angle = 0.0
        self.stage_index = 0
        self._command_joints(self.reference_joints, "RETURN_REFERENCE")

    def _after_return_reference(self) -> None:
        if self.target_index >= len(self.plan):
            self._finish()
            return
        self.last_valid_joints = self.latest_joints.copy()
        self.get_logger().info(f"target {self.plan[self.target_index].name}")
        self._issue_micro_rotation()

    def _issue_micro_rotation(self) -> None:
        current = self._current_transform()
        if current is None:
            self._fail("joint state is unavailable")
            return
        axis, sign = self.plan[self.target_index].stages[self.stage_index]
        remaining = self.rotation_target - self.accumulated_angle
        magnitude = min(self.rotation_step, remaining)
        delta = sign * magnitude
        axis_vector = np.zeros(3)
        axis_vector[axis] = delta
        target = current.copy()
        target[:3, :3] = current[:3, :3] @ so3_exp(axis_vector)
        self.pending_rotation = magnitude
        if not self._command_transform(target, "MICRO_ROTATION"):
            self._rollback("rotation IK failure")

    def _after_micro_rotation(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            self._rollback("bilateral feature became unsafe")
            return
        self.failure_count = 0
        self.accumulated_angle += self.pending_rotation
        self.last_valid_joints = self.latest_joints.copy()
        if abs(feature.x_mid) <= self.x_tolerance:
            self._continue_after_centered(feature)
            return
        self._begin_probing(feature)

    def _begin_probing(self, feature) -> None:
        self.servo_iterations = 0
        self.probe_axis = 0
        self.probe_sensitivities = {}
        self.probe_base_transform = self._current_transform()
        self.probe_base_feature = feature
        self._issue_probe()

    def _issue_probe(self) -> None:
        target = self.probe_base_transform.copy()
        local = np.zeros(3)
        local[self.probe_axis] = self.probe_step
        target[:3, 3] += target[:3, :3] @ local
        if not self._command_transform(target, "PROBE_OUT"):
            self.probe_sensitivities[self.probe_axis] = 0.0
            self._after_probe_back()

    def _after_probe_out(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            self._rollback("probe lost the bilateral feature")
            return
        self.probe_sensitivities[self.probe_axis] = (
            feature.x_mid - self.probe_base_feature.x_mid
        ) / self.probe_step
        if not self._command_transform(self.probe_base_transform, "PROBE_BACK"):
            self._rollback("probe return IK failure")

    def _after_probe_back(self) -> None:
        self.probe_axis += 1
        if self.probe_axis < 3:
            self.probe_base_transform = self._current_transform()
            feature = self._feature()
            if feature is None:
                self._rollback("feature missing after probe")
                return
            self.probe_base_feature = feature
            self._issue_probe()
            return
        try:
            self.servo.choose_axis(self.probe_sensitivities)
        except ValueError:
            self._rollback("no usable translation-servo axis")
            return
        self._issue_servo()

    def _issue_servo(self) -> None:
        if self.servo_iterations >= self.maximum_servo_iterations:
            self._rollback("translation servo iteration limit reached")
            return
        feature = self._feature()
        current = self._current_transform()
        if feature is None or current is None:
            self._rollback("servo input unavailable")
            return
        if abs(feature.x_mid) <= self.x_tolerance:
            self._continue_after_centered(feature)
            return
        step = self.servo.correction(feature.x_mid)
        local = np.zeros(3)
        local[self.servo.axis] = step
        target = current.copy()
        target[:3, 3] += current[:3, :3] @ local
        self.servo_previous_x = feature.x_mid
        self.servo_previous_step = step
        self.servo_iterations += 1
        if not self._command_transform(target, "SERVO"):
            self._rollback("servo IK failure")

    def _after_servo(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            self._rollback("servo left the safe bilateral region")
            return
        self.servo.update(
            feature.x_mid - self.servo_previous_x, self.servo_previous_step
        )
        self.last_valid_joints = self.latest_joints.copy()
        if abs(feature.x_mid) <= self.x_tolerance:
            self._continue_after_centered(feature)
        else:
            self._issue_servo()

    def _continue_after_centered(self, feature) -> None:
        if self.accumulated_angle + 1e-10 < self.rotation_target:
            self._issue_micro_rotation()
            return
        target = self.plan[self.target_index]
        if self.stage_index + 1 < len(target.stages):
            self.stage_index += 1
            self.accumulated_angle = 0.0
            self._issue_micro_rotation()
            return
        self._save_seed(target.name, feature)
        self.target_index += 1
        self._return_reference()

    def _rollback(self, reason: str) -> None:
        self.failure_count += 1
        self.rotation_step = max(self.rotation_step / 2.0, self.rotation_step_minimum)
        self.get_logger().warning(
            f"{reason}; rollback, rotation step={np.rad2deg(self.rotation_step):.2f} deg"
        )
        if self.failure_count >= 6:
            self.get_logger().warning("target abandoned after repeated failures")
            self.failure_count = 0
            self.target_index += 1
            self._return_reference()
            return
        if self.last_valid_joints is None:
            self._fail("no valid rollback pose")
            return
        self._command_joints(self.last_valid_joints, "ROLLBACK")

    def _after_rollback(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            self._fail("rollback did not restore bilateral visibility")
            return
        self._issue_micro_rotation()

    def _save_seed(self, label: str, feature) -> None:
        transform = self._current_transform()
        if transform is None or self.latest_profile is None:
            return
        candidate_rotations = self.seed_rotations + [transform[:3, :3]]
        diversity = rotation_diversity(candidate_rotations)
        if self.seed_rotations and diversity["minimum_pairwise_deg"] < 5.0:
            self.get_logger().warning(f"{label} rejected: insufficient rotation diversity")
            return
        self.seed_rotations.append(transform[:3, :3].copy())
        self.records.append(
            {
                "label": label,
                "R_BF": transform[:3, :3].tolist(),
                "t_BF": transform[:3, 3].tolist(),
                "joints": self.latest_joints.tolist(),
                "profile_points_S": self.latest_profile.tolist(),
                "endpoint_u_S": feature.endpoint_u.tolist(),
                "endpoint_v_S": feature.endpoint_v.tolist(),
                "x_mid": feature.x_mid,
                "roi_margin": feature.roi_margin,
            }
        )
        self.get_logger().info(f"accepted seed {len(self.records)}: {label}")

    def _finish(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(
            json.dumps({"schema_version": 1, "seeds": self.records}, indent=2),
            encoding="utf-8",
        )
        self.state = "DONE"
        self.get_logger().info(
            f"seed collection complete: {len(self.records)} records -> {self.output_file}"
        )

    def _fail(self, reason: str) -> None:
        self.state = "FAILED"
        self.get_logger().error(reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SeedCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
