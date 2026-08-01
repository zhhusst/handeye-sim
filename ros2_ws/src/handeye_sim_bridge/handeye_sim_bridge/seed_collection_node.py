#!/usr/bin/env python3
"""ROS 2 adapter for Phase 0b calibration-free bilateral seed collection."""

from __future__ import annotations

import json
from pathlib import Path
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs_py.point_cloud2 as point_cloud2
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

from calibration_pipeline.geometry import (
    make_transform,
    quaternion_to_matrix,
    so3_exp,
)
from calibration_pipeline.models import SensorROI, TrapezoidDomain
from calibration_pipeline.seed_collection import (
    InitialPoseCriteria,
    RotationTarget,
    TranslationServo,
    adaptive_rotation_plan,
    assess_initial_pose,
    dynamic_preflight_decision,
    evaluate_bilateral_feature,
    local_preflight_is_acceptable,
    preflight_guided_rotation_plan,
    robust_endpoint_inliers,
    rotation_diversity,
    seed_feature_is_acceptable,
)
from handeye_sim_bridge.fanuc_kinematic import (
    JOINT_LIMITS_DEG,
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
        self.declare_parameter("collection_mode", "automatic")
        self.declare_parameter("seed.rotation_target_deg", 6.0)
        self.declare_parameter("seed.rotation_step_deg", 2.0)
        self.declare_parameter("seed.minimum_rotation_step_deg", 0.25)
        self.declare_parameter("seed.minimum_partial_rotation_deg", 2.5)
        self.declare_parameter("seed.probe_step_m", 0.001)
        self.declare_parameter("seed.x_mid_tolerance_m", 0.003)
        self.declare_parameter(
            "seed.rotation_continue_max_abs_x_mid_m", 0.025
        )
        self.declare_parameter(
            "seed.rotation_continue_minimum_domain_margin_m", 0.015
        )
        self.declare_parameter("seed.maximum_translation_step_m", 0.008)
        self.declare_parameter("seed.maximum_servo_iterations", 8)
        self.declare_parameter("seed.measurement_retry_timeout_s", 1.0)
        self.declare_parameter("seed.measurement_batch_size", 20)
        self.declare_parameter("seed.minimum_batch_inliers", 15)
        self.declare_parameter("seed.maximum_batch_frames", 40)
        self.declare_parameter("seed.measurement_batch_timeout_s", 6.0)
        self.declare_parameter("seed.endpoint_mad_multiplier", 3.5)
        self.declare_parameter("seed.maximum_target_failures", 3)
        self.declare_parameter("seed.minimum_profile_points", 5)
        self.declare_parameter("seed.minimum_rotation_separation_deg", 2.0)
        self.declare_parameter("seed.target_count", 6)
        self.declare_parameter("seed.minimum_seed_domain_margin_m", 0.002)
        self.declare_parameter("seed.initial.maximum_abs_x_mid_m", 0.03)
        self.declare_parameter("seed.initial.minimum_z_mid_m", 0.30)
        self.declare_parameter("seed.initial.maximum_z_mid_m", 0.55)
        self.declare_parameter("seed.initial.minimum_domain_margin_m", 0.020)
        self.declare_parameter("seed.initial.minimum_profile_length_m", 0.05)
        self.declare_parameter("seed.initial.maximum_profile_length_m", 0.25)
        self.declare_parameter(
            "seed.initial.minimum_absolute_endpoint_depth_delta_m", 0.015
        )
        self.declare_parameter(
            "seed.initial.minimum_normalized_joint_margin", 0.05
        )
        self.declare_parameter("seed.initial.minimum_local_ik_directions", 3)
        self.declare_parameter("seed.initial.local_ik_test_step_deg", 2.0)
        self.declare_parameter("seed.initial.maximum_local_joint_step_deg", 20.0)
        self.declare_parameter("seed.preflight.mode", "auto")
        self.declare_parameter("seed.preflight.rotation_deg", 2.0)
        self.declare_parameter("seed.preflight.maximum_abs_x_mid_m", 0.030)
        self.declare_parameter("seed.preflight.minimum_domain_margin_m", 0.015)
        self.declare_parameter("seed.preflight.minimum_feasible_directions", 2)
        self.declare_parameter(
            "seed.preflight.auto_skip_maximum_abs_x_mid_m", 0.015
        )
        self.declare_parameter(
            "seed.preflight.auto_skip_minimum_domain_margin_m", 0.030
        )
        self.declare_parameter("seed.motion_duration_s", 0.65)
        self.declare_parameter("seed.motion_timeout_s", 4.0)
        self.declare_parameter("settling_time_s", 0.3)
        self.declare_parameter("manual_max_joint_speed_rad_s", 0.02)
        self.declare_parameter("maximum_measurement_skew_s", 0.1)
        self.declare_parameter("maximum_measurement_age_s", 0.5)
        self.declare_parameter("measurement.pose_source", "topic")
        self.declare_parameter("output_file", "data/seed_measurements.json")
        self.declare_parameter("sensor.min_range_m", 0.27)
        self.declare_parameter("sensor.max_range_m", 0.82)
        self.declare_parameter("sensor.half_fov_deg", 15.0)
        self.declare_parameter("sensor.roi_safe_margin_m", 0.01)
        self.declare_parameter(
            "sensor.hard_trapezoid",
            [-0.292, 0.82, -0.021, -0.22, -0.019, 0.22],
        )
        self.declare_parameter(
            "sensor.safe_trapezoid", [0.27, 0.78, -0.115, -0.19, 0.095, 0.19]
        )

        hard_values = list(self.get_parameter("sensor.hard_trapezoid").value)
        safe_values = list(self.get_parameter("sensor.safe_trapezoid").value)
        hard_domain = TrapezoidDomain(*map(float, hard_values))
        safe_domain = TrapezoidDomain(*map(float, safe_values))
        self.roi = SensorROI(
            min_range=float(self.get_parameter("sensor.min_range_m").value),
            max_range=float(self.get_parameter("sensor.max_range_m").value),
            half_fov_deg=float(self.get_parameter("sensor.half_fov_deg").value),
            safe_margin=float(self.get_parameter("sensor.roi_safe_margin_m").value),
            hard_domain=hard_domain,
            safe_domain=safe_domain,
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
        self.minimum_partial_rotation = np.deg2rad(
            float(self.get_parameter("seed.minimum_partial_rotation_deg").value)
        )
        self.probe_step = float(self.get_parameter("seed.probe_step_m").value)
        self.x_tolerance = float(
            self.get_parameter("seed.x_mid_tolerance_m").value
        )
        self.rotation_continue_max_abs_x_mid = float(
            self.get_parameter(
                "seed.rotation_continue_max_abs_x_mid_m"
            ).value
        )
        self.rotation_continue_minimum_domain_margin = float(
            self.get_parameter(
                "seed.rotation_continue_minimum_domain_margin_m"
            ).value
        )
        self.maximum_translation_step = float(
            self.get_parameter("seed.maximum_translation_step_m").value
        )
        self.maximum_servo_iterations = int(
            self.get_parameter("seed.maximum_servo_iterations").value
        )
        self.measurement_retry_timeout_ns = int(
            float(
                self.get_parameter(
                    "seed.measurement_retry_timeout_s"
                ).value
            )
            * 1e9
        )
        self.seed_measurement_batch_size = int(
            self.get_parameter("seed.measurement_batch_size").value
        )
        self.minimum_batch_inliers = int(
            self.get_parameter("seed.minimum_batch_inliers").value
        )
        self.maximum_batch_frames = int(
            self.get_parameter("seed.maximum_batch_frames").value
        )
        self.measurement_batch_timeout_ns = int(
            float(
                self.get_parameter("seed.measurement_batch_timeout_s").value
            )
            * 1e9
        )
        self.endpoint_mad_multiplier = float(
            self.get_parameter("seed.endpoint_mad_multiplier").value
        )
        if (
            self.seed_measurement_batch_size < 1
            or self.minimum_batch_inliers < 1
            or self.minimum_batch_inliers > self.seed_measurement_batch_size
            or self.maximum_batch_frames < self.seed_measurement_batch_size
            or self.measurement_batch_timeout_ns <= 0
            or self.endpoint_mad_multiplier <= 0.0
        ):
            raise ValueError("invalid stationary seed measurement batch configuration")
        self.maximum_target_failures = int(
            self.get_parameter("seed.maximum_target_failures").value
        )
        self.motion_duration = float(
            self.get_parameter("seed.motion_duration_s").value
        )
        self.motion_timeout_ns = int(
            float(self.get_parameter("seed.motion_timeout_s").value) * 1e9
        )
        self.minimum_profile_points = int(
            self.get_parameter("seed.minimum_profile_points").value
        )
        self.minimum_rotation_separation_deg = float(
            self.get_parameter("seed.minimum_rotation_separation_deg").value
        )
        self.target_count = int(self.get_parameter("seed.target_count").value)
        self.minimum_seed_domain_margin = float(
            self.get_parameter("seed.minimum_seed_domain_margin_m").value
        )
        self.initial_criteria = InitialPoseCriteria(
            maximum_abs_x_mid_m=float(
                self.get_parameter(
                    "seed.initial.maximum_abs_x_mid_m"
                ).value
            ),
            minimum_z_mid_m=float(
                self.get_parameter("seed.initial.minimum_z_mid_m").value
            ),
            maximum_z_mid_m=float(
                self.get_parameter("seed.initial.maximum_z_mid_m").value
            ),
            minimum_domain_margin_m=float(
                self.get_parameter(
                    "seed.initial.minimum_domain_margin_m"
                ).value
            ),
            minimum_profile_length_m=float(
                self.get_parameter(
                    "seed.initial.minimum_profile_length_m"
                ).value
            ),
            maximum_profile_length_m=float(
                self.get_parameter(
                    "seed.initial.maximum_profile_length_m"
                ).value
            ),
            minimum_absolute_endpoint_depth_delta_m=float(
                self.get_parameter(
                    "seed.initial.minimum_absolute_endpoint_depth_delta_m"
                ).value
            ),
            minimum_normalized_joint_margin=float(
                self.get_parameter(
                    "seed.initial.minimum_normalized_joint_margin"
                ).value
            ),
            minimum_local_ik_directions=int(
                self.get_parameter(
                    "seed.initial.minimum_local_ik_directions"
                ).value
            ),
        )
        self.initial_ik_test_step = np.deg2rad(
            float(
                self.get_parameter(
                    "seed.initial.local_ik_test_step_deg"
                ).value
            )
        )
        self.initial_maximum_local_joint_step = np.deg2rad(
            float(
                self.get_parameter(
                    "seed.initial.maximum_local_joint_step_deg"
                ).value
            )
        )
        self.preflight_mode = str(
            self.get_parameter("seed.preflight.mode").value
        ).strip().lower()
        if self.preflight_mode not in {"auto", "always", "off"}:
            raise ValueError("seed.preflight.mode must be auto, always or off")
        self.preflight_rotation = np.deg2rad(
            float(self.get_parameter("seed.preflight.rotation_deg").value)
        )
        self.preflight_maximum_abs_x_mid = float(
            self.get_parameter(
                "seed.preflight.maximum_abs_x_mid_m"
            ).value
        )
        self.preflight_minimum_margin = float(
            self.get_parameter(
                "seed.preflight.minimum_domain_margin_m"
            ).value
        )
        self.preflight_minimum_directions = int(
            self.get_parameter(
                "seed.preflight.minimum_feasible_directions"
            ).value
        )
        self.preflight_auto_skip_maximum_abs_x_mid = float(
            self.get_parameter(
                "seed.preflight.auto_skip_maximum_abs_x_mid_m"
            ).value
        )
        self.preflight_auto_skip_minimum_margin = float(
            self.get_parameter(
                "seed.preflight.auto_skip_minimum_domain_margin_m"
            ).value
        )
        self.settling_time = float(self.get_parameter("settling_time_s").value)
        self.manual_max_joint_speed = float(
            self.get_parameter("manual_max_joint_speed_rad_s").value
        )
        self.maximum_measurement_skew_ns = int(
            float(self.get_parameter("maximum_measurement_skew_s").value) * 1e9
        )
        self.maximum_measurement_age_ns = int(
            float(self.get_parameter("maximum_measurement_age_s").value) * 1e9
        )
        self.pose_source = str(
            self.get_parameter("measurement.pose_source").value
        ).strip().lower()
        if self.pose_source not in {"topic", "tf"}:
            raise ValueError("measurement.pose_source must be topic or tf")
        self.output_file = Path(str(self.get_parameter("output_file").value))
        self.collection_mode = str(
            self.get_parameter("collection_mode").value
        ).strip().lower()
        if self.collection_mode not in {"automatic", "manual"}:
            raise ValueError(
                "collection_mode must be either 'automatic' or 'manual'"
            )

        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.create_subscription(PointCloud2, "/gocator/profile", self._profile_callback, 10)
        self.create_subscription(
            PointCloud2,
            "/calibration/endpoints",
            self._endpoint_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/calibration/flange_pose",
            self._flange_pose_callback,
            10,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._trajectory = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.create_service(Trigger, "~/start", self._start_callback)
        self.create_service(Trigger, "~/capture", self._capture_callback)
        self.create_service(Trigger, "~/status", self._status_callback)
        self.create_timer(0.05, self._tick)

        self.latest_joints: np.ndarray | None = None
        self.latest_joint_speed = float("inf")
        self.latest_profile: np.ndarray | None = None
        self.latest_endpoints: tuple[np.ndarray, np.ndarray] | None = None
        self.latest_measurement_transform: np.ndarray | None = None
        self.pending_profiles: dict[tuple[int, int], np.ndarray] = {}
        self.pending_endpoint_frames: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = {}
        self.pending_flange_poses: dict[tuple[int, int], np.ndarray] = {}
        self.latest_profile_ns = 0
        self.latest_profile_stamp = None
        self.latest_endpoints_ns = 0
        self.latest_joint_wall_ns = 0
        self.latest_profile_wall_ns = 0
        self.latest_endpoints_wall_ns = 0
        self.measurement_not_before_wall_ns = 0
        self.measurement_retry_deadline_wall_ns = 0
        self.reference_joints: np.ndarray | None = None
        self.reference_transform: np.ndarray | None = None
        self.reference_measurement_transform: np.ndarray | None = None
        self.reference_profile: np.ndarray | None = None
        self.reference_feature = None
        self.last_valid_joints: np.ndarray | None = None
        self.last_valid_transform: np.ndarray | None = None
        self.last_valid_profile: np.ndarray | None = None
        self.last_valid_feature = None
        self.records: list[dict] = []
        self.seed_rotations: list[np.ndarray] = []
        self.seed_capture_frames: list[dict] = []
        self.seed_capture_label = ""
        self.seed_capture_continuation = ""
        self.seed_capture_started_wall_ns = 0
        self.seed_capture_deadline_wall_ns = 0
        self.seed_capture_last_diagnostics = None
        self.pending_partial_label = ""
        self.plan = adaptive_rotation_plan()
        self.preflight_plan = (
            RotationTarget("preflight_rx_negative", ((0, -1),)),
            RotationTarget("preflight_rx_positive", ((0, 1),)),
            RotationTarget("preflight_ry_negative", ((1, -1),)),
            RotationTarget("preflight_ry_positive", ((1, 1),)),
        )
        self.collection_phase = "INITIAL"
        self.preflight_index = 0
        self.preflight_results: list[dict] = []
        self.preflight_was_required: bool | None = (
            False if self.collection_mode == "manual" else None
        )
        self.preflight_decision_reason = (
            "manual_collection"
            if self.collection_mode == "manual"
            else "pending_static_assessment"
        )
        self.initial_assessment_cache = None
        self.initial_assessment_cache_wall_ns = 0
        self.initial_assessment_period_ns = int(0.5e9)
        self.target_index = 0
        self.stage_index = 0
        self.accumulated_angle = 0.0
        self.failure_count = 0
        self.started = bool(self.get_parameter("auto_start").value)
        self.state = "WAIT_MANUAL_INIT"
        self.failure_reason = ""
        self.settle_until_ns = 0
        self.motion_deadline_wall_ns = 0
        self.after_settle = ""
        self.pending_rotation = 0.0
        self.probe_axis = 0
        self.probe_base_transform: np.ndarray | None = None
        self.probe_base_feature = None
        self.probe_sensitivities: dict[int, float] = {}
        self.learned_servo_axis: int | None = None
        self.learned_servo_sensitivity: float | None = None
        self.servo = TranslationServo(maximum_step=self.maximum_translation_step)
        self.servo_from_cache = False
        self.servo_reprobe_attempted = False
        self.servo_previous_x = 0.0
        self.servo_previous_step = 0.0
        self.servo_iterations = 0
        self.get_logger().info(
            "waiting for one manually positioned, stable bilateral profile; "
            f"mode={self.collection_mode}; preflight={self.preflight_mode}; "
            "call ~/start when ready"
        )

    def _joint_callback(self, message: JointState) -> None:
        try:
            indices = [message.name.index(name) for name in JOINT_NAMES]
            self.latest_joints = np.array(
                [message.position[index] for index in indices], dtype=float
            )
            if len(message.velocity) > max(indices):
                self.latest_joint_speed = float(
                    np.max(np.abs([message.velocity[index] for index in indices]))
                )
            else:
                self.latest_joint_speed = 0.0
            self.latest_joint_wall_ns = time.monotonic_ns()
        except (ValueError, IndexError):
            return

    def _profile_callback(self, message: PointCloud2) -> None:
        points = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        if len(points) == 0:
            self.latest_profile = None
            return
        if getattr(points.dtype, "names", None):
            profile = np.column_stack(
                tuple(np.asarray(points[name], dtype=float) for name in ("x", "y", "z"))
            )
        else:
            profile = np.asarray(points, dtype=float).reshape(-1, 3)
        # Keep legacy publishers usable while timestamp matching is preferred.
        self.latest_profile = profile
        self.latest_profile_stamp = message.header.stamp
        key = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        self.pending_profiles[key] = profile
        self._try_accept_matched_frame(key, message.header.stamp)
        self._trim_pending_frames()

    def _endpoint_callback(self, message: PointCloud2) -> None:
        values = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        if len(values) != 2:
            self.latest_endpoints = None
            return
        if getattr(values.dtype, "names", None):
            endpoints = np.column_stack(
                tuple(
                    np.asarray(values[name], dtype=float)
                    for name in ("x", "y", "z")
                )
            )
        else:
            endpoints = np.asarray(values, dtype=float).reshape(2, 3)
        key = (
            int(message.header.stamp.sec),
            int(message.header.stamp.nanosec),
        )
        self.pending_endpoint_frames[key] = (
            endpoints[0].copy(),
            endpoints[1].copy(),
        )
        self._try_accept_matched_frame(key, message.header.stamp)
        self._trim_pending_frames()

    def _flange_pose_callback(self, message: PoseStamped) -> None:
        position = message.pose.position
        orientation = message.pose.orientation
        transform = make_transform(
            quaternion_to_matrix(
                np.array(
                    [
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w,
                    ]
                )
            ),
            np.array([position.x, position.y, position.z]),
        )
        key = (
            int(message.header.stamp.sec),
            int(message.header.stamp.nanosec),
        )
        self.pending_flange_poses[key] = transform
        self._try_accept_matched_frame(key, message.header.stamp)
        self._trim_pending_frames()

    def _try_accept_matched_frame(self, key, stamp) -> None:
        profile = self.pending_profiles.get(key)
        endpoints = self.pending_endpoint_frames.get(key)
        if profile is None or endpoints is None:
            return
        if self.pose_source == "topic":
            transform = self.pending_flange_poses.get(key)
            if transform is None:
                return
        else:
            transform = self._lookup_flange_transform(stamp)
            if transform is None:
                return
        self.pending_profiles.pop(key, None)
        self.pending_endpoint_frames.pop(key, None)
        self.pending_flange_poses.pop(key, None)
        self._accept_matched_frame(profile, endpoints, transform, stamp)

    def _accept_matched_frame(
        self, profile, endpoints, transform, stamp
    ) -> None:
        self.latest_profile = np.asarray(profile, dtype=float)
        self.latest_endpoints = tuple(
            np.asarray(endpoint, dtype=float) for endpoint in endpoints
        )
        self.latest_measurement_transform = np.asarray(
            transform, dtype=float
        ).copy()
        now = self.get_clock().now().nanoseconds
        wall_now = time.monotonic_ns()
        self.latest_profile_ns = now
        self.latest_endpoints_ns = now
        self.latest_profile_wall_ns = wall_now
        self.latest_endpoints_wall_ns = wall_now
        self.latest_profile_stamp = stamp
        if self.state == "CAPTURING_SEED":
            self._collect_seed_frame(wall_now)

    def _trim_pending_frames(self) -> None:
        for pending in (
            self.pending_profiles,
            self.pending_endpoint_frames,
            self.pending_flange_poses,
        ):
            while len(pending) > 20:
                pending.pop(next(iter(pending)))

    def _lookup_flange_transform(self, stamp) -> np.ndarray | None:
        try:
            stamped = self.tf_buffer.lookup_transform(
                "base_link",
                "fanuc_flange",
                Time.from_msg(stamp),
            )
        except Exception:
            return None
        translation = stamped.transform.translation
        quaternion = stamped.transform.rotation
        return make_transform(
            quaternion_to_matrix(
                np.array(
                    [
                        quaternion.x,
                        quaternion.y,
                        quaternion.z,
                        quaternion.w,
                    ]
                )
            ),
            np.array([translation.x, translation.y, translation.z]),
        )

    def _start_callback(self, _request, response):
        if self.state in {"DONE", "FAILED"}:
            response.success = self.state == "DONE"
            response.message = f"collection already finished in state {self.state}"
            return response
        self.started = True
        response.success = True
        if self.collection_mode == "manual":
            response.message = (
                "manual seed collection armed; call ~/capture at each stable pose"
            )
        else:
            response.message = (
                "automatic seed collection armed; waiting for stable bilateral data"
            )
        return response

    def _capture_callback(self, _request, response):
        if self.collection_mode != "manual":
            response.success = False
            response.message = "capture is only available in manual collection mode"
            return response
        if not self.started:
            response.success = False
            response.message = "manual collection is not armed; call ~/start first"
            return response
        if self.state != "WAIT_MANUAL_INIT":
            response.success = self.state == "DONE"
            response.message = f"cannot capture while state={self.state}"
            return response
        feature = self._feature()
        transform = self._current_transform()
        if feature is None or transform is None:
            response.success = False
            response.message = (
                "no synchronized bilateral profile/joint measurement is available"
            )
            return response
        if not feature.safe:
            response.success = False
            response.message = (
                f"bilateral endpoints are outside the safe domain; "
                f"margin={1000.0 * feature.domain_margin:.2f} mm"
            )
            return response
        if self.latest_joint_speed > self.manual_max_joint_speed:
            response.success = False
            response.message = (
                f"robot is still moving; maximum joint speed is "
                f"{self.latest_joint_speed:.4f} rad/s"
            )
            return response
        label = (
            "manual_reference"
            if not self.records
            else f"manual_{len(self.records) + 1}"
        )
        if not self._begin_seed_capture(label, "MANUAL"):
            response.success = False
            response.message = "cannot start synchronized seed frame capture"
            return response
        response.success = True
        response.message = (
            f"capturing {self.seed_measurement_batch_size} synchronized frames "
            f"for {label}"
        )
        return response

    def _status_callback(self, _request, response):
        response.success = self.state != "FAILED"
        feature = self._feature()
        assessment = None
        if feature is None:
            observation = "MISSING"
            feature_details = (
                "x_mid_mm=nan; z_mid_mm=nan; safe_margin_mm=nan; "
                "profile_length_mm=nan; endpoint_depth_delta_mm=nan; "
                "absolute_endpoint_depth_delta_mm=nan; "
                "joint_margin_percent=nan; local_ik=0/4; "
                "initial_ready=false; initial_reasons=measurement_missing"
            )
        else:
            observation = "SAFE" if feature.safe else "UNSAFE"
            assessment = self._initial_pose_assessment(feature)
            if assessment is None:
                initial_ready = False
                initial_reasons = "joints_missing"
                joint_margin = float("nan")
                local_ik_directions = 0
            else:
                initial_ready = assessment.accepted
                initial_reasons = (
                    "none"
                    if assessment.accepted
                    else ",".join(assessment.reasons)
                )
                joint_margin = assessment.normalized_joint_margin
                local_ik_directions = assessment.local_ik_directions
            feature_details = (
                f"x_mid_mm={1000.0 * feature.x_mid:.3f}; "
                f"z_mid_mm={1000.0 * feature.z_mid:.3f}; "
                f"safe_margin_mm={1000.0 * feature.domain_margin:.3f}; "
                f"profile_length_mm={1000.0 * feature.profile_length:.3f}; "
                f"endpoint_depth_delta_mm="
                f"{1000.0 * (feature.endpoint_v[2] - feature.endpoint_u[2]):.3f}; "
                f"absolute_endpoint_depth_delta_mm="
                f"{1000.0 * abs(feature.endpoint_v[2] - feature.endpoint_u[2]):.3f}; "
                f"joint_margin_percent={100.0 * joint_margin:.2f}; "
                f"local_ik={local_ik_directions}/4; "
                f"initial_ready={str(initial_ready).lower()}; "
                f"initial_reasons={initial_reasons}"
            )
        preflight_required = self.preflight_was_required
        preflight_reason = self.preflight_decision_reason
        if preflight_required is None and assessment is not None:
            preflight_required, preflight_reason = (
                self._dynamic_preflight_decision(assessment)
            )
        preflight_required_text = (
            "unknown"
            if preflight_required is None
            else str(preflight_required).lower()
        )
        stable = (
            self.latest_joints is not None
            and self.latest_joint_speed <= self.manual_max_joint_speed
            and time.monotonic_ns() - self.latest_joint_wall_ns
            <= self.maximum_measurement_age_ns
        )
        if self.state == "CAPTURING_SEED" and self.seed_capture_label:
            target_name = self.seed_capture_label
        elif (
            self.collection_mode == "automatic"
            and not self.records
            and self.state == "WAIT_MANUAL_INIT"
        ):
            target_name = "reference"
        elif (
            self.collection_mode == "automatic"
            and self.collection_phase == "PREFLIGHT"
            and self.preflight_index < len(self.preflight_plan)
        ):
            target_name = self.preflight_plan[self.preflight_index].name
        elif self.collection_mode == "automatic" and self.target_index < len(self.plan):
            target_name = self.plan[self.target_index].name
        elif self.collection_mode == "manual":
            target_name = f"manual_{len(self.records) + 1}"
        else:
            target_name = "complete"
        if (
            self.collection_phase == "PREFLIGHT"
            and self.preflight_index < len(self.preflight_plan)
        ):
            displayed_rotation_target = (
                self.preflight_rotation
                * self.preflight_plan[self.preflight_index].angle_scale
            )
        elif self.target_index < len(self.plan):
            displayed_rotation_target = (
                self.rotation_target * self.plan[self.target_index].angle_scale
            )
        else:
            displayed_rotation_target = self.rotation_target
        response.message = (
            f"state={self.state}; mode={self.collection_mode}; "
            f"phase={self.collection_phase}; "
            f"preflight_mode={self.preflight_mode}; "
            f"preflight_required={preflight_required_text}; "
            f"preflight_reason={preflight_reason}; "
            f"seeds={len(self.records)}/{self.target_count}; "
            f"target={target_name}; target_index={self.target_index}/{len(self.plan)}; "
            f"preflight={sum(item['accepted'] for item in self.preflight_results)}/"
            f"{len(self.preflight_plan)}; "
            f"target_failures={self.failure_count}/{self.maximum_target_failures}; "
            f"motion_stage={self.after_settle or '-'}; "
            f"seed_batch={len(self.seed_capture_frames)}/"
            f"{self.seed_measurement_batch_size}; "
            f"rotation_deg={np.rad2deg(self.accumulated_angle):.2f}/"
            f"{np.rad2deg(displayed_rotation_target):.2f}; "
            f"observation={observation}; stable={str(stable).lower()}; "
            f"profile_points={0 if self.latest_profile is None else len(self.latest_profile)}; "
            f"{feature_details}; "
            f"failure_reason={self.failure_reason.replace(';', ',') or 'none'}"
        )
        return response

    def _feature(self):
        now_wall_ns = time.monotonic_ns()
        if (
            self.latest_endpoints is None
            or self.latest_profile is None
            or len(self.latest_profile) < self.minimum_profile_points
            or self.latest_profile_wall_ns < self.measurement_not_before_wall_ns
            or self.latest_endpoints_wall_ns < self.measurement_not_before_wall_ns
            or now_wall_ns - self.latest_profile_wall_ns
            > self.maximum_measurement_age_ns
            or now_wall_ns - self.latest_endpoints_wall_ns
            > self.maximum_measurement_age_ns
        ):
            return None
        if (
            abs(self.latest_profile_ns - self.latest_endpoints_ns)
            > self.maximum_measurement_skew_ns
        ):
            return None
        try:
            return evaluate_bilateral_feature(*self.latest_endpoints, self.roi)
        except ValueError:
            return None

    def _current_transform(self) -> np.ndarray | None:
        if (
            self.latest_joints is None
            or time.monotonic_ns() - self.latest_joint_wall_ns
            > self.maximum_measurement_age_ns
        ):
            return None
        return forward_kinematics_urdf(self.latest_joints)

    def _measurement_transform(self) -> np.ndarray | None:
        if self.latest_measurement_transform is not None:
            return self.latest_measurement_transform.copy()
        if self.latest_profile_stamp is None:
            return None
        return self._lookup_flange_transform(self.latest_profile_stamp)

    def _local_ik_coverage(self) -> int:
        transform = self._current_transform()
        if transform is None or self.latest_joints is None:
            return 0
        feasible = 0
        for axis in (0, 1):
            for sign in (-1, 1):
                axis_vector = np.zeros(3)
                axis_vector[axis] = sign * self.initial_ik_test_step
                target = transform.copy()
                target[:3, :3] = (
                    transform[:3, :3] @ so3_exp(axis_vector)
                )
                solutions = inverse_kinematics_numeric(
                    target, q_init=self.latest_joints
                )
                if (
                    len(solutions) > 0
                    and np.max(np.abs(solutions[0] - self.latest_joints))
                    <= self.initial_maximum_local_joint_step
                ):
                    feasible += 1
        return feasible

    def _initial_pose_assessment(self, feature):
        if self.latest_joints is None:
            return None
        now_wall_ns = time.monotonic_ns()
        if (
            self.initial_assessment_cache is not None
            and now_wall_ns - self.initial_assessment_cache_wall_ns
            < self.initial_assessment_period_ns
        ):
            return self.initial_assessment_cache
        self.initial_assessment_cache = assess_initial_pose(
            feature,
            self.latest_joints,
            np.deg2rad(JOINT_LIMITS_DEG),
            local_ik_directions=self._local_ik_coverage(),
            criteria=self.initial_criteria,
        )
        self.initial_assessment_cache_wall_ns = now_wall_ns
        return self.initial_assessment_cache

    def _dynamic_preflight_decision(self, assessment) -> tuple[bool, str]:
        if self.collection_mode == "manual":
            return False, "manual_collection"
        return dynamic_preflight_decision(
            self.preflight_mode,
            assessment,
            auto_skip_maximum_abs_x_mid_m=(
                self.preflight_auto_skip_maximum_abs_x_mid
            ),
            auto_skip_minimum_domain_margin_m=(
                self.preflight_auto_skip_minimum_margin
            ),
        )

    def _command_transform(self, transform: np.ndarray, after_settle: str) -> bool:
        if self.latest_joints is None:
            return False
        solutions = inverse_kinematics_numeric(transform, q_init=self.latest_joints)
        if len(solutions) == 0:
            self.get_logger().warning("IK rejected the requested flange pose")
            return False
        return self._command_joints(solutions[0], after_settle)

    def _reset_servo(self) -> None:
        """Reuse the measured local sensitivity near the common reference pose."""
        self.servo = TranslationServo(maximum_step=self.maximum_translation_step)
        self.servo_from_cache = (
            self.learned_servo_axis is not None
            and self.learned_servo_sensitivity is not None
        )
        if self.servo_from_cache:
            self.servo.axis = self.learned_servo_axis
            self.servo.sensitivity = self.learned_servo_sensitivity
        self.servo_reprobe_attempted = False

    def _command_joints(self, joints: np.ndarray, after_settle: str) -> bool:
        if not self._trajectory.wait_for_server(timeout_sec=0.5):
            self.get_logger().error("joint trajectory action server is unavailable")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in joints]
        duration_sec = int(self.motion_duration)
        point.time_from_start = Duration(
            sec=duration_sec,
            nanosec=int((self.motion_duration - duration_sec) * 1e9),
        )
        goal.trajectory.points = [point]
        self.after_settle = after_settle
        self.state = "MOVING"
        self.motion_deadline_wall_ns = (
            time.monotonic_ns() + self.motion_timeout_ns
        )
        self._trajectory.send_goal_async(goal).add_done_callback(self._goal_response)
        return True

    def _goal_response(self, future) -> None:
        if self.state != "MOVING":
            return
        try:
            handle = future.result()
        except Exception as error:
            self._fail(f"trajectory goal request failed: {error}")
            return
        if not handle.accepted:
            self._fail("trajectory goal was rejected")
            return
        handle.get_result_async().add_done_callback(self._motion_result)

    def _motion_result(self, future) -> None:
        if self.state != "MOVING":
            return
        try:
            wrapped_result = future.result()
        except Exception as error:
            self._fail(f"trajectory result failed: {error}")
            return
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or wrapped_result.result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self._fail(
                "trajectory execution failed "
                f"(status={wrapped_result.status}, "
                f"error={wrapped_result.result.error_code})"
            )
            return
        self.settle_until_ns = (
            self.get_clock().now().nanoseconds + int(self.settling_time * 1e9)
        )
        self.measurement_not_before_wall_ns = time.monotonic_ns()
        self.measurement_retry_deadline_wall_ns = (
            self.measurement_not_before_wall_ns
            + self.measurement_retry_timeout_ns
        )
        self.state = "SETTLING"

    def _tick(self) -> None:
        if self.state == "MOVING":
            if time.monotonic_ns() >= self.motion_deadline_wall_ns:
                self._fail(
                    "trajectory execution timed out; controller or Gazebo "
                    "communication is unavailable"
                )
            return
        if self.state == "SETTLING":
            if self.get_clock().now().nanoseconds >= self.settle_until_ns:
                next_state = self.after_settle
                measurement_stages = {
                    "MICRO_ROTATION",
                    "PROBE_OUT",
                    "PROBE_BACK",
                    "SERVO",
                    "CACHED_SERVO_RECOVERY",
                    "ROLLBACK",
                    "RETURN_REFERENCE",
                    "PARTIAL_CAPTURE_READY",
                }
                if (
                    next_state in measurement_stages
                    and self._feature() is None
                    and time.monotonic_ns()
                    < self.measurement_retry_deadline_wall_ns
                ):
                    return
                self.after_settle = ""
                getattr(self, f"_after_{next_state.lower()}")()
            return
        if self.state == "CAPTURING_SEED":
            if len(self.seed_capture_frames) >= self.seed_measurement_batch_size:
                if self._try_finish_seed_capture():
                    return
            if time.monotonic_ns() >= self.seed_capture_deadline_wall_ns:
                if not self._try_finish_seed_capture(force=True):
                    self._complete_seed_capture(
                        False,
                        "stationary seed batch timed out: "
                        f"raw={len(self.seed_capture_frames)}, "
                        f"required_inliers={self.minimum_batch_inliers}",
                    )
            return
        if not self.started or self.state != "WAIT_MANUAL_INIT":
            return
        if self.collection_mode == "manual":
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
        assessment = self._initial_pose_assessment(feature)
        if assessment is None or not assessment.accepted:
            return
        measurement_transform = self._measurement_transform()
        if measurement_transform is None:
            return
        self.reference_joints = self.latest_joints.copy()
        self.reference_transform = transform.copy()
        self.reference_measurement_transform = measurement_transform
        self.reference_profile = self.latest_profile.copy()
        self.reference_feature = feature
        self.last_valid_joints = self.latest_joints.copy()
        self._remember_last_valid(feature)
        self.preflight_index = 0
        self.preflight_results = []
        (
            self.preflight_was_required,
            self.preflight_decision_reason,
        ) = self._dynamic_preflight_decision(assessment)
        if self.preflight_was_required:
            self.collection_phase = "PREFLIGHT"
            self.get_logger().info(
                "static initial envelope accepted; starting measured local "
                "+/-X/+/-Y preflight; "
                f"mode={self.preflight_mode}; "
                f"reason={self.preflight_decision_reason}"
            )
            self._return_reference()
            return

        self.collection_phase = "REFERENCE"
        self.plan = adaptive_rotation_plan()
        self.get_logger().info(
            "static initial envelope accepted; standalone dynamic preflight "
            f"skipped; mode={self.preflight_mode}; "
            f"reason={self.preflight_decision_reason}; real seed motions "
            "retain bilateral validation and rollback"
        )
        if not self._begin_seed_capture("reference", "REFERENCE"):
            self._fail("reference multi-frame capture could not start")

    def _return_reference(self) -> None:
        if self.reference_joints is None:
            self._fail("reference joints are unavailable")
            return
        self.rotation_step = self.rotation_step_default
        self.accumulated_angle = 0.0
        self.stage_index = 0
        self.failure_count = 0
        if not self._command_joints(self.reference_joints, "RETURN_REFERENCE"):
            self._fail("cannot return to the reference pose: controller unavailable")

    def _after_return_reference(self) -> None:
        if self.collection_phase == "PREFLIGHT":
            if self.preflight_index >= len(self.preflight_plan):
                feasible = [
                    item for item in self.preflight_results if item["accepted"]
                ]
                if not local_preflight_is_acceptable(
                    self.preflight_results,
                    minimum_feasible_directions=self.preflight_minimum_directions,
                ):
                    details = ",".join(
                        f"{item['name']}={'pass' if item['accepted'] else 'fail'}"
                        for item in self.preflight_results
                    )
                    self._fail(
                        "initial pose dynamic preflight failed: need at least "
                        f"{self.preflight_minimum_directions}/4 safe directions "
                        f"spanning X and Y; {details}"
                    )
                    return
                if not self._begin_seed_capture("reference", "REFERENCE"):
                    self._fail("reference multi-frame capture could not start")
                    return
                self.plan = preflight_guided_rotation_plan(
                    self.preflight_results
                )
                self.get_logger().info(
                    f"dynamic preflight accepted "
                    f"({len(feasible)}/{len(self.preflight_plan)} directions); "
                    "reordered seed plan from measured-safe directions; "
                    "collecting the reference stationary frame batch"
                )
                return
            self.last_valid_joints = self.latest_joints.copy()
            feature = self._feature()
            if feature is not None and feature.safe:
                self._remember_last_valid(feature)
            self._reset_servo()
            target = self.preflight_plan[self.preflight_index]
            self.get_logger().info(f"preflight target {target.name}")
            self._issue_micro_rotation()
            return
        if len(self.records) >= self.target_count or self.target_index >= len(
            self.plan
        ):
            self._finish()
            return
        self.last_valid_joints = self.latest_joints.copy()
        feature = self._feature()
        if feature is not None and feature.safe:
            self._remember_last_valid(feature)
        self._reset_servo()
        self.get_logger().info(f"target {self.plan[self.target_index].name}")
        self._issue_micro_rotation()

    def _issue_micro_rotation(self) -> None:
        current = self._current_transform()
        if current is None:
            self._fail("joint state is unavailable")
            return
        target = (
            self.preflight_plan[self.preflight_index]
            if self.collection_phase == "PREFLIGHT"
            else self.plan[self.target_index]
        )
        axis, sign = target.stages[self.stage_index]
        target_angle = self._current_target_angle()
        remaining = target_angle - self.accumulated_angle
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
        self.accumulated_angle += self.pending_rotation
        self.last_valid_joints = self.latest_joints.copy()
        self._remember_last_valid(feature)
        target_angle = self._current_target_angle()
        if (
            self.accumulated_angle + 1e-10 < target_angle
            and abs(feature.x_mid)
            <= self.rotation_continue_max_abs_x_mid
            and feature.domain_margin
            >= self.rotation_continue_minimum_domain_margin
        ):
            self._issue_micro_rotation()
            return
        if abs(feature.x_mid) <= self._centering_tolerance():
            self._continue_after_centered(feature)
            return
        if self.servo.axis is not None and self.servo.sensitivity is not None:
            self.servo_iterations = 0
            self._issue_servo()
        else:
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
            self.probe_sensitivities[self.probe_axis] = 0.0
            self.get_logger().warning(
                f"probe axis {self.probe_axis} excluded: unsafe feature response"
            )
            if not self._command_transform(self.probe_base_transform, "PROBE_BACK"):
                self._rollback("unsafe probe return IK failure")
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
        self.learned_servo_axis = self.servo.axis
        self.learned_servo_sensitivity = self.servo.sensitivity
        self.servo_from_cache = False
        self._issue_servo()

    def _issue_servo(self) -> None:
        if self.servo_iterations >= self.maximum_servo_iterations:
            if self.servo_from_cache and not self.servo_reprobe_attempted:
                self.get_logger().warning(
                    "cached translation sensitivity did not converge; "
                    "re-probing locally"
                )
                self.servo_reprobe_attempted = True
                feature = self._feature()
                if feature is None:
                    self._rollback("feature missing before servo re-probe")
                    return
                self._begin_probing(feature)
                return
            self._rollback("translation servo iteration limit reached")
            return
        feature = self._feature()
        current = self._current_transform()
        if feature is None or current is None:
            self._rollback("servo input unavailable")
            return
        if abs(feature.x_mid) <= self._centering_tolerance():
            self._continue_after_centered(feature)
            return
        step = self.servo.correction(feature.x_mid)
        self.get_logger().info(
            "translation servo: "
            f"x_mid={1000.0 * feature.x_mid:.2f} mm, "
            f"axis={self.servo.axis}, "
            f"sensitivity={self.servo.sensitivity:.4f}, "
            f"step={1000.0 * step:.2f} mm, "
            f"iteration={self.servo_iterations + 1}/"
            f"{self.maximum_servo_iterations}"
        )
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
            if (
                self.servo_from_cache
                and not self.servo_reprobe_attempted
                and self.last_valid_joints is not None
            ):
                self.get_logger().warning(
                    "cached translation sensitivity left the safe region; "
                    "returning to the last valid pose and re-probing"
                )
                self.learned_servo_axis = None
                self.learned_servo_sensitivity = None
                self.servo_from_cache = False
                self.servo_reprobe_attempted = True
                if not self._command_joints(
                    self.last_valid_joints, "CACHED_SERVO_RECOVERY"
                ):
                    self._rollback("cached servo recovery command failed")
                return
            self._rollback("servo left the safe bilateral region")
            return
        self.servo.update(
            feature.x_mid - self.servo_previous_x, self.servo_previous_step
        )
        self.learned_servo_axis = self.servo.axis
        self.learned_servo_sensitivity = self.servo.sensitivity
        self.last_valid_joints = self.latest_joints.copy()
        self._remember_last_valid(feature)
        if abs(feature.x_mid) <= self._centering_tolerance():
            self._continue_after_centered(feature)
        else:
            self._issue_servo()

    def _after_cached_servo_recovery(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            self._rollback("cached servo recovery lost bilateral observation")
            return
        self._begin_probing(feature)

    def _centering_tolerance(self) -> float:
        """Preflight verifies the working envelope; seeds require centering."""
        if self.collection_phase == "PREFLIGHT":
            return self.preflight_maximum_abs_x_mid
        return self.x_tolerance

    def _current_target_angle(self) -> float:
        if self.collection_phase == "PREFLIGHT":
            target = self.preflight_plan[self.preflight_index]
            return self.preflight_rotation * target.angle_scale
        target = self.plan[self.target_index]
        return self.rotation_target * target.angle_scale

    def _continue_after_centered(self, feature) -> None:
        target_angle = self._current_target_angle()
        if self.accumulated_angle + 1e-10 < target_angle:
            self._issue_micro_rotation()
            return
        if self.collection_phase == "PREFLIGHT":
            target = self.preflight_plan[self.preflight_index]
            axis, sign = target.stages[0]
            accepted = seed_feature_is_acceptable(
                feature,
                maximum_abs_x_mid_m=self.preflight_maximum_abs_x_mid,
                minimum_domain_margin_m=self.preflight_minimum_margin,
            )
            self.preflight_results.append(
                {
                    "name": target.name,
                    "axis": axis,
                    "sign": sign,
                    "accepted": accepted,
                    "domain_margin_m": float(feature.domain_margin),
                }
            )
            self.get_logger().info(
                f"{target.name} preflight "
                f"{'accepted' if accepted else 'rejected'}: "
                f"margin={1000.0 * feature.domain_margin:.2f} mm"
            )
            self.preflight_index += 1
            self._return_reference()
            return
        target = self.plan[self.target_index]
        if self.stage_index + 1 < len(target.stages):
            self.stage_index += 1
            self.accumulated_angle = 0.0
            self._issue_micro_rotation()
            return
        if not self._begin_seed_capture(target.name, "TARGET"):
            self.get_logger().warning(
                f"{target.name} stationary frame capture could not start"
            )
            self.target_index += 1
            self._return_reference()

    def _rollback(self, reason: str) -> None:
        if self.collection_phase == "PREFLIGHT":
            target = self.preflight_plan[self.preflight_index]
            axis, sign = target.stages[0]
            self.preflight_results.append(
                {
                    "name": target.name,
                    "axis": axis,
                    "sign": sign,
                    "accepted": False,
                    "domain_margin_m": float("-inf"),
                    "reason": reason,
                }
            )
            self.get_logger().warning(
                f"{target.name} preflight rejected: {reason}"
            )
            self.preflight_index += 1
            self._return_reference()
            return
        self.failure_count += 1
        self.rotation_step = max(self.rotation_step / 2.0, self.rotation_step_minimum)
        self.get_logger().warning(
            f"{reason}; rollback, rotation step={np.rad2deg(self.rotation_step):.2f} deg"
        )
        if self.failure_count >= self.maximum_target_failures:
            partial_angle = self._last_valid_relative_rotation()
            if partial_angle >= self.minimum_partial_rotation:
                self.pending_partial_label = (
                    f"{self.plan[self.target_index].name}_partial"
                )
                if (
                    self.last_valid_joints is not None
                    and self._command_joints(
                        self.last_valid_joints, "PARTIAL_CAPTURE_READY"
                    )
                ):
                    self.get_logger().warning(
                        "target limit not reached; returning to the last "
                        "centered safe partial orientation for multi-frame capture "
                        f"at {np.rad2deg(partial_angle):.2f} deg"
                    )
                    return
                self.get_logger().warning(
                    "partial orientation could not be restored for capture"
                )
            else:
                self.get_logger().warning("target abandoned after repeated failures")
            self.failure_count = 0
            self.target_index += 1
            self._return_reference()
            return
        if self.last_valid_joints is None:
            self._fail("no valid rollback pose")
            return
        if not self._command_joints(self.last_valid_joints, "ROLLBACK"):
            self._fail(
                f"{reason}; cannot execute rollback because the controller "
                "is unavailable"
            )

    def _after_rollback(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            if (
                self.collection_phase != "PREFLIGHT"
                and self.reference_joints is not None
                and self._command_joints(
                    self.reference_joints, "ROLLBACK_REFERENCE_RECOVERY"
                )
            ):
                self.get_logger().warning(
                    "local rollback did not restore bilateral visibility; "
                    "trying the verified reference pose and skipping this "
                    "target"
                )
                return
            self._fail("rollback did not restore bilateral visibility")
            return
        self._issue_micro_rotation()

    def _after_rollback_reference_recovery(self) -> None:
        feature = self._feature()
        if feature is None or not feature.safe:
            self._fail(
                "neither local rollback nor reference recovery restored "
                "bilateral visibility"
            )
            return
        self.get_logger().warning(
            f"{self.plan[self.target_index].name} abandoned after reference "
            "recovery; continuing with the next preflight-guided branch"
        )
        self.failure_count = 0
        self.target_index += 1
        self._return_reference()

    def _after_partial_capture_ready(self) -> None:
        feature = self._feature()
        if (
            feature is None
            or not seed_feature_is_acceptable(
                feature,
                maximum_abs_x_mid_m=self.x_tolerance,
                minimum_domain_margin_m=self.minimum_seed_domain_margin,
            )
        ):
            self.get_logger().warning(
                "restored partial orientation is not acceptable for seed capture"
            )
            self.target_index += 1
            self._return_reference()
            return
        label = self.pending_partial_label
        self.pending_partial_label = ""
        if not self._begin_seed_capture(label, "PARTIAL"):
            self.get_logger().warning(
                "partial stationary frame capture could not start"
            )
            self.target_index += 1
            self._return_reference()

    def _begin_seed_capture(self, label: str, continuation: str) -> bool:
        if (
            not label
            or self.latest_joints is None
            or self._measurement_transform() is None
        ):
            return False
        self.seed_capture_frames = []
        self.seed_capture_label = label
        self.seed_capture_continuation = continuation
        now = time.monotonic_ns()
        self.seed_capture_started_wall_ns = now
        self.seed_capture_deadline_wall_ns = (
            now + self.measurement_batch_timeout_ns
        )
        self.seed_capture_last_diagnostics = None
        self.state = "CAPTURING_SEED"
        self.get_logger().info(
            f"stationary seed capture started: {label}, "
            f"frames={self.seed_measurement_batch_size}, "
            f"minimum_inliers={self.minimum_batch_inliers}"
        )
        return True

    def _collect_seed_frame(self, wall_now: int) -> None:
        if (
            wall_now <= self.seed_capture_started_wall_ns
            or len(self.seed_capture_frames) >= self.maximum_batch_frames
            or self.latest_profile is None
            or self.latest_endpoints is None
            or self.latest_measurement_transform is None
            or self.latest_joints is None
            or self.latest_joint_speed > self.manual_max_joint_speed
            or len(self.latest_profile) < self.minimum_profile_points
        ):
            return
        endpoint_u, endpoint_v = self.latest_endpoints
        try:
            feature = evaluate_bilateral_feature(
                endpoint_u, endpoint_v, self.roi
            )
        except ValueError:
            return
        if not feature.safe:
            return
        self.seed_capture_frames.append(
            {
                "R_BF": self.latest_measurement_transform[:3, :3].copy(),
                "t_BF": self.latest_measurement_transform[:3, 3].copy(),
                "joints": self.latest_joints.copy(),
                "profile_points_S": self.latest_profile.copy(),
                "endpoint_u_S": endpoint_u.copy(),
                "endpoint_v_S": endpoint_v.copy(),
            }
        )

    def _try_finish_seed_capture(self, *, force: bool = False) -> bool:
        if len(self.seed_capture_frames) < self.minimum_batch_inliers:
            return False
        endpoints_u = np.asarray(
            [frame["endpoint_u_S"] for frame in self.seed_capture_frames]
        )
        endpoints_v = np.asarray(
            [frame["endpoint_v_S"] for frame in self.seed_capture_frames]
        )
        inliers, diagnostics = robust_endpoint_inliers(
            endpoints_u,
            endpoints_v,
            mad_multiplier=self.endpoint_mad_multiplier,
        )
        self.seed_capture_last_diagnostics = diagnostics
        inlier_frames = [
            frame
            for frame, accepted in zip(self.seed_capture_frames, inliers)
            if accepted
        ]
        if len(inlier_frames) < self.minimum_batch_inliers:
            if (
                not force
                and len(self.seed_capture_frames) < self.maximum_batch_frames
            ):
                return False
            return False
        if self._save_seed_batch(
            self.seed_capture_label, inlier_frames, diagnostics
        ):
            self._complete_seed_capture(True, "")
        else:
            self._complete_seed_capture(
                False,
                "pose rejected by rotation diversity after stationary capture",
            )
        return True

    @staticmethod
    def _frame_payload(frame: dict) -> dict:
        return {
            "R_BF": frame["R_BF"].tolist(),
            "t_BF": frame["t_BF"].tolist(),
            "joints": frame["joints"].tolist(),
            "profile_points_S": frame["profile_points_S"].tolist(),
            "endpoint_u_S": frame["endpoint_u_S"].tolist(),
            "endpoint_v_S": frame["endpoint_v_S"].tolist(),
        }

    def _save_seed_batch(
        self, label: str, frames: list[dict], diagnostics
    ) -> bool:
        representative = frames[len(frames) // 2]
        transform_rotation = representative["R_BF"]
        candidate_rotations = self.seed_rotations + [transform_rotation]
        diversity = rotation_diversity(candidate_rotations)
        if (
            self.seed_rotations
            and diversity["minimum_pairwise_deg"]
            < self.minimum_rotation_separation_deg
        ):
            self.get_logger().warning(f"{label} rejected: insufficient rotation diversity")
            return False
        aggregate_feature = evaluate_bilateral_feature(
            diagnostics.median_u, diagnostics.median_v, self.roi
        )
        self.seed_rotations.append(transform_rotation.copy())
        self.records.append(
            {
                "label": label,
                # A representative observation keeps the file inspectable by
                # older tools; schema-v3 consumers use every frame below.
                **self._frame_payload(representative),
                "endpoint_u_S": diagnostics.median_u.tolist(),
                "endpoint_v_S": diagnostics.median_v.tolist(),
                "x_mid": float(aggregate_feature.x_mid),
                "domain_margin": float(aggregate_feature.domain_margin),
                "batch_diagnostics": diagnostics.as_dict(),
                "frames": [
                    self._frame_payload(frame) for frame in frames
                ],
            }
        )
        self.get_logger().info(
            f"accepted physical seed {len(self.records)}: {label}; "
            f"endpoint inliers={len(frames)}/{diagnostics.raw_count}"
        )
        return True

    def _complete_seed_capture(self, success: bool, reason: str) -> None:
        continuation = self.seed_capture_continuation
        label = self.seed_capture_label
        self.seed_capture_frames = []
        self.seed_capture_label = ""
        self.seed_capture_continuation = ""
        if not success:
            self.get_logger().warning(f"{label} capture rejected: {reason}")
        if continuation == "REFERENCE":
            if not success:
                self._fail(f"reference seed capture failed: {reason}")
                return
            self.collection_phase = "COLLECT"
            self.get_logger().info(
                "reference multi-frame seed accepted; starting adaptive "
                "star rotation plan"
            )
            self._return_reference()
            return
        if continuation in {"TARGET", "PARTIAL"}:
            self.failure_count = 0
            self.target_index += 1
            self._return_reference()
            return
        if continuation == "MANUAL":
            if success and len(self.records) >= self.target_count:
                self._finish()
            else:
                self.state = "WAIT_MANUAL_INIT"
            return
        self._fail(f"unknown seed capture continuation {continuation}")

    def _remember_last_valid(self, feature) -> None:
        transform = self._measurement_transform()
        if transform is None:
            transform = self._current_transform()
        if transform is None or self.latest_profile is None:
            return
        self.last_valid_joints = self.latest_joints.copy()
        self.last_valid_transform = transform.copy()
        self.last_valid_profile = self.latest_profile.copy()
        self.last_valid_feature = feature

    def _last_valid_relative_rotation(self) -> float:
        if self.last_valid_transform is None or self.reference_transform is None:
            return 0.0
        relative = self.reference_transform[:3, :3].T @ self.last_valid_transform[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        return float(np.arccos(cosine))

    def _finish(self) -> None:
        raw_diversity = rotation_diversity(self.seed_rotations)
        diversity = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in raw_diversity.items()
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "collection_mode": self.collection_mode,
                    "dynamic_preflight": {
                        "mode": self.preflight_mode,
                        "executed": bool(self.preflight_was_required),
                        "decision_reason": self.preflight_decision_reason,
                        "minimum_feasible_directions": (
                            self.preflight_minimum_directions
                        ),
                        "results": self.preflight_results,
                    },
                    "physical_seed_count": len(self.records),
                    "observation_count": sum(
                        len(record.get("frames", []))
                        for record in self.records
                    ),
                    "measurement_batch_size": self.seed_measurement_batch_size,
                    "rotation_diversity": diversity,
                    "seeds": self.records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.state = "DONE" if len(self.records) >= self.target_count else "FAILED"
        self.get_logger().info(
            f"seed collection complete: {len(self.records)} records -> {self.output_file}"
        )
        if self.state == "FAILED":
            self.failure_reason = (
                f"only {len(self.records)}/{self.target_count} observable "
                "seed targets collected"
            )
            self.get_logger().error(
                self.failure_reason
            )

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.state = "FAILED"
        self.get_logger().error(reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SeedCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Some rmw/rclpy combinations can race a subscription take with the
        # console's SIGINT after DONE. Do not turn a completed collection into
        # a misleading process failure, but preserve real runtime failures.
        if node.state != "DONE" and rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
