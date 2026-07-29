#!/usr/bin/env python3
"""Simulation-only v5 initialization, MoveIt planning, NBV execution and update."""

from __future__ import annotations

from pathlib import Path

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetMotionPlan, GetStateValidity
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs_py.point_cloud2 as point_cloud2
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

from calibration_pipeline.dataset_io import (
    aggregate_seed_group,
    load_seed_dataset_grouped,
    save_result,
)
from calibration_pipeline.geometry import (
    make_transform,
    quaternion_to_matrix,
    rotation_distance_deg,
    so3_exp,
)
from calibration_pipeline.models import (
    FlangePose,
    Measurement,
    SensorROI,
    TrapezoidDomain,
)
from calibration_pipeline.initial_validation import (
    bootstrap_initial_stability,
    inflate_handeye_covariance_from_stability,
)
from calibration_pipeline.nbv.stopping import StopPolicy
from calibration_pipeline.pipeline import ActiveCalibrationPipeline
from calibration_pipeline.simulation.scene_truth import (
    HAND_EYE_ROTATION,
    HAND_EYE_TRANSLATION,
)
from calibration_pipeline.solvers import TwelveDofV2Solver
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


def effective_measurement_timeout_s(
    configured_timeout_s: float,
    measurement_batch_size: int,
    minimum_frame_rate_hz: float,
) -> float:
    """Give a requested frame batch enough time under a loaded simulation."""
    if configured_timeout_s <= 0.0:
        raise ValueError("measurement_timeout_s must be positive")
    if measurement_batch_size < 1:
        raise ValueError("measurement_batch_size must be positive")
    if minimum_frame_rate_hz <= 0.0:
        raise ValueError("minimum_frame_rate_hz must be positive")
    # One extra second covers settling-to-callback and matched-topic latency.
    return max(
        float(configured_timeout_s),
        float(measurement_batch_size) / float(minimum_frame_rate_hz) + 1.0,
    )


class ActiveCalibrationSimNode(Node):
    """Run only against the repository's Gazebo/Gocator simulation topics."""

    def __init__(self) -> None:
        super().__init__("active_calibration_sim")
        self.declare_parameter("auto_start", True)
        self.declare_parameter("seed_file", "/workspace/data/seed_measurements_v5.json")
        self.declare_parameter("output_file", "/workspace/data/calibration_result_v5.json")
        self.declare_parameter("maximum_nbv_poses", 5)
        self.declare_parameter("maximum_scored_candidates", 32)
        self.declare_parameter("maximum_planning_candidates", 32)
        self.declare_parameter("nbv.edge_samples", 4)
        self.declare_parameter("nbv.edge_margin_m", 0.04)
        self.declare_parameter("nbv.alpha_deg", [55.0, 65.0, 75.0])
        self.declare_parameter("nbv.psi_deg", [-15.0, 0.0, 15.0])
        self.declare_parameter("nbv.working_distance_m", [0.33, 0.4, 0.5])
        self.declare_parameter("nbv.minimum_sensor_side_clearance_m", 0.15)
        self.declare_parameter("nbv.minimum_nominal_roi_margin_m", 0.01)
        self.declare_parameter("nbv.maximum_joint_step_deg", 70.0)
        self.declare_parameter("nbv.maximum_joint_distance_rad", 1.6)
        self.declare_parameter("nbv.motion_cost_weight", 1.0)
        self.declare_parameter("nbv.minimum_valid_probability", 0.8)
        self.declare_parameter("nbv.maximum_execution_retries_per_pose", 3)
        self.declare_parameter(
            "nbv.minimum_committed_joint_separation_rad", 0.10
        )
        self.declare_parameter("nbv.measurement_batch_size", 5)
        self.declare_parameter("nbv.minimum_poses", 3)
        self.declare_parameter("nbv.maximum_total_poses", 20)
        self.declare_parameter("nbv.information_gain_threshold", 1e-3)
        self.declare_parameter("nbv.consecutive_low_gain_limit", 3)
        self.declare_parameter("nbv.minimum_effective_eigenvalue", 1e-6)
        self.declare_parameter("nbv.maximum_rotation_std_deg", 0.025)
        self.declare_parameter("nbv.maximum_translation_std_m", 0.00005)
        self.declare_parameter("nbv.maximum_update_rotation_deg", 5.0)
        self.declare_parameter("nbv.maximum_update_translation_m", 0.05)
        self.declare_parameter("nbv.maximum_board_rotation_deg", 10.0)
        # Non-positive means derive the first-NBV correction bound from the
        # declared Phase-0a hand-eye uncertainty.
        self.declare_parameter("nbv.initial_maximum_update_rotation_deg", -1.0)
        self.declare_parameter("nbv.initial_maximum_update_translation_m", -1.0)
        self.declare_parameter("nbv.initial_maximum_board_rotation_deg", -1.0)
        self.declare_parameter("settling_time_s", 0.75)
        self.declare_parameter("measurement_timeout_s", 5.0)
        self.declare_parameter("nbv.minimum_measurement_frame_rate_hz", 2.0)
        self.declare_parameter("board.length_u_m", 0.4)
        self.declare_parameter("board.length_v_m", 0.5)
        self.declare_parameter("handeye_init_rotation_error_deg", 3.0)
        self.declare_parameter("handeye_init_translation_error_mm", 10.0)
        self.declare_parameter("solver.plane_weight", 1.0)
        self.declare_parameter("solver.edge_weight", 1.0)
        self.declare_parameter("solver.endpoint_plane_weight", 1.0)
        self.declare_parameter("solver.max_evaluations", 3000)
        self.declare_parameter("solver.tolerance", 1e-11)
        self.declare_parameter("solver.handeye_rotation_scale_deg", 10.0)
        self.declare_parameter("solver.handeye_translation_scale_m", 0.1)
        self.declare_parameter("solver.plane_rotation_scale_deg", 10.0)
        self.declare_parameter("solver.maximum_condition_number", 1e12)
        self.declare_parameter("initial_validation.bootstrap_trials", 4)
        self.declare_parameter("initial_validation.random_seed", 20260728)
        self.declare_parameter(
            "initial_validation.maximum_rotation_p95_deg", 1.0
        )
        self.declare_parameter(
            "initial_validation.maximum_translation_p95_m", 0.010
        )
        self.declare_parameter(
            "initial_validation.minimum_converged_fraction", 0.8
        )
        self.declare_parameter(
            "sensor.hard_trapezoid",
            [-0.292, 0.82, -0.021, -0.22, -0.019, 0.22],
        )
        self.declare_parameter(
            "sensor.safe_trapezoid", [0.27, 0.78, -0.115, -0.19, 0.095, 0.19]
        )

        self.seed_file = Path(str(self.get_parameter("seed_file").value))
        self.output_file = Path(str(self.get_parameter("output_file").value))
        self.maximum_nbv_poses = int(self.get_parameter("maximum_nbv_poses").value)
        self.maximum_scored = int(
            self.get_parameter("maximum_scored_candidates").value
        )
        self.maximum_planning = int(
            self.get_parameter("maximum_planning_candidates").value
        )
        self.candidate_options = {
            "edge_samples": int(self.get_parameter("nbv.edge_samples").value),
            "edge_margin": float(
                self.get_parameter("nbv.edge_margin_m").value
            ),
            "alphas_deg": tuple(
                map(float, self.get_parameter("nbv.alpha_deg").value)
            ),
            "psis_deg": tuple(
                map(float, self.get_parameter("nbv.psi_deg").value)
            ),
            "working_distances": tuple(
                map(float, self.get_parameter("nbv.working_distance_m").value)
            ),
        }
        self.minimum_sensor_side_clearance = float(
            self.get_parameter("nbv.minimum_sensor_side_clearance_m").value
        )
        self.minimum_nominal_roi_margin = float(
            self.get_parameter("nbv.minimum_nominal_roi_margin_m").value
        )
        self.maximum_joint_step = np.deg2rad(
            float(self.get_parameter("nbv.maximum_joint_step_deg").value)
        )
        self.maximum_joint_distance = float(
            self.get_parameter("nbv.maximum_joint_distance_rad").value
        )
        self.motion_cost_weight = float(
            self.get_parameter("nbv.motion_cost_weight").value
        )
        self.minimum_valid_probability = float(
            self.get_parameter("nbv.minimum_valid_probability").value
        )
        if not 0.0 <= self.minimum_valid_probability <= 1.0:
            raise ValueError("nbv.minimum_valid_probability must be in [0, 1]")
        self.maximum_execution_retries = int(
            self.get_parameter("nbv.maximum_execution_retries_per_pose").value
        )
        self.minimum_committed_joint_separation = float(
            self.get_parameter(
                "nbv.minimum_committed_joint_separation_rad"
            ).value
        )
        self.measurement_batch_size = int(
            self.get_parameter("nbv.measurement_batch_size").value
        )
        if self.measurement_batch_size < 1:
            raise ValueError("nbv.measurement_batch_size must be positive")
        self.stop_policy = StopPolicy(
            minimum_nbv_poses=int(
                self.get_parameter("nbv.minimum_poses").value
            ),
            maximum_total_poses=int(
                self.get_parameter("nbv.maximum_total_poses").value
            ),
            information_gain_threshold=float(
                self.get_parameter("nbv.information_gain_threshold").value
            ),
            consecutive_low_gain_limit=int(
                self.get_parameter("nbv.consecutive_low_gain_limit").value
            ),
            minimum_effective_eigenvalue=float(
                self.get_parameter("nbv.minimum_effective_eigenvalue").value
            ),
            maximum_rotation_std_deg=float(
                self.get_parameter("nbv.maximum_rotation_std_deg").value
            ),
            maximum_translation_std_m=float(
                self.get_parameter("nbv.maximum_translation_std_m").value
            ),
        )
        self.settling_time = float(self.get_parameter("settling_time_s").value)
        self.measurement_timeout_ns = int(
            effective_measurement_timeout_s(
                float(self.get_parameter("measurement_timeout_s").value),
                self.measurement_batch_size,
                float(
                    self.get_parameter(
                        "nbv.minimum_measurement_frame_rate_hz"
                    ).value
                ),
            )
            * 1e9
        )
        self.roi = SensorROI(
            hard_domain=TrapezoidDomain(
                *map(float, self.get_parameter("sensor.hard_trapezoid").value)
            ),
            safe_domain=TrapezoidDomain(
                *map(float, self.get_parameter("sensor.safe_trapezoid").value)
            ),
        )
        rotation_scale = np.deg2rad(
            float(self.get_parameter("solver.handeye_rotation_scale_deg").value)
        )
        translation_scale = float(
            self.get_parameter("solver.handeye_translation_scale_m").value
        )
        plane_scale = np.deg2rad(
            float(self.get_parameter("solver.plane_rotation_scale_deg").value)
        )
        self.solver = TwelveDofV2Solver(
            plane_weight=float(self.get_parameter("solver.plane_weight").value),
            edge_weight=float(self.get_parameter("solver.edge_weight").value),
            endpoint_plane_weight=float(
                self.get_parameter("solver.endpoint_plane_weight").value
            ),
            max_evaluations=int(self.get_parameter("solver.max_evaluations").value),
            tolerance=float(self.get_parameter("solver.tolerance").value),
            state_scale=np.array(
                [rotation_scale] * 3
                + [translation_scale] * 3
                + [plane_scale] * 3
            ),
            maximum_condition_number=float(
                self.get_parameter("solver.maximum_condition_number").value
            ),
        )

        self.create_subscription(JointState, "/joint_states", self._joint_callback, 10)
        self.create_subscription(PointCloud2, "/gocator/profile", self._profile_callback, 10)
        self.create_subscription(
            Float64MultiArray, "/gocator/endpoints", self._endpoint_callback, 10
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self.validity_client = self.create_client(
            GetStateValidity, "/check_state_validity"
        )
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.create_service(Trigger, "~/start", self._start_callback)
        self.create_service(Trigger, "~/status", self._status_callback)
        self.create_timer(0.1, self._tick)

        self.started = bool(self.get_parameter("auto_start").value)
        self.state = "WAIT_SEEDS"
        self.pipeline: ActiveCalibrationPipeline | None = None
        self.ranked = []
        self.planning_index = 0
        self.executing_score = None
        self.latest_joints: np.ndarray | None = None
        self.latest_profile: np.ndarray | None = None
        self.latest_endpoints: tuple[np.ndarray, np.ndarray] | None = None
        self.latest_measurement_transform: np.ndarray | None = None
        self.latest_profile_ns = 0
        self.latest_profile_stamp = None
        self.latest_endpoints_ns = 0
        self.measurement_not_before_ns = 0
        self.settle_until_ns = 0
        self.rollback_joints: np.ndarray | None = None
        self.pending_failure_reason = ""
        self.iteration_history: list[dict] = []
        self.last_result_summary = "not_available"
        self.current_candidate_summary = "not_selected"
        self.stop_reason = ""
        self.candidate_joint_cache: dict[str, np.ndarray] = {}
        self.pending_goal_joints: np.ndarray | None = None
        self.rejection_counts: dict[str, int] = {}
        self.execution_failures_since_commit = 0
        self.execution_failure_history: list[dict] = []
        self.initial_stability_report = None
        self.seed_observation_count = 0
        self.committed_nbv_joints: list[np.ndarray] = []
        self.measurement_batch: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.pending_profiles: dict[tuple[int, int], np.ndarray] = {}
        self.pending_endpoint_frames: dict[tuple[int, int], list[float]] = {}
        self.last_batched_profile_ns = 0
        self.last_batched_endpoints_ns = 0
        self.get_logger().info(
            f"simulation-only active calibrator waiting for {self.seed_file}"
        )

    def _start_callback(self, _request, response):
        self.started = True
        response.success = True
        response.message = f"active calibration armed in state {self.state}"
        return response

    def _status_callback(self, _request, response):
        response.success = self.state != "FAILED"
        count = 0 if self.pipeline is None else self.pipeline.nbv_count
        response.message = (
            f"state={self.state}; nbv={count}/{self.maximum_nbv_poses}; "
            f"planning_attempt={self.planning_index}/{self.maximum_planning}; "
            f"measurement_batch={len(self.measurement_batch)}/"
            f"{self.measurement_batch_size}; "
            f"candidate={self.current_candidate_summary}; "
            f"last_result={self.last_result_summary}; "
            f"stop_reason={self.stop_reason or '-'}"
        )
        return response

    def _joint_callback(self, message: JointState) -> None:
        try:
            self.latest_joints = np.asarray(
                [message.position[message.name.index(name)] for name in JOINT_NAMES],
                dtype=float,
            )
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
            values = np.column_stack(
                tuple(np.asarray(points[name], dtype=float) for name in ("x", "y", "z"))
            )
        else:
            values = np.asarray(points, dtype=float).reshape(-1, 3)
        indices = np.linspace(0, len(values) - 1, min(40, len(values)), dtype=int)
        profile = values[indices]
        # Keep legacy publishers usable while timestamp matching is preferred.
        self.latest_profile = profile
        self.latest_profile_stamp = message.header.stamp
        key = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        self.pending_profiles[key] = profile
        endpoint_data = self.pending_endpoint_frames.pop(key, None)
        if endpoint_data is not None:
            self.pending_profiles.pop(key, None)
            self._accept_matched_frame(profile, endpoint_data, message.header.stamp)
        self._trim_pending_frames()

    def _endpoint_callback(self, message: Float64MultiArray) -> None:
        data = list(message.data)
        if len(data) < 9 or not bool(data[4]) or not bool(data[8]):
            self.latest_endpoints = None
            return
        if len(data) >= 23:
            key = (int(data[21]), int(data[22]))
            profile = self.pending_profiles.pop(key, None)
            if profile is None:
                self.pending_endpoint_frames[key] = data
                self._trim_pending_frames()
                return
            self._accept_matched_frame(profile, data, None)
            return
        # Backward compatibility for legacy simulation publishers.
        self._accept_matched_frame(self.latest_profile, data, None)

    def _accept_matched_frame(self, profile, data, stamp) -> None:
        if profile is None:
            return
        self.latest_profile = np.asarray(profile, dtype=float)
        self.latest_endpoints = (
            np.asarray(data[1:4], dtype=float),
            np.asarray(data[5:8], dtype=float),
        )
        if len(data) >= 21:
            self.latest_measurement_transform = make_transform(
                np.asarray(data[9:18], dtype=float).reshape(3, 3),
                np.asarray(data[18:21], dtype=float),
            )
        now = self.get_clock().now().nanoseconds
        self.latest_profile_ns = now
        self.latest_endpoints_ns = now
        if stamp is not None:
            self.latest_profile_stamp = stamp

    def _trim_pending_frames(self) -> None:
        for pending in (self.pending_profiles, self.pending_endpoint_frames):
            while len(pending) > 20:
                pending.pop(next(iter(pending)))

    def _tick(self) -> None:
        if not self.started:
            return
        if self.state == "WAIT_SEEDS":
            if self.seed_file.exists() and self.latest_joints is not None:
                self._initialize_from_seeds()
            return
        if self.state == "WAIT_JOINTS" and self.latest_joints is not None:
            self._plan_next_candidate()
            return
        if self.state == "WAIT_MEASUREMENT":
            now = self.get_clock().now().nanoseconds
            if now < self.settle_until_ns:
                return
            if now > self.settle_until_ns + self.measurement_timeout_ns:
                self._rollback_after_failure("timed out waiting for a fresh bilateral frame")
                return
            if (
                self.latest_profile is None
                or self.latest_endpoints is None
                or min(self.latest_profile_ns, self.latest_endpoints_ns)
                <= self.measurement_not_before_ns
            ):
                return
            if (
                self.latest_profile_ns <= self.last_batched_profile_ns
                or self.latest_endpoints_ns <= self.last_batched_endpoints_ns
            ):
                return
            endpoint_u, endpoint_v = self.latest_endpoints
            transform = self._measurement_flange_transform()
            if transform is None:
                return
            self.measurement_batch.append(
                (
                    self.latest_profile.copy(),
                    endpoint_u.copy(),
                    endpoint_v.copy(),
                    transform,
                )
            )
            self.last_batched_profile_ns = self.latest_profile_ns
            self.last_batched_endpoints_ns = self.latest_endpoints_ns
            if len(self.measurement_batch) >= self.measurement_batch_size:
                self._commit_observation()
            return
        if self.state == "WAIT_ROLLBACK_MEASUREMENT":
            now = self.get_clock().now().nanoseconds
            if now < self.settle_until_ns:
                return
            if now > self.settle_until_ns + self.measurement_timeout_ns:
                self._fail(
                    f"{self.pending_failure_reason}; rollback completed but "
                    "bilateral recovery measurement timed out"
                )
                return
            if (
                self.latest_profile is None
                or self.latest_endpoints is None
                or min(self.latest_profile_ns, self.latest_endpoints_ns)
                <= self.measurement_not_before_ns
            ):
                return
            endpoint_u, endpoint_v = self.latest_endpoints
            margins = (
                self.roi.margin(endpoint_u),
                self.roi.margin(endpoint_v),
            )
            if min(margins) < 0.0:
                self._fail(
                    f"{self.pending_failure_reason}; rollback pose did not "
                    f"restore the safe bilateral region "
                    f"(margins={1000.0 * margins[0]:.2f}/"
                    f"{1000.0 * margins[1]:.2f} mm)"
                )
                return
            self.get_logger().info(
                "rollback verified with fresh bilateral data; "
                f"safe margins={1000.0 * margins[0]:.2f}/"
                f"{1000.0 * margins[1]:.2f} mm; selecting another candidate"
            )
            if (
                self.execution_failures_since_commit
                >= self.maximum_execution_retries
            ):
                self._no_more_feasible_candidates(
                    "maximum failed candidate executions reached at the "
                    f"current NBV step ({self.maximum_execution_retries}); "
                    "last rollback was safe"
                )
            else:
                self._rank_and_plan()

    def _initialize_from_seeds(self) -> None:
        try:
            dataset = load_seed_dataset_grouped(self.seed_file)
            count = dataset.physical_seed_count
            if count < 6:
                self.get_logger().warning(
                    f"seed file has only {count}/6 physical poses"
                )
                return
            self.get_logger().info(
                f"loaded {count} physical seeds with "
                f"{dataset.observation_count} synchronized observations"
            )
            self.seed_observation_count = dataset.observation_count
            angle = np.deg2rad(
                float(self.get_parameter("handeye_init_rotation_error_deg").value)
            )
            rotation_axis = np.array([1.0, -2.0, 1.5])
            rotation_axis /= np.linalg.norm(rotation_axis)
            translation_error = (
                float(self.get_parameter("handeye_init_translation_error_mm").value)
                / 1000.0
            )
            translation_axis = np.array([1.0, -0.6, 0.8])
            translation_axis /= np.linalg.norm(translation_axis)
            nominal_rotation = HAND_EYE_ROTATION @ so3_exp(rotation_axis * angle)
            nominal_translation = (
                HAND_EYE_TRANSLATION + translation_axis * translation_error
            )
            maximum_update_rotation_deg = float(
                self.get_parameter("nbv.maximum_update_rotation_deg").value
            )
            maximum_update_translation_m = float(
                self.get_parameter("nbv.maximum_update_translation_m").value
            )
            maximum_board_rotation_deg = float(
                self.get_parameter("nbv.maximum_board_rotation_deg").value
            )
            initial_rotation_limit = float(
                self.get_parameter(
                    "nbv.initial_maximum_update_rotation_deg"
                ).value
            )
            initial_translation_limit = float(
                self.get_parameter(
                    "nbv.initial_maximum_update_translation_m"
                ).value
            )
            initial_board_limit = float(
                self.get_parameter(
                    "nbv.initial_maximum_board_rotation_deg"
                ).value
            )
            board_dimensions = (
                float(self.get_parameter("board.length_u_m").value),
                float(self.get_parameter("board.length_v_m").value),
            )
            self.pipeline = ActiveCalibrationPipeline(
                nominal_rotation,
                nominal_translation,
                board_dimensions,
                roi=self.roi,
                solver=self.solver,
                stop_policy=self.stop_policy,
                minimum_seed_poses=count,
                maximum_update_rotation_deg=maximum_update_rotation_deg,
                maximum_update_translation_m=maximum_update_translation_m,
                maximum_board_rotation_deg=maximum_board_rotation_deg,
                initial_maximum_update_rotation_deg=(
                    max(maximum_update_rotation_deg, np.rad2deg(angle))
                    if initial_rotation_limit <= 0.0
                    else initial_rotation_limit
                ),
                initial_maximum_update_translation_m=(
                    max(maximum_update_translation_m, translation_error)
                    if initial_translation_limit <= 0.0
                    else initial_translation_limit
                ),
                initial_maximum_board_rotation_deg=(
                    maximum_board_rotation_deg
                    if initial_board_limit <= 0.0
                    else initial_board_limit
                ),
            )
            for group in dataset.groups:
                pose, measurement = aggregate_seed_group(group)
                self.pipeline.append_seed(pose, measurement)
            result = self.pipeline.initialize()
            if not result.converged:
                self._fail(
                    "seed initialization rejected: "
                    f"rank={result.diagnostics.rank}, "
                    f"condition={result.diagnostics.condition_number:.3e}"
                )
                return
            self.initial_stability_report = bootstrap_initial_stability(
                dataset.groups,
                self.solver,
                result,
                board_dimensions=board_dimensions,
                trials=int(
                    self.get_parameter(
                        "initial_validation.bootstrap_trials"
                    ).value
                ),
                random_seed=int(
                    self.get_parameter(
                        "initial_validation.random_seed"
                    ).value
                ),
                maximum_rotation_p95_deg=float(
                    self.get_parameter(
                        "initial_validation.maximum_rotation_p95_deg"
                    ).value
                ),
                maximum_translation_p95_m=float(
                    self.get_parameter(
                        "initial_validation.maximum_translation_p95_m"
                    ).value
                ),
                minimum_converged_fraction=float(
                    self.get_parameter(
                        "initial_validation.minimum_converged_fraction"
                    ).value
                ),
            )
            result = inflate_handeye_covariance_from_stability(
                result, self.initial_stability_report
            )
            self.pipeline.result = result
            self.get_logger().info(
                "12-DOF-V2 initialized: "
                f"rank={result.diagnostics.rank}, "
                f"condition={result.diagnostics.condition_number:.3e}, "
                f"rotation_error={rotation_distance_deg(result.estimate.handeye_rotation, HAND_EYE_ROTATION):.4f} deg, "
                f"translation_error={1000.0 * np.linalg.norm(result.estimate.handeye_translation - HAND_EYE_TRANSLATION):.4f} mm"
            )
            stability = self.initial_stability_report
            if stability.available:
                self.get_logger().info(
                    "seed bootstrap stability: "
                    f"converged={stability.trials_converged}/"
                    f"{stability.trials_requested}, "
                    f"rotation_p95={stability.rotation_p95_deg:.4f} deg, "
                    f"translation_p95="
                    f"{1000.0 * stability.translation_p95_m:.4f} mm"
                )
            self._record_iteration("initial", result)
            self.iteration_history[-1]["initial_stability"] = (
                stability.as_dict()
            )
            self._write_result("INITIALIZED")
            if not stability.accepted:
                self._fail(
                    "seed initialization is not bootstrap-stable: "
                    f"{stability.reason}; recollect seeds or increase "
                    "seed.measurement_batch_size"
                )
                return
            self._rank_and_plan()
        except Exception as error:
            self._fail(f"initialization failed: {error}")

    def _rank_and_plan(self) -> None:
        assert self.pipeline is not None
        if self.latest_joints is None:
            self.state = "WAIT_JOINTS"
            return
        self.state = "RANKING"
        self.candidate_joint_cache = {}
        self.rejection_counts = {
            "wrong_board_side": 0,
            "roi_margin": 0,
            "ik": 0,
            "joint_step": 0,
            "joint_distance": 0,
            "already_observed": 0,
        }
        self.ranked = self.pipeline.rank_candidates(
            maximum_candidates=self.maximum_scored,
            minimum_valid_probability=self.minimum_valid_probability,
            virtual_batch_size=self.measurement_batch_size,
            candidate_filter=self._candidate_kinematic_filter,
            candidate_options=self.candidate_options,
        )
        for score in self.ranked:
            joint_distance = float(
                np.linalg.norm(
                    self.candidate_joint_cache[score.candidate.candidate_id]
                    - self.latest_joints
                )
            )
            utility = score.information_gain / (
                1.0 + self.motion_cost_weight * joint_distance
            )
            score.metadata["joint_distance_rad"] = joint_distance
            score.metadata["motion_aware_utility"] = float(utility)
        self.ranked.sort(
            key=lambda score: (
                score.metadata["motion_aware_utility"],
                score.information_gain,
                score.minimum_eigenvalue,
            ),
            reverse=True,
        )
        self.get_logger().info(
            f"candidate cascade: scored={len(self.ranked)}, "
            f"prefilter_rejections={self.rejection_counts}"
        )
        if self.pipeline.nbv_count:
            stop, reason = self.pipeline.check_stop(self.ranked)
            if stop:
                self.stop_reason = reason
                self.get_logger().info(f"adaptive stop: {reason}")
                self._finish()
                return
        if not self.ranked:
            self._no_more_feasible_candidates(
                "no local robust information-bearing candidate after "
                f"kinematic filtering: {self.rejection_counts}"
            )
            return
        self.planning_index = 0
        self._plan_next_candidate()

    def _candidate_kinematic_filter(self, candidate) -> bool:
        assert self.pipeline is not None and self.pipeline.result is not None
        assert self.latest_joints is not None
        board = self.pipeline.result.estimate.board
        sensor_origin = candidate.sensor_transform_nominal[:3, 3]
        signed_clearance = float(
            board.normal @ (sensor_origin - board.corner)
        )
        if signed_clearance < self.minimum_sensor_side_clearance:
            self.rejection_counts["wrong_board_side"] += 1
            return False
        if candidate.nominal_margin < self.minimum_nominal_roi_margin:
            self.rejection_counts["roi_margin"] += 1
            return False
        solutions = inverse_kinematics_numeric(
            candidate.flange_transform_command, q_init=self.latest_joints
        )
        if len(solutions) == 0:
            self.rejection_counts["ik"] += 1
            return False
        goal_joints = solutions[0]
        delta = np.abs(goal_joints - self.latest_joints)
        if float(np.max(delta)) > self.maximum_joint_step:
            self.rejection_counts["joint_step"] += 1
            return False
        if float(np.linalg.norm(delta)) > self.maximum_joint_distance:
            self.rejection_counts["joint_distance"] += 1
            return False
        if any(
            float(np.linalg.norm(goal_joints - observed))
            < self.minimum_committed_joint_separation
            for observed in self.committed_nbv_joints
        ):
            self.rejection_counts["already_observed"] += 1
            return False
        self.candidate_joint_cache[candidate.candidate_id] = goal_joints
        return True

    def _plan_next_candidate(self) -> None:
        if self.latest_joints is None:
            self.state = "WAIT_JOINTS"
            return
        if self.planning_index >= min(len(self.ranked), self.maximum_planning):
            self._no_more_feasible_candidates(
                "MoveIt found no collision-free path among all locally "
                f"feasible scored candidates ({self.planning_index} checked)"
            )
            return
        score = self.ranked[self.planning_index]
        self.planning_index += 1
        goal_joints = self.candidate_joint_cache.get(
            score.candidate.candidate_id
        )
        if goal_joints is None:
            self.pipeline.reject_candidate(score.candidate.candidate_id)
            self._plan_next_candidate()
            return
        joint_distance = float(
            score.metadata.get(
                "joint_distance_rad",
                np.linalg.norm(goal_joints - self.latest_joints),
            )
        )
        self.current_candidate_summary = (
            f"{score.candidate.candidate_id},"
            f"gain={score.information_gain:.6g},"
            f"p_valid={score.valid_probability:.3f},"
            f"joint_distance={joint_distance:.3f}rad,"
            f"roi_margin={1000.0 * score.prediction.roi_margin:.1f}mm"
        )
        self.executing_score = score
        self.pending_goal_joints = goal_joints
        request = GetStateValidity.Request()
        request.robot_state = RobotState()
        request.robot_state.joint_state.name = list(JOINT_NAMES)
        request.robot_state.joint_state.position = goal_joints.tolist()
        request.group_name = "arm"
        self.state = "CHECKING_GOAL"
        if not self.validity_client.wait_for_service(timeout_sec=1.0):
            self._fail("MoveIt /check_state_validity service unavailable")
            return
        self.validity_client.call_async(request).add_done_callback(
            self._goal_validity_result
        )

    def _goal_validity_result(self, future) -> None:
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(
                f"goal-state collision check failed: {error}"
            )
            self.pipeline.reject_candidate(
                self.executing_score.candidate.candidate_id
            )
            self._plan_next_candidate()
            return
        if not response.valid:
            contacts = ", ".join(
                f"{contact.contact_body_1}<->{contact.contact_body_2}"
                for contact in response.contacts[:3]
            )
            self.get_logger().warning(
                f"{self.executing_score.candidate.candidate_id} rejected: "
                f"goal state is in collision"
                + (f" ({contacts})" if contacts else "")
            )
            self.pipeline.reject_candidate(
                self.executing_score.candidate.candidate_id
            )
            self._plan_next_candidate()
            return
        self._request_motion_plan(self.pending_goal_joints)

    def _request_motion_plan(self, goal_joints: np.ndarray) -> None:
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = "arm"
        motion.pipeline_id = "ompl"
        motion.planner_id = "RRTConnect"
        motion.num_planning_attempts = 3
        motion.allowed_planning_time = 5.0
        motion.max_velocity_scaling_factor = 0.2
        motion.max_acceleration_scaling_factor = 0.2
        motion.start_state = RobotState()
        motion.start_state.joint_state.name = list(JOINT_NAMES)
        motion.start_state.joint_state.position = self.latest_joints.tolist()
        constraints = Constraints()
        constraints.joint_constraints = [
            JointConstraint(
                joint_name=name,
                position=float(position),
                tolerance_above=1e-3,
                tolerance_below=1e-3,
                weight=1.0,
            )
            for name, position in zip(JOINT_NAMES, goal_joints)
        ]
        motion.goal_constraints = [constraints]
        self.state = "PLANNING"
        if not self.plan_client.wait_for_service(timeout_sec=1.0):
            self._fail("MoveIt /plan_kinematic_path service unavailable")
            return
        self.plan_client.call_async(request).add_done_callback(self._plan_result)

    def _no_more_feasible_candidates(self, reason: str) -> None:
        if self.pipeline is not None and self.pipeline.nbv_count > 0:
            self.stop_reason = reason
            self.get_logger().warning(f"active calibration stopped early: {reason}")
            self._finish()
        else:
            self._fail(reason)

    def _plan_result(self, future) -> None:
        try:
            response = future.result().motion_plan_response
        except Exception as error:
            self.get_logger().warning(f"MoveIt planning call failed: {error}")
            self._plan_next_candidate()
            return
        if (
            response.error_code.val != MoveItErrorCodes.SUCCESS
            or not response.trajectory.joint_trajectory.points
        ):
            candidate_id = self.executing_score.candidate.candidate_id
            self.get_logger().warning(
                f"{candidate_id} path rejected by MoveIt, "
                f"code={response.error_code.val}, "
                f"joint_distance="
                f"{self.executing_score.metadata.get('joint_distance_rad', float('nan')):.3f} rad"
            )
            self.pipeline.reject_candidate(candidate_id)
            self._plan_next_candidate()
            return
        if not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self._fail("trajectory controller action unavailable")
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = response.trajectory.joint_trajectory
        goal.trajectory.header.stamp.sec = 0
        goal.trajectory.header.stamp.nanosec = 0
        self.rollback_joints = self.latest_joints.copy()
        self.state = "EXECUTING"
        self.trajectory_client.send_goal_async(goal).add_done_callback(
            self._goal_response
        )

    def _goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._fail("MoveIt-planned trajectory was rejected by controller")
            return
        handle.get_result_async().add_done_callback(self._motion_result)

    def _motion_result(self, future) -> None:
        wrapped = future.result()
        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or wrapped.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self._fail(
                f"trajectory execution failed: status={wrapped.status}, "
                f"error={wrapped.result.error_code}"
            )
            return
        now = self.get_clock().now().nanoseconds
        self.measurement_not_before_ns = now
        self.settle_until_ns = now + int(self.settling_time * 1e9)
        self.measurement_batch = []
        self.last_batched_profile_ns = now
        self.last_batched_endpoints_ns = now
        self.state = "WAIT_MEASUREMENT"

    def _commit_observation(self) -> None:
        assert self.pipeline is not None
        assert self.latest_joints is not None
        if not self.measurement_batch:
            self._rollback_after_failure("measurement batch is empty")
            return
        profiles = [item[0] for item in self.measurement_batch]
        endpoint_u_frames = [item[1] for item in self.measurement_batch]
        endpoint_v_frames = [item[2] for item in self.measurement_batch]
        endpoint_u = np.mean(endpoint_u_frames, axis=0)
        endpoint_v = np.mean(endpoint_v_frames, axis=0)
        frame_hard_margins = [
            min(self.roi.hard_margin(item[1]), self.roi.hard_margin(item[2]))
            for item in self.measurement_batch
        ]
        if min(frame_hard_margins) < 0.0:
            self._rollback_after_failure(
                "at least one frame in the measurement batch left the hard "
                f"bilateral valid domain (minimum hard margin="
                f"{1000.0 * min(frame_hard_margins):.2f} mm)"
            )
            return
        safe_margins = (
            self.roi.margin(endpoint_u),
            self.roi.margin(endpoint_v),
        )
        hard_margins = (
            self.roi.hard_margin(endpoint_u),
            self.roi.hard_margin(endpoint_v),
        )
        if min(hard_margins) < 0.0:
            self._rollback_after_failure(
                "executed candidate left the hard bilateral valid domain "
                f"(hard margins={1000.0 * hard_margins[0]:.2f}/"
                f"{1000.0 * hard_margins[1]:.2f} mm; "
                f"safe margins={1000.0 * safe_margins[0]:.2f}/"
                f"{1000.0 * safe_margins[1]:.2f} mm)"
            )
            return
        if min(safe_margins) < 0.0:
            self.get_logger().warning(
                "real NBV endpoints crossed the planning-safe boundary but "
                "remain inside the hard valid domain; accepting the measured "
                f"frame (hard margins={1000.0 * hard_margins[0]:.2f}/"
                f"{1000.0 * hard_margins[1]:.2f} mm)"
            )
        poses = [
            FlangePose(item[3][:3, :3], item[3][:3, 3])
            for item in self.measurement_batch
        ]
        measurements = [
            Measurement(profile, endpoint_u_frame, endpoint_v_frame)
            for profile, endpoint_u_frame, endpoint_v_frame, _transform
            in self.measurement_batch
        ]
        try:
            result = self.pipeline.append_nbv_batch(
                poses,
                measurements,
                candidate_id=self.executing_score.candidate.candidate_id,
            )
        except RuntimeError as error:
            self._rollback_after_failure(str(error))
            return
        self.get_logger().info(
            f"NBV {self.pipeline.nbv_count} committed: "
            f"gain={self.executing_score.information_gain:.6g}, "
            f"rotation_error={rotation_distance_deg(result.estimate.handeye_rotation, HAND_EYE_ROTATION):.4f} deg, "
            f"translation_error={1000.0 * np.linalg.norm(result.estimate.handeye_translation - HAND_EYE_TRANSLATION):.4f} mm"
        )
        self.execution_failures_since_commit = 0
        self.committed_nbv_joints.append(self.latest_joints.copy())
        self._record_iteration("nbv", result, self.executing_score)
        self._write_result("RUNNING")
        if self.pipeline.nbv_count >= self.maximum_nbv_poses:
            self.stop_reason = "configured simulation NBV limit reached"
            self.get_logger().info("configured simulation NBV limit reached")
            self._finish()
        else:
            self._rank_and_plan()

    def _measurement_flange_transform(self) -> np.ndarray | None:
        if self.latest_measurement_transform is not None:
            return self.latest_measurement_transform.copy()
        if self.latest_profile_stamp is None:
            return None
        try:
            stamped = self.tf_buffer.lookup_transform(
                "base_link",
                "fanuc_flange",
                Time.from_msg(self.latest_profile_stamp),
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

    def _finish(self) -> None:
        assert self.pipeline is not None and self.pipeline.result is not None
        self._write_result("DONE")
        self.state = "DONE"
        self.get_logger().info(f"simulation calibration complete -> {self.output_file}")

    def _record_iteration(self, phase: str, result, score=None) -> None:
        rotation_error = rotation_distance_deg(
            result.estimate.handeye_rotation, HAND_EYE_ROTATION
        )
        translation_error = 1000.0 * float(
            np.linalg.norm(
                result.estimate.handeye_translation - HAND_EYE_TRANSLATION
            )
        )
        covariance = result.estimate.covariance_x9
        if covariance is None:
            maximum_rotation_std_deg = float("nan")
            maximum_translation_std_mm = float("nan")
        else:
            standard_deviations = np.sqrt(
                np.maximum(np.diag(covariance)[:6], 0.0)
            )
            maximum_rotation_std_deg = float(
                np.max(np.rad2deg(standard_deviations[:3]))
            )
            maximum_translation_std_mm = float(
                1000.0 * np.max(standard_deviations[3:6])
            )
        candidate = None
        if score is not None:
            item = score.candidate
            goal_joints = self.candidate_joint_cache.get(item.candidate_id)
            joint_distance = float(
                score.metadata.get("joint_distance_rad", float("nan"))
            )
            candidate = {
                "id": item.candidate_id,
                "a_m": float(item.a),
                "b_m": float(item.b),
                "alpha_deg": float(np.rad2deg(item.alpha)),
                "psi_deg": float(np.rad2deg(item.psi)),
                "working_distance_m": float(item.working_distance),
                "branch": int(item.branch),
                "valid_probability": float(score.valid_probability),
                "information_gain": float(score.information_gain),
                "minimum_eigenvalue": float(score.minimum_eigenvalue),
                "roi_margin_m": float(score.prediction.roi_margin),
                "edge_margin_m": float(score.prediction.edge_margin),
                "joint_distance_rad": joint_distance,
                "maximum_joint_step_deg": (
                    float("nan")
                    if goal_joints is None or self.rollback_joints is None
                    else float(
                        np.rad2deg(
                            np.max(np.abs(goal_joints - self.rollback_joints))
                        )
                    )
                ),
                "motion_aware_utility": float(
                    score.metadata.get("motion_aware_utility", float("nan"))
                ),
                "goal_joints_rad": (
                    None if goal_joints is None else goal_joints.tolist()
                ),
            }
        record = {
            "phase": phase,
            "nbv_index": 0 if self.pipeline is None else self.pipeline.nbv_count,
            "candidate": candidate,
            "cost": float(result.cost),
            "rank": int(result.diagnostics.rank),
            "condition_number": float(result.diagnostics.condition_number),
            "rotation_error_deg": float(rotation_error),
            "translation_error_mm": float(translation_error),
            "maximum_rotation_std_deg": maximum_rotation_std_deg,
            "maximum_translation_std_mm": maximum_translation_std_mm,
            "handeye_rotation": result.estimate.handeye_rotation.tolist(),
            "handeye_translation_m": result.estimate.handeye_translation.tolist(),
            "board_corner_m": result.estimate.board.corner.tolist(),
        }
        self.iteration_history.append(record)
        self.last_result_summary = (
            f"phase={phase},"
            f"rotation_error_deg={rotation_error:.4f},"
            f"translation_error_mm={translation_error:.4f},"
            f"rotation_std_deg={maximum_rotation_std_deg:.4f},"
            f"translation_std_mm={maximum_translation_std_mm:.4f},"
            f"rank={result.diagnostics.rank},"
            f"condition={result.diagnostics.condition_number:.3e}"
        )

    def _write_result(self, status: str) -> None:
        assert self.pipeline is not None and self.pipeline.result is not None
        result = self.pipeline.result
        save_result(
            self.output_file,
            result,
            extra={
                "mode": "gazebo_only",
                "status": status,
                "seed_count": self.pipeline.seed_count,
                "seed_observation_count": self.seed_observation_count,
                "initial_stability": (
                    None
                    if self.initial_stability_report is None
                    else self.initial_stability_report.as_dict()
                ),
                "nbv_count": self.pipeline.nbv_count,
                "rotation_error_deg": rotation_distance_deg(
                    result.estimate.handeye_rotation, HAND_EYE_ROTATION
                ),
                "translation_error_mm": 1000.0
                * float(
                    np.linalg.norm(
                        result.estimate.handeye_translation
                        - HAND_EYE_TRANSLATION
                    )
                ),
                "stop_reason": self.stop_reason,
                "iterations": self.iteration_history,
                "execution_failures": self.execution_failure_history,
            },
        )

    def _rollback_after_failure(self, reason: str) -> None:
        if self.rollback_joints is None:
            self._fail(reason)
            return
        self.pending_failure_reason = reason
        self.execution_failures_since_commit += 1
        candidate_id = (
            "unknown"
            if self.executing_score is None
            else self.executing_score.candidate.candidate_id
        )
        if self.pipeline is not None and self.executing_score is not None:
            self.pipeline.reject_candidate(candidate_id)
        self.execution_failure_history.append(
            {
                "nbv_index": (
                    0 if self.pipeline is None else self.pipeline.nbv_count + 1
                ),
                "candidate_id": candidate_id,
                "reason": reason,
                "retry": self.execution_failures_since_commit,
            }
        )
        self.get_logger().warning(
            f"{candidate_id} real observation rejected: {reason}; "
            f"rollback and reselect "
            f"({self.execution_failures_since_commit}/"
            f"{self.maximum_execution_retries})"
        )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = self.rollback_joints.tolist()
        point.time_from_start = Duration(sec=2)
        goal.trajectory.points = [point]
        self.state = "ROLLBACK"
        self.trajectory_client.send_goal_async(goal).add_done_callback(
            self._rollback_goal_response
        )

    def _rollback_goal_response(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._fail(f"{self.pending_failure_reason}; rollback goal rejected")
            return
        handle.get_result_async().add_done_callback(self._rollback_motion_result)

    def _rollback_motion_result(self, future) -> None:
        wrapped = future.result()
        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or wrapped.result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self._fail(
                f"{self.pending_failure_reason}; rollback failed "
                f"(status={wrapped.status}, error={wrapped.result.error_code})"
            )
            return
        now = self.get_clock().now().nanoseconds
        self.measurement_not_before_ns = now
        self.settle_until_ns = now + int(self.settling_time * 1e9)
        self.state = "WAIT_ROLLBACK_MEASUREMENT"

    def _fail(self, reason: str) -> None:
        self.state = "FAILED"
        self.stop_reason = reason
        if self.pipeline is not None and self.pipeline.result is not None:
            self._write_result("FAILED")
        self.get_logger().error(reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ActiveCalibrationSimNode()
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
