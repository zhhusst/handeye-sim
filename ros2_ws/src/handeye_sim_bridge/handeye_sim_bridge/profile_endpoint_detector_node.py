#!/usr/bin/env python3
"""Extract persistent bilateral endpoints from a metric Gocator profile."""

from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger

from calibration_pipeline.perception import (
    DualEndpointKalmanConfig,
    DualEndpointKalmanTracker,
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)
from calibration_pipeline.seed_collection.endpoint_tracker import EndpointTracker
from calibration_pipeline.roi_tracking import (
    ROIBreakpointPipeline,
    ROIBreakpointPipelineConfig,
)


ENDPOINT_FIELDS = [
    PointField(
        name="x", offset=0, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="y", offset=4, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="z", offset=8, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="confidence", offset=12, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="sigma", offset=16, datatype=PointField.FLOAT32, count=1
    ),
]


def _profile_array(message: PointCloud2) -> np.ndarray:
    values = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    if len(values) == 0:
        return np.zeros((0, 3), dtype=float)
    if getattr(values.dtype, "names", None):
        return np.column_stack(
            tuple(
                np.asarray(values[name], dtype=float)
                for name in ("x", "y", "z")
            )
        )
    return np.asarray(values, dtype=float).reshape(-1, 3)


class ProfileEndpointDetectorNode(Node):
    """ROS adapter kept independent of simulator truth and robot state."""

    def __init__(self) -> None:
        super().__init__("profile_endpoint_detector")
        self.declare_parameter("endpoint_detection.input_topic", "/gocator/profile")
        self.declare_parameter(
            "endpoint_detection.output_topic", "/calibration/endpoints"
        )
        self.declare_parameter(
            "endpoint_detection.output_surface_topic",
            "/calibration/target_surface_points",
        )
        self.declare_parameter(
            "endpoint_detection.guide_topic",
            "/calibration/detection_guide",
        )
        self.declare_parameter(
            "endpoint_detection.prior_topic",
            "/calibration/detection_prior",
        )
        self.declare_parameter(
            "endpoint_detection.measured_prior_topic",
            "/calibration/detection_measured_prior",
        )
        self.declare_parameter(
            "endpoint_detection.control_topic",
            "/calibration/detection_control",
        )
        self.declare_parameter("endpoint_detection.backend", "classic")
        self.declare_parameter("endpoint_detection.roi_size_m", 0.020)
        self.declare_parameter("endpoint_detection.fail_streak_frames", 3)
        self.declare_parameter("endpoint_detection.tracker_name", "csrt")
        self.declare_parameter("endpoint_detection.raster_res_mm", 0.25)
        self.declare_parameter(
            "endpoint_detection.roi_process_every_n_frames", 1
        )
        self.declare_parameter("endpoint_detection.raster_point_radius_px", 2)
        self.declare_parameter("endpoint_detection.raster_margin_m", 0.020)
        self.declare_parameter(
            "endpoint_detection.raster_maximum_dimension_px", 2400
        )
        self.declare_parameter("endpoint_detection.core_fraction", 0.70)
        self.declare_parameter("endpoint_detection.roi_jump_m", 0.030)
        self.declare_parameter("endpoint_detection.bp_jump_m", 0.030)
        self.declare_parameter(
            "endpoint_detection.roi_tracking_minimum_chord_ratio", 0.20
        )
        self.declare_parameter(
            "endpoint_detection.roi_tracking_maximum_chord_ratio", 6.00
        )
        self.declare_parameter(
            "endpoint_detection.plate_growth_residual_threshold_m", 0.0008
        )
        self.declare_parameter(
            "endpoint_detection.roi_surface_residual_threshold_m", 0.0012
        )
        self.declare_parameter("endpoint_detection.minimum_points", 12)
        self.declare_parameter("endpoint_detection.minimum_segment_points", 10)
        self.declare_parameter(
            "endpoint_detection.minimum_segment_length_m", 0.01
        )
        self.declare_parameter(
            "endpoint_detection.maximum_segment_length_m", 0.35
        )
        self.declare_parameter(
            "endpoint_detection.absolute_neighbor_gap_m", 0.004
        )
        self.declare_parameter(
            "endpoint_detection.neighbor_gap_multiplier", 8.0
        )
        self.declare_parameter(
            "endpoint_detection.residual_mad_multiplier", 3.5
        )
        self.declare_parameter(
            "endpoint_detection.residual_floor_m", 0.00008
        )
        self.declare_parameter(
            "endpoint_detection.maximum_residual_rms_m", 0.0015
        )
        self.declare_parameter(
            "endpoint_detection.endpoint_extension_fraction", 0.5
        )
        self.declare_parameter(
            "endpoint_detection.endpoint_local_fit_points", 24
        )
        self.declare_parameter(
            "endpoint_detection.candidate_ambiguity_ratio", 0.03
        )
        self.declare_parameter(
            "endpoint_detection.identity_ambiguity_ratio", 0.05
        )
        self.declare_parameter(
            "endpoint_detection.initial_first_label", "e1"
        )
        self.declare_parameter(
            "endpoint_detection.minimum_confidence", 0.25
        )
        self.declare_parameter("endpoint_detection.smoothing_window", 5)
        self.declare_parameter("endpoint_detection.local_fit_window", 12)
        self.declare_parameter(
            "endpoint_detection.angle_change_threshold_deg", 10.0
        )
        self.declare_parameter(
            "endpoint_detection.height_jump_threshold_m", 0.0002
        )
        self.declare_parameter(
            "endpoint_detection.breakpoint_cluster_points", 8
        )
        self.declare_parameter(
            "endpoint_detection.maximum_abs_surface_midpoint_x_m", 0.0
        )
        self.declare_parameter("endpoint_detection.guided_enabled", True)
        self.declare_parameter(
            "endpoint_detection.alignment_template_center_x_m", 0.0
        )
        self.declare_parameter(
            "endpoint_detection.alignment_template_center_z_m", 0.42
        )
        self.declare_parameter(
            "endpoint_detection.alignment_template_length_m", 0.10
        )
        self.declare_parameter(
            "endpoint_detection.alignment_template_angle_deg", 25.0
        )
        self.declare_parameter(
            "endpoint_detection.alignment_normal_gate_m", 0.003
        )
        self.declare_parameter(
            "endpoint_detection.alignment_endpoint_gate_m", 0.020
        )
        self.declare_parameter(
            "endpoint_detection.alignment_maximum_angle_difference_deg",
            15.0,
        )
        self.declare_parameter(
            "endpoint_detection.alignment_stability_m", 0.0015
        )
        self.declare_parameter(
            "endpoint_detection.minimum_lock_frames", 5
        )
        self.declare_parameter(
            "endpoint_detection.tracking_normal_gate_m", 0.006
        )
        self.declare_parameter(
            "endpoint_detection.tracking_endpoint_gate_m", 0.025
        )
        self.declare_parameter(
            "endpoint_detection.tracking_maximum_angle_difference_deg",
            25.0,
        )
        self.declare_parameter(
            "endpoint_detection.predicted_normal_gate_m", 0.012
        )
        self.declare_parameter(
            "endpoint_detection.predicted_endpoint_gate_m", 0.050
        )
        self.declare_parameter(
            "endpoint_detection.predicted_maximum_angle_difference_deg",
            35.0,
        )
        self.declare_parameter(
            "endpoint_detection.maximum_lost_frames", 5
        )
        self.declare_parameter(
            "endpoint_detection.reacquire_stable_frames", 3
        )
        self.declare_parameter(
            "endpoint_detection.reacquire_maximum_segment_length_change_m",
            0.020,
        )
        self.declare_parameter(
            "endpoint_detection.reacquire_maximum_segment_angle_change_deg",
            20.0,
        )
        self.declare_parameter(
            "endpoint_detection.reacquire_stability_m", 0.003
        )
        self.declare_parameter(
            "endpoint_detection.temporal_tracking_enabled", True
        )
        self.declare_parameter(
            "endpoint_detection.temporal_initial_position_std_m", 0.0005
        )
        self.declare_parameter(
            "endpoint_detection.temporal_initial_velocity_std_m_s", 0.05
        )
        self.declare_parameter(
            "endpoint_detection.temporal_process_acceleration_std_m_s2", 1.0
        )
        self.declare_parameter(
            "endpoint_detection.temporal_measurement_std_floor_m", 0.00008
        )
        self.declare_parameter(
            "endpoint_detection.temporal_partial_measurement_std_m", 0.0005
        )
        self.declare_parameter(
            "endpoint_detection.temporal_mahalanobis_threshold", 13.82
        )
        self.declare_parameter(
            "endpoint_detection.temporal_maximum_endpoint_speed_m_s", 0.25
        )
        self.declare_parameter(
            "endpoint_detection.temporal_maximum_coast_frames", 5
        )
        self.declare_parameter(
            "endpoint_detection.temporal_minimum_search_radius_m", 0.0015
        )
        self.declare_parameter(
            "endpoint_detection.temporal_search_sigma_multiplier", 3.0
        )
        self.declare_parameter(
            "endpoint_detection.temporal_maximum_local_candidates", 6
        )
        self.declare_parameter(
            "endpoint_detection.tracking_maximum_endpoint_step_m", 0.003
        )
        self.declare_parameter(
            "endpoint_detection.tracking_minimum_reference_length_ratio", 0.60
        )
        self.declare_parameter(
            "endpoint_detection.tracking_maximum_reference_length_ratio", 1.80
        )

        prefix = "endpoint_detection."
        config = EndpointDetectionConfig(
            minimum_points=int(
                self.get_parameter(prefix + "minimum_points").value
            ),
            minimum_segment_points=int(
                self.get_parameter(prefix + "minimum_segment_points").value
            ),
            minimum_segment_length_m=float(
                self.get_parameter(prefix + "minimum_segment_length_m").value
            ),
            maximum_segment_length_m=float(
                self.get_parameter(prefix + "maximum_segment_length_m").value
            ),
            absolute_neighbor_gap_m=float(
                self.get_parameter(prefix + "absolute_neighbor_gap_m").value
            ),
            neighbor_gap_multiplier=float(
                self.get_parameter(prefix + "neighbor_gap_multiplier").value
            ),
            residual_mad_multiplier=float(
                self.get_parameter(prefix + "residual_mad_multiplier").value
            ),
            residual_floor_m=float(
                self.get_parameter(prefix + "residual_floor_m").value
            ),
            maximum_residual_rms_m=float(
                self.get_parameter(prefix + "maximum_residual_rms_m").value
            ),
            endpoint_extension_fraction=float(
                self.get_parameter(
                    prefix + "endpoint_extension_fraction"
                ).value
            ),
            endpoint_local_fit_points=int(
                self.get_parameter(
                    prefix + "endpoint_local_fit_points"
                ).value
            ),
            candidate_ambiguity_ratio=float(
                self.get_parameter(
                    prefix + "candidate_ambiguity_ratio"
                ).value
            ),
            smoothing_window=int(
                self.get_parameter(prefix + "smoothing_window").value
            ),
            local_fit_window=int(
                self.get_parameter(prefix + "local_fit_window").value
            ),
            angle_change_threshold_deg=float(
                self.get_parameter(
                    prefix + "angle_change_threshold_deg"
                ).value
            ),
            height_jump_threshold_m=float(
                self.get_parameter(
                    prefix + "height_jump_threshold_m"
                ).value
            ),
            breakpoint_cluster_points=int(
                self.get_parameter(
                    prefix + "breakpoint_cluster_points"
                ).value
            ),
            maximum_abs_surface_midpoint_x_m=float(
                self.get_parameter(
                    prefix + "maximum_abs_surface_midpoint_x_m"
                ).value
            ),
        )
        self.detector = ProfileEndpointDetector(config)
        self.identity_ambiguity_ratio = float(
            self.get_parameter(
                prefix + "identity_ambiguity_ratio"
            ).value
        )
        self.tracker = EndpointTracker(
            ambiguity_ratio=self.identity_ambiguity_ratio
        )
        self.minimum_confidence = float(
            self.get_parameter(prefix + "minimum_confidence").value
        )
        self.initial_first_label = str(
            self.get_parameter(prefix + "initial_first_label").value
        ).strip().lower()
        if self.initial_first_label not in {"e1", "e2"}:
            raise ValueError(
                "endpoint_detection.initial_first_label must be e1 or e2"
            )
        self.guided_enabled = bool(
            self.get_parameter(prefix + "guided_enabled").value
        )
        self.alignment_center = np.array(
            [
                float(
                    self.get_parameter(
                        prefix + "alignment_template_center_x_m"
                    ).value
                ),
                0.0,
                float(
                    self.get_parameter(
                        prefix + "alignment_template_center_z_m"
                    ).value
                ),
            ]
        )
        self.alignment_length = float(
            self.get_parameter(
                prefix + "alignment_template_length_m"
            ).value
        )
        self.alignment_angle = np.deg2rad(
            float(
                self.get_parameter(
                    prefix + "alignment_template_angle_deg"
                ).value
            )
        )
        self.alignment_normal_gate = float(
            self.get_parameter(prefix + "alignment_normal_gate_m").value
        )
        self.alignment_endpoint_gate = float(
            self.get_parameter(prefix + "alignment_endpoint_gate_m").value
        )
        self.alignment_angle_gate = float(
            self.get_parameter(
                prefix + "alignment_maximum_angle_difference_deg"
            ).value
        )
        self.alignment_stability = float(
            self.get_parameter(prefix + "alignment_stability_m").value
        )
        self.minimum_lock_frames = int(
            self.get_parameter(prefix + "minimum_lock_frames").value
        )
        self.tracking_normal_gate = float(
            self.get_parameter(prefix + "tracking_normal_gate_m").value
        )
        self.tracking_endpoint_gate = float(
            self.get_parameter(prefix + "tracking_endpoint_gate_m").value
        )
        self.tracking_angle_gate = float(
            self.get_parameter(
                prefix + "tracking_maximum_angle_difference_deg"
            ).value
        )
        self.predicted_normal_gate = float(
            self.get_parameter(prefix + "predicted_normal_gate_m").value
        )
        self.predicted_endpoint_gate = float(
            self.get_parameter(prefix + "predicted_endpoint_gate_m").value
        )
        self.predicted_angle_gate = float(
            self.get_parameter(
                prefix + "predicted_maximum_angle_difference_deg"
            ).value
        )
        self.maximum_lost_frames = int(
            self.get_parameter(prefix + "maximum_lost_frames").value
        )
        self.reacquire_stable_frames = int(
            self.get_parameter(prefix + "reacquire_stable_frames").value
        )
        self.reacquire_maximum_length_change = float(
            self.get_parameter(
                prefix + "reacquire_maximum_segment_length_change_m"
            ).value
        )
        self.reacquire_maximum_angle_change = float(
            self.get_parameter(
                prefix + "reacquire_maximum_segment_angle_change_deg"
            ).value
        )
        self.reacquire_stability = float(
            self.get_parameter(prefix + "reacquire_stability_m").value
        )
        self.temporal_tracking_enabled = bool(
            self.get_parameter(prefix + "temporal_tracking_enabled").value
        )
        self.temporal_minimum_search_radius = float(
            self.get_parameter(
                prefix + "temporal_minimum_search_radius_m"
            ).value
        )
        self.temporal_search_sigma_multiplier = float(
            self.get_parameter(
                prefix + "temporal_search_sigma_multiplier"
            ).value
        )
        self.temporal_partial_measurement_std = float(
            self.get_parameter(
                prefix + "temporal_partial_measurement_std_m"
            ).value
        )
        self.temporal_maximum_local_candidates = int(
            self.get_parameter(
                prefix + "temporal_maximum_local_candidates"
            ).value
        )
        self.tracking_maximum_endpoint_step = float(
            self.get_parameter(
                prefix + "tracking_maximum_endpoint_step_m"
            ).value
        )
        self.tracking_minimum_reference_length_ratio = float(
            self.get_parameter(
                prefix + "tracking_minimum_reference_length_ratio"
            ).value
        )
        self.tracking_maximum_reference_length_ratio = float(
            self.get_parameter(
                prefix + "tracking_maximum_reference_length_ratio"
            ).value
        )
        self.temporal_kalman_config = DualEndpointKalmanConfig(
            initial_position_std_m=float(
                self.get_parameter(
                    prefix + "temporal_initial_position_std_m"
                ).value
            ),
            initial_velocity_std_m_s=float(
                self.get_parameter(
                    prefix + "temporal_initial_velocity_std_m_s"
                ).value
            ),
            process_acceleration_std_m_s2=float(
                self.get_parameter(
                    prefix + "temporal_process_acceleration_std_m_s2"
                ).value
            ),
            measurement_std_floor_m=float(
                self.get_parameter(
                    prefix + "temporal_measurement_std_floor_m"
                ).value
            ),
            mahalanobis_threshold=float(
                self.get_parameter(
                    prefix + "temporal_mahalanobis_threshold"
                ).value
            ),
            maximum_endpoint_speed_m_s=float(
                self.get_parameter(
                    prefix + "temporal_maximum_endpoint_speed_m_s"
                ).value
            ),
            assignment_ambiguity_ratio=self.identity_ambiguity_ratio,
        )
        self.temporal_maximum_coast_frames = int(
            self.get_parameter(
                prefix + "temporal_maximum_coast_frames"
            ).value
        )
        positive_values = {
            "alignment_template_length_m": self.alignment_length,
            "alignment_normal_gate_m": self.alignment_normal_gate,
            "alignment_endpoint_gate_m": self.alignment_endpoint_gate,
            "alignment_stability_m": self.alignment_stability,
            "tracking_normal_gate_m": self.tracking_normal_gate,
            "tracking_endpoint_gate_m": self.tracking_endpoint_gate,
            "predicted_normal_gate_m": self.predicted_normal_gate,
            "predicted_endpoint_gate_m": self.predicted_endpoint_gate,
            "reacquire_maximum_segment_length_change_m": (
                self.reacquire_maximum_length_change
            ),
            "reacquire_maximum_segment_angle_change_deg": (
                self.reacquire_maximum_angle_change
            ),
            "reacquire_stability_m": self.reacquire_stability,
            "temporal_minimum_search_radius_m": (
                self.temporal_minimum_search_radius
            ),
            "temporal_search_sigma_multiplier": (
                self.temporal_search_sigma_multiplier
            ),
            "temporal_partial_measurement_std_m": (
                self.temporal_partial_measurement_std
            ),
            "temporal_maximum_endpoint_speed_m_s": (
                self.temporal_kalman_config.maximum_endpoint_speed_m_s
            ),
            "tracking_maximum_endpoint_step_m": (
                self.tracking_maximum_endpoint_step
            ),
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"endpoint_detection.{name} must be positive")
        if (
            self.minimum_lock_frames < 1
            or self.maximum_lost_frames < 1
            or self.reacquire_stable_frames < 1
            or self.temporal_maximum_coast_frames < 1
            or self.temporal_maximum_local_candidates < 1
        ):
            raise ValueError("guided detector frame counts must be positive")
        if self.temporal_minimum_search_radius > self.tracking_endpoint_gate:
            raise ValueError(
                "temporal minimum search radius must not exceed the "
                "tracking endpoint gate"
            )
        if not (
            0.0 < self.tracking_minimum_reference_length_ratio <= 1.0
            <= self.tracking_maximum_reference_length_ratio
        ):
            raise ValueError(
                "tracking reference length ratios must bracket one"
            )
        self.identity_initialized = False
        input_topic = str(
            self.get_parameter(prefix + "input_topic").value
        )
        output_topic = str(
            self.get_parameter(prefix + "output_topic").value
        )
        output_surface_topic = str(
            self.get_parameter(prefix + "output_surface_topic").value
        )
        guide_topic = str(
            self.get_parameter(prefix + "guide_topic").value
        )
        prior_topic = str(
            self.get_parameter(prefix + "prior_topic").value
        )
        measured_prior_topic = str(
            self.get_parameter(prefix + "measured_prior_topic").value
        )
        control_topic = str(
            self.get_parameter(prefix + "control_topic").value
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self.surface_publisher = self.create_publisher(
            PointCloud2, output_surface_topic, 10
        )
        self.guide_publisher = self.create_publisher(
            PointCloud2, guide_topic, 10
        )
        self.diagnostic_publisher = self.create_publisher(
            String, "~/diagnostics", 10
        )
        self.create_subscription(PointCloud2, input_topic, self._callback, 10)
        self.create_subscription(
            PointCloud2, prior_topic, self._prior_callback, 10
        )
        self.create_subscription(
            PointCloud2,
            measured_prior_topic,
            self._measured_prior_callback,
            10,
        )
        self.create_subscription(
            String, control_topic, self._control_callback, 10
        )
        self.create_service(Trigger, "~/status", self._status_callback)
        self.create_service(Trigger, "~/lock", self._lock_callback)
        self.create_service(Trigger, "~/reset", self._reset_callback)

        self.frames = 0
        self.accepted = 0
        self.mode = "ALIGN" if self.guided_enabled else "GLOBAL"
        self.template_endpoints = self._make_alignment_template()
        self.guide_endpoints = self.template_endpoints.copy()
        self.tracking_expected: np.ndarray | None = None
        self.predicted_expected: np.ndarray | None = None
        self.prediction_fallback: np.ndarray | None = None
        self.last_matched: np.ndarray | None = None
        self.previous_alignment_match: np.ndarray | None = None
        self.alignment_stable_frames = 0
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.lost_from_mode = "TRACK"
        self.last_profile_header = None
        self.temporal_tracker = DualEndpointKalmanTracker(
            self.temporal_kalman_config
        )
        self.temporal_tracking_requested = False
        self.temporal_prediction_this_frame = False
        self.temporal_search_radius = self.temporal_minimum_search_radius
        self.temporal_search_radii = np.full(
            2, self.temporal_minimum_search_radius, dtype=float
        )
        self.temporal_last_mahalanobis: tuple[float, float] | None = None
        self.temporal_suspended = False
        self.temporal_fallback_reason = ""
        self.tracking_reference_length: float | None = None
        self.last_profile_time_s: float | None = None
        self.last_status: dict[str, object] = {
            "state": "WAIT_PROFILE",
            "reason": "not_processed",
            "mode": self.mode,
        }
        self.endpoint_backend = str(
            self.get_parameter(prefix + "backend").value
        ).strip().lower()
        if self.endpoint_backend not in {"classic", "roi"}:
            raise ValueError(
                "endpoint_detection.backend must be 'classic' or 'roi'"
            )
        self.roi_pipeline: ROIBreakpointPipeline | None = None
        self.roi_input_frames = 0
        self.roi_process_every_n_frames = max(
            1,
            int(
                self.get_parameter(
                    prefix + "roi_process_every_n_frames"
                ).value
            ),
        )
        if self.endpoint_backend == "roi":
            self.roi_pipeline = ROIBreakpointPipeline(
                ROIBreakpointPipelineConfig(
                    initial_first_label=self.initial_first_label,
                    alignment_template_center_x_m=float(self.alignment_center[0]),
                    alignment_template_center_z_m=float(self.alignment_center[2]),
                    alignment_template_length_m=self.alignment_length,
                    alignment_template_angle_deg=float(
                        np.rad2deg(self.alignment_angle)
                    ),
                    alignment_normal_gate_m=self.alignment_normal_gate,
                    alignment_endpoint_gate_m=self.alignment_endpoint_gate,
                    alignment_maximum_angle_difference_deg=self.alignment_angle_gate,
                    alignment_stability_m=self.alignment_stability,
                    minimum_lock_frames=self.minimum_lock_frames,
                    minimum_segment_length_m=self.detector.config.minimum_segment_length_m,
                    maximum_segment_length_m=self.detector.config.maximum_segment_length_m,
                    tracker_name=str(
                        self.get_parameter(prefix + "tracker_name").value
                    ),
                    roi_size_m=float(
                        self.get_parameter(prefix + "roi_size_m").value
                    ),
                    raster_resolution_m_per_pixel=0.001
                    * float(self.get_parameter(prefix + "raster_res_mm").value),
                    raster_point_radius_px=int(
                        self.get_parameter(prefix + "raster_point_radius_px").value
                    ),
                    raster_margin_m=float(
                        self.get_parameter(prefix + "raster_margin_m").value
                    ),
                    raster_maximum_dimension_px=int(
                        self.get_parameter(
                            prefix + "raster_maximum_dimension_px"
                        ).value
                    ),
                    core_fraction=float(
                        self.get_parameter(prefix + "core_fraction").value
                    ),
                    roi_jump_m=float(
                        self.get_parameter(prefix + "roi_jump_m").value
                    ),
                    breakpoint_jump_m=float(
                        self.get_parameter(prefix + "bp_jump_m").value
                    ),
                    fail_streak_frames=int(
                        self.get_parameter(prefix + "fail_streak_frames").value
                    ),
                    reacquire_stable_frames=self.reacquire_stable_frames,
                    reacquire_endpoint_gate_m=self.tracking_endpoint_gate,
                    reacquire_stability_m=self.reacquire_stability,
                    tracking_minimum_chord_ratio=float(
                        self.get_parameter(
                            prefix + "roi_tracking_minimum_chord_ratio"
                        ).value
                    ),
                    tracking_maximum_chord_ratio=float(
                        self.get_parameter(
                            prefix + "roi_tracking_maximum_chord_ratio"
                        ).value
                    ),
                    plate_residual_threshold_m=float(
                        self.get_parameter(
                            prefix + "plate_growth_residual_threshold_m"
                        ).value
                    ),
                    surface_residual_threshold_m=float(
                        self.get_parameter(
                            prefix + "roi_surface_residual_threshold_m"
                        ).value
                    ),
                    endpoint_sigma_floor_m=self.detector.config.residual_floor_m,
                )
            )
            self._sync_roi_pipeline_state()
        self.get_logger().info(
            f"raw-profile endpoint detector ready: {input_topic} -> "
            f"{output_topic} + {output_surface_topic}; "
            f"guided_mode={self.guided_enabled}; backend={self.endpoint_backend}; "
            "no simulator truth subscription"
        )

    def _make_alignment_template(self) -> np.ndarray:
        direction = np.array(
            [np.cos(self.alignment_angle), 0.0, np.sin(self.alignment_angle)]
        )
        half = 0.5 * self.alignment_length * direction
        return np.vstack(
            (self.alignment_center - half, self.alignment_center + half)
        )

    def _guide_parameters(self) -> tuple[float, float, float]:
        mode = self.lost_from_mode if self.mode == "LOST" else self.mode
        if (
            self.mode == "LOST"
            and self.lost_from_mode == "TRACK"
            and self.temporal_suspended
        ):
            # Stage-two recovery remains local and model-constrained, but must
            # cover more than the narrow one-frame tracking corridor.
            return (
                self.predicted_normal_gate,
                self.predicted_endpoint_gate,
                min(
                    self.predicted_angle_gate,
                    self.reacquire_maximum_angle_change,
                ),
            )
        if mode == "ALIGN":
            return (
                self.alignment_normal_gate,
                self.alignment_endpoint_gate,
                self.alignment_angle_gate,
            )
        if mode == "PREDICTED_TRACK":
            return (
                self.predicted_normal_gate,
                self.predicted_endpoint_gate,
                self.predicted_angle_gate,
            )
        return (
            self.tracking_normal_gate,
            self.tracking_endpoint_gate,
            self.tracking_angle_gate,
        )

    def _status_context(self) -> dict[str, object]:
        normal_gate, endpoint_gate, angle_gate = self._guide_parameters()
        return {
            "mode": self.mode,
            "locked": self.mode not in {"ALIGN", "GLOBAL"},
            "alignment_stable_frames": self.alignment_stable_frames,
            "minimum_lock_frames": self.minimum_lock_frames,
            "lost_frames": self.lost_frames,
            "reacquire_frames": self.reacquire_frames,
            "guide_first_mm": (
                1000.0 * self.guide_endpoints[0]
            ).tolist(),
            "guide_second_mm": (
                1000.0 * self.guide_endpoints[1]
            ).tolist(),
            "guide_normal_gate_mm": 1000.0 * normal_gate,
            "guide_endpoint_gate_mm": 1000.0 * endpoint_gate,
            "guide_angle_gate_deg": angle_gate,
            "temporal_tracking": bool(
                self.temporal_tracking_requested
                and self.temporal_tracking_enabled
            ),
            "temporal_initialized": self.temporal_tracker.initialized,
            "temporal_missed_frames": self.temporal_tracker.missed_frames,
            "temporal_missed_frames_by_endpoint": (
                self.temporal_tracker.missed_frames_by_endpoint.tolist()
            ),
            "temporal_suspended": self.temporal_suspended,
            "temporal_fallback_reason": self.temporal_fallback_reason,
            "temporal_search_radius_mm": (
                1000.0 * self.temporal_search_radius
            ),
            "temporal_search_radii_mm": (
                1000.0 * self.temporal_search_radii
            ).tolist(),
            "temporal_mahalanobis": (
                None
                if self.temporal_last_mahalanobis is None
                else list(self.temporal_last_mahalanobis)
            ),
            "tracking_reference_length_mm": (
                None
                if self.tracking_reference_length is None
                else 1000.0 * self.tracking_reference_length
            ),
        }

    def _publish_diagnostics(self) -> None:
        diagnostics = String()
        diagnostics.data = json.dumps(
            self.last_status, ensure_ascii=False, sort_keys=True
        )
        self.diagnostic_publisher.publish(diagnostics)

    def _sync_roi_pipeline_state(self) -> None:
        """Expose the selected backend through the node's legacy status fields."""
        if self.roi_pipeline is None:
            return
        pipeline = self.roi_pipeline
        self.frames = pipeline.frames
        self.accepted = pipeline.accepted
        self.mode = pipeline.mode
        self.guide_endpoints = pipeline.guide_endpoints.copy()
        self.last_matched = (
            None if pipeline.last_matched is None else pipeline.last_matched.copy()
        )
        self.alignment_stable_frames = pipeline.alignment_stable_frames
        self.lost_frames = pipeline.lost_frames
        self.reacquire_frames = pipeline.reacquire_frames
        self.tracking_reference_length = pipeline.tracking_reference_length

    def _roi_profile_callback(self, message: PointCloud2) -> None:
        assert self.roi_pipeline is not None
        self.roi_input_frames += 1
        if (
            (self.roi_input_frames - 1) % self.roi_process_every_n_frames
            != 0
        ):
            return
        self.last_profile_header = message.header
        self.last_profile_time_s = self._message_time_seconds(message)
        try:
            profile = _profile_array(message)
            result = self.roi_pipeline.process_profile(
                profile, self.last_profile_time_s
            )
        except Exception as error:
            # A tracker/backend exception must degrade to an empty observation;
            # it must never terminate the detector process while the robot moves.
            self.get_logger().error(f"ROI endpoint backend failed: {error}")
            self._sync_roi_pipeline_state()
            self.last_status = {
                "state": "LOST",
                "reason": f"roi_backend_exception:{error}",
                "backend": "roi",
                "frames": self.frames,
                "accepted": self.accepted,
            }
            self._empty_output(message)
            self._publish_guide(message.header)
            self._publish_diagnostics()
            return
        self._sync_roi_pipeline_state()
        self._publish_guide(message.header)
        self.last_status = result.to_dict()
        self.last_status.update(self.roi_pipeline.status_extras())
        self.last_status["input_frames"] = self.roi_input_frames
        self.last_status["process_every_n_frames"] = (
            self.roi_process_every_n_frames
        )
        self.last_status["initial_first_label"] = self.initial_first_label
        if (
            result.state != "VALID"
            or result.endpoints is None
            or result.surface_points is None
        ):
            self._empty_output(message)
            self._publish_diagnostics()
            return
        confidence = 0.0 if result.confidence is None else float(result.confidence)
        sigma_m = (
            self.roi_pipeline.config.endpoint_sigma_floor_m
            if result.endpoint_sigma_mm is None
            else 0.001 * float(result.endpoint_sigma_mm)
        )
        rows = [
            (
                float(endpoint[0]),
                0.0,
                float(endpoint[2]),
                confidence,
                sigma_m,
            )
            for endpoint in result.endpoints
        ]
        self.publisher.publish(
            point_cloud2.create_cloud(message.header, ENDPOINT_FIELDS, rows)
        )
        self.surface_publisher.publish(
            point_cloud2.create_cloud_xyz32(
                message.header,
                result.surface_points.astype(np.float32).tolist(),
            )
        )
        self._publish_diagnostics()

    def _publish_guide(self, header) -> None:
        self.guide_publisher.publish(
            point_cloud2.create_cloud_xyz32(
                header,
                self.guide_endpoints.astype(np.float32).tolist(),
            )
        )

    def _message_time_seconds(self, message: PointCloud2) -> float:
        stamp = message.header.stamp
        value = float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)
        if value <= 0.0:
            value = 1.0e-9 * float(self.get_clock().now().nanoseconds)
        return value

    def _uses_temporal_tracking(self) -> bool:
        return bool(
            self.temporal_tracking_enabled
            and self.temporal_tracking_requested
            and not self.temporal_suspended
            and (
                self.mode == "TRACK"
                or (
                    self.mode == "LOST"
                    and self.lost_from_mode == "TRACK"
                )
            )
        )

    def _prepare_temporal_prediction(self, timestamp_s: float) -> None:
        self.temporal_prediction_this_frame = False
        self.temporal_last_mahalanobis = None
        if not self._uses_temporal_tracking():
            return
        if not self.temporal_tracker.initialized:
            initial = (
                self.tracking_expected
                if self.tracking_expected is not None
                else self.guide_endpoints
            )
            self.temporal_tracker.reset(initial, timestamp_s=timestamp_s)
        predicted = self.temporal_tracker.predict(timestamp_s)
        trusted = self.tracking_expected
        segment_length = float(np.linalg.norm(predicted[1] - predicted[0]))
        finite = bool(np.all(np.isfinite(predicted)))
        physical_length = bool(
            self.detector.config.minimum_segment_length_m
            <= segment_length
            <= self.detector.config.maximum_segment_length_m
        )
        bounded_from_measurement = bool(
            trusted is not None
            and np.max(np.linalg.norm(predicted - trusted, axis=1))
            <= self.tracking_endpoint_gate
        )
        if not (finite and physical_length and bounded_from_measurement):
            self._suspend_temporal_tracking("nonphysical_prediction")
            return
        self.guide_endpoints = predicted.copy()
        self.temporal_search_radii = (
            self.temporal_tracker.endpoint_search_radii(
                minimum_m=self.temporal_minimum_search_radius,
                maximum_m=self.tracking_endpoint_gate,
                sigma_multiplier=self.temporal_search_sigma_multiplier,
            )
        )
        self.temporal_search_radius = float(
            np.max(self.temporal_search_radii)
        )
        self.temporal_prediction_this_frame = True

    def _suspend_temporal_tracking(self, reason: str) -> None:
        """Fall back to the last measured guide after unsafe prediction."""
        self.temporal_suspended = True
        self.temporal_prediction_this_frame = False
        self.temporal_last_mahalanobis = None
        self.temporal_fallback_reason = str(reason)
        trusted = (
            self.tracking_expected
            if self.tracking_expected is not None
            else self.last_matched
        )
        if trusted is not None and np.all(np.isfinite(trusted)):
            self.guide_endpoints = trusted.copy()
            self.temporal_tracker.reset(
                trusted, timestamp_s=self.last_profile_time_s
            )
        self.temporal_search_radius = self.tracking_endpoint_gate
        self.temporal_search_radii = np.full(
            2, self.tracking_endpoint_gate, dtype=float
        )

    def _restore_temporal_from_measurement(
        self, endpoints: np.ndarray
    ) -> None:
        """Reset every temporal-loss flag from a trusted measured endpoint pair."""
        trusted = np.asarray(endpoints, dtype=float).reshape(2, 3)
        if self.temporal_tracking_enabled:
            self.temporal_tracker.reset(
                trusted, timestamp_s=self.last_profile_time_s
            )
        self.guide_endpoints = trusted.copy()
        self.temporal_prediction_this_frame = False
        self.temporal_last_mahalanobis = None
        self.temporal_search_radius = self.temporal_minimum_search_radius
        self.temporal_search_radii = np.full(
            2, self.temporal_minimum_search_radius, dtype=float
        )
        self.temporal_suspended = False
        self.temporal_fallback_reason = ""

    def _set_tracking_reference(self, endpoints: np.ndarray) -> None:
        values = np.asarray(endpoints, dtype=float).reshape(2, 3)
        length = float(np.linalg.norm(values[1] - values[0]))
        if np.isfinite(length) and length > 1.0e-9:
            self.tracking_reference_length = length

    def _tracking_pair_is_plausible(
        self, measured: np.ndarray, predicted: np.ndarray
    ) -> bool:
        """Reject identity collapse onto inner profile changes."""
        measured = np.asarray(measured, dtype=float).reshape(2, 3)
        predicted = np.asarray(predicted, dtype=float).reshape(2, 3)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(predicted)):
            return False
        measured_vector = measured[1] - measured[0]
        predicted_vector = predicted[1] - predicted[0]
        measured_length = float(np.linalg.norm(measured_vector))
        predicted_length = float(np.linalg.norm(predicted_vector))
        if min(measured_length, predicted_length) <= 1.0e-9:
            return False
        if not (
            self.detector.config.minimum_segment_length_m
            <= measured_length
            <= self.detector.config.maximum_segment_length_m
        ):
            return False
        endpoint_step = np.linalg.norm(measured - predicted, axis=1)
        if float(np.max(endpoint_step)) > self.tracking_maximum_endpoint_step:
            return False

        # Identities are directed: e1 must stay before e2 along the predicted
        # pair axis. This prevents overlapping local windows from assigning
        # both filters to the same internal slope change.
        predicted_direction = predicted_vector / predicted_length
        predicted_midpoint = np.mean(predicted, axis=0)
        first_coordinate = float(
            (measured[0] - predicted_midpoint) @ predicted_direction
        )
        second_coordinate = float(
            (measured[1] - predicted_midpoint) @ predicted_direction
        )
        if first_coordinate >= second_coordinate:
            return False

        if self.tracking_reference_length is not None:
            minimum = max(
                self.detector.config.minimum_segment_length_m,
                self.tracking_minimum_reference_length_ratio
                * self.tracking_reference_length,
            )
            maximum = min(
                self.detector.config.maximum_segment_length_m,
                self.tracking_maximum_reference_length_ratio
                * self.tracking_reference_length,
            )
            if not minimum <= measured_length <= maximum:
                return False
        return True

    def _reset_guidance(self) -> None:
        self.mode = "ALIGN" if self.guided_enabled else "GLOBAL"
        self.guide_endpoints = self.template_endpoints.copy()
        self.tracking_expected = None
        self.predicted_expected = None
        self.prediction_fallback = None
        self.last_matched = None
        self.previous_alignment_match = None
        self.alignment_stable_frames = 0
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.lost_from_mode = "TRACK"
        self.identity_initialized = False
        self.tracker = EndpointTracker(
            ambiguity_ratio=self.identity_ambiguity_ratio
        )
        self.temporal_tracker = DualEndpointKalmanTracker(
            self.temporal_kalman_config
        )
        self.temporal_tracking_requested = False
        self.temporal_prediction_this_frame = False
        self.temporal_search_radius = self.temporal_minimum_search_radius
        self.temporal_search_radii = np.full(
            2, self.temporal_minimum_search_radius, dtype=float
        )
        self.temporal_last_mahalanobis = None
        self.temporal_suspended = False
        self.temporal_fallback_reason = ""
        self.tracking_reference_length = None

    def _reset_callback(self, _request, response):
        if self.roi_pipeline is not None:
            self.roi_pipeline.reset()
            self.roi_input_frames = 0
            self._sync_roi_pipeline_state()
            self.last_status = {
                "state": "WAIT_ALIGNMENT",
                "reason": "alignment_reset",
                **self.roi_pipeline.status_extras(),
                **self._status_context(),
            }
            response.success = True
            response.message = json.dumps(
                self.last_status, ensure_ascii=False, sort_keys=True
            )
            return response
        self._reset_guidance()
        self.frames = 0
        self.accepted = 0
        self.last_status = {
            "state": "WAIT_ALIGNMENT",
            "reason": "alignment_reset",
            "frames": self.frames,
            "accepted": self.accepted,
            **self._status_context(),
        }
        response.success = True
        response.message = json.dumps(
            self.last_status, ensure_ascii=False, sort_keys=True
        )
        return response

    def _lock_callback(self, _request, response):
        if self.roi_pipeline is not None:
            success = self.roi_pipeline.lock()
            self._sync_roi_pipeline_state()
            response.success = success
            response.message = (
                "ROI/CSRT target locked; both trackers initialized from the "
                "current measured profile"
                if success
                else "ROI alignment is not stable or CSRT initialization failed"
            )
            return response
        if not self.guided_enabled:
            response.success = True
            response.message = "guided detection is disabled"
            return response
        if self.mode == "TRACK":
            response.success = True
            response.message = "target segment is already locked"
            return response
        if self.mode != "ALIGN":
            response.success = False
            response.message = f"cannot lock while mode={self.mode}"
            return response
        if (
            self.last_matched is None
            or self.alignment_stable_frames < self.minimum_lock_frames
        ):
            response.success = False
            response.message = (
                "alignment is not yet stable: "
                f"{self.alignment_stable_frames}/"
                f"{self.minimum_lock_frames} valid frames"
            )
            return response
        self.tracking_expected = self.last_matched.copy()
        self.guide_endpoints = self.tracking_expected.copy()
        self.tracker.reset(*self.tracking_expected)
        self.identity_initialized = True
        self.mode = "TRACK"
        self.lost_frames = 0
        self._set_tracking_reference(self.tracking_expected)
        if self.temporal_tracking_requested and self.temporal_tracking_enabled:
            self._restore_temporal_from_measurement(self.tracking_expected)
        response.success = True
        response.message = (
            "target segment locked; seed collection will use local temporal "
            "tracking and will not fall back to global line selection"
        )
        return response

    def _prior_callback(self, message: PointCloud2) -> None:
        try:
            values = _profile_array(message)
        except (TypeError, ValueError):
            return
        if len(values) < 2:
            return
        prior = np.vstack((values[0], values[-1]))
        if np.linalg.norm(prior[1] - prior[0]) <= 1e-9:
            return
        if self.roi_pipeline is not None:
            self.roi_pipeline.handle_prior(prior)
            self._sync_roi_pipeline_state()
            return
        if self.mode != "PREDICTED_TRACK":
            self.prediction_fallback = (
                None
                if self.tracking_expected is None
                else self.tracking_expected.copy()
            )
        self.predicted_expected = prior.copy()
        self.guide_endpoints = prior.copy()
        self.mode = "PREDICTED_TRACK"
        self.lost_from_mode = "PREDICTED_TRACK"
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.last_matched = None
        # A future-view prior belongs to NBV association, not the continuous
        # seed tracker.  The active node also sends SEED_TRACK_STOP, but this
        # defensive transition prevents stale temporal state crossing phases.
        self.temporal_tracking_requested = False

    def _measured_prior_callback(self, message: PointCloud2) -> None:
        """Reacquire endpoints at a pose that was measured previously."""
        try:
            values = _profile_array(message)
        except (TypeError, ValueError):
            return
        if len(values) < 2:
            return
        prior = np.vstack((values[0], values[-1]))
        if np.linalg.norm(prior[1] - prior[0]) <= 1e-9:
            return
        if self.roi_pipeline is not None:
            self.roi_pipeline.handle_measured_prior(prior)
            self._sync_roi_pipeline_state()
            return
        self.prediction_fallback = (
            None
            if self.tracking_expected is None
            else self.tracking_expected.copy()
        )
        self.predicted_expected = prior.copy()
        self.guide_endpoints = prior.copy()
        # This is a rollback to known measured geometry, not an uncertain NBV.
        # Use the narrow tracking gates and require stable reacquisition.
        self.mode = "LOST"
        self.lost_from_mode = "TRACK"
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.last_matched = None
        if self.temporal_tracking_requested and self.temporal_tracking_enabled:
            # This prior was saved from the last physically verified pose.
            # A rollback must clear suspension and velocity together; merely
            # resetting the state vector leaves `_uses_temporal_tracking()`
            # disabled and makes a reached safe pose look permanently lost.
            self._restore_temporal_from_measurement(prior)

    def _control_callback(self, message: String) -> None:
        command = message.data.strip().upper()
        if self.roi_pipeline is not None:
            self.roi_pipeline.handle_control(command)
            self._sync_roi_pipeline_state()
            return
        if command == "SEED_TRACK_START":
            self.temporal_tracking_requested = self.temporal_tracking_enabled
            self.temporal_suspended = False
            self.temporal_fallback_reason = ""
            initial = (
                self.tracking_expected
                if self.tracking_expected is not None
                else self.last_matched
            )
            if self.temporal_tracking_requested and initial is not None:
                self._set_tracking_reference(initial)
                self._restore_temporal_from_measurement(initial)
            self.get_logger().info(
                "dual-breakpoint Kalman seed tracking "
                f"{'enabled' if self.temporal_tracking_requested else 'disabled'}"
            )
        elif command == "SEED_TRACK_STOP":
            self.temporal_tracking_requested = False
            self.temporal_prediction_this_frame = False
            self.temporal_last_mahalanobis = None
            self.temporal_suspended = False
            self.temporal_fallback_reason = ""
            # Never leave a Kalman prediction as the durable ROI guide.  A
            # phase stop restores measured geometry; a following NBV prior can
            # still replace it through the normal prior callback.
            trusted = (
                self.tracking_expected
                if self.tracking_expected is not None
                else self.last_matched
            )
            if (
                trusted is not None
                and np.all(np.isfinite(trusted))
                and (
                    self.mode == "TRACK"
                    or (
                        self.mode == "LOST"
                        and self.lost_from_mode == "TRACK"
                    )
                )
            ):
                self.guide_endpoints = trusted.copy()
                self.temporal_tracker.reset(
                    trusted, timestamp_s=self.last_profile_time_s
                )
            self.get_logger().info(
                "dual-breakpoint Kalman seed tracking stopped"
            )
        elif command == "REFERENCE_REACQUIRE":
            # A verified return to the operator-aligned reference pose is a
            # special recovery case.  Keeping the last local guide after an
            # observation was lost can leave the detector permanently locked
            # onto a stale association even though the physical target is
            # visible again.  Re-arm the *bounded* alignment template using
            # the same narrow gates already verified by the operator; do not
            # fall back to an unconstrained global line search, which could
            # select the table or a plate side wall.  The next
            # PREDICTION_COMMIT replaces the template with measured endpoints.
            self.prediction_fallback = (
                None
                if self.tracking_expected is None
                else self.tracking_expected.copy()
            )
            self.predicted_expected = self.template_endpoints.copy()
            self.guide_endpoints = self.predicted_expected.copy()
            # Enter LOST deliberately so the normal reacquire_stable_frames
            # requirement is applied before any endpoint pair is republished.
            # Use the narrow, operator-verified ALIGN gates at the reference
            # pose.  The wider predicted corridor is for an uncertain future
            # view and may also contain the plate side or workbench.
            self.mode = "LOST"
            self.lost_from_mode = "ALIGN"
            self.last_matched = None
            self.lost_frames = 0
            self.reacquire_frames = 0
            self.get_logger().warning(
                "reference reacquisition armed with the bounded alignment "
                "template"
            )
        elif command == "PREDICTION_COMMIT":
            if (
                self.predicted_expected is None
                and self.mode != "PREDICTED_TRACK"
                and not (
                    self.mode == "LOST"
                    and self.lost_from_mode == "PREDICTED_TRACK"
                )
            ):
                return
            if self.last_matched is not None:
                self.tracking_expected = self.last_matched.copy()
            elif self.predicted_expected is not None:
                self.tracking_expected = self.predicted_expected.copy()
            if self.tracking_expected is not None:
                self.guide_endpoints = self.tracking_expected.copy()
                self.tracker.reset(*self.tracking_expected)
                self.identity_initialized = True
                self.mode = "TRACK"
                self._set_tracking_reference(self.tracking_expected)
                if (
                    self.temporal_tracking_requested
                    and self.temporal_tracking_enabled
                ):
                    self._restore_temporal_from_measurement(
                        self.tracking_expected
                    )
            self.predicted_expected = None
            self.prediction_fallback = None
            self.lost_frames = 0
            self.reacquire_frames = 0
        elif command == "PREDICTION_CANCEL":
            if (
                self.predicted_expected is None
                and self.mode != "PREDICTED_TRACK"
                and not (
                    self.mode == "LOST"
                    and self.lost_from_mode == "PREDICTED_TRACK"
                )
            ):
                return
            if self.prediction_fallback is not None:
                self.tracking_expected = self.prediction_fallback.copy()
                self.guide_endpoints = self.tracking_expected.copy()
                self.tracker.reset(*self.tracking_expected)
                self.identity_initialized = True
                self.mode = "TRACK"
                self._set_tracking_reference(self.tracking_expected)
                if (
                    self.temporal_tracking_requested
                    and self.temporal_tracking_enabled
                ):
                    self._restore_temporal_from_measurement(
                        self.tracking_expected
                    )
            else:
                self._reset_guidance()
            self.predicted_expected = None
            self.prediction_fallback = None
            self.lost_frames = 0
            self.reacquire_frames = 0

    def _empty_output(self, message: PointCloud2) -> None:
        self.publisher.publish(
            point_cloud2.create_cloud(message.header, ENDPOINT_FIELDS, [])
        )
        self.surface_publisher.publish(
            point_cloud2.create_cloud_xyz32(message.header, [])
        )

    def _reject(self, message: PointCloud2, reason: str) -> None:
        if self.temporal_prediction_this_frame:
            self.temporal_tracker.mark_missed()
            if (
                self.temporal_tracker.missed_frames
                >= self.temporal_maximum_coast_frames
            ):
                self._suspend_temporal_tracking(
                    "maximum_coast_frames_exceeded"
                )
        if self.mode == "ALIGN":
            self.alignment_stable_frames = 0
            self.previous_alignment_match = None
        elif self.mode == "LOST":
            self.reacquire_frames = 0
        elif self.guided_enabled and self.mode in {
            "TRACK",
            "PREDICTED_TRACK",
        }:
            self.lost_frames += 1
            if self.lost_frames >= self.maximum_lost_frames:
                self.lost_from_mode = self.mode
                self.mode = "LOST"
                self.reacquire_frames = 0
        self.last_status = {
            "state": "LOST" if self.mode == "LOST" else "REJECTED",
            "reason": reason,
            "frames": self.frames,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / max(self.frames, 1),
            **self._status_context(),
        }
        self._empty_output(message)
        self._publish_diagnostics()

    def _associate(self, detection) -> np.ndarray | None:
        measured = np.vstack((detection.first, detection.second))
        if self.mode == "GLOBAL":
            if not self.identity_initialized:
                matched = (
                    measured
                    if self.initial_first_label == "e1"
                    else measured[::-1]
                ).copy()
                self.tracker.reset(*matched)
                self.identity_initialized = True
                return matched
            matched = self.tracker.match(*measured)
            return None if matched is None else np.vstack(matched)
        if self.mode == "ALIGN" and not self.identity_initialized:
            matched = (
                measured
                if self.initial_first_label == "e1"
                else measured[::-1]
            )
            return matched.copy()
        expected = self.guide_endpoints
        direct = float(np.sum((measured - expected) ** 2))
        swapped = float(np.sum((measured[::-1] - expected) ** 2))
        scale = max(direct, swapped, 1e-12)
        if abs(direct - swapped) / scale < self.identity_ambiguity_ratio:
            return None
        return measured.copy() if direct < swapped else measured[::-1].copy()

    def _reacquisition_geometry_is_continuous(
        self, matched: np.ndarray
    ) -> bool:
        if not (
            self.mode == "LOST"
            and self.lost_from_mode == "TRACK"
            and self.temporal_suspended
        ):
            return True
        expected_vector = self.guide_endpoints[1] - self.guide_endpoints[0]
        measured_vector = matched[1] - matched[0]
        expected_length = float(np.linalg.norm(expected_vector))
        measured_length = float(np.linalg.norm(measured_vector))
        if min(expected_length, measured_length) <= 1.0e-9:
            return False
        if (
            abs(measured_length - expected_length)
            > self.reacquire_maximum_length_change
        ):
            return False
        cosine = float(
            np.clip(
                abs(
                    float(
                        expected_vector @ measured_vector
                        / (expected_length * measured_length)
                    )
                ),
                0.0,
                1.0,
            )
        )
        angle = float(np.rad2deg(np.arccos(cosine)))
        return angle <= self.reacquire_maximum_angle_change

    def _guided_detection(self, profile: np.ndarray):
        if not self.guided_enabled:
            return self.detector.detect(profile)
        if self.temporal_prediction_this_frame:
            return self.detector.detect_temporal_breakpoint_pair(
                profile,
                self.guide_endpoints[0],
                self.guide_endpoints[1],
                endpoint_gate_m=self.temporal_search_radius,
                normal_gate_m=self.tracking_normal_gate,
                maximum_angle_difference_deg=self.tracking_angle_gate,
                selection_mode="seed_temporal_track",
            )
        normal_gate, endpoint_gate, angle_gate = self._guide_parameters()
        selection_mode = {
            "ALIGN": "guided_align",
            "TRACK": "guided_track",
            "PREDICTED_TRACK": "predicted_track",
            "LOST": "local_reacquire",
        }.get(self.mode, "guided")
        return self.detector.detect_guided(
            profile,
            self.guide_endpoints[0],
            self.guide_endpoints[1],
            normal_gate_m=normal_gate,
            endpoint_gate_m=endpoint_gate,
            maximum_angle_difference_deg=angle_gate,
            selection_mode=selection_mode,
        )

    def _try_partial_temporal_update(
        self,
        message: PointCloud2,
        profile: np.ndarray,
        *,
        candidate_sets: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        """Use any visible physical breakpoint without publishing a fake pair."""
        if not self.temporal_prediction_this_frame:
            return False
        if candidate_sets is None:
            candidate_sets = self.detector.temporal_breakpoint_candidates(
                profile,
                self.guide_endpoints,
                endpoint_gate_m=self.temporal_search_radii,
                maximum_candidates_per_endpoint=(
                    self.temporal_maximum_local_candidates
                ),
            )

        selected: dict[int, np.ndarray] = {}
        distances: dict[int, float] = {}
        for endpoint, candidates in enumerate(candidate_sets):
            result = self.temporal_tracker.select_endpoint_candidate(
                endpoint,
                candidates,
                measurement_sigma_m=self.temporal_partial_measurement_std,
            )
            if result is not None:
                selected[endpoint], distances[endpoint] = result
        # A locally close slope change is not automatically the same physical
        # edge. Test each proposed update against the other endpoint's current
        # prediction before allowing the Kalman state to move inward.
        for endpoint in tuple(selected):
            hypothesis = self.guide_endpoints.copy()
            hypothesis[endpoint] = selected[endpoint]
            if not self._tracking_pair_is_plausible(
                hypothesis, self.guide_endpoints
            ):
                selected.pop(endpoint)
                distances.pop(endpoint)
        if not selected:
            return False

        # Candidate searches can overlap after covariance expansion.  Keep a
        # single best endpoint instead of letting both identities collapse
        # onto one change point or an implausible short/long pair.
        if len(selected) == 2:
            pair = np.vstack((selected[0], selected[1]))
            vector = pair[1] - pair[0]
            length = float(np.linalg.norm(vector))
            predicted_vector = self.guide_endpoints[1] - self.guide_endpoints[0]
            predicted_length = float(np.linalg.norm(predicted_vector))
            angle = float("inf")
            if length > 1.0e-12 and predicted_length > 1.0e-12:
                cosine = float(
                    np.clip(
                        abs(float(vector @ predicted_vector))
                        / (length * predicted_length),
                        0.0,
                        1.0,
                    )
                )
                angle = float(np.rad2deg(np.arccos(cosine)))
            if not (
                self._tracking_pair_is_plausible(
                    pair, self.guide_endpoints
                )
                and angle <= self.tracking_angle_gate
            ):
                rejected = max(distances, key=distances.get)
                selected.pop(rejected)
                distances.pop(rejected)

        self.temporal_tracker.update_partial(
            selected,
            measurement_sigma_m=self.temporal_partial_measurement_std,
        )
        self.guide_endpoints = self.temporal_tracker.endpoints()
        self.temporal_search_radii = (
            self.temporal_tracker.endpoint_search_radii(
                minimum_m=self.temporal_minimum_search_radius,
                maximum_m=self.tracking_endpoint_gate,
                sigma_multiplier=self.temporal_search_sigma_multiplier,
            )
        )
        self.temporal_search_radius = float(
            np.max(self.temporal_search_radii)
        )
        self.temporal_last_mahalanobis = tuple(
            distances.get(endpoint) for endpoint in (0, 1)
        )
        if (
            self.temporal_tracker.missed_frames
            >= self.temporal_maximum_coast_frames
        ):
            self._suspend_temporal_tracking(
                "partial_endpoint_coast_exceeded"
            )
            # `_reject` must not mark both endpoints missed a second time.
            self.temporal_prediction_this_frame = False
            self._reject(message, "partial_endpoint_coast_exceeded")
            return True

        observed = "_".join(f"e{endpoint + 1}" for endpoint in selected)
        self.last_status = {
            "state": "REJECTED",
            "reason": f"partial_endpoint_update_{observed}",
            "frames": self.frames,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / max(self.frames, 1),
            "partial_candidates": [
                int(len(candidate_sets[0])), int(len(candidate_sets[1]))
            ],
            **self._status_context(),
        }
        self._empty_output(message)
        self._publish_diagnostics()
        return True

    def _callback(self, message: PointCloud2) -> None:
        if self.roi_pipeline is not None:
            self._roi_profile_callback(message)
            return
        self.frames += 1
        self.last_profile_header = message.header
        self.last_profile_time_s = self._message_time_seconds(message)
        self._prepare_temporal_prediction(self.last_profile_time_s)
        self._publish_guide(message.header)
        try:
            profile = _profile_array(message)
            detection = self._guided_detection(profile)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self._reject(message, f"invalid_profile:{error}")
            return
        if detection is None:
            if self._try_partial_temporal_update(message, profile):
                return
            self._reject(message, self.detector.last_rejection_reason)
            return
        if detection.confidence < self.minimum_confidence:
            self._reject(message, "confidence_below_threshold")
            return
        measured = np.vstack((detection.first, detection.second))
        predicted_pair = self.guide_endpoints.copy()
        if self.temporal_prediction_this_frame:
            ordered = self.temporal_tracker.order_measurement(
                measured,
                measurement_sigma_m=detection.endpoint_sigma_m,
            )
            if ordered is None:
                both = (measured.copy(), measured.copy())
                if self._try_partial_temporal_update(
                    message, profile, candidate_sets=both
                ):
                    return
                self._reject(message, "temporal_endpoint_gate_rejected")
                return
            matched, self.temporal_last_mahalanobis = ordered
        else:
            matched = self._associate(detection)
            if matched is None:
                self._reject(message, "ambiguous_endpoint_identity")
                return
        tracking_geometry = self.mode == "TRACK" or (
            self.mode == "LOST" and self.lost_from_mode == "TRACK"
        )
        if tracking_geometry and not self._tracking_pair_is_plausible(
            matched, predicted_pair
        ):
            if self._try_partial_temporal_update(message, profile):
                return
            self._reject(message, "tracked_endpoint_topology_rejected")
            return
        if not self._reacquisition_geometry_is_continuous(matched):
            self._reject(message, "reacquire_geometry_discontinuity")
            return
        if self.temporal_prediction_this_frame:
            # The filter state guides only the next search. `matched` remains
            # the actual sensor measurement published to calibration.
            self.temporal_tracker.update(
                matched,
                measurement_sigma_m=detection.endpoint_sigma_m,
            )
            self.guide_endpoints = self.temporal_tracker.endpoints()

        if self.mode == "LOST":
            if self.reacquire_frames > 0 and self.last_matched is not None:
                maximum_change = float(
                    np.max(
                        np.linalg.norm(matched - self.last_matched, axis=1)
                    )
                )
                self.reacquire_frames = (
                    self.reacquire_frames + 1
                    if maximum_change <= self.reacquire_stability
                    else 1
                )
            else:
                self.reacquire_frames = 1
            self.last_matched = matched.copy()
            if self.reacquire_frames < self.reacquire_stable_frames:
                self.last_status = {
                    "state": "REJECTED",
                    "reason": "local_reacquire_pending",
                    "frames": self.frames,
                    "accepted": self.accepted,
                    "acceptance_rate": self.accepted / max(self.frames, 1),
                    **self._status_context(),
                }
                self._empty_output(message)
                self._publish_diagnostics()
                return
            self.mode = self.lost_from_mode
            self.reacquire_frames = 0
            self.lost_frames = 0

        if self.mode == "ALIGN":
            if self.previous_alignment_match is None:
                self.alignment_stable_frames = 1
            else:
                maximum_change = float(
                    np.max(
                        np.linalg.norm(
                            matched - self.previous_alignment_match, axis=1
                        )
                    )
                )
                self.alignment_stable_frames = (
                    self.alignment_stable_frames + 1
                    if maximum_change <= self.alignment_stability
                    else 1
                )
            self.previous_alignment_match = matched.copy()
        elif self.mode == "TRACK":
            self.tracking_expected = matched.copy()
            if not self.temporal_prediction_this_frame:
                self.guide_endpoints = matched.copy()
            self.tracker.reset(*matched)
            self.identity_initialized = True
            self.lost_frames = 0
        elif self.mode == "PREDICTED_TRACK":
            # Keep the pre-NBV fallback untouched until the active node sends
            # COMMIT or CANCEL.  Only the live guide follows the measured line.
            self.guide_endpoints = matched.copy()
            self.lost_frames = 0
        self.last_matched = matched.copy()
        if (
            self.mode == "TRACK"
            and self.temporal_tracking_requested
            and self.temporal_tracking_enabled
            and self.temporal_suspended
        ):
            # A bounded, measured local reacquisition is trustworthy enough to
            # restart the filter with zero velocity.
            self._restore_temporal_from_measurement(matched)

        endpoint_u, endpoint_v = matched
        rows = [
            (
                float(endpoint_u[0]),
                0.0,
                float(endpoint_u[2]),
                float(detection.confidence),
                float(detection.endpoint_sigma_m),
            ),
            (
                float(endpoint_v[0]),
                0.0,
                float(endpoint_v[2]),
                float(detection.confidence),
                float(detection.endpoint_sigma_m),
            ),
        ]
        self.publisher.publish(
            point_cloud2.create_cloud(message.header, ENDPOINT_FIELDS, rows)
        )
        self.surface_publisher.publish(
            point_cloud2.create_cloud_xyz32(
                message.header,
                detection.surface_points.astype(np.float32).tolist(),
            )
        )
        self.accepted += 1
        self.last_status = {
            "state": "VALID",
            "reason": "",
            "frames": self.frames,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / self.frames,
            "profile_points": int(len(profile)),
            "support_points": detection.support_count,
            "target_surface_points": int(len(detection.surface_points)),
            "segment_length_mm": 1000.0 * detection.segment_length_m,
            "residual_rms_mm": 1000.0 * detection.residual_rms_m,
            "sample_pitch_mm": 1000.0 * detection.sample_pitch_m,
            "endpoint_sigma_mm": 1000.0 * detection.endpoint_sigma_m,
            "confidence": detection.confidence,
            "breakpoint_count": detection.breakpoint_count,
            "selection_mode": detection.selection_mode,
            "initial_first_label": self.initial_first_label,
            **self._status_context(),
        }
        self._publish_diagnostics()

    def _status_callback(self, _request, response):
        response.success = self.last_status.get("state") == "VALID"
        response.message = json.dumps(
            self.last_status, ensure_ascii=False, sort_keys=True
        )
        return response


def main() -> None:
    rclpy.init()
    node = ProfileEndpointDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
