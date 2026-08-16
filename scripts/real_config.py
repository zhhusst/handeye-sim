#!/usr/bin/env python3
"""Single source of truth for offline-replay parameters.

Every offline tool (replay_tracking_bag.py, visualize_tracking_bag.py)
MUST build its pipeline config from :func:`make_real_config()` so all
benchmarks share one parameter set -- the one actually recorded in the
real-machine experiment:

    data/breakpoint_tracking_runs/20260814_140754_auto_seed/detector_parameters.yaml

Keeping two divergent inline parameter blocks (as was the case: replay used
ratio 0.10/4.00 + e1 while the visualizer used 0.65/1.60 + e2) makes the same
bag produce different results per tool, which is unacceptable for benchmarking.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.perception.tracking_pipeline import (
    TrackingPipelineConfig,
)
from calibration_pipeline.perception.endpoint_detector import (
    EndpointDetectionConfig,
)


def make_real_config() -> TrackingPipelineConfig:
    """1:1 parameters from the 2026-08-14 real-machine run."""
    return TrackingPipelineConfig(
        detector=EndpointDetectionConfig(
            minimum_points=12,
            minimum_segment_points=10,
            minimum_segment_length_m=0.01,
            maximum_segment_length_m=0.25,
            absolute_neighbor_gap_m=0.004,
            neighbor_gap_multiplier=8.0,
            residual_mad_multiplier=3.5,
            residual_floor_m=8.0e-05,
            maximum_residual_rms_m=0.0005,
            endpoint_extension_fraction=0.5,
            endpoint_local_fit_points=24,
            candidate_ambiguity_ratio=0.03,
            smoothing_window=5,
            local_fit_window=12,
            angle_change_threshold_deg=10.0,
            height_jump_threshold_m=0.0002,
            breakpoint_cluster_points=8,
            maximum_abs_surface_midpoint_x_m=0.08,
        ),
        identity_ambiguity_ratio=0.05,
        initial_first_label="e2",
        minimum_confidence=0.25,
        guided_enabled=True,
        # alignment
        alignment_template_center_x_m=0.0,
        alignment_template_center_z_m=0.28,
        alignment_template_length_m=0.08,
        alignment_template_angle_deg=25.0,
        alignment_normal_gate_m=0.003,
        alignment_endpoint_gate_m=0.020,
        alignment_maximum_angle_difference_deg=15.0,
        alignment_stability_m=0.0015,
        minimum_lock_frames=5,
        # tracking
        tracking_normal_gate_m=0.006,
        tracking_endpoint_gate_m=0.025,
        tracking_maximum_angle_difference_deg=25.0,
        maximum_lost_frames=5,
        # predicted (NBV)
        predicted_normal_gate_m=0.012,
        predicted_endpoint_gate_m=0.050,
        predicted_maximum_angle_difference_deg=35.0,
        # reacquisition
        reacquire_stable_frames=3,
        reacquire_maximum_segment_length_change_m=0.05,
        reacquire_maximum_segment_angle_change_deg=20.0,
        reacquire_stability_m=0.003,
        # temporal Kalman (real node enabled it; must be exercised offline)
        temporal_tracking_enabled=True,
        temporal_initial_position_std_m=0.0005,
        temporal_initial_velocity_std_m_s=0.05,
        temporal_process_acceleration_std_m_s2=1.0,
        temporal_measurement_std_floor_m=8.0e-05,
        temporal_partial_measurement_std_m=0.0005,
        temporal_mahalanobis_threshold=13.82,
        temporal_maximum_endpoint_speed_m_s=0.05,
        temporal_maximum_coast_frames=20,
        temporal_minimum_search_radius_m=0.0015,
        temporal_search_sigma_multiplier=3.0,
        temporal_maximum_local_candidates=6,
        # identity / topology
        tracking_maximum_endpoint_step_m=0.003,
        tracking_minimum_reference_length_ratio=0.65,
        tracking_maximum_reference_length_ratio=1.60,
    )


def make_real_config_with(initial_mode: str = "ALIGN") -> TrackingPipelineConfig:
    cfg = make_real_config()
    cfg.initial_mode = initial_mode
    return cfg
