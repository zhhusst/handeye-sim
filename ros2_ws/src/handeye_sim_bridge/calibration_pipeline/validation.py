"""Truth-independent held-out geometric validation for rolling calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CalibrationResult, FlangePose, Measurement
from .v2_backend.corner_projection import build_corner_system


@dataclass(frozen=True)
class ValidationMetrics:
    """Residual statistics evaluated without refitting any parameter."""

    score_m: float
    rms_m: float
    median_pose_rms_m: float
    maximum_pose_rms_m: float
    pose_count: int


def held_out_geometric_metrics(
    result: CalibrationResult,
    poses: list[FlangePose],
    measurements: list[Measurement],
    *,
    weights: dict[str, float] | None = None,
) -> ValidationMetrics | None:
    """Evaluate fixed hand-eye, board frame and corner on held-out batches.

    Each physical pose contributes one RMS value.  The selection score is the
    median pose RMS, so one missed endpoint or sparse outlier cannot replace a
    consistently better calibration.  No Gazebo truth is used.
    """
    if len(poses) != len(measurements):
        raise ValueError("validation poses and measurements must have equal length")
    if not poses:
        return None
    projection_weights = weights or {
        "plane_weight": 1.0,
        "edge_weight": 1.0,
        "endpoint_plane_weight": 1.0,
    }
    pose_rms: list[float] = []
    all_residuals: list[np.ndarray] = []
    corner = result.estimate.board.corner
    for pose, measurement in zip(poses, measurements):
        system, target = build_corner_system(
            result.estimate.x9,
            [pose],
            [measurement],
            **projection_weights,
        )
        residual = system @ corner - target
        all_residuals.append(residual)
        pose_rms.append(float(np.sqrt(np.mean(np.square(residual)))))
    residuals = np.concatenate(all_residuals)
    values = np.asarray(pose_rms, dtype=float)
    return ValidationMetrics(
        score_m=float(np.median(values)),
        rms_m=float(np.sqrt(np.mean(np.square(residuals)))),
        median_pose_rms_m=float(np.median(values)),
        maximum_pose_rms_m=float(np.max(values)),
        pose_count=len(values),
    )
