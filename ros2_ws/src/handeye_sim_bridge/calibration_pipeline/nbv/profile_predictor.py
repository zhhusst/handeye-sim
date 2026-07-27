"""Edge-aware future profile prediction under a fixed flange command."""

from __future__ import annotations

import numpy as np

from ..models import (
    BoardModel,
    CalibrationEstimate,
    Candidate,
    Measurement,
    Prediction,
    SensorROI,
)
def predict_profile(
    sensor_transform: np.ndarray,
    board: BoardModel,
    roi: SensorROI,
    *,
    profile_samples: int = 40,
    edge_safe_margin: float = 0.02,
    intersection_denominator_minimum: float = 1e-4,
    minimum_profile_length: float = 0.02,
    maximum_profile_length: float = 0.8,
) -> Prediction:
    rotation = sensor_transform[:3, :3]
    translation = sensor_transform[:3, 3]
    laser_normal = rotation[:, 1]
    numerator = float(laser_normal @ (translation - board.corner))
    denominator_u = float(laser_normal @ board.u)
    denominator_v = float(laser_normal @ board.v)
    intersection_margin = min(abs(denominator_u), abs(denominator_v))
    if intersection_margin < intersection_denominator_minimum:
        return Prediction(
            False,
            "laser plane is nearly parallel to a trusted edge",
            edge_labels=("u0", "v0"),
            intersection_margin=float(intersection_margin),
        )
    coordinate_u = numerator / denominator_u
    coordinate_v = numerator / denominator_v
    edge_margin = min(
        coordinate_u - edge_safe_margin,
        board.length_u - edge_safe_margin - coordinate_u,
        coordinate_v - edge_safe_margin,
        board.length_v - edge_safe_margin - coordinate_v,
    )
    if edge_margin < 0.0:
        return Prediction(
            False,
            "intersection outside trusted adjacent-edge segments",
            edge_labels=("u0", "v0"),
            edge_margin=float(edge_margin),
            intersection_margin=float(intersection_margin),
        )
    endpoint_u_base = board.corner + coordinate_u * board.u
    endpoint_v_base = board.corner + coordinate_v * board.v
    rotation_sensor_base = rotation.T
    endpoint_u = rotation_sensor_base @ (endpoint_u_base - translation)
    endpoint_v = rotation_sensor_base @ (endpoint_v_base - translation)
    roi_margin = min(roi.margin(endpoint_u), roi.margin(endpoint_v))
    if roi_margin < 0.0:
        return Prediction(
            False,
            "endpoint outside safe sensor measurement domain",
            edge_labels=("u0", "v0"),
            roi_margin=float(roi_margin),
            edge_margin=float(edge_margin),
            intersection_margin=float(intersection_margin),
        )
    length = float(np.linalg.norm(endpoint_u - endpoint_v))
    if not minimum_profile_length <= length <= maximum_profile_length:
        return Prediction(
            False,
            "profile length outside configured quality range",
            edge_labels=("u0", "v0"),
            roi_margin=float(roi_margin),
            edge_margin=float(edge_margin),
            profile_length=length,
            intersection_margin=float(intersection_margin),
        )
    fractions = np.linspace(0.0, 1.0, profile_samples)
    profile = endpoint_u[None, :] + fractions[:, None] * (endpoint_v - endpoint_u)[None, :]
    return Prediction(
        valid=True,
        reason="ok",
        measurement=Measurement(profile, endpoint_u, endpoint_v),
        edge_labels=("u0", "v0"),
        roi_margin=float(roi_margin),
        edge_margin=float(edge_margin),
        profile_length=length,
        intersection_margin=float(intersection_margin),
    )


def predict_candidate(
    candidate: Candidate,
    estimate: CalibrationEstimate,
    roi: SensorROI,
    **prediction_options: float,
) -> Prediction:
    sensor_transform = candidate.flange_transform_command @ estimate.handeye_transform
    return predict_profile(sensor_transform, estimate.board, roi, **prediction_options)
