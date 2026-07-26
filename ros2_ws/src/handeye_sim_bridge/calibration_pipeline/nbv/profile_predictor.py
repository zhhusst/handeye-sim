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
from .finite_board_intersection import intersect_finite_board


def predict_profile(
    sensor_transform: np.ndarray,
    board: BoardModel,
    roi: SensorROI,
    *,
    profile_samples: int = 40,
    edge_safe_margin: float = 0.02,
) -> Prediction:
    rotation = sensor_transform[:3, :3]
    translation = sensor_transform[:3, 3]
    intersections = intersect_finite_board(rotation[:, 1], translation, board)
    if len(intersections) != 2:
        return Prediction(False, f"finite board has {len(intersections)} unique intersections")
    labels = tuple(item[0] for item in intersections)
    if set(labels) != {"u0", "v0"}:
        return Prediction(False, f"wrong physical edges: {labels}", edge_labels=labels)

    by_label = {label: (point, coordinate) for label, point, coordinate in intersections}
    endpoint_u_base, coordinate_u = by_label["u0"]
    endpoint_v_base, coordinate_v = by_label["v0"]
    rotation_sensor_base = rotation.T
    endpoint_u = rotation_sensor_base @ (endpoint_u_base - translation)
    endpoint_v = rotation_sensor_base @ (endpoint_v_base - translation)
    roi_margin = min(roi.margin(endpoint_u), roi.margin(endpoint_v))
    edge_margin = min(
        coordinate_u,
        board.length_u - coordinate_u,
        coordinate_v,
        board.length_v - coordinate_v,
    )
    if roi_margin < roi.safe_margin:
        return Prediction(
            False,
            "endpoint outside safe sensor ROI",
            edge_labels=("u0", "v0"),
            roi_margin=float(roi_margin),
            edge_margin=float(edge_margin),
        )
    if edge_margin < edge_safe_margin:
        return Prediction(
            False,
            "endpoint outside safe board-edge segment",
            edge_labels=("u0", "v0"),
            roi_margin=float(roi_margin),
            edge_margin=float(edge_margin),
        )
    fractions = np.linspace(0.0, 1.0, profile_samples)
    profile = endpoint_u[None, :] + fractions[:, None] * (endpoint_v - endpoint_u)[None, :]
    length = float(np.linalg.norm(endpoint_u - endpoint_v))
    return Prediction(
        valid=True,
        reason="ok",
        measurement=Measurement(profile, endpoint_u, endpoint_v),
        edge_labels=("u0", "v0"),
        roi_margin=float(roi_margin),
        edge_margin=float(edge_margin),
        profile_length=length,
    )


def predict_candidate(
    candidate: Candidate,
    estimate: CalibrationEstimate,
    roi: SensorROI,
    **prediction_options: float,
) -> Prediction:
    sensor_transform = candidate.flange_transform_command @ estimate.handeye_transform
    return predict_profile(sensor_transform, estimate.board, roi, **prediction_options)
