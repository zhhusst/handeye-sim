"""Joint state-uncertainty propagation for candidate validity."""

from __future__ import annotations

import numpy as np

from ..geometry import so3_exp
from ..models import BoardModel, CalibrationEstimate, Candidate, SensorROI
from ..v2_backend.corner_projection import solve_corner
from .profile_predictor import predict_candidate


def sigma_points(mean: np.ndarray, covariance: np.ndarray) -> list[np.ndarray]:
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    square_root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0)))
    scale = np.sqrt(len(mean))
    points = [np.asarray(mean, dtype=float)]
    for column in range(len(mean)):
        offset = scale * square_root[:, column]
        points.extend((mean + offset, mean - offset))
    return points


def candidate_valid_probability(
    candidate: Candidate,
    estimate: CalibrationEstimate,
    poses,
    measurements,
    roi: SensorROI,
    *,
    prediction_options: dict | None = None,
    projection_weights: dict | None = None,
) -> float:
    if estimate.covariance_x9 is None:
        return float(predict_candidate(candidate, estimate, roi).valid)
    prediction_options = prediction_options or {}
    projection_weights = projection_weights or {}
    valid = 0
    states = sigma_points(estimate.x9, estimate.covariance_x9)
    for state in states:
        corner, rank = solve_corner(state, poses, measurements, **projection_weights)
        if rank < 3:
            continue
        sampled = CalibrationEstimate(
            handeye_rotation=so3_exp(state[:3]),
            handeye_translation=state[3:6],
            board=BoardModel(
                corner=corner,
                rotation=so3_exp(state[6:9]),
                length_u=estimate.board.length_u,
                length_v=estimate.board.length_v,
            ),
            x9=state,
            covariance_x9=None,
        )
        valid += int(predict_candidate(candidate, sampled, roi, **prediction_options).valid)
    return valid / len(states)
