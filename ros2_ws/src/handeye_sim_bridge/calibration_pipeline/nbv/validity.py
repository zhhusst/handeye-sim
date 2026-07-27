"""Joint state-uncertainty propagation for candidate validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import so3_exp
from ..models import BoardModel, CalibrationEstimate, Candidate, SensorROI
from ..v2_backend.corner_projection import solve_corner
from .profile_predictor import predict_candidate


@dataclass(frozen=True)
class ValidityResult:
    valid_probability: float
    edge_u_failure_probability: float
    edge_v_failure_probability: float
    domain_failure_probability: float
    degeneracy_probability: float
    quality_failure_probability: float
    sample_count: int


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


def evaluate_candidate_validity(
    candidate: Candidate,
    estimate: CalibrationEstimate,
    poses,
    measurements,
    roi: SensorROI,
    *,
    prediction_options: dict | None = None,
    projection_weights: dict | None = None,
    covariance_inflation: float = 1.0,
) -> ValidityResult:
    if covariance_inflation <= 0.0:
        raise ValueError("covariance_inflation must be positive")
    if estimate.covariance_x9 is None:
        prediction = predict_candidate(candidate, estimate, roi)
        return ValidityResult(
            valid_probability=float(prediction.valid),
            edge_u_failure_probability=0.0,
            edge_v_failure_probability=0.0,
            domain_failure_probability=float("domain" in prediction.reason),
            degeneracy_probability=float("parallel" in prediction.reason),
            quality_failure_probability=float(
                not prediction.valid
                and "domain" not in prediction.reason
                and "parallel" not in prediction.reason
            ),
            sample_count=1,
        )
    prediction_options = prediction_options or {}
    projection_weights = projection_weights or {}
    counts = {
        "valid": 0,
        "edge_u": 0,
        "edge_v": 0,
        "domain": 0,
        "degenerate": 0,
        "quality": 0,
    }
    states = sigma_points(
        estimate.x9, covariance_inflation * estimate.covariance_x9
    )
    for state in states:
        corner, rank = solve_corner(state, poses, measurements, **projection_weights)
        if rank < 3:
            counts["degenerate"] += 1
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
        prediction = predict_candidate(candidate, sampled, roi, **prediction_options)
        if prediction.valid:
            counts["valid"] += 1
        elif "parallel" in prediction.reason:
            counts["degenerate"] += 1
        elif "domain" in prediction.reason:
            counts["domain"] += 1
        elif "trusted adjacent-edge" in prediction.reason:
            # A single analytic failure can involve one or both fixed edges.
            counts["edge_u"] += 1
            counts["edge_v"] += 1
        else:
            counts["quality"] += 1
    total = len(states)
    return ValidityResult(
        valid_probability=counts["valid"] / total,
        edge_u_failure_probability=counts["edge_u"] / total,
        edge_v_failure_probability=counts["edge_v"] / total,
        domain_failure_probability=counts["domain"] / total,
        degeneracy_probability=counts["degenerate"] / total,
        quality_failure_probability=counts["quality"] / total,
        sample_count=total,
    )


def candidate_valid_probability(*args, **kwargs) -> float:
    return evaluate_candidate_validity(*args, **kwargs).valid_probability
