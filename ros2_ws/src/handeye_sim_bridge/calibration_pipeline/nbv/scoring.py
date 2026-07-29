"""Full variable-projection information scoring for feasible candidates."""

from __future__ import annotations

import numpy as np

from ..models import Candidate, CandidateScore, FlangePose, SensorROI
from ..v2_backend.information import effective_handeye_information, information_gain
from ..v2_backend.residual import numerical_jacobian, variable_projection_residual
from .profile_predictor import predict_candidate
from .validity import candidate_valid_probability


def score_candidates(
    candidates: list[Candidate],
    result,
    poses,
    measurements,
    roi: SensorROI,
    *,
    minimum_valid_probability: float = 0.8,
    projection_weights: dict | None = None,
    maximum_candidates: int | None = None,
    state_scale: np.ndarray | None = None,
    virtual_batch_size: int = 1,
) -> list[CandidateScore]:
    if virtual_batch_size < 1:
        raise ValueError("virtual_batch_size must be positive")
    projection_weights = projection_weights or {}
    x9 = result.estimate.x9
    residual = lambda state: variable_projection_residual(
        state, poses, measurements, **projection_weights
    )
    current_jacobian = numerical_jacobian(residual, x9)
    current_information = effective_handeye_information(
        current_jacobian, state_scale=state_scale
    )
    scored: list[CandidateScore] = []
    iterable = candidates if maximum_candidates is None else candidates[:maximum_candidates]
    for candidate in iterable:
        prediction = predict_candidate(candidate, result.estimate, roi)
        if not prediction.valid or prediction.measurement is None:
            continue
        probability = candidate_valid_probability(
            candidate,
            result.estimate,
            poses,
            measurements,
            roi,
            projection_weights=projection_weights,
        )
        if probability < minimum_valid_probability:
            continue
        flange = candidate.flange_transform_command
        virtual_pose = FlangePose(flange[:3, :3], flange[:3, 3])
        augmented_poses = poses + [virtual_pose] * virtual_batch_size
        fixed_virtual_measurement = candidate.virtual_measurement or prediction.measurement
        augmented_measurements = measurements + [
            fixed_virtual_measurement
        ] * virtual_batch_size
        augmented_residual = lambda state: variable_projection_residual(
            state, augmented_poses, augmented_measurements, **projection_weights
        )
        augmented_jacobian = numerical_jacobian(augmented_residual, x9)
        augmented_information = effective_handeye_information(
            augmented_jacobian, state_scale=state_scale
        )
        eigenvalues = np.linalg.eigvalsh(augmented_information)
        scored.append(
            CandidateScore(
                candidate=candidate,
                prediction=prediction,
                valid_probability=float(probability),
                information_gain=information_gain(current_information, augmented_information),
                minimum_eigenvalue=float(eigenvalues[0]),
                metadata={
                    "nominal_margin": candidate.nominal_margin,
                    "intersection_margin": prediction.intersection_margin,
                },
            )
        )
    return sorted(
        scored,
        key=lambda item: (item.information_gain, item.minimum_eigenvalue),
        reverse=True,
    )
