"""Full variable-projection information scoring for feasible candidates."""

from __future__ import annotations

import numpy as np

from ..models import Candidate, CandidateScore, FlangePose, SensorROI
from ..v2_backend.information import (
    effective_handeye_information,
    effective_handeye_information_from_hessian,
    information_gain,
    scaled_jacobian,
)
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
    solver=None,
    board_dimensions: tuple[float, float] | None = None,
) -> list[CandidateScore]:
    if virtual_batch_size < 1:
        raise ValueError("virtual_batch_size must be positive")
    projection_weights = projection_weights or {}
    state = result.estimate.optimization_state
    if solver is None:
        residual = lambda value: variable_projection_residual(
            value, poses, measurements, **projection_weights
        )
    else:
        if board_dimensions is None:
            board_dimensions = (
                result.estimate.board.length_u,
                result.estimate.board.length_v,
            )
        residual = lambda value: solver.residual(
            value,
            poses,
            measurements,
            board_dimensions=board_dimensions,
        )
        state_scale = solver.state_scale
    additive_shared_information = bool(
        solver is not None
        and getattr(solver, "uses_shared_surface", False)
        and getattr(getattr(result, "diagnostics", None), "state_information", None)
        is not None
    )
    if additive_shared_information:
        current_state_information = result.diagnostics.state_information
        prior_information = (
            result.diagnostics.prior_augmented_effective_handeye_information
        )
        current_information = (
            result.diagnostics.effective_handeye_information
            if prior_information is None
            else prior_information
        )
    else:
        current_jacobian = numerical_jacobian(residual, state)
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
            solver=solver,
            board_dimensions=board_dimensions,
        )
        if probability < minimum_valid_probability:
            continue
        flange = candidate.flange_transform_command
        virtual_pose = FlangePose(flange[:3, :3], flange[:3, 3])
        # Always use the measurement predicted by the current estimate.  In
        # shared mode the candidate's construction-time chord is only a flat
        # tangent-plane command and is not the expected physical profile.
        fixed_virtual_measurement = prediction.measurement
        if additive_shared_information:
            virtual_residual = lambda value: solver.observation_residual(
                value,
                [virtual_pose],
                [fixed_virtual_measurement],
                board_dimensions=board_dimensions,
            )
            virtual_jacobian = numerical_jacobian(virtual_residual, state)
            virtual_scaled = scaled_jacobian(virtual_jacobian, state_scale)
            augmented_state_information = (
                current_state_information
                + virtual_batch_size * (virtual_scaled.T @ virtual_scaled)
            )
            augmented_information = effective_handeye_information_from_hessian(
                augmented_state_information
            )
        elif solver is None:
            augmented_poses = poses + [virtual_pose] * virtual_batch_size
            augmented_measurements = measurements + [
                fixed_virtual_measurement
            ] * virtual_batch_size
            augmented_residual = lambda value: variable_projection_residual(
                value,
                augmented_poses,
                augmented_measurements,
                **projection_weights,
            )
        else:
            augmented_poses = poses + [virtual_pose] * virtual_batch_size
            augmented_measurements = measurements + [
                fixed_virtual_measurement
            ] * virtual_batch_size
            augmented_residual = lambda value: solver.residual(
                value,
                augmented_poses,
                augmented_measurements,
                board_dimensions=board_dimensions,
            )
        if not additive_shared_information:
            augmented_jacobian = numerical_jacobian(augmented_residual, state)
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
