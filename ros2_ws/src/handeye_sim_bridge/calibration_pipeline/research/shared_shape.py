"""Legacy research wrapper for joint hand-eye/shared-shape experiments.

Production now uses the unified :class:`TwelveDofV2Solver` shared mode.  This
module remains only to reproduce the earlier matched-vs-generic feasibility
experiments and imports the same production surface basis definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..geometry import rotation_distance_deg, so3_exp, so3_log
from ..models import CalibrationResult, FlangePose, Measurement
from ..v2_backend.shared_surface import SurfaceBasis


@dataclass(frozen=True)
class SharedShapeCalibrationResult:
    handeye_rotation: np.ndarray
    handeye_translation: np.ndarray
    board_rotation: np.ndarray
    board_corner: np.ndarray
    shape_coefficients: np.ndarray
    cost: float
    converged: bool
    message: str
    evaluations: int
    full_rank: int
    condition_number: float
    effective_handeye_information: np.ndarray

    def errors_against(
        self, truth_rotation: np.ndarray, truth_translation: np.ndarray
    ) -> tuple[float, float]:
        return (
            rotation_distance_deg(self.handeye_rotation, truth_rotation),
            1000.0
            * float(
                np.linalg.norm(self.handeye_translation - truth_translation)
            ),
        )


class SharedShapeHandEyeSolver:
    """Full nonlinear refinement of hand-eye, board pose and shared shape."""

    def __init__(
        self,
        basis: SurfaceBasis,
        *,
        plane_weight: float = 1.0,
        edge_weight: float = 0.25,
        endpoint_surface_weight: float = 0.25,
        shape_scale_m: float = 0.0005,
        shape_regularization: float = 1e-2,
        max_evaluations: int = 1200,
        tolerance: float = 1e-10,
    ) -> None:
        if shape_scale_m <= 0.0:
            raise ValueError("shape_scale_m must be positive")
        if shape_regularization < 0.0:
            raise ValueError("shape_regularization must be non-negative")
        self.basis = basis
        self.plane_weight = float(plane_weight)
        self.edge_weight = float(edge_weight)
        self.endpoint_surface_weight = float(endpoint_surface_weight)
        self.shape_scale_m = float(shape_scale_m)
        self.shape_regularization = float(shape_regularization)
        self.max_evaluations = int(max_evaluations)
        self.tolerance = float(tolerance)

    def residual(
        self,
        state: np.ndarray,
        poses: list[FlangePose],
        measurements: list[Measurement],
        *,
        board_dimensions: tuple[float, float],
    ) -> np.ndarray:
        if len(poses) != len(measurements):
            raise ValueError("poses and measurements must have equal length")
        state = np.asarray(state, dtype=float)
        expected = 12 + self.basis.size
        if state.shape != (expected,):
            raise ValueError(f"state must have shape ({expected},)")
        handeye_rotation = so3_exp(state[:3])
        handeye_translation = state[3:6]
        board_rotation = so3_exp(state[6:9])
        corner = state[9:12]
        coefficients = state[12:]
        u, v, normal = board_rotation.T
        width, height = map(float, board_dimensions)
        rows: list[np.ndarray] = []

        def surface_distance(points_base: np.ndarray) -> np.ndarray:
            delta = points_base - corner[None, :]
            xi = (delta @ u) / width
            eta = (delta @ v) / height
            heights = self.basis.evaluate(xi, eta) @ coefficients
            return delta @ normal - heights

        for pose, measurement in zip(poses, measurements):
            sensor_rotation = pose.rotation @ handeye_rotation
            sensor_translation = (
                pose.translation + pose.rotation @ handeye_translation
            )
            points_base = (
                sensor_rotation @ measurement.profile_points.T
            ).T + sensor_translation
            rows.append(
                np.sqrt(self.plane_weight / max(len(points_base), 1))
                * surface_distance(points_base)
            )

            endpoint_u = (
                sensor_rotation @ measurement.endpoint_u + sensor_translation
            )
            endpoint_v = (
                sensor_rotation @ measurement.endpoint_v + sensor_translation
            )
            rows.append(
                np.array(
                    [
                        np.sqrt(self.edge_weight)
                        * float(v @ (endpoint_u - corner)),
                        np.sqrt(self.endpoint_surface_weight)
                        * float(surface_distance(endpoint_u[None, :])[0]),
                        np.sqrt(self.edge_weight)
                        * float(u @ (endpoint_v - corner)),
                        np.sqrt(self.endpoint_surface_weight)
                        * float(surface_distance(endpoint_v[None, :])[0]),
                    ]
                )
            )
        if self.shape_regularization > 0.0:
            rows.append(
                np.sqrt(self.shape_regularization) * coefficients
            )
        return np.concatenate(rows)

    def solve(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        initial_flat_result: CalibrationResult,
        *,
        board_dimensions: tuple[float, float],
    ) -> SharedShapeCalibrationResult:
        estimate = initial_flat_result.estimate
        initial = np.concatenate(
            (
                so3_log(estimate.handeye_rotation),
                estimate.handeye_translation,
                so3_log(estimate.board.rotation),
                estimate.board.corner,
                np.zeros(self.basis.size),
            )
        )
        scale = np.concatenate(
            (
                np.full(3, np.deg2rad(10.0)),
                np.full(3, 0.1),
                np.full(3, np.deg2rad(10.0)),
                np.full(3, 0.1),
                np.full(self.basis.size, self.shape_scale_m),
            )
        )
        function = lambda state: self.residual(
            state,
            poses,
            measurements,
            board_dimensions=board_dimensions,
        )
        optimized = least_squares(
            function,
            initial,
            method="trf",
            x_scale=scale,
            max_nfev=self.max_evaluations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )
        state = optimized.x
        jacobian_scaled = optimized.jac * scale[None, :]
        singular_values = np.linalg.svd(jacobian_scaled, compute_uv=False)
        tolerance = (
            max(jacobian_scaled.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
        rank = int(np.sum(singular_values > tolerance))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values[-1] > tolerance
            else float("inf")
        )
        hessian = jacobian_scaled.T @ jacobian_scaled
        handeye = hessian[:6, :6]
        cross = hessian[:6, 6:]
        nuisance = hessian[6:, 6:]
        effective = handeye - cross @ np.linalg.pinv(
            nuisance + 1e-10 * np.eye(nuisance.shape[0])
        ) @ cross.T
        effective = 0.5 * (effective + effective.T)
        residual = function(state)
        return SharedShapeCalibrationResult(
            handeye_rotation=so3_exp(state[:3]),
            handeye_translation=state[3:6].copy(),
            board_rotation=so3_exp(state[6:9]),
            board_corner=state[9:12].copy(),
            shape_coefficients=state[12:].copy(),
            cost=0.5 * float(residual @ residual),
            converged=bool(optimized.success and rank == len(state)),
            message=str(optimized.message),
            evaluations=int(optimized.nfev),
            full_rank=rank,
            condition_number=condition,
            effective_handeye_information=effective,
        )
