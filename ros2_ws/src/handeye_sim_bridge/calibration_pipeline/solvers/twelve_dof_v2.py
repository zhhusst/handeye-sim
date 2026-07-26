"""12-DOF-V2 solver with a 9-D nonlinear state and projected corner."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..geometry import so3_exp, so3_log
from ..models import (
    BoardModel,
    CalibrationEstimate,
    CalibrationResult,
    FlangePose,
    Measurement,
    SolverDiagnostics,
)
from ..v2_backend.corner_projection import solve_corner
from ..v2_backend.information import (
    covariance_from_jacobian,
    effective_handeye_information,
    observability,
)
from ..v2_backend.residual import numerical_jacobian, variable_projection_residual


def _fit_direction(points: list[np.ndarray]) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("at least two endpoints are required for each physical edge")
    values = np.asarray(points)
    _, _, right = np.linalg.svd(values - values.mean(axis=0), full_matrices=False)
    return right[0] / np.linalg.norm(right[0])


def _initial_board_rotation(
    poses: list[FlangePose],
    measurements: list[Measurement],
    handeye_rotation: np.ndarray,
    handeye_translation: np.ndarray,
) -> np.ndarray:
    endpoints_u: list[np.ndarray] = []
    endpoints_v: list[np.ndarray] = []
    for pose, measurement in zip(poses, measurements):
        sensor_rotation = pose.rotation @ handeye_rotation
        sensor_translation = pose.translation + pose.rotation @ handeye_translation
        endpoints_u.append(sensor_rotation @ measurement.endpoint_u + sensor_translation)
        endpoints_v.append(sensor_rotation @ measurement.endpoint_v + sensor_translation)
    u = _fit_direction(endpoints_u)
    v_raw = _fit_direction(endpoints_v)
    v = v_raw - u * (u @ v_raw)
    if np.linalg.norm(v) < 1e-8:
        raise ValueError("initial endpoint directions are degenerate")
    v /= np.linalg.norm(v)
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    v = np.cross(normal, u)
    return np.column_stack((u, v, normal))


class TwelveDofV2Solver:
    def __init__(
        self,
        *,
        plane_weight: float = 1.0,
        edge_weight: float = 1.0,
        endpoint_plane_weight: float = 1.0,
        max_evaluations: int = 3000,
        tolerance: float = 1e-11,
    ) -> None:
        self.weights = {
            "plane_weight": plane_weight,
            "edge_weight": edge_weight,
            "endpoint_plane_weight": endpoint_plane_weight,
        }
        self.max_evaluations = max_evaluations
        self.tolerance = tolerance

    def solve(
        self,
        poses: list[FlangePose],
        measurements: list[Measurement],
        nominal_handeye_rotation: np.ndarray,
        nominal_handeye_translation: np.ndarray,
        *,
        board_dimensions: tuple[float, float],
        initial_board_rotation: np.ndarray | None = None,
    ) -> CalibrationResult:
        if len(poses) != len(measurements):
            raise ValueError("poses and measurements must have equal length")
        if len(poses) < 4:
            raise ValueError("12-DOF-V2 needs at least four bilateral poses")

        handeye_rotation = np.asarray(nominal_handeye_rotation, dtype=float)
        handeye_translation = np.asarray(nominal_handeye_translation, dtype=float)
        if initial_board_rotation is None:
            initial_board_rotation = _initial_board_rotation(
                poses, measurements, handeye_rotation, handeye_translation
            )
        x0 = np.concatenate(
            (so3_log(handeye_rotation), handeye_translation, so3_log(initial_board_rotation))
        )
        residual_function = lambda state: variable_projection_residual(
            state, poses, measurements, **self.weights
        )
        optimized = least_squares(
            residual_function,
            x0,
            method="trf",
            x_scale="jac",
            max_nfev=self.max_evaluations,
            ftol=self.tolerance,
            xtol=self.tolerance,
            gtol=self.tolerance,
        )
        residual = residual_function(optimized.x)
        jacobian = numerical_jacobian(residual_function, optimized.x)
        corner, corner_rank = solve_corner(
            optimized.x, poses, measurements, **self.weights
        )
        if corner_rank < 3:
            raise RuntimeError("projected corner system is rank deficient")

        covariance, residual_variance = covariance_from_jacobian(jacobian, residual)
        singular_values, rank, condition = observability(jacobian)
        effective = effective_handeye_information(jacobian)
        board = BoardModel(
            corner=corner,
            rotation=so3_exp(optimized.x[6:9]),
            length_u=float(board_dimensions[0]),
            length_v=float(board_dimensions[1]),
        )
        estimate = CalibrationEstimate(
            handeye_rotation=so3_exp(optimized.x[:3]),
            handeye_translation=optimized.x[3:6],
            board=board,
            x9=optimized.x,
            covariance_x9=covariance,
        )
        return CalibrationResult(
            estimate=estimate,
            cost=0.5 * float(residual @ residual),
            converged=bool(optimized.success and rank == 9),
            message=str(optimized.message),
            evaluations=int(optimized.nfev),
            diagnostics=SolverDiagnostics(
                singular_values=singular_values,
                rank=rank,
                condition_number=condition,
                residual_variance=residual_variance,
                effective_handeye_information=effective,
            ),
        )
