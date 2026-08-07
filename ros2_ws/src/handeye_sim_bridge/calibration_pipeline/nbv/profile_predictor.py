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
from ..v2_backend.shared_surface import SurfaceBasis, get_surface_basis


def _predict_shared_surface_profile(
    sensor_transform: np.ndarray,
    board: BoardModel,
    basis: SurfaceBasis,
    coefficients: np.ndarray,
    *,
    profile_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Intersect the fixed estimated surface with the commanded laser plane."""
    rotation = sensor_transform[:3, :3]
    translation = sensor_transform[:3, 3]
    laser_normal = rotation[:, 1]

    constant = float(laser_normal @ (board.corner - translation))
    slope_xi = float(laser_normal @ (board.length_u * board.u))
    slope_eta = float(laser_normal @ (board.length_v * board.v))
    slope_height = float(laser_normal @ board.normal)

    def value(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return (
            constant
            + slope_xi * xi
            + slope_eta * eta
            + slope_height * basis.height(xi, eta, coefficients)
        )

    def newton(
        xi: np.ndarray, eta: np.ndarray, *, solve_xi: bool
    ) -> np.ndarray | None:
        xi = np.asarray(xi, dtype=float).copy()
        eta = np.asarray(eta, dtype=float).copy()
        epsilon = 1e-5
        for _ in range(8):
            residual = value(xi, eta)
            if solve_xi:
                derivative = (
                    value(xi + epsilon, eta) - value(xi - epsilon, eta)
                ) / (2.0 * epsilon)
                if np.any(np.abs(derivative) < 1e-8):
                    return None
                xi -= residual / derivative
            else:
                derivative = (
                    value(xi, eta + epsilon) - value(xi, eta - epsilon)
                ) / (2.0 * epsilon)
                if np.any(np.abs(derivative) < 1e-8):
                    return None
                eta -= residual / derivative
        solved = xi if solve_xi else eta
        if (
            np.any(~np.isfinite(solved))
            or np.any(solved < -1e-7)
            or np.any(solved > 1.0 + 1e-7)
            or np.max(np.abs(value(xi, eta))) > 1e-7
        ):
            return None
        return np.clip(solved, 0.0, 1.0)

    expected_xi = -constant / slope_xi
    expected_eta = -constant / slope_eta
    xi_solution = newton(
        np.array([expected_xi]), np.zeros(1), solve_xi=True
    )
    eta_solution = newton(
        np.zeros(1), np.array([expected_eta]), solve_xi=False
    )
    if xi_solution is None or eta_solution is None:
        return None
    xi_u = float(xi_solution[0])
    eta_v = float(eta_solution[0])
    if xi_u <= 1e-9:
        return None

    xi_values = np.linspace(xi_u, 0.0, profile_samples)
    eta_initial = np.linspace(0.0, eta_v, profile_samples)
    eta_values = newton(xi_values, eta_initial, solve_xi=False)
    if eta_values is None:
        return None
    heights = basis.height(xi_values, eta_values, coefficients)
    points_base = np.asarray(
        board.corner[None, :]
        + board.length_u * xi_values[:, None] * board.u[None, :]
        + board.length_v * eta_values[:, None] * board.v[None, :]
        + heights[:, None] * board.normal[None, :]
    )
    points_sensor = (
        rotation.T @ (points_base - translation[None, :]).T
    ).T
    return points_sensor, points_sensor[0], points_sensor[-1]


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
    surface_basis: SurfaceBasis | None = None,
    shape_coefficients: np.ndarray | None = None,
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
    shared_profile = None
    if surface_basis is not None and shape_coefficients is not None:
        shared_profile = _predict_shared_surface_profile(
            sensor_transform,
            board,
            surface_basis,
            np.asarray(shape_coefficients, dtype=float),
            profile_samples=profile_samples,
        )
        if shared_profile is None:
            return Prediction(
                False,
                "estimated shared surface has no bounded bilateral intersection",
                edge_labels=("u0", "v0"),
                intersection_margin=float(intersection_margin),
            )
        profile, endpoint_u, endpoint_v = shared_profile
        endpoint_u_base = rotation @ endpoint_u + translation
        endpoint_v_base = rotation @ endpoint_v + translation
        coordinate_u = float((endpoint_u_base - board.corner) @ board.u)
        coordinate_v = float((endpoint_v_base - board.corner) @ board.v)
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
    if shared_profile is None:
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
    if shared_profile is None:
        fractions = np.linspace(0.0, 1.0, profile_samples)
        profile = endpoint_u[None, :] + fractions[:, None] * (
            endpoint_v - endpoint_u
        )[None, :]
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
    if estimate.surface_model == "shared":
        prediction_options = dict(prediction_options)
        prediction_options["surface_basis"] = get_surface_basis(
            str(estimate.surface_basis_kind), int(estimate.surface_degree)
        )
        prediction_options["shape_coefficients"] = estimate.shape_coefficients
    return predict_profile(sensor_transform, estimate.board, roi, **prediction_options)
