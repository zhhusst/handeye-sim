"""Observability, covariance and hand-eye information metrics."""

from __future__ import annotations

import numpy as np


def effective_handeye_information(jacobian_x9: np.ndarray, damping: float = 1e-10) -> np.ndarray:
    hessian = np.asarray(jacobian_x9, dtype=float).T @ np.asarray(jacobian_x9, dtype=float)
    handeye = hessian[:6, :6]
    cross = hessian[:6, 6:9]
    board = hessian[6:9, 6:9]
    effective = handeye - cross @ np.linalg.pinv(board + damping * np.eye(3)) @ cross.T
    return 0.5 * (effective + effective.T)


def covariance_from_jacobian(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    damping: float = 1e-9,
    variance_floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    dof = max(len(residual) - jacobian.shape[1], 1)
    variance = max(float(residual @ residual / dof), variance_floor)
    covariance = variance * np.linalg.pinv(jacobian.T @ jacobian + damping * np.eye(jacobian.shape[1]))
    return covariance, variance


def information_gain(
    current: np.ndarray,
    augmented: np.ndarray,
    *,
    regularization: float = 1e-9,
) -> float:
    eye = np.eye(current.shape[0])
    sign_current, logdet_current = np.linalg.slogdet(current + regularization * eye)
    sign_augmented, logdet_augmented = np.linalg.slogdet(augmented + regularization * eye)
    if sign_current <= 0 or sign_augmented <= 0:
        return float("-inf")
    return 0.5 * float(logdet_augmented - logdet_current)


def observability(jacobian: np.ndarray, tolerance: float | None = None) -> tuple[np.ndarray, int, float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if tolerance is None:
        tolerance = max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > tolerance
        else float("inf")
    )
    return singular_values, rank, condition
