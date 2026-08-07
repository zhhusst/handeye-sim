"""Observability, covariance and hand-eye information metrics."""

from __future__ import annotations

import numpy as np


DEFAULT_STATE_SCALE = np.array(
    [
        np.deg2rad(10.0),
        np.deg2rad(10.0),
        np.deg2rad(10.0),
        0.1,
        0.1,
        0.1,
        np.deg2rad(10.0),
        np.deg2rad(10.0),
        np.deg2rad(10.0),
    ],
    dtype=float,
)


def validate_state_scale(state_scale: np.ndarray | None, size: int = 9) -> np.ndarray:
    scale = DEFAULT_STATE_SCALE.copy() if state_scale is None else np.asarray(state_scale, dtype=float)
    if scale.shape != (size,) or np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError(f"state_scale must contain {size} positive finite values")
    return scale


def scaled_jacobian(
    jacobian: np.ndarray, state_scale: np.ndarray | None = None
) -> np.ndarray:
    """Jacobian with respect to dimensionless state ``x_tilde=S^-1 x``."""
    jacobian = np.asarray(jacobian, dtype=float)
    scale = validate_state_scale(state_scale, jacobian.shape[1])
    return jacobian * scale[None, :]


def effective_handeye_information(
    jacobian_state: np.ndarray,
    damping: float = 1e-10,
    *,
    state_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Schur-marginalized information for the first six hand-eye states.

    Every state after index six is nuisance.  For the flat V2 model this is
    only board rotation; for shared-shape V2 it also includes board corner and
    the target-form coefficients.
    """
    jacobian_scaled = scaled_jacobian(jacobian_state, state_scale)
    if jacobian_scaled.shape[1] < 6:
        raise ValueError("hand-eye information needs at least six state columns")
    hessian = jacobian_scaled.T @ jacobian_scaled
    return effective_handeye_information_from_hessian(hessian, damping=damping)


def effective_handeye_information_from_hessian(
    hessian: np.ndarray,
    damping: float = 1e-10,
) -> np.ndarray:
    """Marginalize nuisance states from a dimensionless normal matrix."""
    hessian = np.asarray(hessian, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be square")
    if hessian.shape[0] < 6:
        raise ValueError("hand-eye information needs at least six states")
    handeye = hessian[:6, :6]
    nuisance = hessian[6:, 6:]
    if nuisance.size == 0:
        return 0.5 * (handeye + handeye.T)
    cross = hessian[:6, 6:]
    effective = handeye - cross @ np.linalg.pinv(
        nuisance + damping * np.eye(nuisance.shape[0])
    ) @ cross.T
    return 0.5 * (effective + effective.T)


def covariance_from_jacobian(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    damping: float = 1e-9,
    variance_floor: float = 1e-12,
    state_scale: np.ndarray | None = None,
    fitted_nuisance_parameters: int = 0,
) -> tuple[np.ndarray, float]:
    if fitted_nuisance_parameters < 0:
        raise ValueError("fitted_nuisance_parameters must be non-negative")
    scale = validate_state_scale(state_scale, jacobian.shape[1])
    jacobian_scaled = scaled_jacobian(jacobian, scale)
    # Variable projection removes linear parameters from the nonlinear state,
    # but they were still fitted from the same residuals and must therefore
    # be included in the residual-variance degrees of freedom.
    dof = max(
        len(residual)
        - jacobian.shape[1]
        - int(fitted_nuisance_parameters),
        1,
    )
    variance = max(float(residual @ residual / dof), variance_floor)
    covariance_scaled = variance * np.linalg.pinv(
        jacobian_scaled.T @ jacobian_scaled + damping * np.eye(jacobian.shape[1])
    )
    covariance = scale[:, None] * covariance_scaled * scale[None, :]
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


def observability(
    jacobian: np.ndarray,
    tolerance: float | None = None,
    *,
    state_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, int, float]:
    jacobian_scaled = scaled_jacobian(jacobian, state_scale)
    singular_values = np.linalg.svd(jacobian_scaled, compute_uv=False)
    if tolerance is None:
        tolerance = max(jacobian_scaled.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > tolerance
        else float("inf")
    )
    return singular_values, rank, condition
