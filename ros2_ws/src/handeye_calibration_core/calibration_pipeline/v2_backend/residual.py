"""Variable-projection residual and reduced numerical Jacobian."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..models import FlangePose, Measurement
from .corner_projection import build_corner_system


def variable_projection_residual(
    x9: np.ndarray,
    poses: list[FlangePose],
    measurements: list[Measurement],
    **weights: float,
) -> np.ndarray:
    system, target = build_corner_system(x9, poses, measurements, **weights)
    corner, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    if rank < 3:
        return np.full(len(target), 1e3)
    return system @ corner - target


def numerical_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    step: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    baseline = function(x)
    jacobian = np.empty((len(baseline), len(x)))
    for column in range(len(x)):
        delta = np.zeros_like(x)
        delta[column] = step
        jacobian[:, column] = (function(x + delta) - function(x - delta)) / (2.0 * step)
    return jacobian
