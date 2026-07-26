"""Numerical backend for the 12-DOF-V2 variable-projection model."""

from .corner_projection import build_corner_system, solve_corner
from .information import (
    covariance_from_jacobian,
    effective_handeye_information,
    information_gain,
)
from .residual import numerical_jacobian, variable_projection_residual

__all__ = [
    "build_corner_system",
    "covariance_from_jacobian",
    "effective_handeye_information",
    "information_gain",
    "numerical_jacobian",
    "solve_corner",
    "variable_projection_residual",
]
