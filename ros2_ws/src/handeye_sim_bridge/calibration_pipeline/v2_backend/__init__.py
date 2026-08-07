"""Numerical backend for the 12-DOF-V2 variable-projection model."""

from .corner_projection import build_corner_system, solve_corner
from .information import (
    DEFAULT_STATE_SCALE,
    covariance_from_jacobian,
    effective_handeye_information,
    effective_handeye_information_from_hessian,
    information_gain,
    scaled_jacobian,
)
from .plane_frame import canonicalize_plane_frame, project_to_rotation
from .residual import numerical_jacobian, variable_projection_residual
from .shared_surface import SurfaceBasis, get_surface_basis, shared_surface_residual

__all__ = [
    "build_corner_system",
    "canonicalize_plane_frame",
    "covariance_from_jacobian",
    "DEFAULT_STATE_SCALE",
    "effective_handeye_information",
    "effective_handeye_information_from_hessian",
    "get_surface_basis",
    "information_gain",
    "numerical_jacobian",
    "project_to_rotation",
    "scaled_jacobian",
    "shared_surface_residual",
    "solve_corner",
    "variable_projection_residual",
    "SurfaceBasis",
]
