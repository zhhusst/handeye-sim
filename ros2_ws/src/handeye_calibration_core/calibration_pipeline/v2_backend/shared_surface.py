"""Low-dimensional target form model shared by every calibration view.

The reference plane still owns the corner and the two physical edge axes.
Only the out-of-plane height is modelled here.  Constant and first-order
terms are deliberately omitted because they are gauge-equivalent to moving
or tilting the reference plane.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


class SurfaceBasis:
    """Normalized height basis over board coordinates ``(xi, eta)``."""

    def __init__(self, kind: str = "legendre", *, degree: int = 4) -> None:
        if kind not in {"matched", "legendre"}:
            raise ValueError("surface basis must be 'matched' or 'legendre'")
        if degree < 2:
            raise ValueError("surface degree must be at least two")
        self.kind = str(kind)
        self.degree = int(degree)
        self._terms = (
            []
            if kind == "matched"
            else [
                (x_order, total - x_order)
                for total in range(2, self.degree + 1)
                for x_order in range(total + 1)
            ]
        )
        grid = np.linspace(0.0, 1.0, 81)
        xi, eta = np.meshgrid(grid, grid, indexing="ij")
        raw = self._raw(xi.reshape(-1), eta.reshape(-1))
        scales = np.sqrt(np.mean(raw**2, axis=0))
        if np.any(scales < 1e-10):
            raise ValueError("surface basis contains a degenerate mode")
        self._scales = scales

    @property
    def size(self) -> int:
        return 6 if self.kind == "matched" else len(self._terms)

    def _raw(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).reshape(-1)
        eta = np.asarray(eta, dtype=float).reshape(-1)
        if xi.shape != eta.shape:
            raise ValueError("xi and eta must have equal shape")
        if self.kind == "matched":
            return np.column_stack(
                (
                    np.sin(np.pi * xi),
                    np.sin(np.pi * eta),
                    np.sin(np.pi * xi) * np.sin(np.pi * eta),
                    np.sin(2.0 * np.pi * xi + 0.31) * np.sin(np.pi * eta),
                    np.sin(np.pi * xi) * np.sin(2.0 * np.pi * eta - 0.47),
                    (xi - 0.5) * (eta - 0.5),
                )
            )
        x = 2.0 * xi - 1.0
        y = 2.0 * eta - 1.0
        columns = []
        for x_order, y_order in self._terms:
            x_coefficients = np.zeros(x_order + 1)
            y_coefficients = np.zeros(y_order + 1)
            x_coefficients[-1] = 1.0
            y_coefficients[-1] = 1.0
            columns.append(
                np.polynomial.legendre.legval(x, x_coefficients)
                * np.polynomial.legendre.legval(y, y_coefficients)
            )
        return np.column_stack(columns)

    def evaluate(self, xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
        return self._raw(xi, eta) / self._scales[None, :]

    def height(
        self, xi: np.ndarray, eta: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.size,):
            raise ValueError(
                f"shape coefficients must have shape ({self.size},)"
            )
        return self.evaluate(xi, eta) @ coefficients

    def rms_and_maximum(
        self, coefficients: np.ndarray, *, grid_size: int = 81
    ) -> tuple[float, float]:
        grid = np.linspace(0.0, 1.0, grid_size)
        xi, eta = np.meshgrid(grid, grid, indexing="ij")
        height = self.height(xi.reshape(-1), eta.reshape(-1), coefficients)
        return (
            float(np.sqrt(np.mean(height**2))),
            float(np.max(np.abs(height))),
        )


@lru_cache(maxsize=16)
def get_surface_basis(kind: str = "legendre", degree: int = 4) -> SurfaceBasis:
    """Reuse basis normalization during repeated NBV prediction."""
    return SurfaceBasis(kind, degree=degree)


def shared_surface_residual(
    state: np.ndarray,
    poses,
    measurements,
    *,
    board_dimensions: tuple[float, float],
    basis: SurfaceBasis,
    plane_weight: float = 1.0,
    edge_weight: float = 1.0,
    endpoint_surface_weight: float = 1.0,
    shape_regularization: float = 1e-2,
    include_regularization: bool = True,
) -> np.ndarray:
    """Residual for hand-eye, reference board frame and shared shape.

    State ordering is ``[r_X, t_X, r_B, C_B, beta]``.  The same ``beta`` is
    used for every pose; this sharing is what prevents fixed target form from
    being mistaken for a different hand-eye transform in each observation.
    """
    from ..geometry import so3_exp

    if len(poses) != len(measurements):
        raise ValueError("poses and measurements must have equal length")
    state = np.asarray(state, dtype=float)
    expected = 12 + basis.size
    if state.shape != (expected,):
        raise ValueError(f"state must have shape ({expected},)")
    handeye_rotation = so3_exp(state[:3])
    handeye_translation = state[3:6]
    board_rotation = so3_exp(state[6:9])
    corner = state[9:12]
    coefficients = state[12:]
    u, v, normal = board_rotation.T
    width, height = map(float, board_dimensions)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("board dimensions must be positive")
    rows: list[np.ndarray] = []

    def surface_distance(points_base: np.ndarray) -> np.ndarray:
        points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
        delta = points_base - corner[None, :]
        xi = (delta @ u) / width
        eta = (delta @ v) / height
        return delta @ normal - basis.height(xi, eta, coefficients)

    for pose, measurement in zip(poses, measurements):
        sensor_rotation = pose.rotation @ handeye_rotation
        sensor_translation = pose.translation + pose.rotation @ handeye_translation
        points_base = (
            sensor_rotation @ measurement.profile_points.T
        ).T + sensor_translation
        rows.append(
            np.sqrt(float(plane_weight) / max(len(points_base), 1))
            * surface_distance(points_base)
        )
        endpoint_u = sensor_rotation @ measurement.endpoint_u + sensor_translation
        endpoint_v = sensor_rotation @ measurement.endpoint_v + sensor_translation
        rows.append(
            np.array(
                [
                    np.sqrt(float(edge_weight)) * float(v @ (endpoint_u - corner)),
                    np.sqrt(float(endpoint_surface_weight))
                    * float(surface_distance(endpoint_u[None, :])[0]),
                    np.sqrt(float(edge_weight)) * float(u @ (endpoint_v - corner)),
                    np.sqrt(float(endpoint_surface_weight))
                    * float(surface_distance(endpoint_v[None, :])[0]),
                ]
            )
        )
    if include_regularization and shape_regularization > 0.0:
        rows.append(np.sqrt(float(shape_regularization)) * coefficients)
    if not rows:
        return np.empty(0, dtype=float)
    return np.concatenate(rows)
