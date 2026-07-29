"""Robust statistics for stationary multi-frame seed observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EndpointBatchDiagnostics:
    raw_count: int
    inlier_count: int
    median_u: np.ndarray
    median_v: np.ndarray
    robust_scale_m: np.ndarray
    maximum_robust_distance: float

    @property
    def inlier_fraction(self) -> float:
        return self.inlier_count / max(self.raw_count, 1)

    def as_dict(self) -> dict:
        return {
            "raw_count": int(self.raw_count),
            "inlier_count": int(self.inlier_count),
            "inlier_fraction": float(self.inlier_fraction),
            "median_u_S": self.median_u.tolist(),
            "median_v_S": self.median_v.tolist(),
            "robust_scale_m": self.robust_scale_m.tolist(),
            "maximum_robust_distance": float(self.maximum_robust_distance),
        }


def robust_endpoint_inliers(
    endpoints_u: np.ndarray,
    endpoints_v: np.ndarray,
    *,
    mad_multiplier: float = 3.5,
    minimum_scale_m: float = 1e-6,
) -> tuple[np.ndarray, EndpointBatchDiagnostics]:
    """Reject frame-level endpoint outliers using joint e1/e2 X-Z MAD.

    Physical edge identity is already fixed upstream.  A frame is retained
    only when all four measured endpoint coordinates (e1.x/e1.z/e2.x/e2.z)
    agree with the stationary batch.
    """
    endpoint_u = np.asarray(endpoints_u, dtype=float)
    endpoint_v = np.asarray(endpoints_v, dtype=float)
    if (
        endpoint_u.ndim != 2
        or endpoint_v.ndim != 2
        or endpoint_u.shape != endpoint_v.shape
        or endpoint_u.shape[1] != 3
        or len(endpoint_u) == 0
    ):
        raise ValueError("endpoint batches must have equal non-empty shape (N, 3)")
    if not np.isfinite(endpoint_u).all() or not np.isfinite(endpoint_v).all():
        raise ValueError("endpoint batches must be finite")
    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0.0:
        raise ValueError("mad_multiplier must be positive")
    if not np.isfinite(minimum_scale_m) or minimum_scale_m <= 0.0:
        raise ValueError("minimum_scale_m must be positive")

    features = np.column_stack(
        (
            endpoint_u[:, 0],
            endpoint_u[:, 2],
            endpoint_v[:, 0],
            endpoint_v[:, 2],
        )
    )
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, minimum_scale_m)
    robust_distance = np.max(
        np.abs(features - median) / robust_scale, axis=1
    )
    inliers = robust_distance <= float(mad_multiplier)
    diagnostics = EndpointBatchDiagnostics(
        raw_count=len(features),
        inlier_count=int(np.count_nonzero(inliers)),
        median_u=np.array([median[0], 0.0, median[1]]),
        median_v=np.array([median[2], 0.0, median[3]]),
        robust_scale_m=robust_scale,
        maximum_robust_distance=float(np.max(robust_distance)),
    )
    return inliers, diagnostics
