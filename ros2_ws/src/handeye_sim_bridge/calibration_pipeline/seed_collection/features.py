"""Bilateral profile feedback defined in method document section 5.4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import SensorROI


@dataclass(frozen=True)
class BilateralFeature:
    endpoint_u: np.ndarray
    endpoint_v: np.ndarray
    x_mid: float
    z_mid: float
    profile_length: float
    roi_margin: float
    safe: bool


def evaluate_bilateral_feature(
    endpoint_u: np.ndarray,
    endpoint_v: np.ndarray,
    roi: SensorROI,
    *,
    minimum_profile_length: float = 0.02,
    maximum_profile_length: float = 0.8,
) -> BilateralFeature:
    endpoint_u = np.asarray(endpoint_u, dtype=float)
    endpoint_v = np.asarray(endpoint_v, dtype=float)
    midpoint = 0.5 * (endpoint_u + endpoint_v)
    profile_length = float(np.linalg.norm(endpoint_u - endpoint_v))
    roi_margin = min(roi.margin(endpoint_u), roi.margin(endpoint_v))
    safe = (
        roi_margin >= roi.safe_margin
        and minimum_profile_length <= profile_length <= maximum_profile_length
    )
    return BilateralFeature(
        endpoint_u=endpoint_u,
        endpoint_v=endpoint_v,
        x_mid=float(midpoint[0]),
        z_mid=float(midpoint[2]),
        profile_length=profile_length,
        roi_margin=float(roi_margin),
        safe=safe,
    )
