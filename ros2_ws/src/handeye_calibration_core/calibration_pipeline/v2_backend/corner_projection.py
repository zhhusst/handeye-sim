"""Analytic projection of the shared board corner C."""

from __future__ import annotations

import numpy as np

from ..geometry import so3_exp
from ..models import FlangePose, Measurement


def build_corner_system(
    x9: np.ndarray,
    poses: list[FlangePose],
    measurements: list[Measurement],
    *,
    plane_weight: float = 1.0,
    edge_weight: float = 1.0,
    endpoint_plane_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A C ~= b`` while normalizing dense points per frame."""
    if len(poses) != len(measurements):
        raise ValueError("poses and measurements must have equal length")
    x9 = np.asarray(x9, dtype=float)
    handeye_rotation = so3_exp(x9[:3])
    handeye_translation = x9[3:6]
    board_rotation = so3_exp(x9[6:9])
    u, v, normal = board_rotation.T

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    edge_scale = np.sqrt(edge_weight)
    endpoint_scale = np.sqrt(endpoint_plane_weight)

    for pose, measurement in zip(poses, measurements):
        sensor_rotation = pose.rotation @ handeye_rotation
        sensor_translation = pose.translation + pose.rotation @ handeye_translation
        plane_scale = np.sqrt(plane_weight / max(len(measurement.profile_points), 1))

        for point_sensor in measurement.profile_points:
            point_base = sensor_rotation @ point_sensor + sensor_translation
            rows.append(plane_scale * normal)
            rhs.append(float(plane_scale * normal @ point_base))

        endpoint_u_base = sensor_rotation @ measurement.endpoint_u + sensor_translation
        rows.extend((edge_scale * v, endpoint_scale * normal))
        rhs.extend(
            (
                float(edge_scale * v @ endpoint_u_base),
                float(endpoint_scale * normal @ endpoint_u_base),
            )
        )

        endpoint_v_base = sensor_rotation @ measurement.endpoint_v + sensor_translation
        rows.extend((edge_scale * u, endpoint_scale * normal))
        rhs.extend(
            (
                float(edge_scale * u @ endpoint_v_base),
                float(endpoint_scale * normal @ endpoint_v_base),
            )
        )

    if not rows:
        raise ValueError("at least one bilateral measurement is required")
    return np.asarray(rows), np.asarray(rhs)


def solve_corner(
    x9: np.ndarray,
    poses: list[FlangePose],
    measurements: list[Measurement],
    **weights: float,
) -> tuple[np.ndarray, int]:
    system, target = build_corner_system(x9, poses, measurements, **weights)
    corner, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    return corner, int(rank)
