"""Generate only laser planes that nominally intersect E_u and E_v."""

from __future__ import annotations

from itertools import product

import numpy as np

from ..geometry import invert_transform, make_transform
from ..models import CalibrationEstimate, Candidate, Measurement, SensorROI


def _sensor_transform(
    point_u: np.ndarray,
    point_v: np.ndarray,
    board_normal: np.ndarray,
    alpha: float,
    psi: float,
    working_distance: float,
    branch: int,
) -> np.ndarray | None:
    line = point_v - point_u
    profile_length = np.linalg.norm(line)
    if profile_length < 1e-10:
        return None
    line /= profile_length
    tangent_normal = np.cross(board_normal, line)
    tangent_norm = np.linalg.norm(tangent_normal)
    if tangent_norm < 1e-10:
        return None
    tangent_normal /= tangent_norm
    laser_normal = (
        np.cos(alpha) * board_normal + branch * np.sin(alpha) * tangent_normal
    )
    laser_normal /= np.linalg.norm(laser_normal)
    sensor_x_zero = line
    sensor_z_zero = np.cross(sensor_x_zero, laser_normal)
    if np.linalg.norm(sensor_z_zero) < 1e-10:
        return None
    sensor_z_zero /= np.linalg.norm(sensor_z_zero)
    sensor_x = np.cos(psi) * sensor_x_zero + np.sin(psi) * sensor_z_zero
    sensor_z = -np.sin(psi) * sensor_x_zero + np.cos(psi) * sensor_z_zero
    sensor_x /= np.linalg.norm(sensor_x)
    sensor_z /= np.linalg.norm(sensor_z)
    sensor_y = laser_normal
    # Remove round-off while retaining x/y labels from the construction.
    sensor_x -= sensor_y * float(sensor_y @ sensor_x)
    sensor_x /= np.linalg.norm(sensor_x)
    sensor_z = np.cross(sensor_x, sensor_y)
    sensor_z /= np.linalg.norm(sensor_z)
    sensor_y = np.cross(sensor_z, sensor_x)
    sensor_y /= np.linalg.norm(sensor_y)
    midpoint = 0.5 * (point_u + point_v)
    sensor_translation = midpoint - working_distance * sensor_z
    return make_transform(np.column_stack((sensor_x, sensor_y, sensor_z)), sensor_translation)


def generate_candidates(
    estimate: CalibrationEstimate,
    *,
    roi: SensorROI | None = None,
    edge_samples: int = 4,
    edge_margin: float = 0.04,
    alphas_deg: tuple[float, ...] = (20.0, 35.0, 50.0),
    psis_deg: tuple[float, ...] = (-15.0, 0.0, 15.0),
    working_distances: tuple[float, ...] = (0.4, 0.55, 0.7),
    profile_samples: int = 40,
    minimum_alpha_deg: float = 5.0,
    minimum_profile_length: float = 0.01,
) -> list[Candidate]:
    board = estimate.board
    if edge_margin * 2.0 >= min(board.length_u, board.length_v):
        raise ValueError("edge_margin leaves no usable board edge")
    a_values = np.linspace(edge_margin, board.length_u - edge_margin, edge_samples)
    b_values = np.linspace(edge_margin, board.length_v - edge_margin, edge_samples)
    candidates: list[Candidate] = []
    serial = 0
    for a, b, alpha_deg, psi_deg, distance, branch in product(
        a_values, b_values, alphas_deg, psis_deg, working_distances, (-1, 1)
    ):
        if abs(alpha_deg) < minimum_alpha_deg:
            continue
        point_u = board.corner + a * board.u
        point_v = board.corner + b * board.v
        sensor_transform = _sensor_transform(
            point_u,
            point_v,
            board.normal,
            np.deg2rad(alpha_deg),
            np.deg2rad(psi_deg),
            distance,
            branch,
        )
        if sensor_transform is None:
            continue
        rotation_sensor_base = sensor_transform[:3, :3].T
        translation_sensor_base = sensor_transform[:3, 3]
        endpoint_u = rotation_sensor_base @ (point_u - translation_sensor_base)
        endpoint_v = rotation_sensor_base @ (point_v - translation_sensor_base)
        profile_length = float(np.linalg.norm(endpoint_u - endpoint_v))
        if profile_length < minimum_profile_length:
            continue
        fractions = np.linspace(0.0, 1.0, profile_samples)
        profile = endpoint_u[None, :] + fractions[:, None] * (
            endpoint_v - endpoint_u
        )[None, :]
        virtual_measurement = Measurement(profile, endpoint_u, endpoint_v)
        nominal_margin = float("inf")
        if roi is not None:
            nominal_margin = min(roi.margin(endpoint_u), roi.margin(endpoint_v))
            if nominal_margin < 0.0:
                continue
        flange_transform = sensor_transform @ invert_transform(estimate.handeye_transform)
        candidates.append(
            Candidate(
                candidate_id=f"candidate_{serial:05d}",
                a=float(a),
                b=float(b),
                alpha=float(np.deg2rad(alpha_deg)),
                psi=float(np.deg2rad(psi_deg)),
                working_distance=float(distance),
                branch=int(branch),
                sensor_transform_nominal=sensor_transform,
                flange_transform_command=flange_transform,
                virtual_measurement=virtual_measurement,
                nominal_margin=float(nominal_margin),
            )
        )
        serial += 1
    return candidates
