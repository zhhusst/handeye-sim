"""Generate only laser planes that nominally intersect E_u and E_v."""

from __future__ import annotations

from itertools import product

import numpy as np

from ..geometry import invert_transform, make_transform
from ..models import BoardModel, CalibrationEstimate, Candidate


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
    if np.linalg.norm(line) < 1e-10:
        return None
    line /= np.linalg.norm(line)
    tangent_normal = np.cross(board_normal, line)
    tangent_normal /= np.linalg.norm(tangent_normal)
    laser_normal = np.cos(alpha) * board_normal + branch * np.sin(alpha) * tangent_normal
    laser_normal /= np.linalg.norm(laser_normal)
    in_plane = np.cross(laser_normal, line)
    if np.linalg.norm(in_plane) < 1e-10:
        return None
    in_plane /= np.linalg.norm(in_plane)
    sensor_z = np.cos(psi) * in_plane + np.sin(psi) * line
    sensor_z /= np.linalg.norm(sensor_z)
    sensor_x = np.cross(laser_normal, sensor_z)
    sensor_x /= np.linalg.norm(sensor_x)
    sensor_y = np.cross(sensor_z, sensor_x)
    sensor_y /= np.linalg.norm(sensor_y)
    midpoint = 0.5 * (point_u + point_v)
    sensor_translation = midpoint - working_distance * sensor_z
    return make_transform(np.column_stack((sensor_x, sensor_y, sensor_z)), sensor_translation)


def generate_candidates(
    estimate: CalibrationEstimate,
    *,
    edge_samples: int = 4,
    edge_margin: float = 0.04,
    alphas_deg: tuple[float, ...] = (20.0, 35.0, 50.0),
    psis_deg: tuple[float, ...] = (-15.0, 0.0, 15.0),
    working_distances: tuple[float, ...] = (0.4, 0.55, 0.7),
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
            )
        )
        serial += 1
    return candidates
