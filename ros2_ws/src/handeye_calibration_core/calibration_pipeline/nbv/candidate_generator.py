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
    *,
    sensor_reference: np.ndarray | None = None,
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
    # Measurement direction: points TOWARD the board (laser travels from the
    # sensor window to the surface).  The Gocator sensor frame has Z along the
    # measurement axis and X along the laser line, so sensor_z = laser.
    laser_normal = -(
        np.cos(alpha) * board_normal + branch * np.sin(alpha) * tangent_normal
    )
    laser_normal /= np.linalg.norm(laser_normal)
    sensor_z = laser_normal
    if sensor_reference is None:
        # Free-standing fallback: X along the scan line, projected orthogonal.
        sensor_x_zero = line - sensor_z * float(sensor_z @ line)
        if np.linalg.norm(sensor_x_zero) < 1e-10:
            return None
        sensor_x_zero /= np.linalg.norm(sensor_x_zero)
        sensor_y_zero = np.cross(sensor_z, sensor_x_zero)
        sensor_y_zero /= np.linalg.norm(sensor_y_zero)
    else:
        # Use the calibrated sensor frame (handeye) so the x/y axes match the
        # physical mount: project the reference x/y onto the plane orthogonal
        # to the laser and rotate by psi about the laser axis.
        ref_x = np.asarray(sensor_reference[:3, 0], dtype=float)
        ref_y = np.asarray(sensor_reference[:3, 1], dtype=float)
        sensor_x_zero = ref_x - sensor_z * float(sensor_z @ ref_x)
        if np.linalg.norm(sensor_x_zero) < 1e-10:
            return None
        sensor_x_zero /= np.linalg.norm(sensor_x_zero)
        sensor_y_zero = ref_y - sensor_z * float(sensor_z @ ref_y)
        if np.linalg.norm(sensor_y_zero) < 1e-10:
            return None
        sensor_y_zero /= np.linalg.norm(sensor_y_zero)
        sensor_y_zero -= sensor_x_zero * float(sensor_x_zero @ sensor_y_zero)
        sensor_y_zero /= np.linalg.norm(sensor_y_zero)
    # psi rotation about the measurement axis (keeps the scan-line labels).
    sensor_x = np.cos(psi) * sensor_x_zero + np.sin(psi) * sensor_y_zero
    sensor_y = -np.sin(psi) * sensor_x_zero + np.cos(psi) * sensor_y_zero
    sensor_x /= np.linalg.norm(sensor_x)
    sensor_y /= np.linalg.norm(sensor_y)
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
    reference_sensor_transform: np.ndarray | None = None,
) -> list[Candidate]:
    board = estimate.board
    if edge_margin * 2.0 >= min(board.length_u, board.length_v):
        raise ValueError("edge_margin leaves no usable board edge")
    if reference_sensor_transform is not None:
        # Local NBV: sample scan positions around the current reference pose
        # instead of the full board grid.  The reference laser hit point in
        # board coordinates anchors the scan window.
        n_hat = np.asarray(board.normal, dtype=float)
        t_s = np.asarray(reference_sensor_transform[:3, 3], dtype=float)
        laser = np.asarray(reference_sensor_transform[:3, 2], dtype=float)
        d = -(n_hat @ (t_s - np.asarray(board.corner))) / (n_hat @ laser)
        hit = t_s + d * laser
        a_center = float((hit - np.asarray(board.corner)) @ np.asarray(board.u))
        b_center = float((hit - np.asarray(board.corner)) @ np.asarray(board.v))
        a_center = float(np.clip(a_center, edge_margin, board.length_u - edge_margin))
        b_center = float(np.clip(b_center, edge_margin, board.length_v - edge_margin))
        radius = 0.05
        a_values = np.linspace(max(edge_margin, a_center - radius),
                               min(board.length_u - edge_margin, a_center + radius),
                               edge_samples)
        b_values = np.linspace(max(edge_margin, b_center - radius),
                               min(board.length_v - edge_margin, b_center + radius),
                               edge_samples)
    else:
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
            sensor_reference=(
                reference_sensor_transform
                if reference_sensor_transform is not None
                else None
            ),
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
