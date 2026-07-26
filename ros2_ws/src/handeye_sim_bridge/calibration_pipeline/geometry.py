"""Small, dependency-free SO(3)/SE(3) helpers."""

from __future__ import annotations

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + skew(vector)
    axis = vector / angle
    cross = skew(axis)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def so3_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ) / 2.0
    if np.pi - angle < 1e-6:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    return angle / (2.0 * np.sin(angle)) * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )


def rotation_x(angle_rad: float) -> np.ndarray:
    return so3_exp(np.array([angle_rad, 0.0, 0.0]))


def rotation_y(angle_rad: float) -> np.ndarray:
    return so3_exp(np.array([0.0, angle_rad, 0.0]))


def rotation_z(angle_rad: float) -> np.ndarray:
    return so3_exp(np.array([0.0, 0.0, angle_rad]))


def rpy_to_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Return ``Rx(rx) @ Ry(ry) @ Rz(rz)`` for angles expressed in degrees."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    return rotation_x(rx) @ rotation_y(ry) @ rotation_z(rz)


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:3, :3], dtype=float)
    translation = np.asarray(transform[:3, 3], dtype=float)
    return make_transform(rotation.T, -rotation.T @ translation)


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3]


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.rad2deg(np.linalg.norm(so3_log(np.asarray(first).T @ np.asarray(second)))))
