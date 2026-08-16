"""Pure geometry and safety checks for small PC_TRACK_ALL calibration moves."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fanuc_m20id25_support.fanuc_kinematic import (
    JOINT_LIMITS_DEG,
    forward_kinematics_urdf,
    inverse_kinematics_numeric,
)
from fanuc_m20id25_support.fanuc_transforms import matrix_to_pose_fanuc


def rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    return float(
        math.acos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    )


def base_flange_to_controller_pose(
    base_from_flange: np.ndarray,
    controller_origin_in_base_mm,
) -> np.ndarray:
    """Convert base_link->flange to UF1 Cartesian XYZ/WPR used by PR[10]."""
    base_from_controller = np.eye(4)
    base_from_controller[:3, 3] = (
        np.asarray(controller_origin_in_base_mm, dtype=float) / 1000.0
    )
    controller_from_flange = (
        np.linalg.inv(base_from_controller) @ np.asarray(base_from_flange)
    )
    controller_from_flange_mm = controller_from_flange.copy()
    controller_from_flange_mm[:3, 3] *= 1000.0
    return matrix_to_pose_fanuc(controller_from_flange_mm)


@dataclass(frozen=True)
class SmallMovePlan:
    target_joints_rad: np.ndarray
    target_pose_xyz_wpr: np.ndarray
    maximum_joint_step_deg: float
    joint_distance_rad: float
    translation_mm: float
    rotation_deg: float
    minimum_joint_margin_deg: float
    cartesian_path_samples: int

    def as_dict(self) -> dict:
        return {
            "target_joints_deg": [
                float(value) for value in np.rad2deg(self.target_joints_rad)
            ],
            "target_pose_xyz_wpr": [
                float(value) for value in self.target_pose_xyz_wpr
            ],
            "maximum_joint_step_deg": self.maximum_joint_step_deg,
            "joint_distance_rad": self.joint_distance_rad,
            "translation_mm": self.translation_mm,
            "rotation_deg": self.rotation_deg,
            "minimum_joint_margin_deg": self.minimum_joint_margin_deg,
            "cartesian_path_samples": self.cartesian_path_samples,
        }


def _interpolate_transform(first: np.ndarray, second: np.ndarray, fraction: float):
    transform = np.eye(4)
    transform[:3, 3] = (
        (1.0 - fraction) * first[:3, 3] + fraction * second[:3, 3]
    )
    rotations = Rotation.from_matrix(
        np.stack([first[:3, :3], second[:3, :3]], axis=0)
    )
    transform[:3, :3] = Slerp([0.0, 1.0], rotations)([fraction]).as_matrix()[0]
    return transform


def plan_small_linear_move(
    current_joints_rad,
    target_joints_rad,
    *,
    controller_origin_in_base_mm=(0.0, 0.0, 425.0),
    maximum_joint_step_deg=6.0,
    maximum_joint_distance_rad=0.20,
    maximum_translation_mm=80.0,
    maximum_rotation_deg=10.0,
    minimum_joint_margin_deg=3.0,
    cartesian_path_samples=5,
    check_cartesian_ik=True,
) -> SmallMovePlan:
    """Validate one small target and the linear Cartesian path TP will execute."""
    current = np.asarray(current_joints_rad, dtype=float)
    target = np.asarray(target_joints_rad, dtype=float)
    if current.shape != (6,) or target.shape != (6,):
        raise ValueError("current and target joints must each contain six values")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise ValueError("joint values must be finite")
    limits = np.deg2rad(np.asarray(JOINT_LIMITS_DEG, dtype=float))
    if np.any(target <= limits[:, 0]) or np.any(target >= limits[:, 1]):
        raise ValueError("target exceeds FANUC joint limits")
    margin_deg = float(
        np.min(
            np.rad2deg(
                np.minimum(target - limits[:, 0], limits[:, 1] - target)
            )
        )
    )
    if margin_deg < float(minimum_joint_margin_deg):
        raise ValueError(
            f"target joint margin {margin_deg:.3f} deg is below "
            f"{minimum_joint_margin_deg:.3f} deg"
        )
    delta = target - current
    maximum_step_deg = float(np.max(np.abs(np.rad2deg(delta))))
    joint_distance = float(np.linalg.norm(delta))
    if maximum_step_deg > float(maximum_joint_step_deg):
        raise ValueError(
            f"maximum joint step {maximum_step_deg:.3f} deg exceeds "
            f"{maximum_joint_step_deg:.3f} deg"
        )
    if joint_distance > float(maximum_joint_distance_rad):
        raise ValueError(
            f"joint distance {joint_distance:.4f} rad exceeds "
            f"{maximum_joint_distance_rad:.4f} rad"
        )

    first = forward_kinematics_urdf(current)
    second = forward_kinematics_urdf(target)
    translation_mm = float(1000.0 * np.linalg.norm(second[:3, 3] - first[:3, 3]))
    rotation_deg = float(np.rad2deg(rotation_distance_rad(first, second)))
    if translation_mm > float(maximum_translation_mm):
        raise ValueError(
            f"Cartesian translation {translation_mm:.3f} mm exceeds "
            f"{maximum_translation_mm:.3f} mm"
        )
    if rotation_deg > float(maximum_rotation_deg):
        raise ValueError(
            f"Cartesian rotation {rotation_deg:.3f} deg exceeds "
            f"{maximum_rotation_deg:.3f} deg"
        )

    samples = max(2, int(cartesian_path_samples))
    if check_cartesian_ik:
        seed = current.copy()
        for fraction in np.linspace(0.0, 1.0, samples + 1)[1:]:
            sample = _interpolate_transform(first, second, float(fraction))
            solutions = inverse_kinematics_numeric(
                sample,
                q_init=seed,
                max_iter=80,
                tol_p=2.0e-5,
                tol_r=2.0e-4,
            )
            if len(solutions) == 0:
                raise ValueError(
                    f"linear Cartesian path has no local IK at {fraction:.2f}"
                )
            seed = np.asarray(solutions[0], dtype=float)
            if np.any(seed <= limits[:, 0]) or np.any(seed >= limits[:, 1]):
                raise ValueError(
                    f"linear Cartesian path exceeds a joint limit at {fraction:.2f}"
                )

    target_pose = base_flange_to_controller_pose(
        second, controller_origin_in_base_mm
    )
    return SmallMovePlan(
        target_joints_rad=target.copy(),
        target_pose_xyz_wpr=target_pose,
        maximum_joint_step_deg=maximum_step_deg,
        joint_distance_rad=joint_distance,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        minimum_joint_margin_deg=margin_deg,
        cartesian_path_samples=samples,
    )
