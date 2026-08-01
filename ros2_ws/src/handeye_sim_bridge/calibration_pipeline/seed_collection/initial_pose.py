"""Calibration-free qualification of the Phase-0b reference pose.

The checks intentionally use only the measured bilateral profile, flange
encoder joints and local robot kinematics.  No hand-eye estimate is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import BilateralFeature


@dataclass(frozen=True)
class InitialPoseCriteria:
    maximum_abs_x_mid_m: float = 0.03
    minimum_z_mid_m: float = 0.30
    maximum_z_mid_m: float = 0.55
    minimum_domain_margin_m: float = 0.020
    minimum_profile_length_m: float = 0.05
    maximum_profile_length_m: float = 0.25
    minimum_absolute_endpoint_depth_delta_m: float = 0.015
    minimum_normalized_joint_margin: float = 0.05
    minimum_local_ik_directions: int = 3


@dataclass(frozen=True)
class InitialPoseAssessment:
    accepted: bool
    reasons: tuple[str, ...]
    x_mid_m: float
    z_mid_m: float
    domain_margin_m: float
    profile_length_m: float
    endpoint_depth_delta_m: float
    absolute_endpoint_depth_delta_m: float
    normalized_joint_margin: float
    local_ik_directions: int


def normalized_joint_limit_margin(
    joints: np.ndarray, joint_limits: np.ndarray
) -> float:
    """Return the smallest distance to a joint limit, normalized by its span."""
    joints = np.asarray(joints, dtype=float)
    limits = np.asarray(joint_limits, dtype=float)
    if joints.shape != (6,) or limits.shape != (6, 2):
        raise ValueError("expected six joints and a (6, 2) joint-limit array")
    spans = limits[:, 1] - limits[:, 0]
    if np.any(spans <= 0.0):
        raise ValueError("joint-limit spans must be positive")
    margins = np.minimum(joints - limits[:, 0], limits[:, 1] - joints) / spans
    return float(np.min(margins))


def assess_initial_pose(
    feature: BilateralFeature,
    joints: np.ndarray,
    joint_limits: np.ndarray,
    *,
    local_ik_directions: int,
    criteria: InitialPoseCriteria | None = None,
) -> InitialPoseAssessment:
    """Evaluate a conservative sufficient operating envelope for Phase 0b."""
    criteria = criteria or InitialPoseCriteria()
    endpoint_depth_delta = float(
        feature.endpoint_v[2] - feature.endpoint_u[2]
    )
    joint_margin = normalized_joint_limit_margin(joints, joint_limits)
    reasons: list[str] = []
    if not feature.safe:
        reasons.append("bilateral_not_safe")
    if abs(feature.x_mid) > criteria.maximum_abs_x_mid_m:
        reasons.append("x_mid")
    if not (
        criteria.minimum_z_mid_m
        <= feature.z_mid
        <= criteria.maximum_z_mid_m
    ):
        reasons.append("z_mid")
    if feature.domain_margin < criteria.minimum_domain_margin_m:
        reasons.append("domain_margin")
    if not (
        criteria.minimum_profile_length_m
        <= feature.profile_length
        <= criteria.maximum_profile_length_m
    ):
        reasons.append("profile_length")
    if (
        abs(endpoint_depth_delta)
        < criteria.minimum_absolute_endpoint_depth_delta_m
    ):
        reasons.append("absolute_endpoint_depth_delta")
    if joint_margin < criteria.minimum_normalized_joint_margin:
        reasons.append("joint_margin")
    if local_ik_directions < criteria.minimum_local_ik_directions:
        reasons.append("local_ik")
    return InitialPoseAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        x_mid_m=float(feature.x_mid),
        z_mid_m=float(feature.z_mid),
        domain_margin_m=float(feature.domain_margin),
        profile_length_m=float(feature.profile_length),
        endpoint_depth_delta_m=endpoint_depth_delta,
        absolute_endpoint_depth_delta_m=abs(endpoint_depth_delta),
        normalized_joint_margin=joint_margin,
        local_ik_directions=int(local_ik_directions),
    )


def seed_feature_is_acceptable(
    feature: BilateralFeature,
    *,
    maximum_abs_x_mid_m: float,
    minimum_domain_margin_m: float = 0.0,
) -> bool:
    """Apply the v5 stability conditions before accepting full or partial seeds."""
    return bool(
        feature.safe
        and abs(feature.x_mid) <= maximum_abs_x_mid_m
        and feature.domain_margin >= minimum_domain_margin_m
    )


def local_preflight_is_acceptable(
    direction_results: list[dict],
    *,
    minimum_feasible_directions: int = 2,
) -> bool:
    """Require measured safe reserve in enough signed X/Y neighborhoods."""
    accepted = [item for item in direction_results if item["accepted"]]
    axes = {int(item["axis"]) for item in accepted}
    return bool(
        len(accepted) >= minimum_feasible_directions and axes == {0, 1}
    )


def dynamic_preflight_decision(
    mode: str,
    assessment: InitialPoseAssessment,
    *,
    auto_skip_maximum_abs_x_mid_m: float,
    auto_skip_minimum_domain_margin_m: float,
) -> tuple[bool, str]:
    """Decide whether the optional measured preflight should run.

    ``auto`` skips the four standalone trial motions only when the static
    observation has ample image-domain reserve and all four local IK probes
    are feasible.  This decision never disables the bilateral checks and
    rollback used during the real seed motions.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "always", "off"}:
        raise ValueError("preflight mode must be auto, always or off")
    if auto_skip_maximum_abs_x_mid_m < 0.0:
        raise ValueError("auto preflight x-mid threshold must be non-negative")
    if auto_skip_minimum_domain_margin_m < 0.0:
        raise ValueError("auto preflight margin threshold must be non-negative")
    if normalized_mode == "always":
        return True, "configured_always"
    if normalized_mode == "off":
        return False, "configured_off"

    reasons: list[str] = []
    if abs(assessment.x_mid_m) > auto_skip_maximum_abs_x_mid_m:
        reasons.append("x_mid_not_centered")
    if assessment.domain_margin_m < auto_skip_minimum_domain_margin_m:
        reasons.append("domain_margin_not_wide")
    if assessment.local_ik_directions < 4:
        reasons.append("local_ik_not_four_of_four")
    if reasons:
        return True, ",".join(reasons)
    return False, "static_reserve_sufficient"
