"""Independent precision-sphere validation for a hand-eye estimate.

The sphere is an evaluation artefact only.  None of the functions in this
module updates the hand-eye transform.  A fixed transform maps every selected
profile point into the robot base frame, where all views must agree with one
stationary sphere of engraved radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SphereArtifact:
    """Metrology information engraved on one physical sphere."""

    artifact_id: str
    diameter_m: float
    roundness_m: float | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must not be empty")
        if self.diameter_m <= 0.0:
            raise ValueError("diameter_m must be positive")
        if self.roundness_m is not None and self.roundness_m < 0.0:
            raise ValueError("roundness_m must be non-negative")

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    def as_dict_mm(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "model": self.model,
            "engraved_diameter_mm": 1000.0 * self.diameter_m,
            "engraved_radius_mm": 1000.0 * self.radius_m,
            "engraved_roundness_mm": (
                None
                if self.roundness_m is None
                else 1000.0 * self.roundness_m
            ),
        }


@dataclass(frozen=True)
class SphereSegmentParameters:
    """Geometry-only selection limits for a sphere arc in one raw profile."""

    minimum_points: int = 16
    minimum_chord_m: float = 0.004
    maximum_arc_length_m: float = 0.12
    absolute_neighbor_gap_m: float = 0.0015
    neighbor_gap_multiplier: float = 6.0
    minimum_slice_radius_fraction: float = 0.25
    maximum_slice_radius_overrun_m: float = 0.0015
    maximum_circle_rms_m: float = 0.00030
    robust_scale_m: float = 0.00010

    def __post_init__(self) -> None:
        if self.minimum_points < 6:
            raise ValueError("minimum_points must be at least six")
        positive = (
            self.minimum_chord_m,
            self.maximum_arc_length_m,
            self.absolute_neighbor_gap_m,
            self.neighbor_gap_multiplier,
            self.minimum_slice_radius_fraction,
            self.maximum_circle_rms_m,
            self.robust_scale_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("sphere segment limits must be positive")
        if self.maximum_slice_radius_overrun_m < 0.0:
            raise ValueError("maximum_slice_radius_overrun_m must be non-negative")


@dataclass(frozen=True)
class CircleFit:
    center_xz_m: np.ndarray
    radius_m: float
    rms_m: float
    p95_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xz_m, dtype=float)
        if center.shape != (2,):
            raise ValueError("center_xz_m must have shape (2,)")
        object.__setattr__(self, "center_xz_m", center)


@dataclass(frozen=True)
class SegmentedSphereProfile:
    points_sensor_m: np.ndarray
    sample_indices: np.ndarray
    circle: CircleFit
    chord_m: float
    arc_length_m: float
    candidate_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_sensor_m, dtype=float)
        indices = np.asarray(self.sample_indices, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_sensor_m must have shape (N, 3)")
        if indices.shape != (len(points),):
            raise ValueError("sample_indices must match the selected points")
        object.__setattr__(self, "points_sensor_m", points)
        object.__setattr__(self, "sample_indices", indices)


@dataclass(frozen=True)
class SphereValidationThresholds:
    minimum_poses: int = 7
    maximum_fixed_radius_rmse_m: float = 0.00010
    maximum_fixed_radius_p95_m: float = 0.00020
    maximum_free_diameter_error_m: float = 0.00010

    def __post_init__(self) -> None:
        if self.minimum_poses < 4:
            raise ValueError("minimum_poses must be at least four")
        if min(
            self.maximum_fixed_radius_rmse_m,
            self.maximum_fixed_radius_p95_m,
            self.maximum_free_diameter_error_m,
        ) <= 0.0:
            raise ValueError("validation thresholds must be positive")


def _finite_profile(
    points_sensor_m: np.ndarray,
    sample_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_sensor_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("profile points must have shape (N, 3)")
    if sample_indices is None:
        indices = np.arange(len(points), dtype=np.int64)
    else:
        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.shape != (len(points),):
            raise ValueError("sample_indices must have one value per point")
    keep = np.all(np.isfinite(points), axis=1)
    points = points[keep]
    indices = indices[keep]
    if len(points) == 0:
        return points, indices
    order = np.argsort(indices, kind="stable")
    return points[order], indices[order]


def _circle_initial(points_xz: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack(
        (2.0 * points_xz[:, 0], 2.0 * points_xz[:, 1], np.ones(len(points_xz)))
    )
    target = np.sum(points_xz * points_xz, axis=1)
    state, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError("profile arc is geometrically degenerate")
    center = state[:2]
    radius_squared = float(state[2] + center @ center)
    if radius_squared <= 0.0 or not np.isfinite(radius_squared):
        raise ValueError("profile arc produced an invalid circle")
    return center, float(np.sqrt(radius_squared))


def fit_profile_circle(
    points_sensor_m: np.ndarray,
    *,
    robust_scale_m: float = 0.00010,
) -> CircleFit:
    """Fit a free 2-D circle to one X-Z profile arc using a robust loss."""
    points = np.asarray(points_sensor_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        raise ValueError("at least six 3-D profile points are required")
    points_xz = points[:, (0, 2)]
    center0, radius0 = _circle_initial(points_xz)

    def residual(state: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points_xz - state[:2], axis=1) - state[2]

    result = least_squares(
        residual,
        np.r_[center0, radius0],
        bounds=(np.array([-np.inf, -np.inf, 1e-9]), np.full(3, np.inf)),
        loss="soft_l1",
        f_scale=robust_scale_m,
        max_nfev=300,
    )
    values = residual(result.x)
    absolute = np.abs(values)
    return CircleFit(
        center_xz_m=result.x[:2],
        radius_m=float(result.x[2]),
        rms_m=float(np.sqrt(np.mean(values * values))),
        p95_m=float(np.percentile(absolute, 95.0)),
    )


def _contiguous_segments(
    points: np.ndarray,
    indices: np.ndarray,
    parameters: SphereSegmentParameters,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(points) < parameters.minimum_points:
        return []
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    adjacent_index = np.diff(indices) == 1
    nominal = distances[adjacent_index & np.isfinite(distances)]
    nominal_pitch = (
        float(np.median(nominal))
        if nominal.size
        else parameters.absolute_neighbor_gap_m
    )
    gap_limit = max(
        parameters.absolute_neighbor_gap_m,
        parameters.neighbor_gap_multiplier * nominal_pitch,
    )
    breaks = np.flatnonzero((~adjacent_index) | (distances > gap_limit)) + 1
    point_groups = np.split(points, breaks)
    index_groups = np.split(indices, breaks)
    return [
        (group, group_indices)
        for group, group_indices in zip(point_groups, index_groups)
        if len(group) >= parameters.minimum_points
    ]


def select_sphere_profile_segment(
    points_sensor_m: np.ndarray,
    artifact: SphereArtifact,
    *,
    sample_indices: np.ndarray | None = None,
    parameters: SphereSegmentParameters | None = None,
) -> SegmentedSphereProfile:
    """Select the most plausible sphere arc without using the hand-eye result.

    Selection occurs entirely in the sensor X-Z plane.  The engraved sphere
    radius is used only as a physical upper bound for the radius of a planar
    sphere slice; no base-frame residual participates in segment selection.
    """
    limits = parameters or SphereSegmentParameters()
    points, indices = _finite_profile(points_sensor_m, sample_indices)
    groups = _contiguous_segments(points, indices, limits)
    candidates: list[tuple[float, SegmentedSphereProfile]] = []
    for group, group_indices in groups:
        chord = float(np.linalg.norm(group[-1] - group[0]))
        arc_length = float(np.sum(np.linalg.norm(np.diff(group, axis=0), axis=1)))
        if chord < limits.minimum_chord_m or arc_length > limits.maximum_arc_length_m:
            continue
        try:
            circle = fit_profile_circle(group, robust_scale_m=limits.robust_scale_m)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if circle.rms_m > limits.maximum_circle_rms_m:
            continue
        if circle.radius_m < limits.minimum_slice_radius_fraction * artifact.radius_m:
            continue
        if circle.radius_m > artifact.radius_m + limits.maximum_slice_radius_overrun_m:
            continue
        # Prefer a long, well-supported, low-residual arc.  Radius proximity is
        # deliberately not part of the score because off-centre sphere slices
        # are physically smaller than the engraved radius.
        score = chord * np.sqrt(len(group)) / max(
            circle.rms_m, 0.25 * limits.robust_scale_m
        )
        candidates.append(
            (
                float(score),
                SegmentedSphereProfile(
                    group,
                    group_indices,
                    circle,
                    chord,
                    arc_length,
                    0,
                ),
            )
        )
    if not candidates:
        raise ValueError("no profile segment satisfies the precision-sphere geometry")
    selected = max(candidates, key=lambda item: item[0])[1]
    return SegmentedSphereProfile(
        selected.points_sensor_m,
        selected.sample_indices,
        selected.circle,
        selected.chord_m,
        selected.arc_length_m,
        len(candidates),
    )


def transform_profile_to_base(
    points_sensor_m: np.ndarray,
    flange_rotation: np.ndarray,
    flange_translation_m: np.ndarray,
    handeye_rotation: np.ndarray,
    handeye_translation_m: np.ndarray,
) -> np.ndarray:
    """Apply ``p_B = R_BF (R_FS p_S + t_FS) + t_BF``."""
    points = np.asarray(points_sensor_m, dtype=float)
    rotation_bf = np.asarray(flange_rotation, dtype=float)
    translation_bf = np.asarray(flange_translation_m, dtype=float)
    rotation_fs = np.asarray(handeye_rotation, dtype=float)
    translation_fs = np.asarray(handeye_translation_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_sensor_m must have shape (N, 3)")
    if rotation_bf.shape != (3, 3) or rotation_fs.shape != (3, 3):
        raise ValueError("flange and hand-eye rotations must have shape (3, 3)")
    if translation_bf.shape != (3,) or translation_fs.shape != (3,):
        raise ValueError("flange and hand-eye translations must have shape (3,)")
    points_flange = (rotation_fs @ points.T).T + translation_fs
    return (rotation_bf @ points_flange.T).T + translation_bf


def _linear_sphere_initial(points: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.column_stack((2.0 * points, np.ones(len(points))))
    target = np.sum(points * points, axis=1)
    state, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 4:
        raise ValueError("multi-view sphere points are geometrically degenerate")
    center = state[:3]
    radius_squared = float(state[3] + center @ center)
    if radius_squared <= 0.0 or not np.isfinite(radius_squared):
        raise ValueError("multi-view points produced an invalid sphere")
    return center, float(np.sqrt(radius_squared))


def _fixed_radius_center(
    points: np.ndarray,
    radius_m: float,
    robust_scale_m: float,
    initial_center: np.ndarray | None = None,
) -> np.ndarray:
    if initial_center is None:
        initial_center, _ = _linear_sphere_initial(points)

    def residual(center: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - center, axis=1) - radius_m

    result = least_squares(
        residual,
        np.asarray(initial_center, dtype=float),
        loss="soft_l1",
        f_scale=robust_scale_m,
        max_nfev=500,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError("fixed-radius sphere centre fit did not converge")
    return result.x


def _free_sphere(
    points: np.ndarray,
    robust_scale_m: float,
) -> tuple[np.ndarray, float]:
    center0, radius0 = _linear_sphere_initial(points)

    def residual(state: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - state[:3], axis=1) - state[3]

    result = least_squares(
        residual,
        np.r_[center0, radius0],
        bounds=(np.r_[np.full(3, -np.inf), 1e-9], np.full(4, np.inf)),
        loss="soft_l1",
        f_scale=robust_scale_m,
        max_nfev=500,
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError("free-radius sphere fit did not converge")
    return result.x[:3], float(result.x[3])


def _residual_summary_m(values: np.ndarray) -> dict:
    signed = np.asarray(values, dtype=float)
    absolute = np.abs(signed)
    return {
        "count": int(len(signed)),
        "signed_mean_m": float(np.mean(signed)),
        "rmse_m": float(np.sqrt(np.mean(signed * signed))),
        "median_abs_m": float(np.median(absolute)),
        "p95_abs_m": float(np.percentile(absolute, 95.0)),
        "maximum_abs_m": float(np.max(absolute)),
    }


def _summary_mm(summary_m: dict) -> dict:
    return {
        key[:-1] + "mm" if key.endswith("_m") else key: (
            1000.0 * value if key.endswith("_m") else value
        )
        for key, value in summary_m.items()
    }


def _direction_coverage(points: np.ndarray, center: np.ndarray) -> dict:
    directions = points - center
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-12] / norms[norms > 1e-12, None]
    gram = directions.T @ directions / max(len(directions), 1)
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "direction_gram_eigenvalues": eigenvalues.tolist(),
        "minimum_direction_gram_eigenvalue": float(eigenvalues[0]),
    }


def validate_sphere_views(
    pose_points_base_m: Sequence[np.ndarray],
    artifact: SphereArtifact,
    *,
    robust_scale_m: float = 0.00010,
    thresholds: SphereValidationThresholds | None = None,
    bootstrap_trials: int = 100,
    random_seed: int = 20260813,
) -> dict:
    """Evaluate one immutable hand-eye estimate on independent sphere views.

    The primary error is computed on every geometrically selected point.  A
    secondary MAD inlier summary is included for diagnosis, but it is never
    used for the pass/fail decision, so a bad robot view cannot be hidden by
    global outlier deletion.
    """
    limits = thresholds or SphereValidationThresholds()
    groups = [np.asarray(group, dtype=float) for group in pose_points_base_m]
    if len(groups) < 4:
        raise ValueError("at least four sphere views are required for evaluation")
    if any(group.ndim != 2 or group.shape[1] != 3 or len(group) < 6 for group in groups):
        raise ValueError("each sphere view must contain at least six 3-D points")
    points = np.vstack(groups)
    fixed_center = _fixed_radius_center(
        points, artifact.radius_m, robust_scale_m
    )
    fixed_residual = np.linalg.norm(points - fixed_center, axis=1) - artifact.radius_m
    primary_m = _residual_summary_m(fixed_residual)

    median = float(np.median(fixed_residual))
    mad = float(np.median(np.abs(fixed_residual - median)))
    robust_sigma = max(1.4826 * mad, 0.25 * robust_scale_m)
    inlier_mask = np.abs(fixed_residual - median) <= 3.5 * robust_sigma
    secondary_m = _residual_summary_m(fixed_residual[inlier_mask])

    free_center, free_radius = _free_sphere(points, robust_scale_m)
    pose_metrics = []
    offset = 0
    for pose_index, group in enumerate(groups):
        count = len(group)
        values = fixed_residual[offset : offset + count]
        pose_metrics.append(
            {
                "pose_index": pose_index + 1,
                **_summary_mm(_residual_summary_m(values)),
            }
        )
        offset += count

    leave_one_out_centers = []
    leave_one_out_rmse = []
    if len(groups) >= 5:
        for omitted in range(len(groups)):
            training = np.vstack(
                [group for index, group in enumerate(groups) if index != omitted]
            )
            center = _fixed_radius_center(
                training,
                artifact.radius_m,
                robust_scale_m,
                initial_center=fixed_center,
            )
            held_out = (
                np.linalg.norm(groups[omitted] - center, axis=1)
                - artifact.radius_m
            )
            leave_one_out_centers.append(center)
            leave_one_out_rmse.append(float(np.sqrt(np.mean(held_out * held_out))))

    bootstrap_centers = []
    bootstrap_radii = []
    if bootstrap_trials > 0:
        rng = np.random.default_rng(random_seed)
        for _ in range(bootstrap_trials):
            selection = rng.integers(0, len(groups), size=len(groups))
            sampled = np.vstack([groups[int(index)] for index in selection])
            try:
                center, radius = _free_sphere(sampled, robust_scale_m)
            except (ValueError, np.linalg.LinAlgError):
                continue
            bootstrap_centers.append(center)
            bootstrap_radii.append(radius)

    free_diameter_error_m = 2.0 * (free_radius - artifact.radius_m)
    checks = {
        "minimum_pose_count": len(groups) >= limits.minimum_poses,
        "fixed_radius_rmse": (
            primary_m["rmse_m"] <= limits.maximum_fixed_radius_rmse_m
        ),
        "fixed_radius_p95": (
            primary_m["p95_abs_m"] <= limits.maximum_fixed_radius_p95_m
        ),
        "free_diameter_error": (
            abs(free_diameter_error_m)
            <= limits.maximum_free_diameter_error_m
        ),
    }
    return {
        "schema_version": 1,
        "artifact": artifact.as_dict_mm(),
        "pose_count": len(groups),
        "point_count": len(points),
        "fixed_radius": {
            "center_base_m": fixed_center.tolist(),
            "all_points": _summary_mm(primary_m),
            "robust_inliers_diagnostic_only": {
                **_summary_mm(secondary_m),
                "inlier_fraction": float(np.mean(inlier_mask)),
            },
        },
        "free_radius_diagnostic": {
            "center_base_m": free_center.tolist(),
            "fitted_radius_mm": 1000.0 * free_radius,
            "fitted_diameter_mm": 2000.0 * free_radius,
            "diameter_error_mm": 1000.0 * free_diameter_error_m,
        },
        "per_pose": pose_metrics,
        "view_coverage": _direction_coverage(points, fixed_center),
        "leave_one_pose_out": {
            "center_spread_rms_mm": (
                None
                if not leave_one_out_centers
                else 1000.0
                * float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (
                                    np.asarray(leave_one_out_centers)
                                    - np.mean(leave_one_out_centers, axis=0)
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                )
            ),
            "held_out_rmse_mm": [1000.0 * value for value in leave_one_out_rmse],
        },
        "pose_bootstrap": {
            "requested_trials": int(bootstrap_trials),
            "successful_trials": len(bootstrap_radii),
            "center_axis_std_mm": (
                None
                if not bootstrap_centers
                else (1000.0 * np.std(bootstrap_centers, axis=0, ddof=0)).tolist()
            ),
            "diameter_std_mm": (
                None
                if not bootstrap_radii
                else 2000.0 * float(np.std(bootstrap_radii, ddof=0))
            ),
        },
        "thresholds": {
            "minimum_poses": limits.minimum_poses,
            "maximum_fixed_radius_rmse_mm": (
                1000.0 * limits.maximum_fixed_radius_rmse_m
            ),
            "maximum_fixed_radius_p95_mm": (
                1000.0 * limits.maximum_fixed_radius_p95_m
            ),
            "maximum_free_diameter_error_mm": (
                1000.0 * limits.maximum_free_diameter_error_m
            ),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "interpretation": {
            "primary_metric_uses_all_selected_points": True,
            "sphere_does_not_update_handeye": True,
            "roundness_is_not_diameter_uncertainty": True,
        },
    }
