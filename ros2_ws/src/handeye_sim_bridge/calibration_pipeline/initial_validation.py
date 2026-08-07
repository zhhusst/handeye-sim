"""Truth-independent stability checks for the multi-frame seed initialization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .dataset_io import SeedObservationGroup, aggregate_seed_group
from .geometry import rotation_distance_deg
from .models import CalibrationResult


@dataclass(frozen=True)
class InitialStabilityReport:
    available: bool
    accepted: bool
    trials_requested: int
    trials_converged: int
    rotation_p95_deg: float
    translation_p95_m: float
    rotation_max_deg: float
    translation_max_m: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "available": bool(self.available),
            "accepted": bool(self.accepted),
            "trials_requested": int(self.trials_requested),
            "trials_converged": int(self.trials_converged),
            "rotation_p95_deg": float(self.rotation_p95_deg),
            "translation_p95_mm": float(1000.0 * self.translation_p95_m),
            "rotation_max_deg": float(self.rotation_max_deg),
            "translation_max_mm": float(1000.0 * self.translation_max_m),
            "reason": self.reason,
        }


def inflate_handeye_covariance_from_stability(
    result: CalibrationResult,
    report: InitialStabilityReport,
    *,
    p95_normal_quantile: float = 1.96,
) -> CalibrationResult:
    """Keep first-NBV uncertainty from being overconfident after bootstrapping."""
    covariance = result.estimate.covariance_x9
    if (
        not report.available
        or covariance is None
        or p95_normal_quantile <= 0.0
    ):
        return result
    inflated = np.asarray(covariance, dtype=float).copy()
    rotation_variance = (
        np.deg2rad(report.rotation_p95_deg) / p95_normal_quantile
    ) ** 2
    translation_variance = (
        report.translation_p95_m / p95_normal_quantile
    ) ** 2
    for index in range(3):
        inflated[index, index] = max(
            inflated[index, index], rotation_variance
        )
    for index in range(3, 6):
        inflated[index, index] = max(
            inflated[index, index], translation_variance
        )
    covariance_state = result.estimate.covariance_state
    inflated_state = None
    if covariance_state is not None:
        inflated_state = np.asarray(covariance_state, dtype=float).copy()
        inflated_state[:9, :9] = inflated
        for index in range(3):
            inflated_state[index, index] = max(
                inflated_state[index, index], rotation_variance
            )
        for index in range(3, 6):
            inflated_state[index, index] = max(
                inflated_state[index, index], translation_variance
            )
    estimate = replace(
        result.estimate,
        covariance_x9=inflated,
        covariance_state=inflated_state,
    )
    return replace(result, estimate=estimate)


def bootstrap_initial_stability(
    groups: tuple[SeedObservationGroup, ...] | list[SeedObservationGroup],
    solver,
    reference: CalibrationResult,
    *,
    board_dimensions: tuple[float, float],
    trials: int = 6,
    random_seed: int = 20260728,
    maximum_rotation_p95_deg: float = 1.0,
    maximum_translation_p95_m: float = 0.005,
    minimum_converged_fraction: float = 0.8,
) -> InitialStabilityReport:
    """Bootstrap frames within each fixed physical seed pose.

    The spread is measured around the full-batch solution and never uses
    simulation truth.  It detects a seed solution whose value depends too
    strongly on the particular endpoint frames that happened to be sampled.
    """
    if trials < 2:
        raise ValueError("bootstrap trials must be at least two")
    if not 0.0 < minimum_converged_fraction <= 1.0:
        raise ValueError("minimum_converged_fraction must be in (0, 1]")
    if maximum_rotation_p95_deg <= 0.0 or maximum_translation_p95_m <= 0.0:
        raise ValueError("bootstrap stability limits must be positive")
    if not groups or min(len(group.poses) for group in groups) < 2:
        return InitialStabilityReport(
            available=False,
            accepted=True,
            trials_requested=trials,
            trials_converged=0,
            rotation_p95_deg=float("nan"),
            translation_p95_m=float("nan"),
            rotation_max_deg=float("nan"),
            translation_max_m=float("nan"),
            reason="legacy or single-frame seeds: bootstrap unavailable",
        )

    rng = np.random.default_rng(random_seed)
    rotation_changes: list[float] = []
    translation_changes: list[float] = []
    for _ in range(trials):
        sampled_poses = []
        sampled_measurements = []
        for group in groups:
            indices = rng.integers(0, len(group.poses), len(group.poses))
            sampled_group = SeedObservationGroup(
                group.label,
                tuple(group.poses[int(index)] for index in indices),
                tuple(
                    group.measurements[int(index)] for index in indices
                ),
            )
            pose, measurement = aggregate_seed_group(sampled_group)
            sampled_poses.append(pose)
            sampled_measurements.append(measurement)
        try:
            trial = solver.solve(
                sampled_poses,
                sampled_measurements,
                reference.estimate.handeye_rotation,
                reference.estimate.handeye_translation,
                board_dimensions=board_dimensions,
                initial_board_rotation=reference.estimate.board.rotation,
                initial_estimate=reference.estimate,
            )
        except Exception:
            continue
        if not trial.converged:
            continue
        rotation_changes.append(
            rotation_distance_deg(
                reference.estimate.handeye_rotation,
                trial.estimate.handeye_rotation,
            )
        )
        translation_changes.append(
            float(
                np.linalg.norm(
                    reference.estimate.handeye_translation
                    - trial.estimate.handeye_translation
                )
            )
        )

    converged = len(rotation_changes)
    required = int(np.ceil(trials * minimum_converged_fraction))
    if converged == 0:
        return InitialStabilityReport(
            available=True,
            accepted=False,
            trials_requested=trials,
            trials_converged=0,
            rotation_p95_deg=float("inf"),
            translation_p95_m=float("inf"),
            rotation_max_deg=float("inf"),
            translation_max_m=float("inf"),
            reason="no bootstrap solve converged",
        )
    rotation_p95 = float(np.percentile(rotation_changes, 95.0))
    translation_p95 = float(np.percentile(translation_changes, 95.0))
    accepted = (
        converged >= required
        and rotation_p95 <= maximum_rotation_p95_deg
        and translation_p95 <= maximum_translation_p95_m
    )
    reasons = []
    if converged < required:
        reasons.append(f"only {converged}/{trials} bootstrap solves converged")
    if rotation_p95 > maximum_rotation_p95_deg:
        reasons.append(
            f"rotation p95 {rotation_p95:.4f} deg exceeds "
            f"{maximum_rotation_p95_deg:.4f} deg"
        )
    if translation_p95 > maximum_translation_p95_m:
        reasons.append(
            f"translation p95 {1000.0 * translation_p95:.4f} mm exceeds "
            f"{1000.0 * maximum_translation_p95_m:.4f} mm"
        )
    return InitialStabilityReport(
        available=True,
        accepted=accepted,
        trials_requested=trials,
        trials_converged=converged,
        rotation_p95_deg=rotation_p95,
        translation_p95_m=translation_p95,
        rotation_max_deg=float(np.max(rotation_changes)),
        translation_max_m=float(np.max(translation_changes)),
        reason="accepted" if accepted else "; ".join(reasons),
    )
