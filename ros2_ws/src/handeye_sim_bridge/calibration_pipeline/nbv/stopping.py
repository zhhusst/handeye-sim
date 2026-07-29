"""Adaptive stop policy from method document section 14."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StopPolicy:
    minimum_nbv_poses: int = 3
    maximum_total_poses: int = 20
    information_gain_threshold: float = 1e-3
    consecutive_low_gain_limit: int = 3
    minimum_effective_eigenvalue: float = 1e-6
    maximum_rotation_std_deg: float = 0.1
    maximum_translation_std_m: float = 0.001
    _low_gain_count: int = field(default=0, init=False)

    def evaluate(
        self,
        *,
        total_poses: int,
        nbv_poses: int,
        effective_rank: int,
        best_information_gain: float,
        minimum_effective_eigenvalue: float,
        handeye_covariance: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        if total_poses >= self.maximum_total_poses:
            return True, "maximum pose protection limit reached"
        if best_information_gain < self.information_gain_threshold:
            self._low_gain_count += 1
        else:
            self._low_gain_count = 0
        covariance_ready = False
        if handeye_covariance is not None:
            covariance = np.asarray(handeye_covariance, dtype=float)
            if covariance.shape != (6, 6):
                raise ValueError("handeye_covariance must have shape (6, 6)")
            standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            covariance_ready = bool(
                np.all(
                    np.rad2deg(standard_deviations[:3])
                    <= self.maximum_rotation_std_deg
                )
                and np.all(
                    standard_deviations[3:6] <= self.maximum_translation_std_m
                )
            )
        observable = (
            nbv_poses >= self.minimum_nbv_poses
            and effective_rank == 6
            and minimum_effective_eigenvalue >= self.minimum_effective_eigenvalue
        )
        if observable and covariance_ready:
            return True, "hand-eye uncertainty target reached"
        if (
            observable
            and self._low_gain_count >= self.consecutive_low_gain_limit
        ):
            return True, "information gain saturated"
        return False, "continue"
