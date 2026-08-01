"""Minimal adaptive stop policy based on relative information saturation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StopPolicy:
    minimum_nbv_poses: int = 3
    maximum_total_poses: int = 20
    relative_information_gain_threshold: float = 0.05
    consecutive_low_gain_limit: int = 3
    _low_gain_count: int = field(default=0, init=False)
    _reference_information_gain: float | None = field(default=None, init=False)

    def evaluate(
        self,
        *,
        total_poses: int,
        nbv_poses: int,
        effective_rank: int,
        best_information_gain: float,
    ) -> tuple[bool, str]:
        if total_poses >= self.maximum_total_poses:
            return True, "emergency maximum pose protection reached"
        if (
            self._reference_information_gain is None
            and np.isfinite(best_information_gain)
            and best_information_gain > 0.0
        ):
            self._reference_information_gain = float(best_information_gain)
        relative_gain = float("inf")
        if self._reference_information_gain is not None:
            relative_gain = (
                float(best_information_gain) / self._reference_information_gain
            )
        if (
            relative_gain < self.relative_information_gain_threshold
        ):
            self._low_gain_count += 1
        else:
            self._low_gain_count = 0
        observable = (
            nbv_poses >= self.minimum_nbv_poses
            and effective_rank == 6
        )
        if (
            observable
            and self._low_gain_count >= self.consecutive_low_gain_limit
        ):
            return True, "information gain saturated"
        return False, "continue"
