"""Adaptive stop policy from method document section 14."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StopPolicy:
    minimum_nbv_poses: int = 3
    maximum_total_poses: int = 20
    information_gain_threshold: float = 1e-3
    consecutive_low_gain_limit: int = 3
    minimum_effective_eigenvalue: float = 1e-6
    _low_gain_count: int = field(default=0, init=False)

    def evaluate(
        self,
        *,
        total_poses: int,
        nbv_poses: int,
        effective_rank: int,
        best_information_gain: float,
        minimum_effective_eigenvalue: float,
    ) -> tuple[bool, str]:
        if total_poses >= self.maximum_total_poses:
            return True, "maximum pose protection limit reached"
        if best_information_gain < self.information_gain_threshold:
            self._low_gain_count += 1
        else:
            self._low_gain_count = 0
        ready = (
            nbv_poses >= self.minimum_nbv_poses
            and effective_rank == 6
            and minimum_effective_eigenvalue >= self.minimum_effective_eigenvalue
            and self._low_gain_count >= self.consecutive_low_gain_limit
        )
        return (True, "information gain saturated") if ready else (False, "continue")
