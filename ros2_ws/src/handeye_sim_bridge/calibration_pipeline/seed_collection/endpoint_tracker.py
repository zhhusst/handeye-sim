"""Temporal e_u/e_v identity tracking from method document section 5.7."""

from __future__ import annotations

import numpy as np


class EndpointTracker:
    def __init__(self, ambiguity_ratio: float = 0.05) -> None:
        self.ambiguity_ratio = ambiguity_ratio
        self._previous: tuple[np.ndarray, np.ndarray] | None = None

    def reset(self, endpoint_u: np.ndarray, endpoint_v: np.ndarray) -> None:
        self._previous = (np.asarray(endpoint_u, dtype=float), np.asarray(endpoint_v, dtype=float))

    def match(
        self, first: np.ndarray, second: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        if self._previous is None:
            self.reset(first, second)
            return first, second
        previous_u, previous_v = self._previous
        direct = float(np.sum((first - previous_u) ** 2) + np.sum((second - previous_v) ** 2))
        swapped = float(np.sum((second - previous_u) ** 2) + np.sum((first - previous_v) ** 2))
        scale = max(direct, swapped, 1e-12)
        if abs(direct - swapped) / scale < self.ambiguity_ratio:
            return None
        matched = (first, second) if direct < swapped else (second, first)
        self._previous = matched
        return matched
