"""Rotation-diversity checks used to accept seed observations."""

from __future__ import annotations

import numpy as np

from ..geometry import rotation_distance_deg


def rotation_diversity(rotations: list[np.ndarray]) -> dict[str, float | np.ndarray]:
    if len(rotations) < 2:
        return {
            "minimum_pairwise_deg": 0.0,
            "gram_eigenvalues": np.zeros(3),
            "minimum_gram_eigenvalue": 0.0,
        }
    gram = np.zeros((3, 3))
    distances: list[float] = []
    for first_index, first in enumerate(rotations):
        for second in rotations[first_index + 1 :]:
            delta = np.asarray(first) - np.asarray(second)
            gram += delta.T @ delta
            distances.append(rotation_distance_deg(first, second))
    eigenvalues = np.linalg.eigvalsh(gram)
    return {
        "minimum_pairwise_deg": float(min(distances)),
        "gram_eigenvalues": eigenvalues,
        "minimum_gram_eigenvalue": float(eigenvalues[0]),
    }
