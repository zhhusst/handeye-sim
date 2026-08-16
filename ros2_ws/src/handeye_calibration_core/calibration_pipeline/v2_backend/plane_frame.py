"""Canonicalize the optimized plane frame without changing edge identities."""

from __future__ import annotations

from itertools import product

import numpy as np


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest right-handed rotation matrix in Frobenius norm."""
    left, _, right = np.linalg.svd(np.asarray(matrix, dtype=float))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def canonicalize_plane_frame(
    rotation: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    endpoint_u_offsets: np.ndarray | None = None,
    endpoint_v_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Fix column signs while preserving the ``u/e1`` and ``v/e2`` mapping.

    Only the four proper diagonal sign transformations are considered.  Axis
    permutations are deliberately forbidden because they would exchange the
    two physical edge labels.
    """
    rotation = project_to_rotation(rotation)
    reference_rotation = (
        None if reference is None else project_to_rotation(np.asarray(reference, dtype=float))
    )
    offsets_u = (
        np.empty((0, 3))
        if endpoint_u_offsets is None
        else np.asarray(endpoint_u_offsets, dtype=float).reshape(-1, 3)
    )
    offsets_v = (
        np.empty((0, 3))
        if endpoint_v_offsets is None
        else np.asarray(endpoint_v_offsets, dtype=float).reshape(-1, 3)
    )

    best: np.ndarray | None = None
    best_score = float("-inf")
    for sign_u, sign_v in product((-1.0, 1.0), repeat=2):
        signs = np.array([sign_u, sign_v, sign_u * sign_v])
        candidate = rotation @ np.diag(signs)
        score = 0.0
        if reference_rotation is not None:
            score += 10.0 * float(np.trace(reference_rotation.T @ candidate))
        if len(offsets_u):
            score += float(np.sum(candidate[:, 0] @ offsets_u.T > 0.0))
            score += float(np.median(candidate[:, 0] @ offsets_u.T))
        if len(offsets_v):
            score += float(np.sum(candidate[:, 1] @ offsets_v.T > 0.0))
            score += float(np.median(candidate[:, 1] @ offsets_v.T))
        if reference_rotation is None and not len(offsets_u) and not len(offsets_v):
            # Deterministic fallback: prefer positive alignment to base axes.
            score += float(candidate[0, 0] + candidate[1, 1] + candidate[2, 2])
        if score > best_score:
            best = candidate
            best_score = score
    assert best is not None
    return best
