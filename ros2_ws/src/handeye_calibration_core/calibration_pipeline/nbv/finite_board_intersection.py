"""Intersect a laser plane with all four edges of a finite rectangle."""

from __future__ import annotations

import numpy as np

from ..models import BoardModel


def _edge_intersection(
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
    edge_origin: np.ndarray,
    edge_direction: np.ndarray,
    edge_length: float,
) -> tuple[np.ndarray, float] | None:
    denominator = float(plane_normal @ edge_direction)
    if abs(denominator) < 1e-10:
        return None
    distance = float(plane_normal @ (plane_point - edge_origin) / denominator)
    if distance < -1e-8 or distance > edge_length + 1e-8:
        return None
    distance = float(np.clip(distance, 0.0, edge_length))
    return edge_origin + distance * edge_direction, distance


def intersect_finite_board(
    plane_normal: np.ndarray,
    plane_point: np.ndarray,
    board: BoardModel,
) -> list[tuple[str, np.ndarray, float]]:
    definitions = (
        ("u0", board.corner, board.u, board.length_u),
        ("uW", board.corner + board.length_v * board.v, board.u, board.length_u),
        ("v0", board.corner, board.v, board.length_v),
        ("vL", board.corner + board.length_u * board.u, board.v, board.length_v),
    )
    intersections: list[tuple[str, np.ndarray, float]] = []
    for label, origin, direction, length in definitions:
        result = _edge_intersection(plane_normal, plane_point, origin, direction, length)
        if result is None:
            continue
        point, coordinate = result
        duplicate = next(
            (index for index, (_, existing, _) in enumerate(intersections)
             if np.linalg.norm(point - existing) < 1e-7),
            None,
        )
        if duplicate is None:
            intersections.append((label, point, coordinate))
        else:
            # A corner has two labels. Prefer the two target edges at C.
            existing_label, _, _ = intersections[duplicate]
            if label in {"u0", "v0"} and existing_label not in {"u0", "v0"}:
                intersections[duplicate] = (label, point, coordinate)
    return intersections
