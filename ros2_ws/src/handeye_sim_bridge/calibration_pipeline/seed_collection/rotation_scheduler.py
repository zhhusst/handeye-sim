"""Star-shaped local-flange rotation plan from method section 5.5."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RotationTarget:
    name: str
    stages: tuple[tuple[int, int], ...]


def star_rotation_plan() -> tuple[RotationTarget, ...]:
    """Return targets after the reference pose; axes are X=0 and Y=1."""
    return (
        RotationTarget("rx_positive", ((0, 1),)),
        RotationTarget("rx_negative", ((0, -1),)),
        RotationTarget("ry_positive", ((1, 1),)),
        RotationTarget("ry_negative", ((1, -1),)),
        RotationTarget("rx_ry_positive", ((0, 1), (1, 1))),
    )


def adaptive_rotation_plan() -> tuple[RotationTarget, ...]:
    """Return the default star followed by non-parallel fallback branches.

    The fallback set changes signs and stage order.  It is used only when a
    default branch is unobservable or rejected by the rotation-diversity test.
    All commands remain local-flange rotations and therefore do not require a
    hand-eye estimate.
    """
    return star_rotation_plan() + (
        RotationTarget("rx_ry_opposite", ((0, 1), (1, -1))),
        RotationTarget("rx_negative_ry_positive", ((0, -1), (1, 1))),
        RotationTarget("rx_ry_negative", ((0, -1), (1, -1))),
        RotationTarget("ry_rx_positive", ((1, 1), (0, 1))),
        RotationTarget("ry_positive_rx_negative", ((1, 1), (0, -1))),
        RotationTarget("ry_negative_rx_positive", ((1, -1), (0, 1))),
        RotationTarget("ry_rx_negative", ((1, -1), (0, -1))),
    )
