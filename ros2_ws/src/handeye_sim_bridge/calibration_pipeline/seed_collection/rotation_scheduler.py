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
