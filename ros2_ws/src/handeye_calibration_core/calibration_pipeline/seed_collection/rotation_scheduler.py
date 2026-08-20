"""Star-shaped local-flange rotation plan from method section 5.5."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RotationTarget:
    name: str
    stages: tuple[tuple[tuple[int, int], ...], ...]
    # Each stage is a SET of (axis, sign) pairs rotated TOGETHER in one
    # micro step (single-axis stages are the common case; multi-axis
    # stages rotate X and Y simultaneously with no intermediate pose).
    angle_scale: float = 1.0


def star_rotation_plan() -> tuple[RotationTarget, ...]:
    """Return targets after the reference pose; axes are X=0 and Y=1.

    Order is deliberately Y-positive / Y-negative FIRST: on the real rig,
    positive local-Y rotation ("camera pitch-up") is the motion that most
    often disturbs the ROI trackers (the laser chord runs toward the plate
    diagonal and the breakpoints separate fastest).  Running those targets
    early surfaces any tracking problem before the easier single-axis X
    branches, so a failure cannot eat the whole tail of the batch.
    """
    return (
        RotationTarget("ry_positive", ((1, 1),)),
        RotationTarget("ry_negative", ((1, -1),)),
        RotationTarget("rx_positive", ((0, 1),)),
        RotationTarget("rx_negative", ((0, -1),)),
        RotationTarget("rx_ry_positive", (((0, 1), (1, 1)),)),
    )


def adaptive_rotation_plan() -> tuple[RotationTarget, ...]:
    """Return the default star followed by non-parallel fallback branches.

    Simultaneous X/Y stages are unordered: ``(+X, -Y)`` and ``(-Y, +X)``
    produce the same rotation vector.  The fallback set therefore contains
    each signed diagonal once, followed by lower-amplitude axial poses that
    are genuinely distinct under the rotation-diversity threshold.  All
    commands remain local-flange rotations and therefore do not require a
    hand-eye estimate.
    """
    return star_rotation_plan() + (
        RotationTarget("rx_ry_opposite", (((0, 1), (1, -1)),)),
        RotationTarget("rx_negative_ry_positive", (((0, -1), (1, 1)),)),
        RotationTarget("rx_ry_negative", (((0, -1), (1, -1)),)),
        RotationTarget("ry_positive_half", ((1, 1),), 0.5),
        RotationTarget("ry_negative_half", ((1, -1),), 0.5),
        RotationTarget("rx_positive_half", ((0, 1),), 0.5),
        RotationTarget("rx_negative_half", ((0, -1),), 0.5),
    )


def _rotation_target_key(
    target: RotationTarget,
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], float]:
    """Return the commanded-motion identity of a rotation target."""
    canonical_stages: list[tuple[tuple[int, int], ...]] = []
    for stage in target.stages:
        if stage and isinstance(stage[0], tuple):
            # Axis components in one simultaneous rotation form a set.
            canonical_stages.append(tuple(sorted(stage)))
        else:
            canonical_stages.append((stage,))
    return tuple(canonical_stages), float(target.angle_scale)


def preflight_guided_rotation_plan(
    preflight_results: list[dict],
) -> tuple[RotationTarget, ...]:
    """Prioritize measured-safe signed directions and their combinations.

    The dynamic preflight is an experiment, not merely an admission check.
    Reusing its result avoids repeatedly commanding a signed direction already
    shown to lose the bilateral profile.  The full adaptive plan is retained
    as a final fallback, so no reachable branch is removed.
    """
    safe_directions: list[tuple[int, int]] = []
    for item in preflight_results:
        direction = (int(item["axis"]), int(item["sign"]))
        if bool(item.get("accepted")) and direction not in safe_directions:
            safe_directions.append(direction)
    guided: list[RotationTarget] = []
    # Information comes primarily from pose variation, so try the full
    # single-axis targets first. Small targets and small compositions remain
    # measured-safe fallbacks, not substitutes for all strong excitation.
    for axis, sign in safe_directions:
        axis_name = "rx" if axis == 0 else "ry"
        sign_name = "positive" if sign > 0 else "negative"
        guided.append(
            RotationTarget(
                f"{axis_name}_{sign_name}",
                ((axis, sign),),
            )
        )
    for axis, sign in safe_directions:
        axis_name = "rx" if axis == 0 else "ry"
        sign_name = "positive" if sign > 0 else "negative"
        guided.append(
            RotationTarget(
                f"{axis_name}_{sign_name}_half",
                ((axis, sign),),
                0.5,
            )
        )
    for first in safe_directions:
        for second in safe_directions:
            if first[0] == second[0]:
                continue
            guided.append(
                RotationTarget(
                    "preflight_combo_"
                    f"{first[0]}_{first[1]:+d}_{second[0]}_{second[1]:+d}",
                    ((first, second),),
                )
            )
    for first in safe_directions:
        for second in safe_directions:
            if first[0] == second[0]:
                continue
            guided.append(
                RotationTarget(
                    "preflight_combo_half_"
                    f"{first[0]}_{first[1]:+d}_{second[0]}_{second[1]:+d}",
                    ((first, second),),
                    0.5,
                )
            )
    unique: list[RotationTarget] = []
    seen: set[tuple[tuple[tuple[tuple[int, int], ...], ...], float]] = set()
    for target in tuple(guided) + adaptive_rotation_plan():
        key = _rotation_target_key(target)
        if key in seen:
            continue
        unique.append(target)
        seen.add(key)
    return tuple(unique)
