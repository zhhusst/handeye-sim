"""Versioned JSON I/O for simulation seed observations and calibration results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .geometry import so3_exp, so3_log
from .models import FlangePose, Measurement


@dataclass(frozen=True)
class SeedObservationGroup:
    label: str
    poses: tuple[FlangePose, ...]
    measurements: tuple[Measurement, ...]

    def __post_init__(self) -> None:
        if not self.poses or len(self.poses) != len(self.measurements):
            raise ValueError("a physical seed needs equal non-empty frame lists")


@dataclass(frozen=True)
class LoadedSeedDataset:
    groups: tuple[SeedObservationGroup, ...]
    schema_version: int

    @property
    def physical_seed_count(self) -> int:
        return len(self.groups)

    @property
    def observation_count(self) -> int:
        return sum(len(group.poses) for group in self.groups)

    @property
    def poses(self) -> list[FlangePose]:
        return [pose for group in self.groups for pose in group.poses]

    @property
    def measurements(self) -> list[Measurement]:
        return [
            measurement
            for group in self.groups
            for measurement in group.measurements
        ]


def aggregate_seed_group(
    group: SeedObservationGroup,
    *,
    maximum_profile_points: int = 200,
) -> tuple[FlangePose, Measurement]:
    """Form one robust stationary observation from a physical seed batch."""
    if maximum_profile_points < 2:
        raise ValueError("maximum_profile_points must be at least two")
    reference_rotation = group.poses[0].rotation
    rotation_offsets = np.asarray(
        [
            so3_log(reference_rotation.T @ pose.rotation)
            for pose in group.poses
        ]
    )
    mean_rotation = reference_rotation @ so3_exp(
        np.mean(rotation_offsets, axis=0)
    )
    mean_translation = np.mean(
        [pose.translation for pose in group.poses], axis=0
    )
    profiles = np.vstack(
        [measurement.profile_points for measurement in group.measurements]
    )
    if len(profiles) > maximum_profile_points:
        indices = np.linspace(
            0, len(profiles) - 1, maximum_profile_points, dtype=int
        )
        profiles = profiles[indices]
    endpoint_u = np.mean(
        [measurement.endpoint_u for measurement in group.measurements], axis=0
    )
    endpoint_v = np.mean(
        [measurement.endpoint_v for measurement in group.measurements], axis=0
    )
    endpoint_u[1] = 0.0
    endpoint_v[1] = 0.0
    return (
        FlangePose(mean_rotation, mean_translation),
        Measurement(profiles, endpoint_u, endpoint_v),
    )


def split_stationary_group(
    group: SeedObservationGroup,
    *,
    validation_frame_count: int,
) -> tuple[SeedObservationGroup, SeedObservationGroup | None]:
    """Split one synchronized stationary batch into fit and held-out frames.

    Validation frames are distributed through the acquisition instead of
    taking one contiguous tail.  A legacy one-frame observation remains a fit
    observation and has no held-out part.
    """
    frame_count = len(group.poses)
    if validation_frame_count < 0:
        raise ValueError("validation_frame_count must be non-negative")
    validation_count = min(int(validation_frame_count), max(frame_count - 1, 0))
    if validation_count == 0:
        return group, None
    validation_indices = set(
        int(index)
        for index in np.linspace(
            0, frame_count - 1, validation_count + 2, dtype=int
        )[1:-1]
    )
    # Integer rounding can duplicate indices for very small batches.
    if len(validation_indices) < validation_count:
        for index in range(frame_count - 1, -1, -1):
            if index not in validation_indices:
                validation_indices.add(index)
            if len(validation_indices) >= validation_count:
                break
    fit_indices = [
        index for index in range(frame_count) if index not in validation_indices
    ]
    held_out_indices = sorted(validation_indices)
    fit = SeedObservationGroup(
        group.label,
        tuple(group.poses[index] for index in fit_indices),
        tuple(group.measurements[index] for index in fit_indices),
    )
    held_out = SeedObservationGroup(
        f"{group.label}_validation",
        tuple(group.poses[index] for index in held_out_indices),
        tuple(group.measurements[index] for index in held_out_indices),
    )
    return fit, held_out


def _parse_observation(record: dict, description: str) -> tuple[FlangePose, Measurement]:
    try:
        return (
            FlangePose(
                np.asarray(record["R_BF"], dtype=float),
                np.asarray(record["t_BF"], dtype=float),
            ),
            Measurement(
                np.asarray(record["profile_points_S"], dtype=float),
                np.asarray(record["endpoint_u_S"], dtype=float),
                np.asarray(record["endpoint_v_S"], dtype=float),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {error}") from error


def load_seed_dataset_grouped(path: str | Path) -> LoadedSeedDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 1))
    if version not in {1, 2, 3}:
        raise ValueError(f"unsupported seed schema version {version}")
    records = payload.get("seeds")
    if not isinstance(records, list):
        raise ValueError("seed file must contain a seeds list")
    groups: list[SeedObservationGroup] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"invalid seed record {index}: expected an object")
        label = str(record.get("label", f"seed_{index + 1}"))
        frame_records = record.get("frames") if version >= 3 else None
        if frame_records is None:
            frame_records = [record]
        if not isinstance(frame_records, list) or not frame_records:
            raise ValueError(f"invalid seed record {index}: frames must be non-empty")
        poses: list[FlangePose] = []
        measurements: list[Measurement] = []
        for frame_index, frame in enumerate(frame_records):
            if not isinstance(frame, dict):
                raise ValueError(
                    f"invalid seed record {index} frame {frame_index}: "
                    "expected an object"
                )
            pose, measurement = _parse_observation(
                frame, f"seed record {index} frame {frame_index}"
            )
            poses.append(pose)
            measurements.append(measurement)
        groups.append(
            SeedObservationGroup(label, tuple(poses), tuple(measurements))
        )
    return LoadedSeedDataset(tuple(groups), version)


def load_seed_dataset(path: str | Path) -> tuple[list[FlangePose], list[Measurement]]:
    dataset = load_seed_dataset_grouped(path)
    return dataset.poses, dataset.measurements


def result_payload(result, *, extra: dict | None = None) -> dict:
    estimate = result.estimate
    payload = {
        "schema_version": 1,
        "converged": bool(result.converged),
        "message": result.message,
        "cost": float(result.cost),
        "handeye": {
            "rotation": estimate.handeye_rotation.tolist(),
            "translation": estimate.handeye_translation.tolist(),
        },
        "board": {
            "corner": estimate.board.corner.tolist(),
            "rotation": estimate.board.rotation.tolist(),
        },
        "diagnostics": {
            "rank": int(result.diagnostics.rank),
            "condition_number": float(result.diagnostics.condition_number),
            "singular_values": result.diagnostics.singular_values.tolist(),
            "weakest_direction": result.diagnostics.weakest_direction.tolist(),
            "residual_variance": float(result.diagnostics.residual_variance),
        },
    }
    if extra:
        payload["simulation"] = extra
    return payload


def save_result(path: str | Path, result, *, extra: dict | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result_payload(result, extra=extra), indent=2),
        encoding="utf-8",
    )
