"""Versioned JSON I/O for simulation seed observations and calibration results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import FlangePose, Measurement


def load_seed_dataset(path: str | Path) -> tuple[list[FlangePose], list[Measurement]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 1))
    if version not in {1, 2}:
        raise ValueError(f"unsupported seed schema version {version}")
    records = payload.get("seeds")
    if not isinstance(records, list):
        raise ValueError("seed file must contain a seeds list")
    poses: list[FlangePose] = []
    measurements: list[Measurement] = []
    for index, record in enumerate(records):
        try:
            poses.append(
                FlangePose(
                    np.asarray(record["R_BF"], dtype=float),
                    np.asarray(record["t_BF"], dtype=float),
                )
            )
            measurements.append(
                Measurement(
                    np.asarray(record["profile_points_S"], dtype=float),
                    np.asarray(record["endpoint_u_S"], dtype=float),
                    np.asarray(record["endpoint_v_S"], dtype=float),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid seed record {index}: {error}") from error
    return poses, measurements


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
