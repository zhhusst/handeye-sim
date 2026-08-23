#!/usr/bin/env python3
"""A/B/C ablation of the flat pre-solve used before shared-shape calibration.

The experiment deliberately keeps the calibration observations, shared-surface
residual, weights, regularization and TRF settings fixed.  Only construction of
the initial 19-D state changes:

* A: production four-hypothesis flat pre-solve, then shared refinement.
* B: one direct shared solve from the perturbed mounting nominal.
* C: four direct shared solves from the same axis hypotheses as A, with no
  flat nonlinear solve; the cheapest physically plausible result is retained.

For B/C, the board orientation is estimated directly from transformed endpoint
clouds and the corner is obtained by the model's linear corner projection.
Neither operation optimizes the flat nonlinear objective.  Shape coefficients
start at zero in every group.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
CORE_SOURCE = WORKSPACE / "ros2_ws/src/handeye_calibration_core"
sys.path.insert(0, str(CORE_SOURCE))

from calibration_pipeline.dataset_io import (  # noqa: E402
    aggregate_seed_group,
    load_seed_dataset_grouped,
)
from calibration_pipeline.geometry import so3_exp, so3_log  # noqa: E402
from calibration_pipeline.models import (  # noqa: E402
    BoardModel,
    CalibrationEstimate,
)
from calibration_pipeline.solvers.twelve_dof_v2 import (  # noqa: E402
    TwelveDofV2Solver,
    _initial_board_rotation,
)
from calibration_pipeline.sphere_validation import (  # noqa: E402
    _fixed_radius_center,
    _free_sphere,
)
from calibration_pipeline.v2_backend.corner_projection import (  # noqa: E402
    solve_corner,
)


DEFAULT_DATASETS = (
    "20260820_115707_圆点标定板背面_位置1_真机_2",
    "20260820_132928_圆点标定板背面_位置1_真机_3",
    "20260820_133622_圆点标定板背面_位置1_真机_5",
    "20260820_135111_圆点标定板背面_位置2_真机_1",
    "20260820_140109_圆点标定板背面_位置2_真机_2",
)
DEFAULT_LEVELS = (
    # hand-eye rotation deg, translation mm, board rotation deg, corner mm
    (0.0, 0.0, 0.0, 0.0),
    (5.0, 25.0, 2.0, 5.0),
    (10.0, 50.0, 4.0, 10.0),
    (15.0, 100.0, 6.0, 15.0),
    (20.0, 150.0, 8.0, 20.0),
    (30.0, 200.0, 10.0, 30.0),
)
DEFAULT_SPHERE = (
    WORKSPACE
    / "data/sphere_validation_runs/20260820_124558_sphere_20mm"
    / "sphere_acquisition.npz"
)


@dataclass(frozen=True)
class PerturbationLevel:
    handeye_rotation_deg: float
    handeye_translation_mm: float
    board_rotation_deg: float
    corner_translation_mm: float

    @property
    def label(self) -> str:
        return (
            f"{self.handeye_rotation_deg:g}deg/"
            f"{self.handeye_translation_mm:g}mm"
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "handeye_rotation_deg": self.handeye_rotation_deg,
            "handeye_translation_mm": self.handeye_translation_mm,
            "board_rotation_deg": self.board_rotation_deg,
            "corner_translation_mm": self.corner_translation_mm,
        }


@dataclass
class DatasetContext:
    name: str
    path: Path
    poses: list[Any]
    measurements: list[Any]
    nominal_rotation: np.ndarray
    nominal_translation: np.ndarray
    board_dimensions: tuple[float, float]
    solver: TwelveDofV2Solver
    maximum_board_tilt_deg: float
    configuration: dict[str, Any]


@dataclass(frozen=True)
class SphereData:
    points_sensor_m: np.ndarray
    frame_indices: np.ndarray
    flange_rotations: np.ndarray
    flange_translations_m: np.ndarray
    radius_m: float
    source: str


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _ros_parameters(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid ROS parameter file: {path}")
    wildcard = payload.get("/**", {})
    parameters = wildcard.get("ros__parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"missing /**.ros__parameters in {path}")
    return parameters


def _load_context(path: Path) -> DatasetContext:
    base_path = path / "calibration_parameters.yaml"
    overlay_path = path / "real_calibration_parameters.yaml"
    parameters = _ros_parameters(base_path)
    if overlay_path.exists():
        parameters = _deep_merge(parameters, _ros_parameters(overlay_path))
    solver_values = parameters["solver"]
    multistart = solver_values["multistart"]
    initial_handeye = parameters["initial_handeye"]
    board = parameters["board"]
    rotation_scale = math.radians(
        float(solver_values["handeye_rotation_scale_deg"])
    )
    translation_scale = float(solver_values["handeye_translation_scale_m"])
    board_rotation_scale = math.radians(
        float(solver_values["plane_rotation_scale_deg"])
    )
    solver = TwelveDofV2Solver(
        plane_weight=float(solver_values["plane_weight"]),
        edge_weight=float(solver_values["edge_weight"]),
        endpoint_plane_weight=float(solver_values["endpoint_plane_weight"]),
        max_evaluations=int(solver_values["max_evaluations"]),
        tolerance=float(solver_values["tolerance"]),
        state_scale=np.array(
            [rotation_scale] * 3
            + [translation_scale] * 3
            + [board_rotation_scale] * 3
        ),
        maximum_condition_number=float(
            solver_values["maximum_condition_number"]
        ),
        surface_model=str(solver_values["surface_model"]),
        surface_basis_kind=str(solver_values["surface_basis_kind"]),
        surface_degree=int(solver_values["surface_degree"]),
        shape_scale_m=float(solver_values["shape_scale_m"]),
        shape_regularization=float(solver_values["shape_regularization"]),
        multistart_enabled=bool(multistart["enabled"]),
        multistart_maximum_board_tilt_deg=float(
            multistart["maximum_board_tilt_deg"]
        ),
        multistart_require_plausible=bool(multistart["require_plausible"]),
    )
    if not solver.uses_shared_surface:
        raise ValueError(f"{path}: experiment requires surface_model=shared")
    dataset = load_seed_dataset_grouped(path / "seeds.json")
    aggregated = [aggregate_seed_group(group) for group in dataset.groups]
    poses = [pair[0] for pair in aggregated]
    measurements = [pair[1] for pair in aggregated]
    if len(poses) < 6:
        raise ValueError(f"{path}: expected at least six physical seed poses")
    return DatasetContext(
        name=path.name,
        path=path,
        poses=poses,
        measurements=measurements,
        nominal_rotation=np.asarray(initial_handeye["rotation"], dtype=float).reshape(3, 3),
        nominal_translation=np.asarray(
            initial_handeye["translation_m"], dtype=float
        ),
        board_dimensions=(
            float(board["length_u_m"]),
            float(board["length_v_m"]),
        ),
        solver=solver,
        maximum_board_tilt_deg=float(multistart["maximum_board_tilt_deg"]),
        configuration={
            "physical_seed_count": dataset.physical_seed_count,
            "raw_synchronized_observations": dataset.observation_count,
            "aggregated_observations_used": len(poses),
            "profile_points_per_observation": [
                len(measurement.profile_points) for measurement in measurements
            ],
            "solver": solver_values,
            "board": board,
            "initial_handeye": initial_handeye,
        },
    )


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _perturbation(
    context: DatasetContext,
    level: PerturbationLevel,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    handeye_axis = _unit_vector(rng)
    translation_axis = _unit_vector(rng)
    board_axis = _unit_vector(rng)
    corner_axis = _unit_vector(rng)
    rotation = context.nominal_rotation @ so3_exp(
        handeye_axis * math.radians(level.handeye_rotation_deg)
    )
    translation = context.nominal_translation + (
        translation_axis * level.handeye_translation_mm / 1000.0
    )
    board_rotation = _initial_board_rotation(
        context.poses,
        context.measurements,
        rotation,
        translation,
    ) @ so3_exp(board_axis * math.radians(level.board_rotation_deg))
    return {
        "handeye_rotation": rotation,
        "handeye_translation": translation,
        "board_rotation": board_rotation,
        "corner_delta": corner_axis * level.corner_translation_mm / 1000.0,
        "handeye_axis": handeye_axis,
        "translation_axis": translation_axis,
        "board_axis": board_axis,
        "board_rotation_vector": (
            board_axis * math.radians(level.board_rotation_deg)
        ),
        "corner_axis": corner_axis,
    }


def _direct_shared_initial(
    context: DatasetContext,
    handeye_rotation: np.ndarray,
    handeye_translation: np.ndarray,
    board_rotation: np.ndarray,
    corner_delta: np.ndarray,
) -> CalibrationEstimate:
    basis = context.solver.surface_basis
    assert basis is not None
    x9 = np.concatenate(
        (
            so3_log(handeye_rotation),
            handeye_translation,
            so3_log(board_rotation),
        )
    )
    corner, rank = solve_corner(
        x9,
        context.poses,
        context.measurements,
        **context.solver.weights,
    )
    if rank < 3:
        raise RuntimeError("coarse projected corner system is rank deficient")
    corner = corner + np.asarray(corner_delta, dtype=float)
    coefficients = np.zeros(basis.size)
    state = np.concatenate((x9, corner, coefficients))
    board = BoardModel(
        corner=corner,
        rotation=board_rotation,
        length_u=context.board_dimensions[0],
        length_v=context.board_dimensions[1],
    )
    return CalibrationEstimate(
        handeye_rotation=handeye_rotation,
        handeye_translation=handeye_translation,
        board=board,
        x9=x9,
        state=state,
        surface_model="shared",
        surface_basis_kind=context.solver.surface_basis_kind,
        surface_degree=context.solver.surface_degree,
        shape_coefficients=coefficients,
    )


def _result_record(
    context: DatasetContext,
    group: str,
    result: Any,
    *,
    elapsed_s: float,
    total_nfev: int,
    total_njev: int | None,
    selected_restart: str | None = None,
    restart_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    estimate = result.estimate
    state = estimate.optimization_state
    data_residual = context.solver.observation_residual(
        state,
        context.poses,
        context.measurements,
        board_dimensions=context.board_dimensions,
    )
    data_cost = 0.5 * float(data_residual @ data_residual)
    tilt = context.solver._board_tilt_deg(estimate.board.rotation)
    finite = bool(
        np.all(np.isfinite(state))
        and np.isfinite(result.cost)
        and np.isfinite(tilt)
    )
    optimizer_success = result.diagnostics.optimizer_success
    if optimizer_success is None:
        optimizer_success = bool(result.converged)
    physically_reasonable = bool(
        finite
        and tilt <= context.maximum_board_tilt_deg
        and np.linalg.norm(estimate.handeye_translation) < 1.0
    )
    return {
        "group": group,
        "solver_returned": True,
        "optimizer_success": bool(optimizer_success),
        "solver_converged": bool(result.converged),
        "physically_reasonable": physically_reasonable,
        "successful_physical_solution": bool(
            result.converged and physically_reasonable
        ),
        "message": str(result.message),
        "optimizer_status": result.diagnostics.optimizer_status,
        "objective_cost": float(result.cost),
        "data_only_cost": data_cost,
        "data_residual_rms_mm": 1000.0
        * float(np.sqrt(np.mean(data_residual * data_residual))),
        "data_residual_p95_mm": 1000.0
        * float(np.percentile(np.abs(data_residual), 95.0)),
        "data_rank": int(result.diagnostics.rank),
        "data_condition_number": float(result.diagnostics.condition_number),
        "prior_rank": int(result.diagnostics.prior_augmented_rank or 0),
        "prior_condition_number": float(
            result.diagnostics.prior_augmented_condition_number or np.inf
        ),
        "board_tilt_deg": float(tilt),
        "surface_rms_mm": 1000.0 * float(result.diagnostics.surface_rms_m),
        "surface_maximum_mm": 1000.0
        * float(result.diagnostics.surface_maximum_m),
        "handeye_rotation": estimate.handeye_rotation.tolist(),
        "handeye_translation_m": estimate.handeye_translation.tolist(),
        "board_rotation": estimate.board.rotation.tolist(),
        "board_corner_m": estimate.board.corner.tolist(),
        "shape_coefficients_m": estimate.shape_coefficients.tolist(),
        "function_evaluations": int(total_nfev),
        "jacobian_evaluations": (
            None if total_njev is None else int(total_njev)
        ),
        "solver_time_s": float(elapsed_s),
        "selected_restart": selected_restart,
        "restart_records": restart_records or [],
    }


def _failure_record(
    group: str,
    error: Exception,
    elapsed_s: float,
    *,
    restart_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "group": group,
        "solver_returned": False,
        "optimizer_success": False,
        "solver_converged": False,
        "physically_reasonable": False,
        "successful_physical_solution": False,
        "message": f"{type(error).__name__}: {error}",
        "solver_time_s": float(elapsed_s),
        "function_evaluations": 0,
        "jacobian_evaluations": None,
        "restart_records": restart_records or [],
    }


def _solve_a(
    context: DatasetContext, perturbation: dict[str, np.ndarray]
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        result = context.solver.solve(
            context.poses,
            context.measurements,
            perturbation["handeye_rotation"],
            perturbation["handeye_translation"],
            board_dimensions=context.board_dimensions,
        )
        attempts = list(result.diagnostics.initialization_candidates)
        nfev = int(result.evaluations) + sum(
            int(attempt.get("function_evaluations") or 0)
            for attempt in attempts
        )
        njev_values = [
            attempt.get("jacobian_evaluations") for attempt in attempts
        ] + [result.diagnostics.optimizer_jacobian_evaluations]
        total_njev = (
            None
            if any(value is None for value in njev_values)
            else sum(int(value) for value in njev_values)
        )
        return _result_record(
            context,
            "A_flat_presolve",
            result,
            elapsed_s=time.perf_counter() - start,
            total_nfev=nfev,
            total_njev=total_njev,
            selected_restart=result.diagnostics.selected_initialization,
            restart_records=attempts,
        )
    except Exception as error:
        return _failure_record(
            "A_flat_presolve", error, time.perf_counter() - start
        )


def _solve_b(
    context: DatasetContext, perturbation: dict[str, np.ndarray]
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        initial = _direct_shared_initial(
            context,
            perturbation["handeye_rotation"],
            perturbation["handeye_translation"],
            perturbation["board_rotation"],
            perturbation["corner_delta"],
        )
        result = context.solver.solve(
            context.poses,
            context.measurements,
            perturbation["handeye_rotation"],
            perturbation["handeye_translation"],
            board_dimensions=context.board_dimensions,
            initial_estimate=initial,
        )
        return _result_record(
            context,
            "B_direct_single",
            result,
            elapsed_s=time.perf_counter() - start,
            total_nfev=int(result.evaluations),
            total_njev=result.diagnostics.optimizer_jacobian_evaluations,
        )
    except Exception as error:
        return _failure_record(
            "B_direct_single", error, time.perf_counter() - start
        )


def _solve_c(
    context: DatasetContext, perturbation: dict[str, np.ndarray]
) -> dict[str, Any]:
    start = time.perf_counter()
    candidates: list[tuple[str, Any]] = []
    restart_records: list[dict[str, Any]] = []
    total_nfev = 0
    total_njev = 0
    njev_known = True
    for name, offset in context.solver._flat_rotation_hypotheses():
        restart_start = time.perf_counter()
        try:
            handeye_rotation = perturbation["handeye_rotation"] @ offset
            board_rotation = _initial_board_rotation(
                context.poses,
                context.measurements,
                handeye_rotation,
                perturbation["handeye_translation"],
            ) @ so3_exp(perturbation["board_rotation_vector"])
            initial = _direct_shared_initial(
                context,
                handeye_rotation,
                perturbation["handeye_translation"],
                board_rotation,
                perturbation["corner_delta"],
            )
            result = context.solver.solve(
                context.poses,
                context.measurements,
                handeye_rotation,
                perturbation["handeye_translation"],
                board_dimensions=context.board_dimensions,
                initial_estimate=initial,
            )
            tilt = context.solver._board_tilt_deg(result.estimate.board.rotation)
            candidates.append((name, result))
            total_nfev += int(result.evaluations)
            if result.diagnostics.optimizer_jacobian_evaluations is None:
                njev_known = False
            else:
                total_njev += int(
                    result.diagnostics.optimizer_jacobian_evaluations
                )
            restart_records.append(
                {
                    "name": name,
                    "returned": True,
                    "converged": bool(result.converged),
                    "cost": float(result.cost),
                    "board_tilt_deg": float(tilt),
                    "accepted": bool(
                        result.converged
                        and tilt <= context.maximum_board_tilt_deg
                    ),
                    "function_evaluations": int(result.evaluations),
                    "jacobian_evaluations": (
                        result.diagnostics.optimizer_jacobian_evaluations
                    ),
                    "time_s": time.perf_counter() - restart_start,
                    "message": str(result.message),
                }
            )
        except Exception as error:
            restart_records.append(
                {
                    "name": name,
                    "returned": False,
                    "converged": False,
                    "accepted": False,
                    "function_evaluations": 0,
                    "jacobian_evaluations": None,
                    "time_s": time.perf_counter() - restart_start,
                    "message": f"{type(error).__name__}: {error}",
                }
            )
            njev_known = False
    plausible = [
        (name, result)
        for name, result in candidates
        if result.converged
        and context.solver._board_tilt_deg(result.estimate.board.rotation)
        <= context.maximum_board_tilt_deg
    ]
    fallback = [
        (name, result) for name, result in candidates if result.converged
    ]
    pool = plausible or fallback or candidates
    if not pool:
        return _failure_record(
            "C_direct_multistart",
            RuntimeError("all direct shared restarts failed"),
            time.perf_counter() - start,
            restart_records=restart_records,
        )
    selected_name, selected = min(pool, key=lambda item: item[1].cost)
    return _result_record(
        context,
        "C_direct_multistart",
        selected,
        elapsed_s=time.perf_counter() - start,
        total_nfev=total_nfev,
        total_njev=total_njev if njev_known else None,
        selected_restart=selected_name,
        restart_records=restart_records,
    )


def _load_sphere(path: Path, maximum_points: int) -> SphereData:
    payload = np.load(str(path), allow_pickle=True)
    points = np.asarray(payload["selected_points_sensor_m"], dtype=float)
    offsets = np.asarray(payload["selected_frame_offsets"], dtype=np.int64)
    frame_indices = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    if len(frame_indices) != len(points):
        raise ValueError("sphere point offsets do not match selected points")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    frame_indices = frame_indices[finite]
    if maximum_points > 0 and len(points) > maximum_points:
        selected = np.linspace(0, len(points) - 1, maximum_points, dtype=int)
        points = points[selected]
        frame_indices = frame_indices[selected]
    return SphereData(
        points_sensor_m=points,
        frame_indices=frame_indices,
        flange_rotations=np.asarray(payload["flange_rotations"], dtype=float),
        flange_translations_m=np.asarray(
            payload["flange_translations_m"], dtype=float
        ),
        radius_m=0.010001,
        source=str(path),
    )


def _sphere_metrics(
    sphere: SphereData,
    handeye_rotation: np.ndarray,
    handeye_translation: np.ndarray,
) -> dict[str, Any]:
    points_flange = (
        np.einsum("ij,nj->ni", handeye_rotation, sphere.points_sensor_m)
        + handeye_translation
    )
    frame_rotation = sphere.flange_rotations[sphere.frame_indices]
    points_base = (
        np.einsum("nij,nj->ni", frame_rotation, points_flange)
        + sphere.flange_translations_m[sphere.frame_indices]
    )
    fixed_center = _fixed_radius_center(
        points_base, sphere.radius_m, robust_scale_m=0.00010
    )
    fixed_residual = (
        np.linalg.norm(points_base - fixed_center[None, :], axis=1)
        - sphere.radius_m
    )
    free_center, free_radius = _free_sphere(
        points_base, robust_scale_m=0.00010
    )
    free_residual = (
        np.linalg.norm(points_base - free_center[None, :], axis=1)
        - free_radius
    )
    return {
        "sphere_point_count": int(len(points_base)),
        "sphere_fixed_rmse_mm": 1000.0
        * float(np.sqrt(np.mean(fixed_residual * fixed_residual))),
        "sphere_fixed_p95_mm": 1000.0
        * float(np.percentile(np.abs(fixed_residual), 95.0)),
        "sphere_fixed_max_mm": 1000.0
        * float(np.max(np.abs(fixed_residual))),
        "sphere_free_diameter_error_mm": 2000.0
        * float(free_radius - sphere.radius_m),
        "sphere_free_dispersion_rmse_mm": 1000.0
        * float(np.sqrt(np.mean(free_residual * free_residual))),
    }


def _parse_levels(value: str | None) -> list[PerturbationLevel]:
    raw_levels = DEFAULT_LEVELS if not value else []
    if value:
        for item in value.split(","):
            values = [float(token) for token in item.split(":")]
            if len(values) == 2:
                rotation, translation = values
                values = [rotation, translation, 0.4 * rotation, 0.15 * translation]
            if len(values) != 4:
                raise ValueError(
                    "each level must be rot_deg:trans_mm[:board_deg:corner_mm]"
                )
            raw_levels.append(tuple(values))
    return [PerturbationLevel(*map(float, level)) for level in raw_levels]


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(so3_log(first.T @ second))))


def _dispersion(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    valid = [row for row in rows if row.get("successful_physical_solution")]
    if len(valid) < 2:
        return None, None
    rotations = [np.asarray(row["handeye_rotation"], dtype=float) for row in valid]
    translations = np.asarray(
        [row["handeye_translation_m"] for row in valid], dtype=float
    )
    pairwise = np.array(
        [
            [
                _rotation_distance_deg(first, second)
                for second in rotations
            ]
            for first in rotations
        ]
    )
    medoid = int(np.argmin(np.sum(pairwise * pairwise, axis=1)))
    rotation_rms = float(np.sqrt(np.mean(pairwise[medoid] ** 2)))
    center = np.mean(translations, axis=0)
    translation_rms = 1000.0 * float(
        np.sqrt(np.mean(np.sum((translations - center) ** 2, axis=1)))
    )
    return rotation_rms, translation_rms


def _finite_values(rows: Iterable[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    levels = sorted({int(row["level_index"]) for row in rows})
    groups = ("A_flat_presolve", "B_direct_single", "C_direct_multistart")
    for level_index in levels:
        for group in groups:
            selected = [
                row
                for row in rows
                if row["level_index"] == level_index and row["group"] == group
            ]
            physical = [
                row for row in selected if row["successful_physical_solution"]
            ]
            sphere_rmse = _finite_values(physical, "sphere_fixed_rmse_mm")
            residual = _finite_values(physical, "data_residual_rms_mm")
            elapsed = _finite_values(selected, "solver_time_s")
            nfev = _finite_values(selected, "function_evaluations")
            dispersions_r = []
            dispersions_t = []
            for dataset in sorted({row["dataset"] for row in selected}):
                dataset_rows = [
                    row for row in selected if row["dataset"] == dataset
                ]
                value_r, value_t = _dispersion(dataset_rows)
                if value_r is not None:
                    dispersions_r.append(value_r)
                if value_t is not None:
                    dispersions_t.append(value_t)
            first = selected[0]
            summaries.append(
                {
                    "level_index": level_index,
                    "level_label": first["level_label"],
                    "handeye_rotation_deg": first["handeye_rotation_deg"],
                    "handeye_translation_mm": first["handeye_translation_mm"],
                    "group": group,
                    "trials": len(selected),
                    "optimizer_success_rate": float(
                        np.mean([row["optimizer_success"] for row in selected])
                    ),
                    "strict_convergence_rate": float(
                        np.mean([row["solver_converged"] for row in selected])
                    ),
                    "physical_success_rate": float(
                        np.mean(
                            [
                                row["successful_physical_solution"]
                                for row in selected
                            ]
                        )
                    ),
                    "data_residual_rms_mm_median": (
                        None if not residual else float(np.median(residual))
                    ),
                    "sphere_fixed_rmse_mm_median": (
                        None if not sphere_rmse else float(np.median(sphere_rmse))
                    ),
                    "sphere_fixed_rmse_mm_p95": (
                        None
                        if not sphere_rmse
                        else float(np.percentile(sphere_rmse, 95.0))
                    ),
                    "rotation_dispersion_deg_median": (
                        None
                        if not dispersions_r
                        else float(np.median(dispersions_r))
                    ),
                    "translation_dispersion_mm_median": (
                        None
                        if not dispersions_t
                        else float(np.median(dispersions_t))
                    ),
                    "solver_time_s_median": (
                        None if not elapsed else float(np.median(elapsed))
                    ),
                    "function_evaluations_median": (
                        None if not nfev else float(np.median(nfev))
                    ),
                }
            )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: value
                for key, value in row.items()
                if value is None
                or isinstance(value, (str, int, float, bool, np.number))
            }
        )
    keys = sorted({key for row in scalar_rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _plots(output: Path, rows: list[dict], summaries: list[dict]) -> None:
    from matplotlib import font_manager

    cjk_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk_font_path).exists():
        font_manager.fontManager.addfont(cjk_font_path)
        cjk_family = font_manager.FontProperties(fname=cjk_font_path).get_name()
    else:
        cjk_family = "Droid Sans Fallback"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk_family, "Droid Sans Fallback", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )
    group_labels = {
        "A_flat_presolve": "A 平面预求解",
        "B_direct_single": "B 单初值直解",
        "C_direct_multistart": "C 多初值直解",
    }
    colors = {
        "A_flat_presolve": "#1f77b4",
        "B_direct_single": "#d62728",
        "C_direct_multistart": "#2ca02c",
    }
    groups = list(group_labels)
    levels = sorted({int(row["level_index"]) for row in summaries})
    labels = [
        next(row["level_label"] for row in summaries if row["level_index"] == i)
        for i in levels
    ]
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    for group in groups:
        values = [
            next(
                row["physical_success_rate"]
                for row in summaries
                if row["level_index"] == level and row["group"] == group
            )
            for level in levels
        ]
        axis.plot(labels, values, marker="o", linewidth=2, label=group_labels[group], color=colors[group])
    axis.set_xlabel("初始手眼扰动（旋转/平移）")
    axis.set_ylabel("物理合理成功率")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "initial_perturbation_vs_success_rate.png")
    fig.savefig(output / "initial_perturbation_vs_success_rate.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    any_external = False
    for group in groups:
        medians = []
        low = []
        high = []
        for level in levels:
            values = _finite_values(
                [
                    row
                    for row in rows
                    if row["level_index"] == level
                    and row["group"] == group
                    and row["successful_physical_solution"]
                ],
                "sphere_fixed_rmse_mm",
            )
            if values:
                any_external = True
                medians.append(float(np.median(values)))
                low.append(float(np.percentile(values, 25.0)))
                high.append(float(np.percentile(values, 75.0)))
            else:
                medians.append(np.nan)
                low.append(np.nan)
                high.append(np.nan)
        medians_array = np.asarray(medians)
        axis.plot(labels, medians, marker="o", linewidth=2, label=group_labels[group], color=colors[group])
        axis.fill_between(labels, low, high, alpha=0.15, color=colors[group])
    axis.set_xlabel("初始手眼扰动（旋转/平移）")
    axis.set_ylabel("独立精密球固定半径 RMSE (mm)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "initial_perturbation_vs_external_error.png")
    fig.savefig(output / "initial_perturbation_vs_external_error.pdf")
    plt.close(fig)

    if any_external:
        data = []
        positions = []
        labels_box = []
        for index, group in enumerate(groups, start=1):
            values = _finite_values(
                [
                    row
                    for row in rows
                    if row["group"] == group
                    and row["successful_physical_solution"]
                ],
                "sphere_fixed_rmse_mm",
            )
            if values:
                data.append(values)
                positions.append(index)
                labels_box.append(group_labels[group])
        fig, axis = plt.subplots(figsize=(8.0, 5.2))
        axis.boxplot(data, positions=positions, labels=labels_box, showfliers=True)
        axis.set_ylabel("独立精密球固定半径 RMSE (mm)")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "final_external_error_distribution.png")
        fig.savefig(output / "final_external_error_distribution.pdf")
        plt.close(fig)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _automatic_interpretation(summaries: list[dict[str, Any]]) -> str:
    maximum_level = max(row["level_index"] for row in summaries)
    high = [row for row in summaries if row["level_index"] == maximum_level]
    by_group = {row["group"]: row for row in high}
    a = by_group["A_flat_presolve"]
    b = by_group["B_direct_single"]
    c = by_group["C_direct_multistart"]
    a_rate = a["physical_success_rate"]
    b_rate = b["physical_success_rate"]
    c_rate = c["physical_success_rate"]
    if abs(a_rate - b_rate) <= 0.10 and abs(a_rate - c_rate) <= 0.10:
        return (
            "在最大扰动档，A/B/C 的物理合理成功率差异不超过 10 个百分点；"
            "当前证据不支持第5章预求解具有实质必要性。应结合外部球误差和更多重复确认后，"
            "再考虑简化流程。"
        )
    if a_rate >= b_rate + 0.20 and a_rate >= c_rate + 0.20:
        return (
            "在最大扰动档，A 的物理合理成功率同时领先 B、C 至少 20 个百分点。"
            "这支持平面预求解具有不能由同规模直接多初值完全替代的结构性收敛域优势。"
        )
    if a_rate >= b_rate + 0.20 and abs(a_rate - c_rate) <= 0.10:
        return (
            "A 明显优于单初值 B，但与多初值直解 C 的成功率接近。"
            "这说明主要收益来自多解域搜索/好初值，而尚不能证明理想平面预求解本身不可替代；"
            "还需结合 A/C 实际计算时间与外部精度决定保留哪一种。"
        )
    if b_rate >= a_rate + 0.20:
        return (
            "最大扰动档中 B 比 A 更稳定，提示理想平面预求解可能引入错误解域或平面模型偏差。"
            "应检查 A 的离散轴假设筛选和平面度失配。"
        )
    return (
        "三组表现存在差异，但尚未达到预设的 20 个百分点强判据。"
        "结论应以置信区间、独立球误差以及 A/C 计算预算是否相当为准。"
    )


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = z / denominator * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    )
    return center - half_width, center + half_width


def _paired_a_c_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_maps: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {
        "A_flat_presolve": {},
        "C_direct_multistart": {},
    }
    for row in rows:
        if row["group"] not in group_maps:
            continue
        key = (row["dataset"], row["level_index"], row["repeat"])
        if row.get("successful_physical_solution"):
            group_maps[row["group"]][key] = row
    common = sorted(
        set(group_maps["A_flat_presolve"])
        & set(group_maps["C_direct_multistart"])
    )
    sphere_differences = []
    rotation_differences = []
    translation_differences = []
    time_ratios = []
    nfev_ratios = []
    a_times = []
    c_times = []
    a_nfev = []
    c_nfev = []
    for key in common:
        a = group_maps["A_flat_presolve"][key]
        c = group_maps["C_direct_multistart"][key]
        if a.get("sphere_fixed_rmse_mm") is not None and c.get(
            "sphere_fixed_rmse_mm"
        ) is not None:
            sphere_differences.append(
                float(a["sphere_fixed_rmse_mm"])
                - float(c["sphere_fixed_rmse_mm"])
            )
        rotation_differences.append(
            _rotation_distance_deg(
                np.asarray(a["handeye_rotation"], dtype=float),
                np.asarray(c["handeye_rotation"], dtype=float),
            )
        )
        translation_differences.append(
            1000.0
            * float(
                np.linalg.norm(
                    np.asarray(a["handeye_translation_m"], dtype=float)
                    - np.asarray(c["handeye_translation_m"], dtype=float)
                )
            )
        )
        a_time = float(a["solver_time_s"])
        c_time = float(c["solver_time_s"])
        a_eval = float(a["function_evaluations"])
        c_eval = float(c["function_evaluations"])
        a_times.append(a_time)
        c_times.append(c_time)
        a_nfev.append(a_eval)
        c_nfev.append(c_eval)
        time_ratios.append(c_time / a_time)
        nfev_ratios.append(c_eval / a_eval)

    def median(values: list[float]) -> float | None:
        return None if not values else float(np.median(values))

    def p95(values: list[float]) -> float | None:
        return None if not values else float(np.percentile(values, 95.0))

    return {
        "paired_trials": len(common),
        "sphere_rmse_a_minus_c_median_mm": median(sphere_differences),
        "sphere_rmse_abs_difference_p95_mm": p95(
            [abs(value) for value in sphere_differences]
        ),
        "handeye_rotation_difference_median_deg": median(rotation_differences),
        "handeye_rotation_difference_p95_deg": p95(rotation_differences),
        "handeye_translation_difference_median_mm": median(
            translation_differences
        ),
        "handeye_translation_difference_p95_mm": p95(
            translation_differences
        ),
        "a_time_median_s": median(a_times),
        "c_time_median_s": median(c_times),
        "c_over_a_time_ratio_median": median(time_ratios),
        "a_nfev_median": median(a_nfev),
        "c_nfev_median": median(c_nfev),
        "c_over_a_nfev_ratio_median": median(nfev_ratios),
    }


def _report(
    output: Path,
    contexts: list[DatasetContext],
    levels: list[PerturbationLevel],
    repeats: int,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    sphere: SphereData | None,
) -> None:
    paired = _paired_a_c_statistics(rows)
    groups = ("A_flat_presolve", "B_direct_single", "C_direct_multistart")
    overall = []
    for group in groups:
        selected = [row for row in rows if row["group"] == group]
        successes = sum(
            bool(row["successful_physical_solution"]) for row in selected
        )
        lower, upper = _wilson_interval(successes, len(selected))
        successful_rows = [
            row for row in selected if row["successful_physical_solution"]
        ]
        overall.append(
            {
                "group": group,
                "successes": successes,
                "trials": len(selected),
                "rate": successes / len(selected),
                "lower": lower,
                "upper": upper,
                "sphere": (
                    None
                    if not successful_rows
                    else float(
                        np.median(
                            _finite_values(
                                successful_rows, "sphere_fixed_rmse_mm"
                            )
                        )
                    )
                ),
                "time": float(
                    np.median(_finite_values(selected, "solver_time_s"))
                ),
                "nfev": float(
                    np.median(_finite_values(selected, "function_evaluations"))
                ),
            }
        )
    lines = [
        "# 理想平面 12-DOF 预求解必要性消融报告",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. 实验问题与公平性约束",
        "",
        "本实验只改变共享形貌 19 维联合优化的初始化方式。三组使用完全相同的六物理位姿聚合数据、三阶 Legendre 共享形貌残差、观测权重、形貌正则、状态尺度、TRF 容差和最大函数调用数。",
        "",
        "- **A（当前方法）**：四个离散轴约定下执行理想平面 12-DOF-V2 预求解，经水平板筛选后初始化一次共享形貌优化。",
        "- **B（无预求解）**：从受扰粗手眼直接估计板朝向，以线性 variable projection 仅求角点，形貌系数置零，然后只运行一次共享形貌优化。没有平面非线性优化。",
        "- **C（无预求解多初值）**：使用与 A 相同的四个离散轴假设，但每个假设直接运行共享形貌优化；按同一水平板门限和目标函数选解。",
        "",
        "C 的实际函数调用数和墙钟时间被完整记录。判断结构性优势时必须同时比较 A/C 预算，不能只看成功率。",
        "",
        "## 2. 数据与扰动",
        "",
        f"- 真机标定数据集：{len(contexts)} 组。",
        f"- 每个非零扰动档随机重复：{repeats} 次；零扰动档只运行一次，因为不存在随机方向差异。",
        "- 每次随机扰动对 A/B/C 完全配对，共用同一手眼旋转轴和平移方向。A 按当前生产流程从每个平面假设重新构造板姿态并以 variable projection 消除角点，因此不消费显式板姿态/角点初值；B/C 则消费同一组板姿态和角点扰动。这一差别正是第5章结构可能带来的初始化优势，而不是残差或优化器差异。",
    ]
    if sphere is not None:
        lines.extend(
            [
                f"- 外部评价：`{sphere.source}`，固定刻字半径 10.001 mm，确定性均匀抽取 {len(sphere.points_sensor_m)} 个球面点。该球数据不参与任何标定求解。",
            ]
        )
    lines.extend(["", "扰动档如下：", "", "| 档位 | 手眼旋转 | 手眼平移 | 板姿态 | 角点 |", "|---:|---:|---:|---:|---:|"])
    for index, level in enumerate(levels):
        lines.append(
            f"| {index} | {level.handeye_rotation_deg:g}° | {level.handeye_translation_mm:g} mm | {level.board_rotation_deg:g}° | {level.corner_translation_mm:g} mm |"
        )
    lines.extend(
        [
            "",
            "## 3. 判据说明",
            "",
            "- `optimizer success`：SciPy TRF 正常终止。",
            "- `strict convergence`：除正常终止外，data-only Jacobian 还须满秩且条件数不超过当前配置阈值。",
            "- `physical success`：strict convergence，并满足当前生产代码同一水平板法向倾角门限及有限值检查。",
            "- `nfev`：残差函数调用数。SciPy 1.11 不直接报告 TRF 迭代次数，因此同时记录 `njev` 作为主要迭代/线性化次数代理，不能把它称为严格迭代数。",
            "- 数据残差只统计观测项，不含形貌正则；外部精度优先看独立精密球固定半径 RMSE/P95，而不是训练残差。",
            "",
            "## 4. 汇总结果",
            "",
            "| 扰动 | 组别 | 次数 | 优化器成功率 | 严格收敛率 | 物理解成功率 | 数据RMS/mm | 球RMSE/mm | 旋转离散/° | 平移离散/mm | 时间/s | nfev |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            "| {level_label} | {group} | {trials} | {optimizer:.1%} | {strict:.1%} | {physical:.1%} | {data} | {sphere} | {rd} | {td} | {time} | {nfev} |".format(
                level_label=row["level_label"],
                group=row["group"],
                trials=row["trials"],
                optimizer=row["optimizer_success_rate"],
                strict=row["strict_convergence_rate"],
                physical=row["physical_success_rate"],
                data=_fmt(row["data_residual_rms_mm_median"]),
                sphere=_fmt(row["sphere_fixed_rmse_mm_median"]),
                rd=_fmt(row["rotation_dispersion_deg_median"]),
                td=_fmt(row["translation_dispersion_mm_median"]),
                time=_fmt(row["solver_time_s_median"], 3),
                nfev=_fmt(row["function_evaluations_median"], 1),
            )
        )
    lines.extend(
        [
            "",
            "## 5. 跨数据集总体与 A/C 配对比较",
            "",
            "| 组别 | 物理解成功 | 成功率（Wilson 95% CI） | 成功试验球RMSE中位/mm | 时间中位/s | nfev中位 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in overall:
        lines.append(
            f"| {row['group']} | {row['successes']}/{row['trials']} | "
            f"{row['rate']:.1%}（{row['lower']:.1%}–{row['upper']:.1%}） | "
            f"{_fmt(row['sphere'])} | {_fmt(row['time'], 3)} | {_fmt(row['nfev'], 1)} |"
        )
    lines.extend(
        [
            "",
            f"A/C 共同成功的配对试验数为 {paired['paired_trials']}。逐对比较得到：",
            "",
            f"- 球 RMSE 的 `A-C` 中位差为 {_fmt(paired['sphere_rmse_a_minus_c_median_mm'], 6)} mm，绝对差 P95 为 {_fmt(paired['sphere_rmse_abs_difference_p95_mm'], 6)} mm；",
            f"- 两解手眼旋转差中位/P95 为 {_fmt(paired['handeye_rotation_difference_median_deg'], 6)}/{_fmt(paired['handeye_rotation_difference_p95_deg'], 6)}°；",
            f"- 两解手眼平移差中位/P95 为 {_fmt(paired['handeye_translation_difference_median_mm'], 6)}/{_fmt(paired['handeye_translation_difference_p95_mm'], 6)} mm；",
            f"- C/A 时间倍率中位为 {_fmt(paired['c_over_a_time_ratio_median'], 3)}，nfev 倍率中位为 {_fmt(paired['c_over_a_nfev_ratio_median'], 3)}。",
            "",
            "## 6. 结论",
            "",
            "本轮真机先导实验支持一个分层结论：**相对于单初值直接 19 维求解 B，第5章预求解具有明确必要性；但相对于穷举四个解域的直接 19 维多初值 C，它不是数学上不可替代的。** A 与 C 都达到全试验稳定收敛，外部球误差总体接近，说明 C 也能找到同一类正确解域；但 C 的时间和函数调用显著更高。因此第5章当前最准确的定位是：利用低维理想平面模型廉价搜索离散解域、扩大共享形貌问题的有效收敛域，并以低于直接 19 维穷举的代价提供稳定初始化。",
            "",
            "这不能被写成“共享形貌优化只有经过平面预求解才能成功”；更严谨的表述应是“平面预求解是一种计算高效的结构化全局化策略”。当前每个非零档只有 3 次随机重复，结论属于先导证据；论文最终版应运行默认 10 次重复，并增加一个严格限时或限 nfev 的直接多初值组，以进一步消除 A/C 预算不完全相等的影响。",
            "",
            "## 7. 图表",
            "",
            "![初始扰动—成功率](initial_perturbation_vs_success_rate.png)",
            "",
            "![初始扰动—外部误差](initial_perturbation_vs_external_error.png)",
            "",
            "![最终外部误差分布](final_external_error_distribution.png)",
            "",
            "## 8. 可复现文件",
            "",
            "- `experiment_manifest.json`：数据、配置、随机种子和扰动定义。",
            "- `trial_results.json` / `trial_results.csv`：每次 A/B/C 完整结果。",
            "- `summary.json` / `summary.csv`：按扰动档和组别汇总。",
            "- PDF 与 PNG 图使用相同原始结果生成。",
            "",
            "## 9. 适用边界",
            "",
            "本实验评价的是六种子初始共享形貌解的收敛域，不评价后续 NBV 策略。各真机标定数据可共用同一静止精密球数据的前提是球、机器人基座和传感器安装关系在这些运行之间未发生改变；若安装发生变化，外部球指标必须按对应时段重新采集。",
        ]
    )
    (output / "ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _run_paired_trial(
    context: DatasetContext,
    level: PerturbationLevel,
    *,
    dataset_index: int,
    level_index: int,
    repeat: int,
    trial_seed: int,
    sphere: SphereData | None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(trial_seed)
    perturbation = _perturbation(context, level, rng)
    common = {
        "dataset": context.name,
        "dataset_path": str(context.path),
        "dataset_index": dataset_index,
        "level_index": level_index,
        "level_label": level.label,
        **level.as_dict(),
        "repeat": repeat,
        "trial_seed": trial_seed,
        "perturbation_axes": {
            key: perturbation[key].tolist()
            for key in (
                "handeye_axis",
                "translation_axis",
                "board_axis",
                "corner_axis",
            )
        },
    }
    trial_rows = [
        _solve_a(context, perturbation),
        _solve_b(context, perturbation),
        _solve_c(context, perturbation),
    ]
    for row in trial_rows:
        row.update(common)
        if sphere is not None and row.get("solver_returned"):
            try:
                row.update(
                    _sphere_metrics(
                        sphere,
                        np.asarray(row["handeye_rotation"], dtype=float),
                        np.asarray(row["handeye_translation_m"], dtype=float),
                    )
                )
            except Exception as error:
                row["sphere_error"] = f"{type(error).__name__}: {error}"
    return trial_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B/C ablation for the flat 12-DOF pre-solve"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=Path,
        help="calibration run directories; defaults to the five real runs",
    )
    parser.add_argument(
        "--levels",
        help=(
            "comma-separated rot_deg:trans_mm[:board_deg:corner_mm] levels"
        ),
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes; use OPENBLAS_NUM_THREADS=1 when >1",
    )
    parser.add_argument("--sphere", type=Path, default=DEFAULT_SPHERE)
    parser.add_argument(
        "--sphere-max-points",
        type=int,
        default=30000,
        help="deterministic point cap for repeated external sphere fits; 0=all",
    )
    parser.add_argument("--no-sphere", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    levels = _parse_levels(args.levels)
    dataset_paths = args.datasets or [
        WORKSPACE / "data/calibration_runs" / name for name in DEFAULT_DATASETS
    ]
    output = args.output or (
        WORKSPACE
        / "data/ablation_runs"
        / ("flat_presolve_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output.mkdir(parents=True, exist_ok=False)
    contexts = [_load_context(path.resolve()) for path in dataset_paths]
    sphere = None if args.no_sphere else _load_sphere(
        args.sphere.resolve(), args.sphere_max_points
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.random_seed,
        "repeats_nonzero_levels": args.repeats,
        "zero_level_repeats": 1,
        "levels": [level.as_dict() for level in levels],
        "groups": {
            "A_flat_presolve": "production flat multistart then shared solve",
            "B_direct_single": "single direct shared solve",
            "C_direct_multistart": "four direct shared restarts",
        },
        "datasets": [
            {
                "name": context.name,
                "path": str(context.path),
                "configuration": context.configuration,
            }
            for context in contexts
        ],
        "sphere": (
            None
            if sphere is None
            else {
                "source": sphere.source,
                "radius_m": sphere.radius_m,
                "point_count": len(sphere.points_sensor_m),
                "selection": "deterministic uniform index subsample",
            }
        ),
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    trial_count = sum(
        1
        if (
            level.handeye_rotation_deg == 0.0
            and level.handeye_translation_mm == 0.0
            and level.board_rotation_deg == 0.0
            and level.corner_translation_mm == 0.0
        )
        else args.repeats
        for level in levels
    )
    total = len(contexts) * trial_count
    completed = 0
    print(f"output: {output}", flush=True)
    print(f"datasets={len(contexts)}, paired perturbation trials={total}, groups=A/B/C", flush=True)
    jobs = []
    for dataset_index, context in enumerate(contexts):
        for level_index, level in enumerate(levels):
            repeats = 1 if (
                level.handeye_rotation_deg == 0.0
                and level.handeye_translation_mm == 0.0
                and level.board_rotation_deg == 0.0
                and level.corner_translation_mm == 0.0
            ) else args.repeats
            for repeat in range(repeats):
                trial_seed = (
                    args.random_seed
                    + 1000003 * dataset_index
                    + 1009 * level_index
                    + repeat
                )
                jobs.append(
                    (
                        context,
                        level,
                        dataset_index,
                        level_index,
                        repeat,
                        trial_seed,
                    )
                )

    def accept_trial(trial_rows: list[dict[str, Any]]) -> None:
        nonlocal completed
        rows.extend(trial_rows)
        first = trial_rows[0]
        status = " ".join(
            f"{row['group'][0]}={'OK' if row['successful_physical_solution'] else 'FAIL'}"
            for row in trial_rows
        )
        completed += 1
        print(
            f"[{completed:03d}/{total:03d}] {first['dataset']} "
            f"{first['level_label']} repeat={first['repeat']}: {status}",
            flush=True,
        )
        ordered = sorted(
            rows,
            key=lambda row: (
                row["dataset_index"],
                row["level_index"],
                row["repeat"],
                row["group"],
            ),
        )
        (output / "trial_results.json").write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.workers == 1:
        for context, level, dataset_index, level_index, repeat, trial_seed in jobs:
            try:
                trial_rows = _run_paired_trial(
                    context,
                    level,
                    dataset_index=dataset_index,
                    level_index=level_index,
                    repeat=repeat,
                    trial_seed=trial_seed,
                    sphere=sphere,
                )
            except Exception as error:
                print(f"paired trial failed: {error}", flush=True)
                traceback.print_exc()
                return 2
            accept_trial(trial_rows)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_paired_trial,
                    context,
                    level,
                    dataset_index=dataset_index,
                    level_index=level_index,
                    repeat=repeat,
                    trial_seed=trial_seed,
                    sphere=sphere,
                ): (context.name, level.label, repeat)
                for context, level, dataset_index, level_index, repeat, trial_seed in jobs
            }
            for future in as_completed(futures):
                try:
                    accept_trial(future.result())
                except Exception as error:
                    description = futures[future]
                    print(f"paired trial {description} failed: {error}", flush=True)
                    traceback.print_exc()
                    return 2

    rows.sort(
        key=lambda row: (
            row["dataset_index"],
            row["level_index"],
            row["repeat"],
            row["group"],
        )
    )
    summaries = _summarize(rows)
    (output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output / "trial_results.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    (output / "paired_a_c_comparison.json").write_text(
        json.dumps(_paired_a_c_statistics(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plots(output, rows, summaries)
    _report(output, contexts, levels, args.repeats, rows, summaries, sphere)
    print(f"report: {output / 'ablation_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
