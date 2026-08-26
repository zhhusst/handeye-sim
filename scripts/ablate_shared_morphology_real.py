#!/usr/bin/env python3
"""Real-data ablation of ideal, shared and pose-specific target morphology.

All three groups use the same six physical seed poses, the same stationary
frame bootstrap sample and the same flat multi-start initialization.  They
then run the same SciPy TRF optimizer with identical tolerances, observation
weights and physical state scales.  Only the target surface model changes:

* A: ideal plane, beta fixed to zero;
* B: one degree-three Legendre beta shared by every physical pose;
* C: one independent degree-three Legendre beta_i per physical pose.

The independent precision sphere is never used during calibration.
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
from scipy.optimize import least_squares


WORKSPACE = Path(__file__).resolve().parents[1]
CORE_SOURCE = WORKSPACE / "ros2_ws/src/handeye_calibration_core"
sys.path.insert(0, str(CORE_SOURCE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from ablate_flat_presolve_initialization import (  # noqa: E402
    DEFAULT_SPHERE,
    SphereData,
    _deep_merge,
    _load_sphere,
    _ros_parameters,
)
from calibration_pipeline.dataset_io import (  # noqa: E402
    SeedObservationGroup,
    aggregate_seed_group,
    load_seed_dataset_grouped,
)
from calibration_pipeline.geometry import so3_exp, so3_log  # noqa: E402
from calibration_pipeline.solvers.twelve_dof_v2 import (  # noqa: E402
    TwelveDofV2Solver,
)
from calibration_pipeline.sphere_validation import (  # noqa: E402
    _fixed_radius_center,
    _free_sphere,
)
from calibration_pipeline.v2_backend.information import (  # noqa: E402
    observability,
)
from calibration_pipeline.v2_backend.shared_surface import (  # noqa: E402
    SurfaceBasis,
    get_surface_basis,
)


DEFAULT_DATASETS = (
    "20260820_115707_圆点标定板背面_位置1_真机_2",
    "20260820_132928_圆点标定板背面_位置1_真机_3",
    "20260820_133622_圆点标定板背面_位置1_真机_5",
    "20260820_135111_圆点标定板背面_位置2_真机_1",
    "20260820_140109_圆点标定板背面_位置2_真机_2",
    "20260820_141546_原点标定板背面_位置3_真机_1",
    "20260820_142718_知象光电宣传册_位置1_真机_1_nbv初始化失败",
    "20260820_144753_焊接书_位置1_真机_1",
    "20260820_150032_生锈的刚板_位置1_真机_1",
)
GROUPS = (
    "A_ideal_plane",
    "B_shared_morphology",
    "C_pose_specific_morphology",
)
GROUP_SHORT = {
    "A_ideal_plane": "A Ideal plane",
    "B_shared_morphology": "B Shared morphology",
    "C_pose_specific_morphology": "C Pose-specific morphology",
}


@dataclass
class DatasetContext:
    name: str
    path: Path
    groups: tuple[SeedObservationGroup, ...]
    nominal_rotation: np.ndarray
    nominal_translation: np.ndarray
    board_dimensions: tuple[float, float]
    flat_solver: TwelveDofV2Solver
    basis: SurfaceBasis
    shape_regularization: float
    maximum_board_tilt_deg: float
    maximum_condition_number: float
    max_evaluations: int
    tolerance: float
    geometric_scale: np.ndarray
    shape_scale_m: float
    weights: dict[str, float]
    configuration: dict[str, Any]


def _load_context(path: Path) -> DatasetContext:
    parameters = _ros_parameters(path / "calibration_parameters.yaml")
    overlay = path / "real_calibration_parameters.yaml"
    if overlay.exists():
        parameters = _deep_merge(parameters, _ros_parameters(overlay))
    solver_values = parameters["solver"]
    multistart = solver_values["multistart"]
    handeye = parameters["initial_handeye"]
    board = parameters["board"]
    rotation_scale = math.radians(
        float(solver_values["handeye_rotation_scale_deg"])
    )
    translation_scale = float(solver_values["handeye_translation_scale_m"])
    board_rotation_scale = math.radians(
        float(solver_values["plane_rotation_scale_deg"])
    )
    geometric_scale = np.array(
        [rotation_scale] * 3
        + [translation_scale] * 3
        + [board_rotation_scale] * 3
        + [translation_scale] * 3,
        dtype=float,
    )
    weights = {
        "plane_weight": float(solver_values["plane_weight"]),
        "edge_weight": float(solver_values["edge_weight"]),
        "endpoint_plane_weight": float(
            solver_values["endpoint_plane_weight"]
        ),
    }
    flat_solver = TwelveDofV2Solver(
        **weights,
        max_evaluations=int(solver_values["max_evaluations"]),
        tolerance=float(solver_values["tolerance"]),
        state_scale=geometric_scale[:9],
        maximum_condition_number=float(
            solver_values["maximum_condition_number"]
        ),
        surface_model="flat",
        multistart_enabled=True,
        multistart_maximum_board_tilt_deg=float(
            multistart["maximum_board_tilt_deg"]
        ),
        # The failed brochure run is still solved and reported.  If no
        # horizontal hypothesis exists, the cheapest converged flat result is
        # retained as a common initialization and marked implausible.
        multistart_require_plausible=False,
    )
    dataset = load_seed_dataset_grouped(path / "seeds.json")
    if dataset.physical_seed_count != 6:
        raise ValueError(
            f"{path}: expected six physical seeds, got "
            f"{dataset.physical_seed_count}"
        )
    basis = get_surface_basis(
        str(solver_values["surface_basis_kind"]),
        int(solver_values["surface_degree"]),
    )
    return DatasetContext(
        name=path.name,
        path=path,
        groups=dataset.groups,
        nominal_rotation=np.asarray(handeye["rotation"], dtype=float).reshape(3, 3),
        nominal_translation=np.asarray(handeye["translation_m"], dtype=float),
        board_dimensions=(
            float(board["length_u_m"]),
            float(board["length_v_m"]),
        ),
        flat_solver=flat_solver,
        basis=basis,
        shape_regularization=float(solver_values["shape_regularization"]),
        maximum_board_tilt_deg=float(multistart["maximum_board_tilt_deg"]),
        maximum_condition_number=float(
            solver_values["maximum_condition_number"]
        ),
        max_evaluations=int(solver_values["max_evaluations"]),
        tolerance=float(solver_values["tolerance"]),
        geometric_scale=geometric_scale,
        shape_scale_m=float(solver_values["shape_scale_m"]),
        weights=weights,
        configuration={
            "physical_seed_count": dataset.physical_seed_count,
            "synchronized_frame_count": dataset.observation_count,
            "frames_per_pose": [len(group.poses) for group in dataset.groups],
            "solver": solver_values,
            "board": board,
            "initial_handeye": handeye,
        },
    )


def _bootstrap_aggregate(
    context: DatasetContext,
    repeat: int,
    random_seed: int,
) -> tuple[list[Any], list[Any], list[list[int]]]:
    rng = np.random.default_rng(random_seed)
    poses = []
    measurements = []
    sampled_indices = []
    for group in context.groups:
        frame_count = len(group.poses)
        if repeat == 0:
            indices = np.arange(frame_count, dtype=int)
        else:
            indices = rng.integers(0, frame_count, size=frame_count)
        sampled = SeedObservationGroup(
            group.label,
            tuple(group.poses[int(index)] for index in indices),
            tuple(group.measurements[int(index)] for index in indices),
        )
        pose, measurement = aggregate_seed_group(sampled)
        poses.append(pose)
        measurements.append(measurement)
        sampled_indices.append(indices.tolist())
    return poses, measurements, sampled_indices


def _common_flat_initialization(
    context: DatasetContext,
    poses: list[Any],
    measurements: list[Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    flat, attempts, selected_name, selected_tilt = (
        context.flat_solver._solve_flat_multistart(
            poses,
            measurements,
            context.nominal_rotation,
            context.nominal_translation,
            board_dimensions=context.board_dimensions,
            initial_board_rotation=None,
        )
    )
    initial = np.concatenate((flat.estimate.x9, flat.estimate.board.corner))
    return initial, {
        "selected": selected_name,
        "selected_tilt_deg": float(selected_tilt),
        "selected_cost": float(flat.cost),
        "physically_plausible": bool(
            selected_tilt <= context.maximum_board_tilt_deg
        ),
        "attempts": list(attempts),
    }


def _model_state_scale(
    context: DatasetContext,
    model: str,
    pose_count: int,
) -> np.ndarray:
    if model == "ideal":
        return context.geometric_scale.copy()
    shape_count = context.basis.size * (1 if model == "shared" else pose_count)
    return np.concatenate(
        (
            context.geometric_scale,
            np.full(shape_count, context.shape_scale_m),
        )
    )


def _initial_state(
    context: DatasetContext,
    model: str,
    pose_count: int,
    geometric_initial: np.ndarray,
) -> np.ndarray:
    if model == "ideal":
        return geometric_initial.copy()
    shape_count = context.basis.size * (1 if model == "shared" else pose_count)
    return np.concatenate((geometric_initial, np.zeros(shape_count)))


def _residual_components(
    state: np.ndarray,
    poses: list[Any],
    measurements: list[Any],
    *,
    context: DatasetContext,
    model: str,
    include_regularization: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted observation, unweighted surface and prior residuals."""
    state = np.asarray(state, dtype=float)
    pose_count = len(poses)
    expected = 12
    if model == "shared":
        expected += context.basis.size
    elif model == "pose_specific":
        expected += pose_count * context.basis.size
    elif model != "ideal":
        raise ValueError(f"unknown morphology model {model}")
    if state.shape != (expected,):
        raise ValueError(f"{model} state must have shape ({expected},)")
    handeye_rotation = so3_exp(state[:3])
    handeye_translation = state[3:6]
    board_rotation = so3_exp(state[6:9])
    corner = state[9:12]
    u, v, normal = board_rotation.T
    width, height = context.board_dimensions
    data_rows: list[np.ndarray] = []
    surface_rows: list[np.ndarray] = []
    plane_weight = context.weights["plane_weight"]
    edge_weight = context.weights["edge_weight"]
    endpoint_weight = context.weights["endpoint_plane_weight"]

    def coefficients_for(index: int) -> np.ndarray:
        if model == "ideal":
            return np.zeros(context.basis.size)
        if model == "shared":
            return state[12 : 12 + context.basis.size]
        start = 12 + index * context.basis.size
        return state[start : start + context.basis.size]

    def surface_distance(points_base: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        delta = np.asarray(points_base, dtype=float).reshape(-1, 3) - corner[None, :]
        xi = (delta @ u) / width
        eta = (delta @ v) / height
        height_values = (
            np.zeros(len(delta))
            if model == "ideal"
            else context.basis.height(xi, eta, coefficients)
        )
        return delta @ normal - height_values

    for pose_index, (pose, measurement) in enumerate(zip(poses, measurements)):
        coefficients = coefficients_for(pose_index)
        sensor_rotation = pose.rotation @ handeye_rotation
        sensor_translation = pose.translation + pose.rotation @ handeye_translation
        points_base = (
            sensor_rotation @ measurement.profile_points.T
        ).T + sensor_translation
        profile_surface = surface_distance(points_base, coefficients)
        surface_rows.append(profile_surface)
        data_rows.append(
            math.sqrt(plane_weight / max(len(points_base), 1))
            * profile_surface
        )
        endpoint_u = sensor_rotation @ measurement.endpoint_u + sensor_translation
        endpoint_v = sensor_rotation @ measurement.endpoint_v + sensor_translation
        endpoint_u_surface = float(
            surface_distance(endpoint_u[None, :], coefficients)[0]
        )
        endpoint_v_surface = float(
            surface_distance(endpoint_v[None, :], coefficients)[0]
        )
        data_rows.append(
            np.array(
                [
                    math.sqrt(edge_weight) * float(v @ (endpoint_u - corner)),
                    math.sqrt(endpoint_weight) * endpoint_u_surface,
                    math.sqrt(edge_weight) * float(u @ (endpoint_v - corner)),
                    math.sqrt(endpoint_weight) * endpoint_v_surface,
                ]
            )
        )
    prior = np.empty(0, dtype=float)
    if model != "ideal" and include_regularization and context.shape_regularization > 0:
        prior = math.sqrt(context.shape_regularization) * state[12:]
    return np.concatenate(data_rows), np.concatenate(surface_rows), prior


def _solve_model(
    context: DatasetContext,
    model: str,
    poses: list[Any],
    measurements: list[Any],
    geometric_initial: np.ndarray,
) -> dict[str, Any]:
    pose_count = len(poses)
    initial = _initial_state(context, model, pose_count, geometric_initial)
    scale = _model_state_scale(context, model, pose_count)

    def objective(value: np.ndarray) -> np.ndarray:
        data, _surface, prior = _residual_components(
            value,
            poses,
            measurements,
            context=context,
            model=model,
            include_regularization=True,
        )
        return np.concatenate((data, prior))

    start = time.perf_counter()
    optimized = least_squares(
        objective,
        initial,
        method="trf",
        x_scale=scale,
        max_nfev=context.max_evaluations,
        ftol=context.tolerance,
        xtol=context.tolerance,
        gtol=context.tolerance,
    )
    elapsed = time.perf_counter() - start
    state = optimized.x.copy()
    data, surface, prior = _residual_components(
        state,
        poses,
        measurements,
        context=context,
        model=model,
        include_regularization=True,
    )
    prior_count = len(prior)
    data_jacobian = (
        optimized.jac[:-prior_count] if prior_count else optimized.jac
    )
    data_singular, data_rank, data_condition = observability(
        data_jacobian, state_scale=scale
    )
    prior_singular, prior_rank, prior_condition = observability(
        optimized.jac, state_scale=scale
    )
    handeye_rotation = so3_exp(state[:3])
    handeye_translation = state[3:6]
    board_rotation = so3_exp(state[6:9])
    board_normal = board_rotation[:, 2]
    board_tilt = math.degrees(
        math.acos(float(np.clip(abs(board_normal[2]), 0.0, 1.0)))
    )
    finite = bool(np.all(np.isfinite(state)) and np.all(np.isfinite(data)))
    physically_reasonable = bool(
        finite
        and board_tilt <= context.maximum_board_tilt_deg
        and np.linalg.norm(handeye_translation) < 1.0
    )
    strict = bool(
        optimized.success
        and data_rank == len(state)
        and data_condition <= context.maximum_condition_number
    )
    return {
        "model": model,
        "optimizer_success": bool(optimized.success),
        "optimizer_status": int(optimized.status),
        "strict_data_convergence": strict,
        "physically_reasonable": physically_reasonable,
        "usable_solution": bool(optimized.success and physically_reasonable),
        "message": str(optimized.message),
        "state_size": int(len(state)),
        "data_rank": int(data_rank),
        "data_condition_number": float(data_condition),
        "prior_rank": int(prior_rank),
        "prior_condition_number": float(prior_condition),
        "data_singular_values": data_singular.tolist(),
        "prior_singular_values": prior_singular.tolist(),
        "objective_cost": 0.5 * float(objective(state) @ objective(state)),
        "data_cost": 0.5 * float(data @ data),
        "weighted_data_residual_rms_mm": 1000.0
        * float(np.sqrt(np.mean(data * data))),
        "surface_residual_rms_mm": 1000.0
        * float(np.sqrt(np.mean(surface * surface))),
        "surface_residual_mean_abs_mm": 1000.0
        * float(np.mean(np.abs(surface))),
        "surface_residual_max_abs_mm": 1000.0
        * float(np.max(np.abs(surface))),
        "board_tilt_deg": float(board_tilt),
        "handeye_rotation": handeye_rotation.tolist(),
        "handeye_translation_m": handeye_translation.tolist(),
        "board_rotation": board_rotation.tolist(),
        "board_corner_m": state[9:12].tolist(),
        "shape_coefficients_m": state[12:].tolist(),
        "shape_coefficient_rms_mm": (
            0.0
            if len(state) == 12
            else 1000.0 * float(np.sqrt(np.mean(state[12:] ** 2)))
        ),
        "function_evaluations": int(optimized.nfev),
        "jacobian_evaluations": (
            None if optimized.njev is None else int(optimized.njev)
        ),
        "solver_time_s": float(elapsed),
    }


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
    fixed = (
        np.linalg.norm(points_base - fixed_center[None, :], axis=1)
        - sphere.radius_m
    )
    free_center, free_radius = _free_sphere(
        points_base, robust_scale_m=0.00010
    )
    free = (
        np.linalg.norm(points_base - free_center[None, :], axis=1)
        - free_radius
    )
    return {
        "sphere_point_count": int(len(points_base)),
        "sphere_fixed_rmse_mm": 1000.0
        * float(np.sqrt(np.mean(fixed * fixed))),
        "sphere_fixed_mean_abs_mm": 1000.0 * float(np.mean(np.abs(fixed))),
        "sphere_fixed_p95_mm": 1000.0
        * float(np.percentile(np.abs(fixed), 95.0)),
        "sphere_fixed_max_abs_mm": 1000.0 * float(np.max(np.abs(fixed))),
        "sphere_free_radius_mm": 1000.0 * float(free_radius),
        "sphere_radius_error_mm": 1000.0
        * float(free_radius - sphere.radius_m),
        "sphere_abs_radius_error_mm": 1000.0
        * abs(float(free_radius - sphere.radius_m)),
        "sphere_free_rmse_mm": 1000.0
        * float(np.sqrt(np.mean(free * free))),
    }


def _run_paired_repeat(
    context: DatasetContext,
    repeat: int,
    random_seed: int,
    sphere: SphereData,
) -> list[dict[str, Any]]:
    poses, measurements, indices = _bootstrap_aggregate(
        context, repeat, random_seed
    )
    start = time.perf_counter()
    try:
        geometric_initial, initialization = _common_flat_initialization(
            context, poses, measurements
        )
    except Exception as error:
        return [
            {
                "group": group,
                "model": model,
                "dataset": context.name,
                "dataset_path": str(context.path),
                "repeat": repeat,
                "random_seed": random_seed,
                "sampled_frame_indices": indices,
                "common_initialization_failed": True,
                "optimizer_success": False,
                "strict_data_convergence": False,
                "physically_reasonable": False,
                "usable_solution": False,
                "message": f"common flat initialization failed: {type(error).__name__}: {error}",
                "common_initialization_time_s": time.perf_counter() - start,
            }
            for group, model in zip(GROUPS, ("ideal", "shared", "pose_specific"))
        ]
    initialization_time = time.perf_counter() - start
    rows = []
    for group, model in zip(GROUPS, ("ideal", "shared", "pose_specific")):
        try:
            row = _solve_model(
                context,
                model,
                poses,
                measurements,
                geometric_initial,
            )
            if row["optimizer_success"]:
                row.update(
                    _sphere_metrics(
                        sphere,
                        np.asarray(row["handeye_rotation"], dtype=float),
                        np.asarray(row["handeye_translation_m"], dtype=float),
                    )
                )
        except Exception as error:
            row = {
                "model": model,
                "optimizer_success": False,
                "strict_data_convergence": False,
                "physically_reasonable": False,
                "usable_solution": False,
                "message": f"{type(error).__name__}: {error}",
            }
        row.update(
            {
                "group": group,
                "dataset": context.name,
                "dataset_path": str(context.path),
                "repeat": repeat,
                "random_seed": random_seed,
                "sampled_frame_indices": indices,
                "common_initialization_failed": False,
                "common_initialization": initialization,
                "common_initialization_time_s": initialization_time,
            }
        )
        rows.append(row)
    return rows


def _finite(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return values


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    return math.degrees(float(np.linalg.norm(so3_log(relative))))


def _dispersion(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    valid = [row for row in rows if row.get("usable_solution")]
    if len(valid) < 2:
        return None, None
    rotations = [np.asarray(row["handeye_rotation"], dtype=float) for row in valid]
    translations = np.asarray(
        [row["handeye_translation_m"] for row in valid], dtype=float
    )
    pairwise = np.array(
        [
            [_rotation_distance_deg(first, second) for second in rotations]
            for first in rotations
        ]
    )
    medoid = int(np.argmin(np.sum(pairwise**2, axis=1)))
    rotation_rms = float(np.sqrt(np.mean(pairwise[medoid] ** 2)))
    center = np.mean(translations, axis=0)
    translation_rms = 1000.0 * float(
        np.sqrt(np.mean(np.sum((translations - center) ** 2, axis=1)))
    )
    return rotation_rms, translation_rms


def _median_or_none(rows: list[dict], key: str) -> float | None:
    values = _finite(rows, key)
    return None if not values else float(np.median(values))


def _dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for group in GROUPS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["group"] == group
            ]
            usable = [row for row in selected if row.get("usable_solution")]
            rotation_dispersion, translation_dispersion = _dispersion(selected)
            output.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "trials": len(selected),
                    "optimizer_success_rate": float(
                        np.mean([row.get("optimizer_success", False) for row in selected])
                    ),
                    "strict_data_convergence_rate": float(
                        np.mean(
                            [
                                row.get("strict_data_convergence", False)
                                for row in selected
                            ]
                        )
                    ),
                    "usable_solution_rate": float(
                        np.mean([row.get("usable_solution", False) for row in selected])
                    ),
                    "sphere_fixed_rmse_mm": _median_or_none(
                        usable, "sphere_fixed_rmse_mm"
                    ),
                    "sphere_fixed_mean_abs_mm": _median_or_none(
                        usable, "sphere_fixed_mean_abs_mm"
                    ),
                    "sphere_fixed_p95_mm": _median_or_none(
                        usable, "sphere_fixed_p95_mm"
                    ),
                    "sphere_fixed_max_abs_mm": _median_or_none(
                        usable, "sphere_fixed_max_abs_mm"
                    ),
                    "sphere_radius_error_mm": _median_or_none(
                        usable, "sphere_radius_error_mm"
                    ),
                    "sphere_abs_radius_error_mm": _median_or_none(
                        usable, "sphere_abs_radius_error_mm"
                    ),
                    "surface_residual_rms_mm": _median_or_none(
                        usable, "surface_residual_rms_mm"
                    ),
                    "weighted_data_residual_rms_mm": _median_or_none(
                        usable, "weighted_data_residual_rms_mm"
                    ),
                    "rotation_dispersion_deg": rotation_dispersion,
                    "translation_dispersion_mm": translation_dispersion,
                    "data_rank": _median_or_none(usable, "data_rank"),
                    "state_size": _median_or_none(usable, "state_size"),
                    "solver_time_s": _median_or_none(selected, "solver_time_s"),
                }
            )
    return output


def _paired_dataset_differences(
    summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["group"]): row for row in summary}
    output = []
    metrics = (
        "sphere_fixed_rmse_mm",
        "sphere_fixed_mean_abs_mm",
        "sphere_fixed_p95_mm",
        "sphere_fixed_max_abs_mm",
        "sphere_radius_error_mm",
        "sphere_abs_radius_error_mm",
        "surface_residual_rms_mm",
        "rotation_dispersion_deg",
        "translation_dispersion_mm",
    )
    for dataset in sorted({row["dataset"] for row in summary}):
        a = by_key[(dataset, "A_ideal_plane")]
        b = by_key[(dataset, "B_shared_morphology")]
        row: dict[str, Any] = {"dataset": dataset}
        for metric in metrics:
            a_value = a.get(metric)
            b_value = b.get(metric)
            row[f"A_{metric}"] = a_value
            row[f"B_{metric}"] = b_value
            row[f"B_minus_A_{metric}"] = (
                None
                if a_value is None or b_value is None
                else float(b_value - a_value)
            )
        output.append(row)
    return output


def _bootstrap_difference(
    values: list[float],
    *,
    trials: int,
    random_seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "mean_ci95": [None, None],
            "median_ci95": [None, None],
        }
    rng = np.random.default_rng(random_seed)
    samples = array[rng.integers(0, len(array), size=(trials, len(array)))]
    means = np.mean(samples, axis=1)
    medians = np.median(samples, axis=1)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "absolute_p95": float(np.percentile(np.abs(array), 95.0)),
        "improved_fraction": float(np.mean(array < 0.0)),
        "mean_ci95": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "median_ci95": [
            float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)),
        ],
    }


def _paired_statistics(
    differences: list[dict[str, Any]],
    *,
    bootstrap_trials: int,
    random_seed: int,
) -> dict[str, Any]:
    metrics = (
        "sphere_fixed_rmse_mm",
        "sphere_fixed_mean_abs_mm",
        "sphere_fixed_p95_mm",
        "sphere_fixed_max_abs_mm",
        "sphere_radius_error_mm",
        "sphere_abs_radius_error_mm",
        "surface_residual_rms_mm",
        "rotation_dispersion_deg",
        "translation_dispersion_mm",
    )
    output = {}
    for index, metric in enumerate(metrics):
        values = [
            row[f"B_minus_A_{metric}"]
            for row in differences
            if row.get(f"B_minus_A_{metric}") is not None
        ]
        output[metric] = _bootstrap_difference(
            values,
            trials=bootstrap_trials,
            random_seed=random_seed + 104729 * index,
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar = [
        {
            key: value
            for key, value in row.items()
            if value is None or isinstance(value, (str, int, float, bool, np.number))
        }
        for row in rows
    ]
    keys = sorted({key for row in scalar for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(scalar)


def _font_setup() -> None:
    from matplotlib import font_manager

    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
    else:
        family = "Droid Sans Fallback"
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Droid Sans Fallback", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )


def _plots(
    output: Path,
    summary: list[dict[str, Any]],
    differences: list[dict[str, Any]],
) -> None:
    _font_setup()
    datasets = sorted({row["dataset"] for row in summary})
    short_names = [f"D{index + 1}" for index in range(len(datasets))]
    colors = ("#777777", "#1f77b4", "#d62728")
    x = np.arange(len(datasets), dtype=float)
    width = 0.25
    fig, axis = plt.subplots(figsize=(12.0, 5.8))
    for group_index, group in enumerate(GROUPS):
        values = []
        for dataset in datasets:
            row = next(
                item
                for item in summary
                if item["dataset"] == dataset and item["group"] == group
            )
            values.append(row["sphere_fixed_rmse_mm"])
        axis.bar(
            x + (group_index - 1) * width,
            [np.nan if value is None else value for value in values],
            width,
            color=colors[group_index],
            label=GROUP_SHORT[group],
        )
    axis.set_xticks(x, short_names)
    axis.set_ylabel("独立球固定半径 RMSE (mm)")
    axis.set_xlabel("真机标定数据集")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "per_dataset_external_sphere_abc.png")
    fig.savefig(output / "per_dataset_external_sphere_abc.pdf")
    plt.close(fig)

    metrics = (
        ("sphere_fixed_rmse_mm", "Δ球面RMSE (mm)"),
        ("sphere_abs_radius_error_mm", "Δ|半径误差| (mm)"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.2), sharex=True)
    for axis, (metric, label) in zip(axes, metrics):
        values = [
            next(row for row in differences if row["dataset"] == dataset).get(
                f"B_minus_A_{metric}"
            )
            for dataset in datasets
        ]
        colors_delta = [
            "#2ca02c" if value is not None and value < 0.0 else "#d62728"
            for value in values
        ]
        axis.bar(
            short_names,
            [np.nan if value is None else value for value in values],
            color=colors_delta,
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("真机标定数据集（B-A，负值表示共享形貌更好）")
    fig.tight_layout()
    fig.savefig(output / "paired_a_b_external_precision_difference.png")
    fig.savefig(output / "paired_a_b_external_precision_difference.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 6.2))
    markers = ("o", "s", "^")
    for group_index, group in enumerate(GROUPS):
        selected = [row for row in summary if row["group"] == group]
        x_values = [row["surface_residual_rms_mm"] for row in selected]
        y_values = [row["sphere_fixed_rmse_mm"] for row in selected]
        axis.scatter(
            x_values,
            y_values,
            s=60,
            marker=markers[group_index],
            color=colors[group_index],
            label=GROUP_SHORT[group],
            alpha=0.85,
        )
        for short, x_value, y_value in zip(short_names, x_values, y_values):
            if x_value is not None and y_value is not None:
                axis.annotate(short, (x_value, y_value), fontsize=8, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("标定数据内部 surface residual RMS (mm)")
    axis.set_ylabel("独立球固定半径 RMSE (mm)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "internal_residual_vs_external_error.png")
    fig.savefig(output / "internal_residual_vs_external_error.pdf")
    plt.close(fig)

    mapping = [f"- D{index + 1}: `{dataset}`" for index, dataset in enumerate(datasets)]
    (output / "dataset_label_mapping.md").write_text(
        "# 图中数据集缩写\n\n" + "\n".join(mapping) + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _conclusion(
    summary: list[dict[str, Any]], paired: dict[str, Any]
) -> str:
    rmse = paired["sphere_fixed_rmse_mm"]
    radius = paired["sphere_abs_radius_error_mm"]
    surface = paired["surface_residual_rms_mm"]
    c_rows = [row for row in summary if row["group"] == "C_pose_specific_morphology"]
    b_rows = [row for row in summary if row["group"] == "B_shared_morphology"]
    c_external = _finite(c_rows, "sphere_fixed_rmse_mm")
    b_external = _finite(b_rows, "sphere_fixed_rmse_mm")
    c_surface = _finite(c_rows, "surface_residual_rms_mm")
    b_surface = _finite(b_rows, "surface_residual_rms_mm")
    external_supported = bool(
        rmse["count"] >= 5
        and rmse["improved_fraction"] >= 0.7
        and rmse["mean_ci95"][1] < 0.0
        and radius["improved_fraction"] >= 0.7
    )
    only_fit = bool(
        surface["median"] is not None
        and surface["median"] < 0.0
        and (
            rmse["mean_ci95"][0] <= 0.0 <= rmse["mean_ci95"][1]
            or abs(rmse["median"]) < 0.005
        )
    )
    c_overfits = bool(
        c_surface
        and b_surface
        and np.median(c_surface) < np.median(b_surface)
        and c_external
        and b_external
        and np.median(c_external) > np.median(b_external)
    )
    if external_supported:
        first = (
            "B 相比 A 在多数数据集上同时降低独立球 RMSE 和绝对半径误差，且 RMSE 的配对 bootstrap 均值差 95% CI 完全小于零。"
            "这支持固定板面非理想形貌会污染手眼估计，而跨位姿共享形貌能够隔离该偏差并提高外参精度。"
        )
    elif only_fit:
        first = (
            "B 明显降低内部表面残差，但独立球 RMSE 的 A-B 配对差置信区间跨越零或效应量很小。"
            "当前证据只支持拟合改善，不支持把共享形貌作为核心精度创新。"
        )
    else:
        first = (
            "A-B 的独立球指标未形成跨数据集一致优势，当前真机数据不足以证明共享形貌稳定提升外参精度。"
        )
    second = (
        "C 虽获得更低内部残差，但外部球误差高于 B，支持形貌参数必须跨位姿共享；逐位姿独立形貌会吸收位姿相关约束并导致过拟合。"
        if c_overfits
        else "C 尚未同时表现出“内部残差最低、外部精度更差”的稳定模式，因此跨位姿共享的必要性仍需结合秩、条件数和结果离散度判断。"
    )
    return first + second


def _report(
    output: Path,
    contexts: list[DatasetContext],
    repeats: int,
    summary: list[dict[str, Any]],
    differences: list[dict[str, Any]],
    paired: dict[str, Any],
    sphere: SphereData,
) -> None:
    lines = [
        "# 位姿共享平板形貌模型真机消融报告",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. 实验口径",
        "",
        "九组真机运行统一使用自动 NBV 之前的六物理种子数据。原因是后续 NBV 位姿由共享形貌模型闭环选择，且各运行 NBV 数量不同；直接复用这些数据会让观测集合依赖 B 组并排除初始化失败数据，违反只改变板面模型的消融原则。",
        "",
        "每次配对重复先在六个位姿内以同一随机索引重采样同步帧并聚合，然后只运行一次公共理想平面多初值预解。A/B/C 从完全相同的 12 维几何初值出发，使用同一 TRF 优化器、容差、最大函数调用数、观测权重和物理状态尺度。",
        "",
        "- A：理想平面，形貌系数固定为零；",
        "- B：所有六个位姿共享一组 7 维三阶 Legendre 系数；",
        "- C：每个位姿独立一组 7 维三阶 Legendre 系数，共 42 个形貌干扰参数。",
        "",
        f"每组运行 {repeats} 次，其中 repeat 0 使用全部原始帧，其余为位姿内 bootstrap。外部评价固定使用 `{sphere.source}` 的 {len(sphere.points_sensor_m)} 个确定性抽样球面点，刻字半径为 {1000.0 * sphere.radius_m:.3f} mm。",
        "",
        "## 2. 逐数据集 A/B/C 外部球验证",
        "",
        "下表数值为各重复的中位数。球指标不参与标定优化。",
        "",
        "| 数据集 | 组 | 可用率 | data-only秩/维数 | 内部RMS/mm | 球RMSE/mm | 平均绝对距离/mm | 最大距离/mm | 半径误差/mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_index, context in enumerate(contexts):
        for group in GROUPS:
            row = next(
                item
                for item in summary
                if item["dataset"] == context.name and item["group"] == group
            )
            lines.append(
                f"| D{dataset_index + 1} | {group[0]} | {row['usable_solution_rate']:.0%} | {_fmt(row['data_rank'], 0)}/{_fmt(row['state_size'], 0)} | {_fmt(row['surface_residual_rms_mm'])} | {_fmt(row['sphere_fixed_rmse_mm'])} | {_fmt(row['sphere_fixed_mean_abs_mm'])} | {_fmt(row['sphere_fixed_max_abs_mm'])} | {_fmt(row['sphere_radius_error_mm'])} |"
            )
    lines.extend(
        [
            "",
            "### 2.1 重复求解的手眼结果离散度",
            "",
            "| 数据集 | 组 | 旋转离散/° | 平移离散/mm |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset_index, context in enumerate(contexts):
        for group in GROUPS:
            row = next(
                item
                for item in summary
                if item["dataset"] == context.name and item["group"] == group
            )
            lines.append(
                f"| D{dataset_index + 1} | {group[0]} | {_fmt(row['rotation_dispersion_deg'], 6)} | {_fmt(row['translation_dispersion_mm'], 6)} |"
            )
    lines.extend(
        [
            "",
            "## 3. A-B 配对差值",
            "",
            "所有差值定义为 `B-A`；对误差和残差而言，负值表示共享形貌更好。",
            "",
            "| 数据集 | Δ球RMSE/mm | Δ球平均绝对距离/mm | Δ球P95/mm | Δ球最大距离/mm | Δ绝对半径误差/mm | Δsurface RMS/mm |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(differences):
        lines.append(
            f"| D{index + 1} | {_fmt(row['B_minus_A_sphere_fixed_rmse_mm'], 6)} | {_fmt(row['B_minus_A_sphere_fixed_mean_abs_mm'], 6)} | {_fmt(row['B_minus_A_sphere_fixed_p95_mm'], 6)} | {_fmt(row['B_minus_A_sphere_fixed_max_abs_mm'], 6)} | {_fmt(row['B_minus_A_sphere_abs_radius_error_mm'], 6)} | {_fmt(row['B_minus_A_surface_residual_rms_mm'], 6)} |"
        )
    lines.extend(
        [
            "",
            "### 3.1 跨数据集配对统计",
            "",
            "| 指标（B-A） | n | 均值 | 中位数 | P95 | 改善比例 | 均值 bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    metric_labels = {
        "sphere_fixed_rmse_mm": "球RMSE/mm",
        "sphere_fixed_mean_abs_mm": "球平均绝对距离/mm",
        "sphere_fixed_p95_mm": "球P95/mm",
        "sphere_fixed_max_abs_mm": "球最大距离/mm",
        "sphere_abs_radius_error_mm": "绝对半径误差/mm",
        "surface_residual_rms_mm": "surface RMS/mm",
        "rotation_dispersion_deg": "旋转离散/°",
        "translation_dispersion_mm": "平移离散/mm",
    }
    for metric, label in metric_labels.items():
        stats = paired[metric]
        lines.append(
            f"| {label} | {stats['count']} | {_fmt(stats['mean'], 6)} | {_fmt(stats['median'], 6)} | {_fmt(stats['p95'], 6)} | {_fmt(100.0 * stats.get('improved_fraction', 0.0), 1)}% | [{_fmt(stats['mean_ci95'][0], 6)}, {_fmt(stats['mean_ci95'][1], 6)}] |"
        )
    lines.extend(
        [
            "",
            "## 4. 结论",
            "",
            _conclusion(summary, paired),
            "",
            "内部 surface residual 只能说明模型对标定数据的解释能力，不能单独证明手眼精度提高。结论以独立球指标、配对差值和重复离散度为主。",
            "每次求解的完整手眼旋转矩阵、平移向量、形貌系数、秩、条件数、函数调用次数和耗时均保存在 `trial_results.json`；便于统计的软件字段保存在 `trial_results.csv`。",
            "",
            "## 5. 图表",
            "",
            "![逐数据集外部球结果](per_dataset_external_sphere_abc.png)",
            "",
            "![A-B外部精度差值](paired_a_b_external_precision_difference.png)",
            "",
            "![内部残差与外部误差](internal_residual_vs_external_error.png)",
            "",
            "## 6. 数据集缩写",
            "",
        ]
    )
    lines.extend(
        f"- D{index + 1}: `{context.name}`"
        for index, context in enumerate(contexts)
    )
    lines.extend(
        [
            "",
            "## 7. 适用边界",
            "",
            "本报告验证的是不受待比较模型影响的六种子初始求解。若要评价 A/B/C 各自完整主动标定闭环，必须让三种模型分别在真机上重新选择并采集 NBV；当前由 B 生成的历史 NBV 不能作为严格公平的离线闭环消融数据。精密球数据可跨运行复用的前提是传感器安装、机器人基座和球位置在期间未发生变化。",
        ]
    )
    (output / "shared_morphology_ablation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real A/B/C ablation of shared target morphology"
    )
    parser.add_argument("--datasets", nargs="*", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-trials", type=int, default=10000)
    parser.add_argument("--sphere", type=Path, default=DEFAULT_SPHERE)
    parser.add_argument("--sphere-max-points", type=int, default=30000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1 or args.bootstrap_trials < 100:
        parser.error("repeats/workers must be positive and bootstrap-trials >= 100")
    dataset_paths = args.datasets or [
        WORKSPACE / "data/calibration_runs" / name for name in DEFAULT_DATASETS
    ]
    contexts = [_load_context(path.resolve()) for path in dataset_paths]
    sphere = _load_sphere(args.sphere.resolve(), args.sphere_max_points)
    output = args.output or (
        WORKSPACE
        / "data/ablation_runs"
        / ("shared_morphology_real_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.random_seed,
        "repeats": args.repeats,
        "bootstrap_trials": args.bootstrap_trials,
        "data_scope": "six physical seed poses before model-dependent NBV",
        "groups": {
            GROUPS[0]: "ideal plane, beta fixed zero",
            GROUPS[1]: "one shared degree-3 Legendre beta",
            GROUPS[2]: "one independent degree-3 Legendre beta per pose",
        },
        "datasets": [
            {
                "name": context.name,
                "path": str(context.path),
                "configuration": context.configuration,
            }
            for context in contexts
        ],
        "sphere": {
            "source": sphere.source,
            "radius_m": sphere.radius_m,
            "point_count": len(sphere.points_sensor_m),
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    jobs = []
    for dataset_index, context in enumerate(contexts):
        for repeat in range(args.repeats):
            seed = args.random_seed + 1000003 * dataset_index + repeat
            jobs.append((dataset_index, context, repeat, seed))
    rows: list[dict[str, Any]] = []
    completed = 0
    print(
        f"output={output}; datasets={len(contexts)}; paired repeats={len(jobs)}",
        flush=True,
    )

    def accept(trial_rows: list[dict[str, Any]]) -> None:
        nonlocal completed
        rows.extend(trial_rows)
        completed += 1
        first = trial_rows[0]
        status = " ".join(
            f"{row['group'][0]}={'OK' if row.get('usable_solution') else 'FAIL'}"
            for row in trial_rows
        )
        print(
            f"[{completed:02d}/{len(jobs):02d}] {first['dataset']} "
            f"repeat={first['repeat']}: {status}",
            flush=True,
        )
        ordered = sorted(
            rows,
            key=lambda row: (row["dataset"], row["repeat"], row["group"]),
        )
        (output / "trial_results.json").write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.workers == 1:
        for _dataset_index, context, repeat, seed in jobs:
            accept(_run_paired_repeat(context, repeat, seed, sphere))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_run_paired_repeat, context, repeat, seed, sphere): (
                    context.name,
                    repeat,
                )
                for _dataset_index, context, repeat, seed in jobs
            }
            for future in as_completed(futures):
                try:
                    accept(future.result())
                except Exception as error:
                    print(f"paired repeat {futures[future]} failed: {error}", flush=True)
                    traceback.print_exc()
                    return 2
    rows.sort(key=lambda row: (row["dataset"], row["repeat"], row["group"]))
    summary = _dataset_summary(rows)
    differences = _paired_dataset_differences(summary)
    paired = _paired_statistics(
        differences,
        bootstrap_trials=args.bootstrap_trials,
        random_seed=args.random_seed,
    )
    (output / "trial_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "paired_a_b_differences.json").write_text(
        json.dumps(differences, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "paired_bootstrap_statistics.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output / "trial_results.csv", rows)
    _write_csv(output / "dataset_summary.csv", summary)
    _write_csv(output / "paired_a_b_differences.csv", differences)
    _plots(output, summary, differences)
    _report(output, contexts, args.repeats, summary, differences, paired, sphere)
    print(f"report={output / 'shared_morphology_ablation_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
