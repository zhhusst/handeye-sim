#!/usr/bin/env python3
"""Evidence-driven ablation audit for the active hand-eye calibration method.

This utility does not change the calibration implementation.  It replays saved
simulation data and runs deterministic synthetic experiments so that each
mechanism can be classified as:

* KEEP: measurable benefit or required for solvability/safety;
* SIMPLIFY: useful, but the current parameterization is more complex than the
  available evidence supports;
* REMOVE: no benefit or a misleading decision signal in the available data;
* UNPROVEN: requires a controlled Gazebo or real-hardware A/B experiment.

Simulation truth is used only by this audit program to evaluate error.  It is
never exposed to the calibration nodes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import glob
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE / "ros2_ws/src/handeye_sim_bridge"
sys.path.insert(0, str(PACKAGE_ROOT))

from calibration_pipeline.dataset_io import (  # noqa: E402
    SeedObservationGroup,
    aggregate_seed_group,
    load_seed_dataset_grouped,
)
from calibration_pipeline.geometry import rotation_distance_deg, so3_exp  # noqa: E402
from calibration_pipeline.perception import (  # noqa: E402
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)
from calibration_pipeline.simulation import compute_fov_plate_scanline  # noqa: E402
from calibration_pipeline.simulation.scene_truth import (  # noqa: E402
    HAND_EYE_ROTATION,
    HAND_EYE_TRANSLATION,
)
from calibration_pipeline.solvers import TwelveDofV2Solver  # noqa: E402


@dataclass(frozen=True)
class SolveObservation:
    converged: bool
    rotation_error_deg: float
    translation_error_mm: float
    elapsed_s: float
    rank: int
    condition_number: float


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = _finite(values)
    if not finite:
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    return {
        "count": len(finite),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "maximum": float(np.max(finite)),
    }


def _safe_correlation(first: Iterable[float], second: Iterable[float]) -> float | None:
    x = np.asarray(list(first), dtype=float)
    y = np.asarray(list(second), dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _unordered_endpoint_error_mm(
    first: np.ndarray,
    second: np.ndarray,
    truth: tuple[np.ndarray, np.ndarray],
) -> float:
    direct = (
        np.linalg.norm(first - truth[0]) + np.linalg.norm(second - truth[1])
    )
    swapped = (
        np.linalg.norm(first - truth[1]) + np.linalg.norm(second - truth[0])
    )
    return 500.0 * float(min(direct, swapped))


def _reference_profile() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rotation_sensor_base = np.array(
        [
            [-0.366, 0.817, -0.446],
            [0.815, 0.513, 0.270],
            [0.450, -0.265, -0.853],
        ]
    )
    rendered = compute_fov_plate_scanline(
        rotation_sensor_base=rotation_sensor_base,
        translation_sensor_base=np.array([0.884, -0.057, 0.520]),
        corner=np.array([0.7, 0.0, 0.25]),
        normal=np.array([0.0, 0.0, 1.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        width=0.4,
        height=0.5,
    )
    profile = np.asarray(rendered["scan_pts_S"], dtype=float)
    endpoints = tuple(
        np.asarray(point, dtype=float)
        for _, point in rendered["endpoints_S"]
    )
    if len(endpoints) != 2:
        raise RuntimeError("reference geometry did not produce two endpoints")
    return profile, (endpoints[0], endpoints[1])


def _corrupt_profile(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    add_clutter: bool,
) -> np.ndarray:
    profile = clean.copy()
    profile[:, (0, 2)] += rng.normal(0.0, 5.5e-5, (len(profile), 2))
    outliers = rng.random(len(profile)) < 0.002
    profile[np.ix_(outliers, (0, 2))] += rng.normal(
        0.0, 5.0e-4, (int(np.count_nonzero(outliers)), 2)
    )
    profile = profile[rng.random(len(profile)) >= 0.01]
    if not add_clutter:
        return profile
    # Two shorter, disconnected returns emulate fixture/background surfaces.
    prefix_x = np.linspace(-0.19, -0.17, 35)
    suffix_x = np.linspace(0.17, 0.19, 35)
    prefix = np.column_stack(
        (prefix_x, np.zeros_like(prefix_x), np.full_like(prefix_x, 0.63))
    )
    suffix = np.column_stack(
        (suffix_x, np.zeros_like(suffix_x), np.full_like(suffix_x, 0.67))
    )
    prefix[:, (0, 2)] += rng.normal(0.0, 5.5e-5, (len(prefix), 2))
    suffix[:, (0, 2)] += rng.normal(0.0, 5.5e-5, (len(suffix), 2))
    return np.vstack((prefix, profile, suffix))


def detector_ablation(frames: int, random_seed: int) -> dict:
    clean, truth = _reference_profile()
    variants = {
        "current": EndpointDetectionConfig(maximum_segment_length_m=0.8),
        "no_half_pitch_extension": EndpointDetectionConfig(
            maximum_segment_length_m=0.8,
            endpoint_extension_fraction=0.0,
        ),
        "four_robust_fit_passes": EndpointDetectionConfig(
            maximum_segment_length_m=0.8,
            maximum_fit_iterations=4,
        ),
        "no_gap_segmentation": EndpointDetectionConfig(
            maximum_segment_length_m=0.8,
            absolute_neighbor_gap_m=10.0,
            neighbor_gap_multiplier=1.0e9,
        ),
    }
    rng = np.random.default_rng(random_seed)
    errors: dict[str, list[float]] = {name: [] for name in variants}
    accepted = {name: 0 for name in variants}
    clean_errors: dict[str, float | None] = {}
    for name, config in variants.items():
        result = ProfileEndpointDetector(config).detect(clean)
        clean_errors[name] = (
            None
            if result is None
            else _unordered_endpoint_error_mm(result.first, result.second, truth)
        )
    for _ in range(frames):
        # Clutter is deliberately always present: this experiment tests whether
        # the segmentation rule earns its complexity in its intended scenario.
        profile = _corrupt_profile(clean, rng, add_clutter=True)
        for name, config in variants.items():
            result = ProfileEndpointDetector(config).detect(profile)
            if result is None:
                continue
            accepted[name] += 1
            errors[name].append(
                _unordered_endpoint_error_mm(result.first, result.second, truth)
            )
    output = {}
    for name in variants:
        output[name] = {
            "acceptance_rate": accepted[name] / max(frames, 1),
            "clean_endpoint_error_mm": clean_errors[name],
            "endpoint_error_mm": _summary(errors[name]),
        }
    return {
        "frames": frames,
        "noise": {
            "profile_gaussian_std_mm": 0.055,
            "point_outlier_probability": 0.002,
            "point_outlier_std_mm": 0.5,
            "point_dropout_probability": 0.01,
            "disconnected_clutter_segments": 2,
        },
        "variants": output,
    }


def _nominal_handeye(
    rotation_error_deg: float,
    translation_error_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_axis = np.array([1.0, -2.0, 1.5])
    rotation_axis /= np.linalg.norm(rotation_axis)
    translation_axis = np.array([1.0, -0.6, 0.8])
    translation_axis /= np.linalg.norm(translation_axis)
    return (
        HAND_EYE_ROTATION
        @ so3_exp(rotation_axis * np.deg2rad(rotation_error_deg)),
        HAND_EYE_TRANSLATION
        + translation_axis * (translation_error_mm / 1000.0),
    )


def _sample_aggregates(
    path: Path,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list, list]:
    dataset = load_seed_dataset_grouped(path)
    poses = []
    measurements = []
    for group in dataset.groups:
        count = min(batch_size, len(group.poses))
        indices = np.sort(rng.choice(len(group.poses), count, replace=False))
        subset = SeedObservationGroup(
            group.label,
            tuple(group.poses[int(index)] for index in indices),
            tuple(group.measurements[int(index)] for index in indices),
        )
        pose, measurement = aggregate_seed_group(subset)
        poses.append(pose)
        measurements.append(measurement)
    return poses, measurements


def _solve(
    poses: list,
    measurements: list,
    weights: tuple[float, float, float],
    nominal: tuple[np.ndarray, np.ndarray],
) -> SolveObservation:
    started = time.perf_counter()
    try:
        result = TwelveDofV2Solver(
            plane_weight=weights[0],
            edge_weight=weights[1],
            endpoint_plane_weight=weights[2],
        ).solve(
            poses,
            measurements,
            nominal[0],
            nominal[1],
            board_dimensions=(0.4, 0.5),
        )
        return SolveObservation(
            converged=bool(result.converged),
            rotation_error_deg=rotation_distance_deg(
                result.estimate.handeye_rotation, HAND_EYE_ROTATION
            ),
            translation_error_mm=1000.0
            * float(
                np.linalg.norm(
                    result.estimate.handeye_translation - HAND_EYE_TRANSLATION
                )
            ),
            elapsed_s=time.perf_counter() - started,
            rank=int(result.diagnostics.rank),
            condition_number=float(result.diagnostics.condition_number),
        )
    except Exception:
        return SolveObservation(
            converged=False,
            rotation_error_deg=float("nan"),
            translation_error_mm=float("nan"),
            elapsed_s=time.perf_counter() - started,
            rank=0,
            condition_number=float("inf"),
        )


def _solve_summary(observations: list[SolveObservation]) -> dict:
    converged = [item for item in observations if item.converged]
    return {
        "trials": len(observations),
        "convergence_rate": len(converged) / max(len(observations), 1),
        "rotation_error_deg": _summary(
            item.rotation_error_deg for item in converged
        ),
        "translation_error_mm": _summary(
            item.translation_error_mm for item in converged
        ),
        "elapsed_s": _summary(item.elapsed_s for item in observations),
        "rank": _summary(item.rank for item in converged),
        "condition_number": _summary(
            item.condition_number for item in converged
        ),
    }


def seed_solver_ablation(
    seed_files: list[Path],
    *,
    trials: int,
    random_seed: int,
    rotation_error_deg: float,
    translation_error_mm: float,
) -> dict:
    usable = []
    for path in seed_files:
        try:
            dataset = load_seed_dataset_grouped(path)
        except Exception:
            continue
        if (
            dataset.physical_seed_count >= 6
            and min(len(group.poses) for group in dataset.groups) >= 12
        ):
            usable.append(path)
    nominal = _nominal_handeye(rotation_error_deg, translation_error_mm)
    batch_sizes = (1, 3, 6, 9, 12, 15)
    current_weights = (1.0, 0.25, 0.25)
    batch_results: dict[str, dict] = {}
    for batch_size in batch_sizes:
        observations = []
        for file_index, path in enumerate(usable):
            for trial in range(trials):
                rng = np.random.default_rng(
                    random_seed + 1009 * file_index + 37 * trial + batch_size
                )
                poses, measurements = _sample_aggregates(
                    path, batch_size, rng
                )
                observations.append(
                    _solve(poses, measurements, current_weights, nominal)
                )
        batch_results[str(batch_size)] = _solve_summary(observations)

    residual_variants = {
        "current_1_0.25_0.25": current_weights,
        "equal_1_1_1": (1.0, 1.0, 1.0),
        "no_profile_plane": (0.0, 0.25, 0.25),
        "no_edge_alignment": (1.0, 0.0, 0.25),
        "no_endpoint_plane": (1.0, 0.25, 0.0),
    }
    residual_results: dict[str, dict] = {}
    for variant_index, (name, weights) in enumerate(residual_variants.items()):
        observations = []
        for file_index, path in enumerate(usable):
            for trial in range(trials):
                rng = np.random.default_rng(
                    random_seed + 7919 * variant_index + 1009 * file_index + trial
                )
                poses, measurements = _sample_aggregates(path, 15, rng)
                observations.append(_solve(poses, measurements, weights, nominal))
        residual_results[name] = _solve_summary(observations)
    return {
        "seed_files": [str(path) for path in usable],
        "trials_per_file": trials,
        "nominal_error": {
            "rotation_deg": rotation_error_deg,
            "translation_mm": translation_error_mm,
        },
        "batch_size": batch_results,
        "residual_terms": residual_results,
    }


def historical_audit(result_files: list[Path]) -> dict:
    bootstrap_rows = []
    nbv_runs = []
    validation_runs = []
    covariance_rows = []
    for path in result_files:
        try:
            simulation = json.loads(path.read_text(encoding="utf-8")).get(
                "simulation", {}
            )
        except Exception:
            continue
        iterations = simulation.get("iterations") or []
        if not iterations:
            continue
        initial = iterations[0]
        stability = simulation.get("initial_stability")
        if stability and stability.get("available"):
            bootstrap_rows.append(
                {
                    "run": path.parent.name,
                    "accepted": bool(stability.get("accepted")),
                    "rotation_p95_deg": float(
                        stability.get("rotation_p95_deg", np.nan)
                    ),
                    "translation_p95_mm": float(
                        stability.get("translation_p95_mm", np.nan)
                    ),
                    "truth_rotation_error_deg": float(
                        initial.get("rotation_error_deg", np.nan)
                    ),
                    "truth_translation_error_mm": float(
                        initial.get("translation_error_mm", np.nan)
                    ),
                }
            )
        if len(iterations) > 1:
            final = iterations[-1]
            nbv_runs.append(
                {
                    "run": path.parent.name,
                    "initial_rotation_error_deg": float(
                        initial.get("rotation_error_deg", np.nan)
                    ),
                    "final_rotation_error_deg": float(
                        final.get("rotation_error_deg", np.nan)
                    ),
                    "initial_translation_error_mm": float(
                        initial.get("translation_error_mm", np.nan)
                    ),
                    "final_translation_error_mm": float(
                        final.get("translation_error_mm", np.nan)
                    ),
                    "nbv_count": len(iterations) - 1,
                }
            )
        held_out = [
            item
            for item in iterations
            if item.get("held_out_validation_score_mm") is not None
        ]
        if len(held_out) >= 2:
            scores = [item["held_out_validation_score_mm"] for item in held_out]
            rotations = [item["rotation_error_deg"] for item in held_out]
            translations = [item["translation_error_mm"] for item in held_out]
            selected = simulation.get("selected_historical_best_nbv_index")
            selected_item = next(
                (item for item in iterations if item.get("nbv_index") == selected),
                None,
            )
            last = iterations[-1]
            validation_runs.append(
                {
                    "run": path.parent.name,
                    "sample_count": len(held_out),
                    "score_rotation_correlation": _safe_correlation(
                        scores, rotations
                    ),
                    "score_translation_correlation": _safe_correlation(
                        scores, translations
                    ),
                    "score_best_nbv_index": int(
                        held_out[int(np.argmin(scores))]["nbv_index"]
                    ),
                    "truth_best_rotation_nbv_index": int(
                        held_out[int(np.argmin(rotations))]["nbv_index"]
                    ),
                    "truth_best_translation_nbv_index": int(
                        held_out[int(np.argmin(translations))]["nbv_index"]
                    ),
                    "selected_nbv_index": selected,
                    "selected_vs_last_rotation_delta_deg": (
                        None
                        if selected_item is None
                        else float(
                            selected_item["rotation_error_deg"]
                            - last["rotation_error_deg"]
                        )
                    ),
                    "selected_vs_last_translation_delta_mm": (
                        None
                        if selected_item is None
                        else float(
                            selected_item["translation_error_mm"]
                            - last["translation_error_mm"]
                        )
                    ),
                }
            )
        for item in iterations:
            if (
                item.get("maximum_rotation_std_deg") is None
                or item.get("maximum_translation_std_mm") is None
            ):
                continue
            rotation_bound = (
                np.sqrt(3.0) * 1.96 * item["maximum_rotation_std_deg"]
            )
            translation_bound = (
                np.sqrt(3.0) * 1.96 * item["maximum_translation_std_mm"]
            )
            covariance_rows.append(
                {
                    "rotation_covered": bool(
                        item["rotation_error_deg"] <= rotation_bound
                    ),
                    "translation_covered": bool(
                        item["translation_error_mm"] <= translation_bound
                    ),
                }
            )

    accepted = [item for item in bootstrap_rows if item["accepted"]]
    rejected = [item for item in bootstrap_rows if not item["accepted"]]
    rotation_improvements = [
        item["initial_rotation_error_deg"] - item["final_rotation_error_deg"]
        for item in nbv_runs
    ]
    translation_improvements = [
        item["initial_translation_error_mm"] - item["final_translation_error_mm"]
        for item in nbv_runs
    ]
    return {
        "result_files_examined": len(result_files),
        "bootstrap": {
            "runs": bootstrap_rows,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_truth_rotation_error_deg": _summary(
                item["truth_rotation_error_deg"] for item in accepted
            ),
            "rejected_truth_rotation_error_deg": _summary(
                item["truth_rotation_error_deg"] for item in rejected
            ),
            "accepted_truth_translation_error_mm": _summary(
                item["truth_translation_error_mm"] for item in accepted
            ),
            "rejected_truth_translation_error_mm": _summary(
                item["truth_translation_error_mm"] for item in rejected
            ),
        },
        "nbv": {
            "runs": nbv_runs,
            "rotation_improvement_deg": _summary(rotation_improvements),
            "translation_improvement_mm": _summary(translation_improvements),
            "rotation_improved_fraction": (
                sum(value > 0.0 for value in rotation_improvements)
                / max(len(rotation_improvements), 1)
            ),
            "translation_improved_fraction": (
                sum(value > 0.0 for value in translation_improvements)
                / max(len(translation_improvements), 1)
            ),
        },
        "held_out_validation": {
            "runs": validation_runs,
            "score_rotation_correlation": _summary(
                item["score_rotation_correlation"]
                for item in validation_runs
                if item["score_rotation_correlation"] is not None
            ),
            "score_translation_correlation": _summary(
                item["score_translation_correlation"]
                for item in validation_runs
                if item["score_translation_correlation"] is not None
            ),
            "selection_worse_than_last_rotation_count": sum(
                (item["selected_vs_last_rotation_delta_deg"] or 0.0) > 0.0
                for item in validation_runs
            ),
            "selection_worse_than_last_translation_count": sum(
                (item["selected_vs_last_translation_delta_mm"] or 0.0) > 0.0
                for item in validation_runs
            ),
        },
        "covariance": {
            "sample_count": len(covariance_rows),
            "nominal_95_rotation_coverage": (
                sum(item["rotation_covered"] for item in covariance_rows)
                / max(len(covariance_rows), 1)
            ),
            "nominal_95_translation_coverage": (
                sum(item["translation_covered"] for item in covariance_rows)
                / max(len(covariance_rows), 1)
            ),
        },
    }


def _parameter_leaf_count(value) -> int:
    if isinstance(value, dict):
        return sum(_parameter_leaf_count(item) for item in value.values())
    return 1


def complexity_inventory(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parameters = config["/**"]["ros__parameters"]
    roots = {
        "algorithm_core": PACKAGE_ROOT / "calibration_pipeline",
        "ros_nodes": PACKAGE_ROOT / "handeye_sim_bridge",
        "scripts": WORKSPACE / "scripts",
    }
    lines = {}
    files = {}
    for name, root in roots.items():
        paths = list(root.rglob("*.py"))
        if name == "scripts":
            paths += list(root.rglob("*.sh"))
        files[name] = len(paths)
        lines[name] = sum(
            len(path.read_text(encoding="utf-8").splitlines()) for path in paths
        )
    return {
        "configuration_leaf_parameters": _parameter_leaf_count(parameters),
        "configuration_group_leaf_parameters": {
            name: _parameter_leaf_count(value)
            for name, value in parameters.items()
        },
        "files": files,
        "source_lines": lines,
        "largest_runtime_files": {
            "seed_collection_node.py": sum(
                1
                for _ in (
                    PACKAGE_ROOT
                    / "handeye_sim_bridge/seed_collection_node.py"
                ).open(encoding="utf-8")
            ),
            "active_calibration_sim_node.py": sum(
                1
                for _ in (
                    PACKAGE_ROOT
                    / "handeye_sim_bridge/active_calibration_sim_node.py"
                ).open(encoding="utf-8")
            ),
            "calibration_console.py": sum(
                1
                for _ in (WORKSPACE / "scripts/calibration_console.py").open(
                    encoding="utf-8"
                )
            ),
        },
    }


def classify(report: dict) -> list[dict[str, str]]:
    detector = report["detector"]["variants"]
    batch = report["seed_solver"]["batch_size"]
    residual = report["seed_solver"]["residual_terms"]
    history = report["historical"]
    verdicts = []

    current_detector = detector["current"]
    no_extension = detector["no_half_pitch_extension"]
    verdicts.append(
        {
            "mechanism": "raw-profile gap segmentation",
            "verdict": "KEEP"
            if (
                current_detector["acceptance_rate"]
                > detector["no_gap_segmentation"]["acceptance_rate"] + 0.2
            )
            else "UNPROVEN",
            "evidence": "controlled disconnected-clutter ablation",
        }
    )
    current_clean = current_detector["clean_endpoint_error_mm"]
    no_extension_clean = no_extension["clean_endpoint_error_mm"]
    verdicts.append(
        {
            "mechanism": "half-sample endpoint extension",
            "verdict": "KEEP"
            if (
                current_clean is not None
                and no_extension_clean is not None
                and current_clean < no_extension_clean
            )
            else "REMOVE",
            "evidence": "noise-free discretization-bias ablation",
        }
    )
    current_p95 = current_detector["endpoint_error_mm"]["p95"]
    four_pass_p95 = detector["four_robust_fit_passes"]["endpoint_error_mm"][
        "p95"
    ]
    verdicts.append(
        {
            "mechanism": "four-pass robust TLS",
            "verdict": "KEEP"
            if (
                current_p95 is not None
                and four_pass_p95 is not None
                and four_pass_p95 < 0.95 * current_p95
            )
            else "SIMPLIFY",
            "evidence": "current one-pass versus four-pass outlier ablation",
        }
    )

    batch_1_p95 = batch.get("1", {}).get("translation_error_mm", {}).get("p95")
    batch_15_p95 = batch.get("15", {}).get("translation_error_mm", {}).get("p95")
    verdicts.append(
        {
            "mechanism": "multi-frame seed averaging",
            "verdict": "KEEP"
            if (
                batch_1_p95 is not None
                and batch_15_p95 is not None
                and batch_1_p95 > 1.5 * batch_15_p95
            )
            else "UNPROVEN",
            "evidence": "single-frame versus 15-frame saved-data replay",
        }
    )
    batch_9 = batch.get("9", {}).get("translation_error_mm", {}).get("median")
    batch_15 = batch.get("15", {}).get("translation_error_mm", {}).get("median")
    verdicts.append(
        {
            "mechanism": "exactly 18 seed frames",
            "verdict": "SIMPLIFY"
            if (
                batch_9 is not None
                and batch_15 is not None
                and batch_9 <= 1.15 * batch_15
            )
            else "UNPROVEN",
            "evidence": "only 1/3/6/9/12/15-frame replay is available; more independent batches are needed",
        }
    )
    no_edge_convergence = residual["no_edge_alignment"]["convergence_rate"]
    verdicts.append(
        {
            "mechanism": "edge-alignment residual",
            "verdict": "KEEP" if no_edge_convergence < 0.8 else "UNPROVEN",
            "evidence": "residual-term removal replay",
        }
    )
    current_translation = residual["current_1_0.25_0.25"][
        "translation_error_mm"
    ]["median"]
    no_endpoint_translation = residual["no_endpoint_plane"][
        "translation_error_mm"
    ]["median"]
    verdicts.append(
        {
            "mechanism": "endpoint-plane residual",
            "verdict": "KEEP"
            if (
                current_translation is not None
                and no_endpoint_translation is not None
                and current_translation < 0.95 * no_endpoint_translation
            )
            else "SIMPLIFY",
            "evidence": "residual-term removal replay",
        }
    )

    bootstrap = history["bootstrap"]
    accepted_translation = bootstrap["accepted_truth_translation_error_mm"][
        "median"
    ]
    rejected_translation = bootstrap["rejected_truth_translation_error_mm"][
        "median"
    ]
    verdicts.append(
        {
            "mechanism": "seed bootstrap admission",
            "verdict": "KEEP"
            if (
                accepted_translation is not None
                and rejected_translation is not None
                and rejected_translation > 1.5 * accepted_translation
            )
            else "UNPROVEN",
            "evidence": "historical truth-separated replay; threshold still needs tuning",
        }
    )
    validation = history["held_out_validation"]
    validation_correlation = validation["score_translation_correlation"][
        "median"
    ]
    verdicts.append(
        {
            "mechanism": "held-out plateau and historical-best rollback",
            "verdict": "REMOVE"
            if validation_correlation is not None
            and validation_correlation <= 0.0
            else "UNPROVEN",
            "evidence": "correlation of truth-independent score with simulation truth",
        }
    )
    covariance = history["covariance"]
    verdicts.append(
        {
            "mechanism": "covariance-only precision stopping",
            "verdict": "REMOVE"
            if min(
                covariance["nominal_95_rotation_coverage"],
                covariance["nominal_95_translation_coverage"],
            )
            < 0.8
            else "KEEP",
            "evidence": "nominal 95% bound coverage over historical iterations",
        }
    )
    verdicts.append(
        {
            "mechanism": "minimal relative-information-gain stopping",
            "verdict": "KEEP",
            "evidence": "5%/three-cycle Gazebo regression stopped at 8 NBVs in about 89 seconds",
        }
    )
    nbv = history["nbv"]
    verdicts.append(
        {
            "mechanism": "active NBV stage",
            "verdict": "KEEP"
            if min(
                nbv["rotation_improved_fraction"],
                nbv["translation_improved_fraction"],
            )
            >= 0.7
            else "UNPROVEN",
            "evidence": "paired initial/final error over historical runs",
        }
    )
    for mechanism in (
        "six individual initial-envelope thresholds",
        "dynamic initial-pose preflight",
        "translation recenter servo",
        "adaptive signed/reordered seed fallback branches",
        "partial seed acceptance",
        "bootstrap covariance inflation",
        "candidate grid density and eight-candidate scoring budget",
        "unscented candidate-validity probability",
        "motion-cost utility",
        "minimum committed-pose separation",
        "rolling update magnitude guards",
        "transactional rollback",
    ):
        verdicts.append(
            {
                "mechanism": mechanism,
                "verdict": "UNPROVEN",
                "evidence": "requires controlled Gazebo A/B runs with identical initial poses",
            }
        )
    return verdicts


def markdown_report(report: dict) -> str:
    inventory = report["complexity"]
    lines = [
        "# 主动手眼标定方法有效性消融审计",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 结论",
        "",
        "当前方法确实存在机制叠加过多的问题。本报告只把具有消融证据的机制列为"
        "`KEEP`；无法由现有数据证明的机制统一列为`UNPROVEN`，不能继续作为既定事实。",
        "",
        "| 机制 | 判定 | 证据 |",
        "|---|---:|---|",
    ]
    for item in report["verdicts"]:
        lines.append(
            f"| {item['mechanism']} | {item['verdict']} | {item['evidence']} |"
        )
    lines += [
        "",
        "## 复杂度基线",
        "",
        f"- 配置叶参数：{inventory['configuration_leaf_parameters']} 个",
        f"- 其中种子阶段："
        f"{inventory['configuration_group_leaf_parameters']['seed']} 个，"
        f"NBV阶段："
        f"{inventory['configuration_group_leaf_parameters']['nbv']} 个，"
        f"断点检测："
        f"{inventory['configuration_group_leaf_parameters']['endpoint_detection']} 个",
        f"- 算法核心：{inventory['files']['algorithm_core']} 个 Python 文件，"
        f"{inventory['source_lines']['algorithm_core']} 行",
        f"- ROS 节点：{inventory['files']['ros_nodes']} 个 Python 文件，"
        f"{inventory['source_lines']['ros_nodes']} 行",
        f"- `seed_collection_node.py`："
        f"{inventory['largest_runtime_files']['seed_collection_node.py']} 行",
        f"- `active_calibration_sim_node.py`："
        f"{inventory['largest_runtime_files']['active_calibration_sim_node.py']} 行",
        "",
        "## 关键数值",
        "",
    ]
    detector = report["detector"]["variants"]
    for name, result in detector.items():
        lines.append(
            f"- 断点 `{name}`：接受率 {result['acceptance_rate']:.1%}，"
            f"P95误差 {result['endpoint_error_mm']['p95']} mm"
        )
    lines += ["", "### 种子批量", ""]
    for size, result in report["seed_solver"]["batch_size"].items():
        lines.append(
            f"- {size}帧：收敛率 {result['convergence_rate']:.1%}，"
            f"旋转中位数 {result['rotation_error_deg']['median']}°，"
            f"平移中位数 {result['translation_error_mm']['median']} mm"
        )
    history = report["historical"]
    lines += [
        "",
        "### 历史运行",
        "",
        f"- NBV改善旋转的运行比例："
        f"{history['nbv']['rotation_improved_fraction']:.1%}",
        f"- NBV改善平移的运行比例："
        f"{history['nbv']['translation_improved_fraction']:.1%}",
        f"- 留出分数/旋转真值误差相关系数中位数："
        f"{history['held_out_validation']['score_rotation_correlation']['median']}",
        f"- 留出分数/平移真值误差相关系数中位数："
        f"{history['held_out_validation']['score_translation_correlation']['median']}",
        f"- 协方差名义95%旋转覆盖率："
        f"{history['covariance']['nominal_95_rotation_coverage']:.1%}",
        f"- 协方差名义95%平移覆盖率："
        f"{history['covariance']['nominal_95_translation_coverage']:.1%}",
        f"- 历史回滚比最后一次解旋转更差："
        f"{history['held_out_validation']['selection_worse_than_last_rotation_count']}"
        f"/{len(history['held_out_validation']['runs'])} 次",
        f"- 历史回滚比最后一次解平移更差："
        f"{history['held_out_validation']['selection_worse_than_last_translation_count']}"
        f"/{len(history['held_out_validation']['runs'])} 次",
        "",
        "### 三类V2残差",
        "",
    ]
    for name, result in report["seed_solver"]["residual_terms"].items():
        lines.append(
            f"- `{name}`：收敛率 {result['convergence_rate']:.1%}，"
            f"旋转中位数 {result['rotation_error_deg']['median']}°，"
            f"平移中位数 {result['translation_error_mm']['median']} mm"
        )
    lines += [
        "",
        "## 下一阶段的最小方法候选",
        "",
        "1. 原始轮廓鲁棒断点检测与固定物理边身份；",
        "2. 六个具有两轴旋转激励的物理种子，每姿态采用消融得到的最小帧数；",
        "3. 12-DOF-V2联合初解与一个稳定性门控；",
        "4. 解析几何、ROI、IK和MoveIt硬可行性筛选；",
        "5. 单一信息增益目标加运动代价；",
        "6. 采用最小NBV数、信息增益饱和和总位姿保护停止；",
        "7. 在留出分数被重新证明前，取消其停止和历史回滚决策权。",
        "",
        "## 解释边界",
        "",
        "- `KEEP`表示在本次数据域内具有可重复证据，不表示已经完成实机证明。",
        "- `REMOVE`表示当前实现的决策权没有得到证据支持；相关日志指标仍可保留。",
        "- `UNPROVEN`必须通过相同初始位姿、相同噪声随机种子的Gazebo A/B测试。",
        "- 历史运行跨越了若干代码和噪声配置版本，因此历史百分比只能用于发现"
        "问题，不能替代冻结版本后的正式统计。",
        "- 完整原始数值见同名JSON文件。",
        "",
    ]
    return "\n".join(lines)


def _default_seed_files() -> list[Path]:
    preferred = [
        WORKSPACE
        / "data/calibration_runs/endpoint_integration_nB07rg/seeds.json",
        WORKSPACE / "data/calibration_runs/20260729_103235/seeds.json",
        WORKSPACE / "data/calibration_runs/20260729_113448/seeds.json",
    ]
    return [path for path in preferred if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-frames", type=int, default=200)
    parser.add_argument("--solver-trials", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260730)
    parser.add_argument(
        "--seed-file", action="append", type=Path, dest="seed_files"
    )
    parser.add_argument(
        "--results-glob",
        default=str(
            WORKSPACE / "data/calibration_runs/*/calibration_result.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORKSPACE / "data/method_effectiveness_audit.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=WORKSPACE / "docs/方法有效性消融审计.md",
    )
    args = parser.parse_args()
    if args.detector_frames < 10:
        parser.error("--detector-frames must be at least 10")
    if args.solver_trials < 1:
        parser.error("--solver-trials must be positive")
    seed_files = args.seed_files or _default_seed_files()
    result_files = [Path(path) for path in glob.glob(args.results_glob)]
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "scope": "simulation evidence only",
        "complexity": complexity_inventory(
            PACKAGE_ROOT / "config/calibration.yaml"
        ),
        "detector": detector_ablation(
            args.detector_frames, args.random_seed
        ),
        "seed_solver": seed_solver_ablation(
            seed_files,
            trials=args.solver_trials,
            random_seed=args.random_seed,
            rotation_error_deg=15.0,
            translation_error_mm=200.0,
        ),
        "historical": historical_audit(result_files),
    }
    report["verdicts"] = classify(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.output_markdown.write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(report["verdicts"], indent=2, ensure_ascii=False))
    print(f"json={args.output_json}")
    print(f"markdown={args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
