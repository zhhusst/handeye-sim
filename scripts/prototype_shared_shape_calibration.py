#!/usr/bin/env python3
"""Offline regression for production and legacy shared-shape calibration.

The unified production solver is compared with the flat ablation, a paired
no-flatness oracle and two legacy research refinements:

* matched shape family: knows the simulation basis, but not its coefficients;
* generic Legendre field: does not know the simulation basis or phases.

The matched case is deliberately labelled an upper-bound/inverse-crime check,
not evidence of real-world generalization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE / "ros2_ws/src/handeye_sim_bridge"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from analyze_precision_error_budget import (  # noqa: E402
    _aggregate_physical_pose,
    _detector_config,
    _information_selected_indices,
    _noise_config,
    _nominal_handeye,
    _parameters,
    _render_frame,
    _solver,
)
from calibration_pipeline.geometry import rotation_distance_deg  # noqa: E402
from calibration_pipeline.perception import ProfileEndpointDetector  # noqa: E402
from calibration_pipeline.research import (  # noqa: E402
    SharedShapeHandEyeSolver,
    SurfaceBasis,
)
from calibration_pipeline.solvers import TwelveDofV2Solver  # noqa: E402
from calibration_pipeline.simulation import (  # noqa: E402
    SimulationNoiseConfig,
    SimulationNoiseModel,
)
from calibration_pipeline.simulation.synthetic import (  # noqa: E402
    default_scene,
    generate_seed_dataset,
)


DEFAULT_JSON = WORKSPACE / "data/shared_shape_feasibility.json"
DEFAULT_REPORT = WORKSPACE / "docs/共享形貌手眼联合估计_可行性报告.md"


def _marginal_handeye_information(hessian: np.ndarray) -> np.ndarray:
    handeye = hessian[:6, :6]
    cross = hessian[:6, 6:]
    nuisance = hessian[6:, 6:]
    effective = handeye - cross @ np.linalg.pinv(
        nuisance + 1e-10 * np.eye(nuisance.shape[0])
    ) @ cross.T
    return 0.5 * (effective + effective.T)


def _shape_marginal_selected_indices(
    poses,
    measurements,
    *,
    selected_count: int,
    seed_count: int,
    parameters: dict,
    objective: str = "d_optimal",
) -> list[int]:
    """Greedy D-optimal design after marginalizing board pose and shape.

    Candidate Jacobians use the ideal virtual profiles supplied by the same
    candidate generator as the existing offline NBV audit.  No simulated
    target-shape truth or future noisy measurement is used for selection.
    """
    scene = default_scene()
    nominal_rotation, nominal_translation = _nominal_handeye(parameters, scene)
    flat_solver = _solver(parameters)
    initial_flat = flat_solver.solve(
        list(poses[:seed_count]),
        list(measurements[:seed_count]),
        nominal_rotation,
        nominal_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    basis = SurfaceBasis("legendre", degree=4)
    shared = SharedShapeHandEyeSolver(
        basis,
        plane_weight=flat_solver.weights["plane_weight"],
        edge_weight=flat_solver.weights["edge_weight"],
        endpoint_surface_weight=flat_solver.weights["endpoint_plane_weight"],
        shape_regularization=0.0,
    )
    estimate = initial_flat.estimate
    from calibration_pipeline.geometry import so3_log

    state = np.concatenate(
        (
            so3_log(estimate.handeye_rotation),
            estimate.handeye_translation,
            so3_log(estimate.board.rotation),
            estimate.board.corner,
            np.zeros(basis.size),
        )
    )
    if objective not in {"d_optimal", "goal_a_optimal"}:
        raise ValueError("unknown shape-marginal design objective")
    handeye_scale = (
        np.array([np.deg2rad(0.1)] * 3 + [0.0001] * 3)
        if objective == "goal_a_optimal"
        else np.array([np.deg2rad(10.0)] * 3 + [0.1] * 3)
    )
    scale = np.concatenate(
        (
            handeye_scale,
            np.full(3, np.deg2rad(10.0)),
            np.full(3, 0.1),
            np.full(basis.size, 0.0005),
        )
    )

    def pose_hessian(index: int) -> np.ndarray:
        function = lambda value: shared.residual(
            value,
            [poses[index]],
            [measurements[index]],
            board_dimensions=(scene.board.length_u, scene.board.length_v),
        )
        baseline = function(state)
        jacobian = np.empty((len(baseline), len(state)))
        for column in range(len(state)):
            delta = np.zeros_like(state)
            delta[column] = 1e-5 * scale[column]
            jacobian[:, column] = (
                function(state + delta) - function(state - delta)
            ) / (2.0 * delta[column])
        scaled = jacobian * scale[None, :]
        return scaled.T @ scaled

    hessians = [pose_hessian(index) for index in range(len(poses))]
    selected = list(range(seed_count))
    remaining = list(range(seed_count, len(poses)))
    total = sum((hessians[index] for index in selected), np.zeros_like(hessians[0]))
    # One shape prior, not one prior per candidate observation.
    total[12:, 12:] += (
        1e-2 * np.diag(scale[12:] ** 2)
    )
    while len(selected) < selected_count:
        current = _marginal_handeye_information(total)
        current_sign, current_logdet = np.linalg.slogdet(
            current + 1e-9 * np.eye(6)
        )
        best_index = None
        best_score = (float("-inf"), float("-inf"))
        for index in remaining:
            augmented = _marginal_handeye_information(total + hessians[index])
            sign, logdet = np.linalg.slogdet(augmented + 1e-9 * np.eye(6))
            gain = (
                0.5 * float(logdet - current_logdet)
                if sign > 0 and current_sign > 0
                else float("-inf")
            )
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(augmented)))
            if objective == "goal_a_optimal":
                covariance = np.linalg.pinv(
                    augmented + 1e-12 * np.eye(6)
                )
                score = (-float(np.trace(covariance)), minimum_eigenvalue)
            else:
                score = (gain, minimum_eigenvalue)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            raise RuntimeError("shape-marginal design found no finite candidate")
        selected.append(best_index)
        remaining.remove(best_index)
        total += hessians[best_index]
    return selected


def _condition_config(
    base: SimulationNoiseConfig,
    name: str,
    *,
    random_seed: int,
) -> SimulationNoiseConfig:
    common = dict(
        random_seed=random_seed,
        board_flatness_rms_m=0.0005,
        endpoint_gaussian_std_m=0.0,
        endpoint_outlier_probability=0.0,
        endpoint_outlier_std_m=0.0,
        endpoint_dropout_probability=0.0,
        sync_delay_mean_s=0.0,
        sync_jitter_std_s=0.0,
    )
    if name == "flatness_only":
        return replace(
            base,
            **common,
            profile_gaussian_std_m=0.0,
            robot_translation_std_m=0.0,
            robot_rotation_std_deg=0.0,
            point_outlier_probability=0.0,
            point_outlier_std_m=0.0,
            point_dropout_probability=0.0,
            frame_dropout_probability=0.0,
        )
    if name == "combined_stress":
        return replace(base, **common)
    raise ValueError(f"unknown condition {name}")


def _collect_dataset(
    config: SimulationNoiseConfig,
    physical_poses,
    *,
    frames_per_pose: int,
    parameters: dict,
    trial: int,
):
    scene = default_scene()
    trial_config = replace(
        config,
        random_seed=int(config.random_seed + 104729 * trial),
    )
    model = SimulationNoiseModel(trial_config)
    detector = ProfileEndpointDetector(_detector_config(parameters))
    poses = []
    measurements = []
    mad_multiplier = float(parameters["seed"]["endpoint_mad_multiplier"])
    for pose_index, nominal_pose in enumerate(physical_poses):
        frames = []
        for _ in range(frames_per_pose):
            rendered = _render_frame(nominal_pose, scene, model, detector)
            if rendered is not None:
                frames.append(rendered)
        aggregated = _aggregate_physical_pose(
            f"pose_{pose_index:02d}",
            frames,
            mad_multiplier=mad_multiplier,
        )
        if aggregated is None:
            continue
        pose, measurement, _ = aggregated
        poses.append(pose)
        measurements.append(measurement)
    return poses, measurements, model


def _flat_metrics(result, scene, elapsed_s: float) -> dict:
    return {
        "converged": bool(result.converged),
        "rotation_error_deg": rotation_distance_deg(
            result.estimate.handeye_rotation, scene.handeye_rotation
        ),
        "translation_error_mm": 1000.0
        * float(
            np.linalg.norm(
                result.estimate.handeye_translation
                - scene.handeye_translation
            )
        ),
        "cost": float(result.cost),
        "evaluations": int(result.evaluations),
        "elapsed_s": float(elapsed_s),
    }


def _shape_error_rms_mm(model, basis, coefficients, scene) -> float:
    grid = np.linspace(0.0, 1.0, 61)
    xi, eta = np.meshgrid(grid, grid, indexing="ij")
    xi = xi.reshape(-1)
    eta = eta.reshape(-1)
    points = (
        scene.board.corner[None, :]
        + (xi * scene.board.length_u)[:, None] * scene.board.u[None, :]
        + (eta * scene.board.length_v)[:, None] * scene.board.v[None, :]
    )
    truth = model.flatness_height(
        points,
        corner=scene.board.corner,
        board_u=scene.board.u,
        board_v=scene.board.v,
        width=scene.board.length_u,
        height=scene.board.length_v,
    )
    plane = np.column_stack((np.ones_like(xi), xi, eta))
    fitted = basis.evaluate(xi, eta) @ coefficients
    # Board translation and tilt absorb an arbitrary constant/linear part of
    # the height field.  Compare only the gauge-invariant non-planar remainder.
    difference = fitted - truth
    difference -= plane @ np.linalg.lstsq(plane, difference, rcond=None)[0]
    return 1000.0 * float(np.sqrt(np.mean(difference**2)))


def _shape_metrics(result, scene, model, basis, elapsed_s: float) -> dict:
    rotation_error, translation_error = result.errors_against(
        scene.handeye_rotation, scene.handeye_translation
    )
    return {
        "converged": bool(result.converged),
        "rotation_error_deg": rotation_error,
        "translation_error_mm": translation_error,
        "shape_error_rms_mm": _shape_error_rms_mm(
            model, basis, result.shape_coefficients, scene
        ),
        "cost": float(result.cost),
        "evaluations": int(result.evaluations),
        "rank": int(result.full_rank),
        "condition_number": float(result.condition_number),
        "elapsed_s": float(elapsed_s),
    }


def _production_shape_metrics(result, scene, model, basis, elapsed_s: float) -> dict:
    return {
        "converged": bool(result.converged),
        "rotation_error_deg": rotation_distance_deg(
            result.estimate.handeye_rotation, scene.handeye_rotation
        ),
        "translation_error_mm": 1000.0
        * float(
            np.linalg.norm(
                result.estimate.handeye_translation
                - scene.handeye_translation
            )
        ),
        "shape_error_rms_mm": _shape_error_rms_mm(
            model, basis, result.estimate.shape_coefficients, scene
        ),
        "cost": float(result.cost),
        "evaluations": int(result.evaluations),
        "rank": int(result.diagnostics.rank),
        "condition_number": float(result.diagnostics.condition_number),
        "elapsed_s": float(elapsed_s),
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"median": None, "p95": None, "maximum": None}
    values_array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values_array)),
        "p95": float(np.percentile(values_array, 95.0)),
        "maximum": float(np.max(values_array)),
    }


def _summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("converged")]
    payload = {
        "trials": len(rows),
        "convergence_rate": len(valid) / max(len(rows), 1),
        "rotation_error_deg": _stats(
            [row["rotation_error_deg"] for row in valid]
        ),
        "translation_error_mm": _stats(
            [row["translation_error_mm"] for row in valid]
        ),
        "elapsed_s": _stats([row["elapsed_s"] for row in rows]),
    }
    shape = [row["shape_error_rms_mm"] for row in valid if "shape_error_rms_mm" in row]
    if shape:
        payload["shape_error_rms_mm"] = _stats(shape)
    return payload


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _markdown(payload: dict) -> str:
    lines = [
        "# 共享形貌—手眼联合估计可行性报告",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "本实验同时回归当前生产共享形貌后端和早期研究原型。",
        "",
        "## 方法边界",
        "",
        "- `flat_baseline`：现有理想平面 12DOF-V2。",
        "- `oracle_no_flatness`：保持同一随机种子但移除固定形貌，是形貌被完美补偿后的上界。",
        "- `production_shared_legendre`：当前生产12-DOF-V2共享形貌模式；同一模型也用于六种子初解、滚动更新、候选预测和边际信息。",
        "- `shared_matched`：联合估计六维共享形貌；基函数与仿真生成器同族，属于可行性上界，不能作为泛化证据。",
        "- `shared_legendre`：联合估计通用四阶二维 Legendre 形貌，不知道仿真的正弦基或相位。",
        "- 所有形貌基均不包含独立的常数和两个一次斜率，防止与标定件位置、姿态产生规范自由度。",
        "",
        "## 汇总",
        "",
        "| 工况 | 方法 | 收敛率 | 旋转中位/P95 (°) | 平移中位/P95 (mm) | 去平面形貌RMS中位 (mm) | 时间中位 (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for condition, methods in payload["summary"].items():
        for method, summary in methods.items():
            rotation = summary["rotation_error_deg"]
            translation = summary["translation_error_mm"]
            shape = summary.get("shape_error_rms_mm", {}).get("median")
            lines.append(
                f"| `{condition}` | `{method}` | {summary['convergence_rate']:.1%} | "
                f"{_fmt(rotation['median'])} / {_fmt(rotation['p95'])} | "
                f"{_fmt(translation['median'])} / {_fmt(translation['p95'])} | "
                f"{_fmt(shape)} | {_fmt(summary['elapsed_s']['median'])} |"
            )
    lines += [
        "",
        "## 判读规则",
        "",
        "1. 若 oracle 明显优于 baseline，说明固定形貌确是主要可消除偏差。",
        "2. 若 matched 接近 oracle，说明共享形貌与外参在当前观测几何下原则上可分离。",
        "3. 只有 generic 也稳定改善，才能说明路线不依赖已知仿真函数。",
        "4. 若形貌拟合误差下降但手眼误差不降，说明参数仍有强耦合，下一步必须优化交叉覆盖和边际化位姿设计。",
        "5. 形貌误差在整块标定件网格上评价，并先移除两张形貌之差的最佳拟合平面；常数高度和一次倾斜属于板位姿规范自由度。",
        "",
        f"原始结果：`{payload['output_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--poses", type=int, default=16)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--frames-per-pose", type=int, default=18)
    parser.add_argument("--pose-seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=20260807)
    parser.add_argument(
        "--pose-selection",
        choices=(
            "flat_information",
            "shape_marginal",
            "shape_marginal_goal",
            "fixed",
        ),
        default="flat_information",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=("flatness_only", "combined_stress"),
        dest="conditions",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.trials < 1 or args.poses < 4 or args.frames_per_pose < 4:
        parser.error("trials >= 1, poses >= 4 and frames-per-pose >= 4 are required")
    if args.candidate_pool < args.poses:
        parser.error("candidate-pool must be at least poses")

    parameters = _parameters()
    scene = default_scene()
    all_poses, all_measurements = generate_seed_dataset(
        scene, count=args.candidate_pool, seed=args.pose_seed
    )
    if args.pose_selection == "flat_information":
        indices = _information_selected_indices(
            all_poses,
            all_measurements,
            selected_count=args.poses,
            seed_count=6,
            parameters=parameters,
        )
    elif args.pose_selection in {"shape_marginal", "shape_marginal_goal"}:
        indices = _shape_marginal_selected_indices(
            all_poses,
            all_measurements,
            selected_count=args.poses,
            seed_count=6,
            parameters=parameters,
            objective=(
                "goal_a_optimal"
                if args.pose_selection == "shape_marginal_goal"
                else "d_optimal"
            ),
        )
    else:
        indices = list(range(args.poses))
    physical_poses = [all_poses[index] for index in indices]
    base_noise = _noise_config(parameters)
    nominal_rotation, nominal_translation = _nominal_handeye(parameters, scene)
    conditions = args.conditions or ["flatness_only", "combined_stress"]
    rows = {
        condition: {
            method: []
            for method in (
                "flat_baseline",
                "oracle_no_flatness",
                "production_shared_legendre",
                "shared_matched",
                "shared_legendre",
            )
        }
        for condition in conditions
    }

    for condition in conditions:
        config = _condition_config(
            base_noise, condition, random_seed=args.random_seed
        )
        oracle_config = replace(config, board_flatness_rms_m=0.0)
        print(f"[{condition}]", flush=True)
        for trial in range(args.trials):
            poses, measurements, model = _collect_dataset(
                config,
                physical_poses,
                frames_per_pose=args.frames_per_pose,
                parameters=parameters,
                trial=trial,
            )
            oracle_poses, oracle_measurements, _ = _collect_dataset(
                oracle_config,
                physical_poses,
                frames_per_pose=args.frames_per_pose,
                parameters=parameters,
                trial=trial,
            )
            flat_solver = _solver(parameters)
            started = time.perf_counter()
            flat = flat_solver.solve(
                poses,
                measurements,
                nominal_rotation,
                nominal_translation,
                board_dimensions=(scene.board.length_u, scene.board.length_v),
            )
            rows[condition]["flat_baseline"].append(
                _flat_metrics(flat, scene, time.perf_counter() - started)
            )

            started = time.perf_counter()
            oracle = flat_solver.solve(
                oracle_poses,
                oracle_measurements,
                nominal_rotation,
                nominal_translation,
                board_dimensions=(scene.board.length_u, scene.board.length_v),
            )
            rows[condition]["oracle_no_flatness"].append(
                _flat_metrics(oracle, scene, time.perf_counter() - started)
            )

            production_basis = SurfaceBasis("legendre", degree=4)
            production_solver = TwelveDofV2Solver(
                surface_model="shared",
                surface_basis_kind="legendre",
                surface_degree=4,
                plane_weight=flat_solver.weights["plane_weight"],
                edge_weight=flat_solver.weights["edge_weight"],
                endpoint_plane_weight=flat_solver.weights[
                    "endpoint_plane_weight"
                ],
                max_evaluations=1200,
                tolerance=1e-10,
            )
            started = time.perf_counter()
            production = production_solver.solve(
                poses,
                measurements,
                nominal_rotation,
                nominal_translation,
                board_dimensions=(
                    scene.board.length_u,
                    scene.board.length_v,
                ),
            )
            rows[condition]["production_shared_legendre"].append(
                _production_shape_metrics(
                    production,
                    scene,
                    model,
                    production_basis,
                    time.perf_counter() - started,
                )
            )

            for method, basis in (
                ("shared_matched", SurfaceBasis("matched")),
                ("shared_legendre", SurfaceBasis("legendre", degree=4)),
            ):
                shared_solver = SharedShapeHandEyeSolver(
                    basis,
                    plane_weight=flat_solver.weights["plane_weight"],
                    edge_weight=flat_solver.weights["edge_weight"],
                    endpoint_surface_weight=flat_solver.weights[
                        "endpoint_plane_weight"
                    ],
                )
                started = time.perf_counter()
                refined = shared_solver.solve(
                    poses,
                    measurements,
                    flat,
                    board_dimensions=(scene.board.length_u, scene.board.length_v),
                )
                rows[condition][method].append(
                    _shape_metrics(
                        refined,
                        scene,
                        model,
                        basis,
                        time.perf_counter() - started,
                    )
                )
            trial_summary = " ".join(
                f"{method}:t={values[-1]['translation_error_mm']:.3f}mm"
                for method, values in rows[condition].items()
            )
            print(f"  trial {trial + 1}/{args.trials} {trial_summary}", flush=True)

    summary = {
        condition: {
            method: _summarize(values) for method, values in methods.items()
        }
        for condition, methods in rows.items()
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "trials": args.trials,
            "poses": args.poses,
            "candidate_pool": args.candidate_pool,
            "frames_per_pose": args.frames_per_pose,
            "selected_pose_indices": indices,
            "pose_selection": args.pose_selection,
            "random_seed": args.random_seed,
            "flatness_rms_mm": 0.5,
        },
        "conditions": conditions,
        "condition_noise": {
            condition: asdict(
                _condition_config(
                    base_noise, condition, random_seed=args.random_seed
                )
            )
            for condition in conditions
        },
        "summary": summary,
        "trials": rows,
        "output_json": str(args.output_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_markdown(payload), encoding="utf-8")
    print(f"JSON: {args.output_json}")
    print(f"报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
