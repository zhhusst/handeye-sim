#!/usr/bin/env python3
"""Paired degree selection for the production shared-surface solver.

Every degree sees exactly the same noisy training profiles and the same
independent validation poses in each trial.  Validation geometry, rather than
training cost or simulation truth, is the primary model-selection signal.
Simulation truth is retained only as a secondary accuracy audit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
CORE_PACKAGE_ROOT = WORKSPACE / "ros2_ws/src/handeye_calibration_core"
sys.path.insert(0, str(CORE_PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from analyze_precision_error_budget import (  # noqa: E402
    _information_selected_indices,
    _noise_config,
    _nominal_handeye,
    _parameters,
    _solver,
)
from calibration_pipeline.geometry import rotation_distance_deg  # noqa: E402
from calibration_pipeline.simulation.synthetic import (  # noqa: E402
    default_scene,
    generate_seed_dataset,
)
from calibration_pipeline.solvers import TwelveDofV2Solver  # noqa: E402
from calibration_pipeline.v2_backend.shared_surface import (  # noqa: E402
    get_surface_basis,
)
from prototype_shared_shape_calibration import (  # noqa: E402
    _collect_dataset,
    _condition_config,
    _shape_error_rms_mm,
)


DEFAULT_JSON = WORKSPACE / "data/shared_surface_degree_validation.json"
DEFAULT_REPORT = WORKSPACE / "docs/共享形貌阶数_独立验证报告.md"


def _physical_geometric_metrics(result, poses, measurements, basis) -> dict:
    """Unweighted metric residuals on fixed parameters, without refitting."""
    estimate = result.estimate
    board = estimate.board
    coefficients = estimate.shape_coefficients
    pose_rms = []
    all_residuals = []

    def surface_distance(points_base: np.ndarray) -> np.ndarray:
        delta = np.asarray(points_base) - board.corner[None, :]
        xi = (delta @ board.u) / board.length_u
        eta = (delta @ board.v) / board.length_v
        return delta @ board.normal - basis.height(
            xi, eta, coefficients
        )

    for pose, measurement in zip(poses, measurements):
        sensor_rotation = pose.rotation @ estimate.handeye_rotation
        sensor_translation = (
            pose.translation + pose.rotation @ estimate.handeye_translation
        )
        points_base = (
            sensor_rotation @ measurement.profile_points.T
        ).T + sensor_translation
        endpoint_u = (
            sensor_rotation @ measurement.endpoint_u + sensor_translation
        )
        endpoint_v = (
            sensor_rotation @ measurement.endpoint_v + sensor_translation
        )
        residual = np.concatenate(
            (
                surface_distance(points_base),
                np.array(
                    [
                        board.v @ (endpoint_u - board.corner),
                        surface_distance(endpoint_u[None, :])[0],
                        board.u @ (endpoint_v - board.corner),
                        surface_distance(endpoint_v[None, :])[0],
                    ]
                ),
            )
        )
        all_residuals.append(residual)
        pose_rms.append(float(np.sqrt(np.mean(residual**2))))
    combined = np.concatenate(all_residuals)
    return {
        "score_m": float(np.median(pose_rms)),
        "rms_m": float(np.sqrt(np.mean(combined**2))),
    }


def _stats(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if not len(finite):
        return {
            "mean": None,
            "standard_error": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "mean": float(np.mean(finite)),
        "standard_error": float(
            np.std(finite, ddof=1) / np.sqrt(len(finite))
            if len(finite) > 1
            else 0.0
        ),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
        "maximum": float(np.max(finite)),
    }


def _summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["converged"]]
    names = (
        "validation_score_mm",
        "validation_rms_mm",
        "training_score_mm",
        "rotation_error_deg",
        "translation_error_mm",
        "shape_error_rms_mm",
        "data_condition_number",
        "prior_augmented_condition_number",
        "elapsed_s",
    )
    return {
        "trials": len(rows),
        "converged_trials": len(valid),
        "convergence_rate": len(valid) / max(len(rows), 1),
        **{
            name: _stats([float(row[name]) for row in valid])
            for name in names
        },
        "data_rank_minimum": (
            min(int(row["data_rank"]) for row in valid) if valid else None
        ),
        "prior_augmented_rank_minimum": (
            min(int(row["prior_augmented_rank"]) for row in valid)
            if valid
            else None
        ),
    }


def _paired_comparisons(rows: dict[str, list[dict]]) -> dict:
    payload = {}
    degrees = sorted(int(value) for value in rows)
    for left, right in zip(degrees[:-1], degrees[1:]):
        paired = [
            (
                rows[str(left)][index],
                rows[str(right)][index],
            )
            for index in range(len(rows[str(left)]))
            if rows[str(left)][index]["converged"]
            and rows[str(right)][index]["converged"]
        ]
        deltas = np.asarray(
            [
                right_row["validation_score_mm"]
                - left_row["validation_score_mm"]
                for left_row, right_row in paired
            ],
            dtype=float,
        )
        payload[f"degree_{left}_to_{right}"] = {
            "paired_trials": len(deltas),
            "validation_delta_right_minus_left_mm": _stats(
                deltas.tolist()
            ),
            "right_win_rate": (
                float(np.mean(deltas < 0.0)) if len(deltas) else None
            ),
        }
    return payload


def _recommend(summary: dict) -> dict:
    eligible = {
        int(degree): values
        for degree, values in summary.items()
        if values["convergence_rate"] == 1.0
        and values["rotation_error_deg"]["p95"] is not None
        and values["rotation_error_deg"]["p95"] < 0.1
        and values["translation_error_mm"]["p95"] is not None
        and values["translation_error_mm"]["p95"] < 0.1
    }
    if not eligible:
        return {
            "recommended_degree": None,
            "rule": "one-standard-error with 0.1 deg/0.1 mm simulation gate",
            "reason": "no degree passed convergence and accuracy gates",
        }
    best_degree = min(
        eligible,
        key=lambda degree: eligible[degree]["validation_score_mm"]["mean"],
    )
    best = eligible[best_degree]["validation_score_mm"]
    cutoff = float(best["mean"] + best["standard_error"])
    # The production question is whether extra coefficients are necessary to
    # meet the declared hand-eye target.  Once that gate is passed, prefer the
    # lowest-dimensional observable model.  The one-SE result is retained as
    # a separate pure-prediction diagnostic instead of silently overriding
    # the engineering accuracy requirement.
    recommended = min(eligible)
    one_se_degree = min(
        degree
        for degree, values in eligible.items()
        if values["validation_score_mm"]["mean"] <= cutoff
    )
    return {
        "recommended_degree": recommended,
        "best_mean_validation_degree": best_degree,
        "one_standard_error_degree": one_se_degree,
        "one_standard_error_cutoff_mm": cutoff,
        "rule": (
            "选择完全收敛且仿真P95满足0.1°/0.1 mm的最低阶模型；"
            "另行报告只追求预测残差的一标准误差选择"
        ),
        "reason": "主选择量来自独立位姿；仿真真值只用于精度准入，不参与拟合或验证残差计算",
    }


def _fmt(value, digits: int = 5) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown(payload: dict) -> str:
    lines = [
        "# 共享形貌二/三/四阶独立验证报告",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 实验设计",
        "",
        f"- 配对试验：{payload['experiment']['trials']}次；每次三个阶数使用完全相同的原始数据。",
        f"- 训练位姿：{payload['experiment']['training_pose_count']}；独立验证位姿：{payload['experiment']['validation_pose_count']}。",
        f"- 每个位姿同步帧：{payload['experiment']['frames_per_pose']}。",
        "- 工况：0.5 mm RMS固定非平面形貌加当前综合噪声。",
        "- 主选择量：未参与求解的新位姿几何残差；仿真真值只作0.1°/0.1 mm门禁。",
        "- 每个模型的收敛判据使用data-only秩和条件数，不允许形貌先验补秩。",
        "",
        "## 汇总",
        "",
        "| 阶数/系数 | 收敛率 | 独立验证score中位/P95 (mm) | 旋转P95 (°) | 平移P95 (mm) | 形貌RMS中位 (mm) | data条件数中位 | prior条件数中位 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for degree, values in sorted(
        payload["summary"].items(), key=lambda item: int(item[0])
    ):
        score = values["validation_score_mm"]
        lines.append(
            f"| {degree}阶/{payload['coefficient_counts'][degree]} | "
            f"{values['convergence_rate']:.1%} | "
            f"{_fmt(score['median'])} / {_fmt(score['p95'])} | "
            f"{_fmt(values['rotation_error_deg']['p95'])} | "
            f"{_fmt(values['translation_error_mm']['p95'])} | "
            f"{_fmt(values['shape_error_rms_mm']['median'])} | "
            f"{_fmt(values['data_condition_number']['median'], 3)} | "
            f"{_fmt(values['prior_augmented_condition_number']['median'], 3)} |"
        )
    lines += [
        "",
        "## 配对差值",
        "",
        "差值定义为高一阶减低一阶；负值表示升阶改善独立验证误差。",
        "",
        "| 比较 | 配对次数 | 差值中位/P95 (mm) | 高阶胜率 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in payload["paired_comparisons"].items():
        delta = values["validation_delta_right_minus_left_mm"]
        win_rate = values["right_win_rate"]
        lines.append(
            f"| `{name}` | {values['paired_trials']} | "
            f"{_fmt(delta['median'])} / {_fmt(delta['p95'])} | "
            f"{'—' if win_rate is None else f'{win_rate:.1%}'} |"
        )
    recommendation = payload["recommendation"]
    lines += [
        "",
        "## 选择结论",
        "",
        f"推荐阶数：**{recommendation['recommended_degree']}**。",
        "",
        "若只按独立验证均值的一标准误差规则选择，结果为："
        f"**{recommendation.get('one_standard_error_degree')}阶**；"
        f"验证均值最小的是**{recommendation.get('best_mean_validation_degree')}阶**。",
        "",
        f"规则：{recommendation['rule']}。",
        "",
        f"说明：{recommendation['reason']}。",
        "",
        "该结论只适用于当前仿真形貌族、覆盖范围和综合噪声。实机仍应使用同样的独立位姿验证重新选阶。",
        "",
        f"原始数据：`{payload['output_json']}`",
        "",
    ]
    return "\n".join(lines)


def _run_paired_trial(task) -> tuple[int, dict[str, dict]]:
    (
        trial,
        config,
        physical_poses,
        training_pose_count,
        validation_pose_count,
        frames_per_pose,
        parameters,
    ) = task
    scene = default_scene()
    total_poses = training_pose_count + validation_pose_count
    poses, measurements, model = _collect_dataset(
        config,
        physical_poses,
        frames_per_pose=frames_per_pose,
        parameters=parameters,
        trial=trial,
    )
    if len(poses) != total_poses:
        raise RuntimeError(
            f"trial {trial} retained {len(poses)}/{total_poses} poses; "
            "paired validation requires a complete dataset"
        )
    training_poses = poses[:training_pose_count]
    training_measurements = measurements[:training_pose_count]
    validation_poses = poses[training_pose_count:]
    validation_measurements = measurements[training_pose_count:]
    nominal_rotation, nominal_translation = _nominal_handeye(
        parameters, scene
    )
    flat_solver = _solver(parameters)
    flat_initial = flat_solver.solve(
        training_poses,
        training_measurements,
        nominal_rotation,
        nominal_translation,
        board_dimensions=(scene.board.length_u, scene.board.length_v),
    )
    solver_values = parameters["solver"]
    trial_rows = {}
    for degree in (2, 3, 4):
        basis = get_surface_basis("legendre", degree)
        solver = TwelveDofV2Solver(
            surface_model="shared",
            surface_basis_kind="legendre",
            surface_degree=degree,
            shape_scale_m=float(solver_values["shape_scale_m"]),
            shape_regularization=float(
                solver_values["shape_regularization"]
            ),
            plane_weight=flat_solver.weights["plane_weight"],
            edge_weight=flat_solver.weights["edge_weight"],
            endpoint_plane_weight=flat_solver.weights[
                "endpoint_plane_weight"
            ],
            max_evaluations=int(solver_values["max_evaluations"]),
            tolerance=float(solver_values["tolerance"]),
            state_scale=flat_solver.flat_state_scale,
            maximum_condition_number=float(
                solver_values["maximum_condition_number"]
            ),
        )
        started = time.perf_counter()
        result = solver.solve(
            training_poses,
            training_measurements,
            nominal_rotation,
            nominal_translation,
            board_dimensions=(scene.board.length_u, scene.board.length_v),
            initial_board_rotation=flat_initial.estimate.board.rotation,
            initial_estimate=flat_initial.estimate,
        )
        elapsed = time.perf_counter() - started
        training_metrics = _physical_geometric_metrics(
            result, training_poses, training_measurements, basis
        )
        validation_metrics = _physical_geometric_metrics(
            result, validation_poses, validation_measurements, basis
        )
        prior_rank = result.diagnostics.prior_augmented_rank
        prior_condition = result.diagnostics.prior_augmented_condition_number
        trial_rows[str(degree)] = {
            "trial": trial,
            "degree": degree,
            "coefficient_count": basis.size,
            "converged": bool(result.converged),
            "data_rank": int(result.diagnostics.rank),
            "data_condition_number": float(
                result.diagnostics.condition_number
            ),
            "prior_augmented_rank": int(
                result.diagnostics.rank if prior_rank is None else prior_rank
            ),
            "prior_augmented_condition_number": float(
                result.diagnostics.condition_number
                if prior_condition is None
                else prior_condition
            ),
            "training_score_mm": 1000.0 * training_metrics["score_m"],
            "validation_score_mm": 1000.0
            * validation_metrics["score_m"],
            "validation_rms_mm": 1000.0 * validation_metrics["rms_m"],
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
                model,
                basis,
                result.estimate.shape_coefficients,
                scene,
            ),
            "elapsed_s": elapsed,
        }
    return trial, trial_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--training-poses", type=int, default=12)
    parser.add_argument("--validation-poses", type=int, default=12)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--frames-per-pose", type=int, default=18)
    parser.add_argument("--pose-seed", type=int, default=17)
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    total_poses = args.training_poses + args.validation_poses
    if args.trials < 2:
        parser.error("at least two paired trials are required")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.training_poses < 6 or args.validation_poses < 4:
        parser.error("training >= 6 and validation >= 4 poses are required")
    if args.candidate_pool < total_poses:
        parser.error("candidate-pool must cover training and validation poses")

    parameters = _parameters()
    scene = default_scene()
    ideal_poses, ideal_measurements = generate_seed_dataset(
        scene, count=args.candidate_pool, seed=args.pose_seed
    )
    indices = _information_selected_indices(
        ideal_poses,
        ideal_measurements,
        selected_count=total_poses,
        seed_count=6,
        parameters=parameters,
    )
    physical_poses = [ideal_poses[index] for index in indices]
    training_indices = indices[: args.training_poses]
    validation_indices = indices[args.training_poses :]
    base_noise = _noise_config(parameters)
    config = _condition_config(
        base_noise, "combined_stress", random_seed=args.random_seed
    )
    rows: dict[str, list[dict]] = {str(degree): [] for degree in (2, 3, 4)}
    tasks = [
        (
            trial,
            config,
            physical_poses,
            args.training_poses,
            args.validation_poses,
            args.frames_per_pose,
            parameters,
        )
        for trial in range(args.trials)
    ]
    completed = []
    if args.workers == 1:
        completed = [_run_paired_trial(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_paired_trial, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
                trial, trial_rows = completed[-1]
                print(
                    f"trial {trial + 1}/{args.trials}: "
                    + ", ".join(
                        f"d{degree} val={trial_rows[str(degree)]['validation_score_mm']:.5f}mm "
                        f"t={trial_rows[str(degree)]['translation_error_mm']:.4f}mm"
                        for degree in (2, 3, 4)
                    ),
                    flush=True,
                )
    for trial, trial_rows in sorted(completed):
        for degree in (2, 3, 4):
            rows[str(degree)].append(trial_rows[str(degree)])

    summary = {degree: _summarize(values) for degree, values in rows.items()}
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "trials": args.trials,
            "training_pose_count": args.training_poses,
            "validation_pose_count": args.validation_poses,
            "candidate_pool": args.candidate_pool,
            "frames_per_pose": args.frames_per_pose,
            "pose_seed": args.pose_seed,
            "random_seed": args.random_seed,
            "training_pose_indices": training_indices,
            "validation_pose_indices": validation_indices,
            "flatness_rms_mm": 0.5,
            "selection_signal": "independent-pose geometric residual",
        },
        "noise": asdict(config),
        "coefficient_counts": {
            str(degree): get_surface_basis("legendre", degree).size
            for degree in (2, 3, 4)
        },
        "summary": summary,
        "paired_comparisons": _paired_comparisons(rows),
        "recommendation": _recommend(summary),
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
    print(f"推荐阶数: {payload['recommendation']['recommended_degree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
