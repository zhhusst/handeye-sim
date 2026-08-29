#!/usr/bin/env python3
"""Exploratory shared-morphology study on low-quality real targets.

The precision-sphere data are evaluated only after every calibration solve.
This script deliberately reports every candidate instead of silently choosing
one by sphere error: a setting inspected on these sphere data must be frozen
and confirmed on a newly acquired calibration/validation pair before it can
support a precision claim.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
CORE_SOURCE = WORKSPACE / "ros2_ws/src/handeye_calibration_core"
sys.path.insert(0, str(CORE_SOURCE))
sys.path.insert(0, str(WORKSPACE / "scripts"))

from ablate_shared_morphology_real import (  # noqa: E402
    DEFAULT_SPHERE,
    _bootstrap_aggregate,
    _common_flat_initialization,
    _load_context,
    _load_sphere,
    _solve_model,
    _sphere_metrics,
)
from calibration_pipeline.geometry import so3_log  # noqa: E402
from calibration_pipeline.solvers.twelve_dof_v2 import (  # noqa: E402
    _initial_board_rotation,
)
from calibration_pipeline.v2_backend.corner_projection import (  # noqa: E402
    solve_corner,
)
from calibration_pipeline.v2_backend.shared_surface import (  # noqa: E402
    get_surface_basis,
)


DEFAULT_DATASETS = (
    "20260820_142718_知象光电宣传册_位置1_真机_1_nbv初始化失败",
    "20260820_144753_焊接书_位置1_真机_1",
    "20260820_150032_生锈的刚板_位置1_真机_1",
)
DEFAULT_REFERENCE = (
    "20260820_115707_圆点标定板背面_位置1_真机_2"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _reference_handeye(path: Path) -> tuple[np.ndarray, np.ndarray]:
    context = _load_context(path)
    poses, measurements, _ = _bootstrap_aggregate(context, 0, 0)
    initial, _ = _common_flat_initialization(context, poses, measurements)
    result = _solve_model(context, "ideal", poses, measurements, initial)
    if not result["usable_solution"]:
        raise RuntimeError(f"reference calibration is not usable: {path}")
    return (
        np.asarray(result["handeye_rotation"], dtype=float),
        np.asarray(result["handeye_translation_m"], dtype=float),
    )


def _reference_seeded_initial(
    context: Any,
    poses: list[Any],
    measurements: list[Any],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    board_rotation = _initial_board_rotation(
        poses, measurements, rotation, translation
    )
    x9 = np.concatenate(
        (so3_log(rotation), translation, so3_log(board_rotation))
    )
    corner, rank = solve_corner(x9, poses, measurements, **context.weights)
    if rank < 3:
        raise RuntimeError("reference-seeded corner solve is rank deficient")
    return np.concatenate((x9, corner))


def _solve_and_score(
    context: Any,
    poses: list[Any],
    measurements: list[Any],
    sphere: Any,
    initial: np.ndarray,
    *,
    model: str,
    degree: int | None,
    regularization: float | None,
    initialization: str,
) -> dict[str, Any]:
    if model == "shared":
        assert degree is not None and regularization is not None
        context.basis = get_surface_basis("legendre", degree)
        context.shape_regularization = regularization
    result = _solve_model(context, model, poses, measurements, initial)
    if result["optimizer_success"]:
        result.update(
            _sphere_metrics(
                sphere,
                np.asarray(result["handeye_rotation"], dtype=float),
                np.asarray(result["handeye_translation_m"], dtype=float),
            )
        )
    keep = {
        "optimizer_success",
        "strict_data_convergence",
        "physically_reasonable",
        "usable_solution",
        "message",
        "state_size",
        "data_rank",
        "data_condition_number",
        "prior_rank",
        "prior_condition_number",
        "objective_cost",
        "data_cost",
        "weighted_data_residual_rms_mm",
        "surface_residual_rms_mm",
        "surface_residual_mean_abs_mm",
        "surface_residual_max_abs_mm",
        "board_tilt_deg",
        "handeye_rotation",
        "handeye_translation_m",
        "board_rotation",
        "board_corner_m",
        "shape_coefficients_m",
        "shape_coefficient_rms_mm",
        "function_evaluations",
        "solver_time_s",
        "sphere_fixed_rmse_mm",
        "sphere_fixed_mean_abs_mm",
        "sphere_fixed_p95_mm",
        "sphere_fixed_max_abs_mm",
        "sphere_free_radius_mm",
        "sphere_radius_error_mm",
        "sphere_abs_radius_error_mm",
        "sphere_free_rmse_mm",
    }
    row = {key: value for key, value in result.items() if key in keep}
    row.update(
        {
            "model": model,
            "degree": degree,
            "shape_regularization": regularization,
            "initialization": initialization,
        }
    )
    return row


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        if not np.isfinite(float(value)):
            return "—"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.{digits}f}"


def _plots(output: Path, rows: list[dict[str, Any]]) -> None:
    usable = [
        row
        for row in rows
        if row["initialization"] == "current_flat_presolve"
        and row.get("usable_solution")
        and row.get("sphere_fixed_rmse_mm") is not None
    ]
    datasets = list(dict.fromkeys(row["dataset"] for row in usable))
    short_labels = {name: f"D{index + 1}" for index, name in enumerate(datasets)}
    fig, axes = plt.subplots(
        max(len(datasets), 1), 1, figsize=(9.5, 3.4 * max(len(datasets), 1))
    )
    axes = np.atleast_1d(axes)
    for axis, dataset in zip(axes, datasets):
        selected = [row for row in usable if row["dataset"] == dataset]
        ideal = next((row for row in selected if row["model"] == "ideal"), None)
        for degree, marker in ((2, "o"), (3, "s"), (4, "^")):
            values = sorted(
                (row for row in selected if row.get("degree") == degree),
                key=lambda row: row["shape_regularization"],
            )
            if values:
                axis.semilogx(
                    [row["shape_regularization"] for row in values],
                    [row["sphere_fixed_rmse_mm"] for row in values],
                    marker=marker,
                    label=f"degree {degree}",
                )
        if ideal is not None:
            axis.axhline(
                ideal["sphere_fixed_rmse_mm"],
                color="black",
                linestyle="--",
                label="ideal plane",
            )
        axis.set_title(short_labels[dataset])
        axis.set_xlabel("shape regularization λ")
        axis.set_ylabel("independent sphere RMSE (mm)")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "regularization_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 6.0))
    for row in usable:
        if row["model"] != "shared":
            continue
        axis.scatter(
            row["surface_residual_rms_mm"],
            row["sphere_fixed_rmse_mm"],
            marker={2: "o", 3: "s", 4: "^"}[row["degree"]],
            alpha=0.8,
        )
    axis.set_xlabel("calibration surface residual RMS (mm)")
    axis.set_ylabel("independent sphere RMSE (mm)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "internal_vs_external.png", dpi=180)
    plt.close(fig)


def _report(output: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# 低精度标定件共享形貌专项探索",
        "",
        f"> 生成时间：{manifest['created_utc']}",
        "",
        "## 实验边界",
        "",
        "本实验只使用每次真机运行在 NBV 之前的六个物理种子位姿。所有候选使用相同观测、初始手眼、残差权重和 TRF 设置；只改变 Legendre 阶数与形貌正则化。精密球数据不参与任何求解，但本轮已经查看了球误差，因此由本轮挑出的参数只能算探索性候选，必须在新采数据上盲测确认。",
        "",
        "`reference_seeded_diagnostic` 只在当前平面预求解不合理时启用：它用另一组高质量标定件的离线外参把求解送入正确几何邻域，用于区分“初始化失败”和“进入正确邻域后仍有模型偏差”。它不是可部署流程，也不能计入方法精度。",
        "",
        "## 当前平面预求解下的结果",
        "",
        "| 数据集 | 模型 | degree | λ | 板倾角/° | data rank | condition | surface RMS/mm | 球RMSE/mm | 半径误差/mm | 可用 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    current = [
        row
        for row in rows
        if row["initialization"] == "current_flat_presolve"
        and row["repeat"] == 0
    ]
    for row in current:
        label = "ideal" if row["model"] == "ideal" else "shared"
        lines.append(
            "| {dataset} | {label} | {degree} | {reg} | {tilt} | {rank}/{size} | {condition} | {surface} | {sphere} | {radius} | {usable} |".format(
                dataset=row["dataset"],
                label=label,
                degree="—" if row["degree"] is None else row["degree"],
                reg="—" if row["shape_regularization"] is None else f"{row['shape_regularization']:g}",
                tilt=_fmt(row.get("board_tilt_deg"), 2),
                rank=row.get("data_rank", "—"),
                size=row.get("state_size", "—"),
                condition=(f"{row['data_condition_number']:.2e}" if row.get("data_condition_number") is not None else "—"),
                surface=_fmt(row.get("surface_residual_rms_mm")),
                sphere=_fmt(row.get("sphere_fixed_rmse_mm")),
                radius=_fmt(row.get("sphere_radius_error_mm")),
                usable="是" if row.get("usable_solution") else "否",
            )
        )
    diagnostic = [
        row
        for row in rows
        if row["initialization"] == "reference_seeded_diagnostic"
        and row["repeat"] == 0
    ]
    if diagnostic:
        lines.extend(
            [
                "",
                "## 正确几何邻域诊断（不计入方法结果）",
                "",
                "| 数据集 | 模型 | degree | λ | 板倾角/° | surface RMS/mm | 球RMSE/mm | 半径误差/mm |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in diagnostic:
            lines.append(
                f"| {row['dataset']} | {row['model']} | {row['degree'] if row['degree'] is not None else '—'} | {row['shape_regularization'] if row['shape_regularization'] is not None else '—'} | {_fmt(row.get('board_tilt_deg'), 2)} | {_fmt(row.get('surface_residual_rms_mm'))} | {_fmt(row.get('sphere_fixed_rmse_mm'))} | {_fmt(row.get('sphere_radius_error_mm'))} |"
            )
    if manifest["repeats"] > 1:
        lines.extend(
            [
                "",
                "## 位姿内 bootstrap 稳定性",
                "",
                "差值为相同重复内 `shared - ideal`；负值表示共享形貌的球 RMSE 更低。",
                "",
                "| 数据集 | degree | λ | 可用率 | 球RMSE中位数/mm | Δ均值/mm | Δ中位数/mm | 改善比例 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        datasets = list(dict.fromkeys(row["dataset"] for row in rows))
        for dataset in datasets:
            baseline = {
                row["repeat"]: row
                for row in rows
                if row["dataset"] == dataset
                and row["initialization"] == "current_flat_presolve"
                and row["model"] == "ideal"
                and row.get("usable_solution")
            }
            candidates = sorted(
                {
                    (row["degree"], row["shape_regularization"])
                    for row in rows
                    if row["dataset"] == dataset
                    and row["initialization"] == "current_flat_presolve"
                    and row["model"] == "shared"
                }
            )
            for degree, regularization in candidates:
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["initialization"] == "current_flat_presolve"
                    and row["model"] == "shared"
                    and row["degree"] == degree
                    and row["shape_regularization"] == regularization
                ]
                usable = [row for row in selected if row.get("usable_solution")]
                paired = [
                    row["sphere_fixed_rmse_mm"]
                    - baseline[row["repeat"]]["sphere_fixed_rmse_mm"]
                    for row in usable
                    if row["repeat"] in baseline
                ]
                lines.append(
                    f"| {dataset} | {degree} | {regularization:g} | {len(usable) / max(len(selected), 1):.0%} | {_fmt(np.median([row['sphere_fixed_rmse_mm'] for row in usable]) if usable else None)} | {_fmt(np.mean(paired) if paired else None, 6)} | {_fmt(np.median(paired) if paired else None, 6)} | {(sum(value < 0 for value in paired) / len(paired) if paired else 0.0):.0%} |"
                )
    lines.extend(["", "## 本轮直接结论", ""])
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    for dataset in datasets:
        ideal = next(
            (
                row
                for row in rows
                if row["dataset"] == dataset
                and row["repeat"] == 0
                and row["initialization"] == "current_flat_presolve"
                and row["model"] == "ideal"
            ),
            None,
        )
        candidate = next(
            (
                row
                for row in rows
                if row["dataset"] == dataset
                and row["repeat"] == 0
                and row["initialization"] == "current_flat_presolve"
                and row["model"] == "shared"
                and row["degree"] == 4
                and math.isclose(row["shape_regularization"], 0.01)
            ),
            None,
        )
        if ideal is None:
            continue
        if not ideal.get("usable_solution"):
            ref_rows = [
                row
                for row in rows
                if row["dataset"] == dataset
                and row["initialization"] == "reference_seeded_diagnostic"
                and row.get("sphere_fixed_rmse_mm") is not None
            ]
            best_reference = min(
                (row["sphere_fixed_rmse_mm"] for row in ref_rows),
                default=None,
            )
            lines.append(
                f"- `{dataset}`：当前平面预求解给出 {ideal['board_tilt_deg']:.1f}° 的非物理解；即使借用外部正确邻域，候选的最好球 RMSE 仍为 {_fmt(best_reference)} mm，因此不能靠调整形貌阶数或 λ 挽救。"
            )
        elif candidate is not None:
            difference = (
                candidate["sphere_fixed_rmse_mm"]
                - ideal["sphere_fixed_rmse_mm"]
            )
            lines.append(
                f"- `{dataset}`：四阶、λ=0.01 相对理想平面的球 RMSE 差为 {difference:+.4f} mm（{ideal['sphere_fixed_rmse_mm']:.4f} → {candidate['sphere_fixed_rmse_mm']:.4f} mm）。"
            )
    lines.extend(
        [
            "",
            "因此，当前结果只构成“锈钢板上存在稳定改善信号”的初步证据；尚未证明低精度标定件经共享形貌后能达到高精度标定件水平，也尚未证明该收益能够跨标定件泛化。",
        ]
    )
    lines.extend(
        [
            "",
            "## 如何解释",
            "",
            "1. 内部 surface RMS 下降只是拟合能力增强，不能证明外参更准；主判断必须看独立球指标。",
            "2. 若降低 λ 后形貌幅值迅速增大、condition 恶化而球误差不降，说明额外自由度正在吸收手眼误差。",
            "3. 柔性书本或宣传册可能随放置、翻曲或接触发生位姿相关形变；单一跨位姿共享的静态形貌无法解释动态形变。锈钢板更接近“固定但不理想”的目标，是检验共享静态形貌的首选。",
            "4. 本轮任何表现最好的参数都已经接触过球评价，必须冻结后在新采集的低质量刚性板数据和新球数据上复验。",
            "",
            "## 输出",
            "",
            "- 完整数值：`candidate_results.json`、`candidate_results.csv`",
            "- 正则敏感性：`regularization_sensitivity.png`",
            "- 内部残差与外部误差：`internal_vs_external.png`",
        ]
    )
    (output / "低精度标定件共享形貌专项探索.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", type=Path)
    parser.add_argument("--degrees", nargs="+", type=int, default=(2, 3, 4))
    parser.add_argument(
        "--regularizations",
        nargs="+",
        type=float,
        default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument("--sphere", type=Path, default=DEFAULT_SPHERE)
    parser.add_argument("--sphere-max-points", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=WORKSPACE / "data/calibration_runs" / DEFAULT_REFERENCE,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dataset_paths = args.datasets or [
        WORKSPACE / "data/calibration_runs" / name for name in DEFAULT_DATASETS
    ]
    output = args.output or (
        WORKSPACE
        / "data/ablation_runs"
        / ("low_quality_morphology_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    output.mkdir(parents=True, exist_ok=False)
    sphere = _load_sphere(args.sphere.resolve(), args.sphere_max_points)
    reference_rotation, reference_translation = _reference_handeye(
        args.reference_dataset.resolve()
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory; sphere-inspected candidates require new blind confirmation",
        "datasets": [str(path.resolve()) for path in dataset_paths],
        "degrees": args.degrees,
        "regularizations": args.regularizations,
        "repeats": args.repeats,
        "random_seed": args.random_seed,
        "sphere": str(args.sphere.resolve()),
        "reference_dataset_for_failure_diagnosis": str(
            args.reference_dataset.resolve()
        ),
    }
    rows: list[dict[str, Any]] = []
    for dataset_index, path in enumerate(dataset_paths):
        context = _load_context(path.resolve())
        for repeat in range(args.repeats):
            random_seed = args.random_seed + dataset_index * 1000003 + repeat
            poses, measurements, _ = _bootstrap_aggregate(
                context, repeat, random_seed
            )
            current_initial, initialization = _common_flat_initialization(
                context, poses, measurements
            )
            strategies = [("current_flat_presolve", current_initial)]
            if repeat == 0 and not initialization["physically_plausible"]:
                strategies.append(
                    (
                        "reference_seeded_diagnostic",
                        _reference_seeded_initial(
                            context,
                            poses,
                            measurements,
                            reference_rotation,
                            reference_translation,
                        ),
                    )
                )
            for strategy_name, initial in strategies:
                ideal = _solve_and_score(
                    context,
                    poses,
                    measurements,
                    sphere,
                    initial,
                    model="ideal",
                    degree=None,
                    regularization=None,
                    initialization=strategy_name,
                )
                ideal.update(
                    {
                        "dataset": context.name,
                        "repeat": repeat,
                        "random_seed": random_seed,
                        "flat_initial_tilt_deg": initialization["selected_tilt_deg"],
                        "flat_initial_plausible": initialization["physically_plausible"],
                    }
                )
                rows.append(ideal)
                for degree in args.degrees:
                    for regularization in args.regularizations:
                        row = _solve_and_score(
                            context,
                            poses,
                            measurements,
                            sphere,
                            initial,
                            model="shared",
                            degree=degree,
                            regularization=regularization,
                            initialization=strategy_name,
                        )
                        row.update(
                            {
                                "dataset": context.name,
                                "repeat": repeat,
                                "random_seed": random_seed,
                                "flat_initial_tilt_deg": initialization["selected_tilt_deg"],
                                "flat_initial_plausible": initialization["physically_plausible"],
                            }
                        )
                        rows.append(row)
                        print(
                            f"{context.name}: repeat={repeat} {strategy_name} "
                            f"degree={degree} lambda={regularization:g} sphere="
                            f"{row.get('sphere_fixed_rmse_mm', float('nan')):.4f} mm",
                            flush=True,
                        )

    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "candidate_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(output / "candidate_results.csv", rows)
    _plots(output, rows)
    _report(output, rows, manifest)
    print(f"report={output / '低精度标定件共享形貌专项探索.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
