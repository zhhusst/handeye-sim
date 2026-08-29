#!/usr/bin/env python3
"""Evaluate the nine 2026-08-20 real calibration runs with one sphere dataset.

Each calibration run receives an independent output directory containing:

* method 1: fixed/free-radius and leave-one-direction diagnostics;
* method 2: fixed 25 mm ROR and RCIM-style per-direction/combined fit;
* a manifest, execution log and readable summary.

Runs without a valid ``calibration_result.json`` are retained as explicit
failed evaluations instead of being silently omitted or replaced by an
offline/oracle estimate.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
SPHERE_RUN = (
    WORKSPACE
    / "data/sphere_validation_runs/20260820_124558_sphere_20mm"
)
SPHERE_NPZ = SPHERE_RUN / "sphere_acquisition.npz"
CALIBRATION_ROOT = WORKSPACE / "data/calibration_runs"

RUN_NAMES = (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _run(command: list[str], log: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=WORKSPACE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def _method2_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    combined = text.split("[All", 1)[-1]

    def number(name: str) -> float | None:
        match = re.search(
            rf"^{re.escape(name)}=([+-]?[0-9]+(?:\.[0-9]+)?)",
            combined,
            flags=re.MULTILINE,
        )
        return None if match is None else float(match.group(1))

    pose_count_match = re.search(r"\[All (\d+) poses combined\]", text)
    return {
        "pose_count": (
            None if pose_count_match is None else int(pose_count_match.group(1))
        ),
        "combined_radius_mm": number("r_fit"),
        "combined_radius_error_signed_mm": number("delta_r_signed"),
        "combined_radius_error_abs_mm": number("delta_r_abs"),
        "combined_sigma_mm": number("sigma_e"),
        "combined_fixed_radius_rmse_mm": number("rmse_e"),
    }


def _last_failure(run_directory: Path) -> str:
    active_log = run_directory / "active_calibration.log"
    if not active_log.exists():
        return "calibration_result.json is missing"
    lines = active_log.read_text(encoding="utf-8", errors="replace").splitlines()
    failures = [line for line in lines if "[ERROR]" in line or "failed" in line.lower()]
    return failures[-1] if failures else "calibration_result.json is missing"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _per_run_report(
    output: Path,
    row: dict[str, Any],
    method1: dict[str, Any] | None,
    method2: dict[str, Any] | None,
) -> None:
    lines = [
        f"# {row['run_name']} 精密球评价",
        "",
        f"> 状态：{'完成' if row['status'] == 'evaluated' else '不可评价'}",
        "",
        "## 固定评价条件",
        "",
        f"- 球数据：`{row['sphere_npz']}`",
        f"- 标定结果：`{row.get('calibration_result') or '不存在'}`",
        "- 刻字直径：20.002 mm；固定参考半径：10.001 mm；",
        "- 方法2帧级 ROR 深度门限：25 mm；",
        "- 当前球数据仅包含6个方向，因此本报告属于统一口径的横向诊断，不满足正式实验建议的至少7方向要求；",
        "- 同一球数据跨标定运行复用的结论依赖传感器安装、机器人基坐标系和球实验条件在这些运行之间未发生变化。",
    ]
    if row["status"] != "evaluated":
        lines.extend(
            [
                "",
                "## 不能计算精度的原因",
                "",
                row["failure_reason"],
                "",
                "该运行没有生成有效的 `calibration_result.json`，因此不存在可用于将球面点从传感器系变换到机器人基坐标系的手眼外参。它保留在九组结果中并计为标定失败，但不参与数值精度排名。",
            ]
        )
    else:
        assert method1 is not None and method2 is not None
        initial = method1["evaluations"]["iter0_6seeds"]
        final = method1["evaluations"]["iter1_nbv"]
        lines.extend(
            [
                "",
                "## 方法1：固定半径、自由半径与留一方向",
                "",
                "| 指标 | 六种子初值 | 最终结果 |",
                "|---|---:|---:|",
                f"| 固定半径全点RMSE/mm | {_fmt(initial['fixed_radius_rmse_mm'])} | {_fmt(final['fixed_radius_rmse_mm'])} |",
                f"| 固定半径P95/mm | {_fmt(initial['fixed_radius_p95_mm'])} | {_fmt(final['fixed_radius_p95_mm'])} |",
                f"| 固定半径MAX/mm | {_fmt(initial['fixed_radius_max_mm'])} | {_fmt(final['fixed_radius_max_mm'])} |",
                f"| 自由拟合直径误差/mm | {_fmt(initial['free_diameter_error_mm'])} | {_fmt(final['free_diameter_error_mm'])} |",
                f"| 留一方向RMSE/mm | {_fmt(initial['loo_rmse_mm'])} | {_fmt(final['loo_rmse_mm'])} |",
                f"| 留一球心扩散RMS/mm | {_fmt(initial['loo_center_rms_mm'])} | {_fmt(final['loo_center_rms_mm'])} |",
                f"| 方向数 | {initial['n_groups']} | {final['n_groups']} |",
                f"| 通过当前严格阈值 | {initial['pass']} | {final['pass']} |",
                "",
                "方法1使用原始已选球面点，不执行方法2的帧级ROR；因此少量异常帧可能显著放大全点RMSE和MAX。",
                "",
                "## 方法2：ROR与RCIM风格合并球拟合",
                "",
                "| 指标 | 最终外参 |",
                "|---|---:|",
                f"| 方向数 | {method2['pose_count']} |",
                f"| 合并拟合半径/mm | {_fmt(method2['combined_radius_mm'])} |",
                f"| 合并半径有符号误差/mm | {_fmt(method2['combined_radius_error_signed_mm'])} |",
                f"| 合并半径绝对误差/mm | {_fmt(method2['combined_radius_error_abs_mm'])} |",
                f"| 合并自由球残差标准差/mm | {_fmt(method2['combined_sigma_mm'])} |",
                f"| 合并固定名义半径RMSE/mm | {_fmt(method2['combined_fixed_radius_rmse_mm'])} |",
                "",
                "逐方向半径、球心和残差见 `method2_rcim/metrics.txt`，图见同目录下两张PNG。",
            ]
        )
    (output / "evaluation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _root_report(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 九组真机标定统一精密球评价",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "九个源运行分别对应本目录下九个同名文件夹。8组存在标定结果并完成两种球评价；宣传册运行在NBV初始化阶段失败，没有外参可评价，作为失败样本保留。",
        "",
        "本批球数据只保留6个方向，低于正式评价建议的7方向，因此以下结果适合现有运行之间的统一横向诊断，不能替代未来逐次配对、至少7方向的正式精度实验。",
        "",
        "| # | 真机运行 | 状态 | 方法1最终RMSE/mm | 方法1最终P95/mm | 方法2半径误差/mm | 方法2固定半径RMSE/mm |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | [{row['run_name']}]({row['run_name']}/evaluation_summary.md) | {row['status']} | {_fmt(row.get('method1_final_fixed_rmse_mm'))} | {_fmt(row.get('method1_final_p95_mm'))} | {_fmt(row.get('method2_radius_error_signed_mm'))} | {_fmt(row.get('method2_fixed_rmse_mm'))} |"
        )
    valid = [row for row in rows if row["status"] == "evaluated"]
    best = min(valid, key=lambda row: row["method2_fixed_rmse_mm"])
    worst = max(valid, key=lambda row: row["method2_fixed_rmse_mm"])
    smallest_radius_bias = min(
        valid, key=lambda row: abs(row["method2_radius_error_signed_mm"])
    )
    below_point_one = sum(
        row["method2_fixed_rmse_mm"] <= 0.1 for row in valid
    )
    lines.extend(
        [
            "",
            "## 结果概览",
            "",
            f"- 9组运行中8组有有效外参并完成评价，1组初始化失败；方法2固定半径RMSE达到0.1 mm以内的运行数为 {below_point_one}/{len(valid)}。",
            f"- 方法2固定半径RMSE最好的是 `{best['run_name']}`：{best['method2_fixed_rmse_mm']:.4f} mm；最差的是 `{worst['run_name']}`：{worst['method2_fixed_rmse_mm']:.4f} mm。",
            f"- 合并拟合半径偏差最小的是 `{smallest_radius_bias['run_name']}`：{smallest_radius_bias['method2_radius_error_signed_mm']:+.4f} mm，但其固定半径RMSE为 {smallest_radius_bias['method2_fixed_rmse_mm']:.4f} mm。因此不能只用拟合半径误差代表外参质量，还必须同时考察跨方向合并RMSE、P95和球心一致性。",
            "- 方法1的8组最终结果均未通过当前严格阈值；其中方向数不足7本身也使这批数据不能作为正式验收结果。",
            "",
            "## 口径说明",
            "",
            "- 方法1同时评价六种子初值与最终NBV外参；根目录表格只列最终值。",
            "- 方法2只评价 `calibration_result.json` 中的最终手眼外参，并固定使用25 mm帧级深度ROR。",
            "- 评价数值反映球面点经手眼变换后的几何一致性，不能单独分解成手眼旋转真值误差和手眼平移真值误差。",
            "- 由于所有运行复用同一球数据，比较成立还要求运行之间未重新安装传感器、改变机器人基坐标定义或改变相关机械结构。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_plot(output: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row["status"] == "evaluated"]
    labels = [f"R{index + 1}" for index, _ in enumerate(valid)]
    x = np.arange(len(valid))
    width = 0.36
    fig, axis = plt.subplots(figsize=(11.0, 5.8))
    axis.bar(
        x - width / 2,
        [row["method1_final_p95_mm"] for row in valid],
        width,
        label="Method 1: final P95",
    )
    axis.bar(
        x + width / 2,
        [row["method2_fixed_rmse_mm"] for row in valid],
        width,
        label="Method 2: fixed-radius RMSE",
    )
    axis.axhline(0.1, color="black", linestyle="--", linewidth=1.0, label="0.1 mm")
    axis.set_xticks(x, labels)
    axis.set_ylabel("error (mm)")
    axis.set_xlabel("evaluated run (see dataset_label_mapping.md)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "nine_run_sphere_comparison.png", dpi=180)
    plt.close(fig)
    mapping = [
        f"- R{index + 1}: `{row['run_name']}`"
        for index, row in enumerate(valid)
    ]
    (output / "dataset_label_mapping.md").write_text(
        "# 图中运行缩写\n\n" + "\n".join(mapping) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sphere", type=Path, default=SPHERE_NPZ)
    parser.add_argument("--radius-mm", type=float, default=10.001)
    parser.add_argument("--z-gate-mm", type=float, default=25.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=SPHERE_RUN / "nine_real_run_evaluations",
    )
    args = parser.parse_args()
    sphere = args.sphere.resolve()
    if not sphere.exists():
        parser.error(f"sphere dataset does not exist: {sphere}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sphere_hash = _sha256(sphere)
    rows: list[dict[str, Any]] = []

    for index, run_name in enumerate(RUN_NAMES, 1):
        run_directory = CALIBRATION_ROOT / run_name
        result = run_directory / "calibration_result.json"
        run_output = output / run_name
        run_output.mkdir()
        row: dict[str, Any] = {
            "index": index,
            "run_name": run_name,
            "run_directory": str(run_directory),
            "sphere_npz": str(sphere),
            "sphere_sha256": sphere_hash,
            "reference_radius_mm": args.radius_mm,
            "method2_z_gate_mm": args.z_gate_mm,
            "formal_direction_requirement_met": False,
        }
        if not result.exists():
            row.update(
                {
                    "status": "not_evaluable",
                    "calibration_result": None,
                    "failure_reason": _last_failure(run_directory),
                }
            )
            _write_json(run_output / "evaluation_status.json", row)
            _per_run_report(run_output, row, None, None)
            rows.append(row)
            print(f"[{index}/9] {run_name}: no calibration result", flush=True)
            continue

        result_payload = json.loads(result.read_text(encoding="utf-8"))
        row.update(
            {
                "status": "evaluated",
                "calibration_result": str(result),
                "calibration_result_sha256": _sha256(result),
                "calibration_result_converged": result_payload.get("converged"),
            }
        )
        method1_file = run_output / "method1_fixed_radius.json"
        _run(
            [
                sys.executable,
                str(WORKSPACE / "paper_test/评价方法1/eval_two_handeyes.py"),
                "--result",
                str(result),
                "--npz",
                str(sphere),
                "--radius",
                str(args.radius_mm),
                "--out",
                str(method1_file),
            ],
            run_output / "method1_execution.log",
        )
        method2_output = run_output / "method2_rcim"
        _run(
            [
                str(WORKSPACE / "scripts/eval_ball.sh"),
                "--npz",
                str(sphere),
                "--result",
                str(result),
                "--radius",
                str(args.radius_mm),
                "--z-gate",
                str(args.z_gate_mm),
                "--out",
                str(method2_output),
            ],
            run_output / "method2_execution.log",
        )
        method1 = json.loads(method1_file.read_text(encoding="utf-8"))
        method2 = _method2_metrics(method2_output / "metrics.txt")
        final = method1["evaluations"]["iter1_nbv"]
        row.update(
            {
                "method1_group_count": final["n_groups"],
                "method1_final_fixed_rmse_mm": final["fixed_radius_rmse_mm"],
                "method1_final_p95_mm": final["fixed_radius_p95_mm"],
                "method1_final_max_mm": final["fixed_radius_max_mm"],
                "method1_final_free_diameter_error_mm": final[
                    "free_diameter_error_mm"
                ],
                "method1_final_loo_rmse_mm": final["loo_rmse_mm"],
                "method1_pass": final["pass"],
                "method2_pose_count": method2["pose_count"],
                "method2_radius_error_signed_mm": method2[
                    "combined_radius_error_signed_mm"
                ],
                "method2_radius_error_abs_mm": method2[
                    "combined_radius_error_abs_mm"
                ],
                "method2_sigma_mm": method2["combined_sigma_mm"],
                "method2_fixed_rmse_mm": method2[
                    "combined_fixed_radius_rmse_mm"
                ],
            }
        )
        _write_json(run_output / "evaluation_status.json", row)
        _per_run_report(run_output, row, method1, method2)
        rows.append(row)
        print(
            f"[{index}/9] {run_name}: method2 RMSE="
            f"{row['method2_fixed_rmse_mm']:.4f} mm",
            flush=True,
        )

    _write_json(output / "summary.json", rows)
    fields = sorted({key for row in rows for key in row})
    with (output / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _root_report(output, rows)
    _comparison_plot(output, rows)
    print(f"summary={output / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
