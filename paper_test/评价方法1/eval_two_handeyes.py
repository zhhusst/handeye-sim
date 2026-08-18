#!/usr/bin/env python3
"""评价方法1：用新标定结果的两个手眼（6种子初始化 / NBV后）评价球验证数据。

口径与 flat_vs_shared 实验一致：
  - 球面点用手眼变换到基座系
  - validate_sphere_views：固定半径球拟合 RMSE/P95/MAX + 自由球拟合 + 留一法
输入：
  --result   calibration_result.json（含 iterations[0]=6种子, iterations[1]=NBV后）
  --npz      球验证数据 sphere_acquisition.npz
  --radius   球半径 mm（默认 10.001）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.sphere_validation import (
    SphereArtifact,
    SphereValidationThresholds,
    transform_profile_to_base,
    validate_sphere_views,
)


def load_ball_views(npz_path: Path, R_fs, t_fs):
    """把球验证 npz 的每帧选中点用手眼变换到基座系，按组返回。"""
    npz = np.load(str(npz_path), allow_pickle=True)
    offsets = npz["selected_frame_offsets"]
    points = npz["selected_points_sensor_m"]
    fpi = npz["frame_pose_indices"]
    rots = npz["flange_rotations"]
    ts = npz["flange_translations_m"]

    groups = []
    for p in np.unique(fpi):
        frames = np.where(fpi == p)[0]
        base = []
        for f in frames:
            s = points[offsets[f]:offsets[f + 1]]
            s = s[np.all(np.isfinite(s), axis=1)]
            if len(s) < 5:
                continue
            base.append(transform_profile_to_base(
                s, rots[f], ts[f], R_fs, t_fs))
        if base:
            groups.append(np.vstack(base))
    return groups


def evaluate(handeye_R, handeye_t, npz_path, artifact, thresholds):
    groups = load_ball_views(npz_path, np.asarray(handeye_R), np.asarray(handeye_t))
    report = validate_sphere_views(
        groups, artifact,
        robust_scale_m=0.00010,
        thresholds=thresholds,
        bootstrap_trials=100,
        random_seed=20260813,
    )
    return report, groups


def main() -> int:
    parser = argparse.ArgumentParser(description="评价方法1：双手眼球验证")
    parser.add_argument("--result", type=Path, required=True,
                        help="calibration_result.json")
    parser.add_argument("--npz", type=Path,
                        default=Path("/workspace/data/sphere_validation_runs/"
                                     "20260816_142845_sphere_20mm/sphere_acquisition.npz"))
    parser.add_argument("--radius", type=float, default=10.001)
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/tmp_diag/eval_20260818_110341.json"))
    args = parser.parse_args()

    res = json.load(open(args.result))
    artifact = SphereArtifact(
        artifact_id="sphere_20mm",
        diameter_m=0.001 * 2 * args.radius,
        roundness_m=0.001 * 0.002,
        model="D20GZ",
    )
    thresholds = SphereValidationThresholds()

    iterations = res["simulation"]["iterations"]
    phases = {
        "iter0_6seeds": iterations[0],
        "iter1_nbv": iterations[-1],
    }

    output = {"result_file": str(args.result), "npz": str(args.npz),
              "radius_mm": args.radius, "evaluations": {}}
    for label, it in phases.items():
        R_fs = np.array(it["handeye_rotation"])
        t_fs = np.array(it["handeye_translation_m"])
        report, groups = evaluate(R_fs, t_fs, args.npz, artifact, thresholds)

        # 提取关键指标
        fixed = report["fixed_radius"]["all_points"]
        free = report["free_radius_diagnostic"]
        loo = report.get("leave_one_pose_out", {})
        eval_entry = {
            "handeye_translation_mm": (1000.0 * t_fs).tolist(),
            "nbv_index": it.get("nbv_index"),
            "phase": it.get("phase"),
            "surface_rms_mm": it.get("surface_rms_mm"),
            "fixed_radius_rmse_mm": fixed["rmse_mm"],
            "fixed_radius_p95_mm": fixed["p95_abs_mm"],
            "fixed_radius_max_mm": fixed["maximum_abs_mm"],
            "fixed_radius_mean_mm": fixed["signed_mean_mm"],
            "free_radius_mm": free.get("fitted_radius_mm"),
            "free_diameter_error_mm": free.get("diameter_error_mm"),
            "free_center_mm": free.get("center_base_m"),
            "loo_rmse_mm": (np.mean(loo.get("held_out_rmse_mm"))
                            if loo.get("held_out_rmse_mm") else None),
            "loo_center_rms_mm": loo.get("center_spread_rms_mm"),
            "pass": report.get("passed"),
            "interpretation": report.get("interpretation"),
            "n_points": sum(len(g) for g in groups),
            "n_groups": len(groups),
        }
        output["evaluations"][label] = eval_entry

        print(f"\n===== {label}（手眼 t={eval_entry['handeye_translation_mm']}） =====")
        print(f"  固定半径球拟合：RMSE {fixed['rmse_mm']:.4f} mm | "
              f"P95 {fixed['p95_abs_mm']:.4f} mm | MAX {fixed['maximum_abs_mm']:.4f} mm")
        print(f"  自由球拟合：r={free.get('fitted_radius_mm')} mm | "
              f"直径误差 {free.get('diameter_error_mm')} mm")
        if loo:
            loo_rmse = np.mean(loo["held_out_rmse_mm"]) if loo.get("held_out_rmse_mm") else None
            print(f"  留一法：held-out RMSE {loo_rmse:.4f} mm | "
                  f"球心扩散 {loo.get('center_spread_rms_mm'):.4f} mm")
        print(f"  pass={report.get('passed')} | 组数 {len(groups)} | "
              f"点数 {sum(len(g) for g in groups)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已保存: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
