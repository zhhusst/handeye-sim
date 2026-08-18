#!/usr/bin/env python3
"""评价方法2：Gautam et al. RCIM 2025 §8 论文式球验证。

流程（对应论文 §8）：
  1. 每组（一次移动扫描 = 论文一个板方向）独立自由球拟合：
     - 拟合半径 R_i、球心 C_i
     - 半径偏差 Δr_i = R_i - R_true
     - 组内 form error（点到拟合球的峰谷/分散）
  2. 合并所有组做自由球拟合（论文组合数据）：
     - size error（直径口径）、form error、dispersion RMSE
  3. 逐组统计：平均 Δr、std、范围（论文 Fig.15 Middle）

输入：sphere_acquisition.npz（与 _save_dataset 格式一致，每组=移动扫描）
手眼：calibration_result.json 或参数指定
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
    _free_sphere,
    transform_profile_to_base,
)

TRUE_DIAMETER_MM = 20.002
TRUE_RADIUS_MM = 10.001


def load_npz_groups(npz_path: Path):
    npz = np.load(str(npz_path), allow_pickle=True)
    return npz


def build_base_groups(npz, R_fs, t_fs):
    """把每组移动扫描的点变换到基座系（按 frame_pose_indices 分组）。"""
    offsets = npz["selected_frame_offsets"]
    points = npz["selected_points_sensor_m"]
    fpi = npz["frame_pose_indices"]
    rots = npz["flange_rotations"]
    ts = npz["flange_translations_m"]
    groups = []
    for pose_index in np.unique(fpi):
        frames = np.where(fpi == pose_index)[0]
        pose_pts = []
        for f in frames:
            pts_s = points[offsets[f]:offsets[f + 1]]
            pts_s = pts_s[np.all(np.isfinite(pts_s), axis=1)]
            if len(pts_s) == 0:
                continue
            pose_pts.append(transform_profile_to_base(
                pts_s, rots[f], ts[f], R_fs, t_fs
            ))
        if pose_pts:
            groups.append(np.vstack(pose_pts))
    return groups


def fit_one_group(points, robust_scale_m=0.00010):
    center, radius = _free_sphere(points, robust_scale_m=robust_scale_m)
    residual = np.linalg.norm(points - center, axis=1) - radius
    return {
        "center_base_m": center.tolist(),
        "radius_mm": 1000.0 * float(radius),
        "delta_r_mm": 1000.0 * float(radius) - TRUE_RADIUS_MM,
        "form_error_peak_valley_mm": 1000.0 * float(np.max(residual) - np.min(residual)),
        "dispersion_rmse_mm": 1000.0 * float(np.sqrt(np.mean(residual ** 2))),
        "point_count": int(len(points)),
    }


def evaluate(npz_path: Path, R_fs, t_fs, artifact_mm=TRUE_RADIUS_MM,
             robust_scale_m=0.00010) -> dict:
    npz = load_npz_groups(npz_path)
    groups = build_base_groups(npz, R_fs, t_fs)

    per_group = [fit_one_group(g, robust_scale_m) for g in groups]

    # 合并拟合
    all_points = np.vstack(groups)
    combined = fit_one_group(all_points, robust_scale_m)
    combined["size_error_mm"] = 2.0 * combined["delta_r_mm"]  # 直径口径

    delta_rs = np.array([g["delta_r_mm"] for g in per_group])
    radii = np.array([g["radius_mm"] for g in per_group])
    summary = {
        "group_count": len(groups),
        "mean_radius_mm": float(np.mean(radii)),
        "std_radius_mm": float(np.std(radii)),
        "mean_delta_r_mm": float(np.mean(delta_rs)),
        "std_delta_r_mm": float(np.std(delta_rs)),
        "min_delta_r_mm": float(np.min(delta_rs)),
        "max_delta_r_mm": float(np.max(delta_rs)),
        "abs_mean_delta_r_mm": float(np.mean(np.abs(delta_rs))),
        "total_points": int(len(all_points)),
    }
    return {
        "true_radius_mm": artifact_mm,
        "combined": combined,
        "per_group": per_group,
        "per_group_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评价方法2（论文式球验证）")
    parser.add_argument("--npz", type=Path, required=True,
                        help="球采集 npz（每组=移动扫描）")
    parser.add_argument("--result", type=Path, required=True,
                        help="calibration_result.json（手眼来源）")
    parser.add_argument("--radius-mm", type=float, default=TRUE_RADIUS_MM)
    parser.add_argument("--out", type=Path,
                        default=Path("/workspace/tmp_diag/method2_result.json"))
    args = parser.parse_args()

    result = json.load(open(args.result))
    R_fs = np.array(result["handeye"]["rotation"])
    t_fs = np.array(result["handeye"]["translation"])
    evaluation = evaluate(args.npz, R_fs, t_fs, artifact_mm=args.radius_mm)

    print(f"=== 评价方法2（论文式）: {args.npz.name} ===")
    print(f"手眼: {args.result}")
    print(f"组数: {evaluation['per_group_summary']['group_count']}，"
          f"总点数: {evaluation['per_group_summary']['total_points']}")
    c = evaluation["combined"]
    print(f"\n合并拟合: 半径={c['radius_mm']:.4f}mm Δr={c['delta_r_mm']:+.4f}mm "
          f"size_error={c['size_error_mm']:+.4f}mm "
          f"form={c['form_error_peak_valley_mm']:.4f}mm "
          f"dispersion_RMSE={c['dispersion_rmse_mm']:.4f}mm")
    s = evaluation["per_group_summary"]
    print(f"逐组: 平均Δr={s['mean_delta_r_mm']:+.4f}mm std={s['std_delta_r_mm']:.4f}mm "
          f"范围[{s['min_delta_r_mm']:+.4f}, {s['max_delta_r_mm']:+.4f}]mm "
          f"|Δr|平均={s['abs_mean_delta_r_mm']:.4f}mm")
    print("\n逐组明细:")
    for g in evaluation["per_group"]:
        print(f"  pose {g['point_count']:6d}点 r={g['radius_mm']:8.4f}mm "
              f"Δr={g['delta_r_mm']:+8.4f}mm form={g['form_error_peak_valley_mm']:7.4f}mm")

    args.out.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False))
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
