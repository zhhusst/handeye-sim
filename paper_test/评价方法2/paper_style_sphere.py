#!/usr/bin/env python3
"""论文风格球验证：对齐 Gautam et al. RCIM 2025 §8 方法（修正版）。

论文方法（§8）：
  - 球固定在板中心，7 个板方向扫描（我们有 10 个位姿）
  - 手眼变换把 2D 轮廓注册为 3D 点云（Eq.1）
  - 最小二乘球回归：
      * size error: 拟合半径 vs 已知半径（论文单传感器合并 0.0239mm）
      * form error: 点到拟合球面的分散度（论文 0.0844mm）
      * Fig.15 逐组统计: 每组拟合半径/球心 → 平均偏差、std、范围
        （论文单传感器平均半径偏差 0.1929mm，范围 0.1040-0.2396mm）

对齐修正（重要）：
  - 论文每组 = 一次完整球扫描（球沿法向移动），单组覆盖球面大片区域，
    可自由拟合半径；我们每个位姿 12 帧只覆盖球的一条窄带，单组自由
    拟合数学上退化（半径发散到 10^4 mm 级），因此：
      * 合并拟合 = 对齐论文"组合数据"（Table 2 / Fig.16 bottom）
      * 逐组统计 = 用已知半径固定球心拟合（论文 Fig.15 的球心稳定性语义）
  我们的球: D20GZ, 半径 10.001mm (直径 20.002mm)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.sphere_validation import (
    SphereArtifact,
    _free_sphere,
    transform_profile_to_base,
)

SPHERE_RUN = Path("/workspace/data/sphere_validation_runs/20260816_142845_sphere_20mm")
OUT = Path("/workspace/tmp_diag/paper_style_sphere_validation.json")

ARTIFACT = SphereArtifact(
    artifact_id="sphere_20mm", model="D20GZ",
    diameter_m=0.020002, roundness_m=0.0007,
)
TRUE_RADIUS_MM = 1000.0 * ARTIFACT.radius_m  # 10.001 mm


def load_handeye(path: str):
    d = json.load(open(path))
    return np.array(d["handeye"]["rotation"]), np.array(d["handeye"]["translation"])


def build_groups_base(R_fs, t_fs):
    npz = np.load(str(SPHERE_RUN / "sphere_acquisition.npz"), allow_pickle=True)
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
            pose_pts.append(transform_profile_to_base(
                pts_s, rots[f], ts[f], R_fs, t_fs
            ))
        groups.append(np.vstack(pose_pts))
    return groups


def _fixed_radius_center(points, radius_m, initial_center=None):
    if initial_center is None:
        center0 = np.mean(points, axis=0)
    else:
        center0 = initial_center

    def residual(center):
        return np.linalg.norm(points - center, axis=1) - radius_m

    result = least_squares(residual, center0, max_nfev=200)
    return result.x


def fit_report(groups, label):
    """合并自由拟合（对齐论文组合数据）+ 逐组固定半径拟合（对齐Fig.15球心）。"""
    all_points = np.vstack(groups)

    # --- 合并自由拟合: size error / form error ---
    center, radius = _free_sphere(all_points, robust_scale_m=0.00010)
    residual = np.linalg.norm(all_points - center, axis=1) - radius
    radius_mm = 1000.0 * radius
    delta_r_mm = radius_mm - TRUE_RADIUS_MM
    size_error_mm = 2.0 * delta_r_mm  # 直径口径
    form_error_mm = float(np.max(residual) - np.min(residual)) * 1000.0
    dispersion_rmse_mm = float(np.sqrt(np.mean(residual ** 2))) * 1000.0

    # --- 逐组固定半径拟合: 球心一致性 ---
    per_group = []
    for gi, group in enumerate(groups):
        g_center = _fixed_radius_center(group, ARTIFACT.radius_m,
                                        initial_center=center)
        g_residual = np.linalg.norm(group - g_center, axis=1) - ARTIFACT.radius_m
        per_group.append({
            "pose_index": gi + 1,
            "point_count": int(len(group)),
            "center_base_m": g_center.tolist(),
            "center_offset_from_combined_mm": 1000.0 * float(
                np.linalg.norm(g_center - center)),
            "group_rmse_mm": 1000.0 * float(np.sqrt(np.mean(g_residual ** 2))),
        })
    centers = np.array([g["center_base_m"] for g in per_group])
    center_spread_rms = float(
        np.sqrt(np.mean(np.sum((centers - np.mean(centers, axis=0)) ** 2, axis=1)))
    ) * 1000.0

    report = {
        "label": label,
        "true_radius_mm": TRUE_RADIUS_MM,
        "combined": {
            "point_count": int(len(all_points)),
            "fitted_radius_mm": radius_mm,
            "delta_r_mm": delta_r_mm,
            "size_error_mm": size_error_mm,
            "form_error_peak_valley_mm": form_error_mm,
            "dispersion_rmse_mm": dispersion_rmse_mm,
            "center_base_m": center.tolist(),
        },
        "per_group": per_group,
        "per_group_summary": {
            "center_spread_rms_mm": center_spread_rms,
            "max_center_offset_mm": float(
                np.max([g["center_offset_from_combined_mm"] for g in per_group])),
            "mean_group_rmse_mm": float(
                np.mean([g["group_rmse_mm"] for g in per_group])),
        },
    }
    return report


def main() -> None:
    final = json.load(open("/workspace/data/calibration_runs/20260817_134739/calibration_result.json"))
    iters = final["simulation"]["iterations"]

    candidates = {
        "A_6seed_init_shared": (
            np.array(iters[0]["handeye_rotation"]),
            np.array(iters[0]["handeye_translation_m"]),
            "6种子初始化 (shared, NBV前)",
        ),
        "B_6seed_plus_NBV_shared": (
            np.array(final["handeye"]["rotation"]),
            np.array(final["handeye"]["translation"]),
            "6种子+NBV×6 (shared, 生产最终)",
        ),
        "C_ball_baseline": (
            *load_handeye(str(SPHERE_RUN / "calibration_result_ball.json")),
            "球标定 (基准)",
        ),
    }

    results = {"schema_version": 1, "artifact": ARTIFACT.as_dict_mm(),
               "true_radius_mm": TRUE_RADIUS_MM, "candidates": {}}
    for key, (R_fs, t_fs, desc) in candidates.items():
        groups = build_groups_base(R_fs, t_fs)
        report = fit_report(groups, desc)
        results["candidates"][key] = report
        c = report["combined"]
        s = report["per_group_summary"]
        print(f"\n=== {key}: {desc} ===")
        print(f"  合并: 半径={c['fitted_radius_mm']:.4f}mm "
              f"Δr={c['delta_r_mm']:+.4f}mm "
              f"size_error={c['size_error_mm']:+.4f}mm "
              f"form_error={c['form_error_peak_valley_mm']:.4f}mm "
              f"dispersion_RMSE={c['dispersion_rmse_mm']:.4f}mm")
        print(f"  逐组(固定半径): 球心扩散RMS={s['center_spread_rms_mm']:.4f}mm "
              f"最大球心偏移={s['max_center_offset_mm']:.4f}mm "
              f"平均组RMSE={s['mean_group_rmse_mm']:.4f}mm")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
