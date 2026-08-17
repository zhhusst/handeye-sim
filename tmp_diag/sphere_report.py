#!/usr/bin/env python3
"""离线球验证: 用 20260816_142845 采集的标定球数据
+ 6种子+NBV 共享形貌12-DOF-V2 最终手眼 (20260817_134739)
计算球拟合精度, 生成完整测试报告。

对比:
  A. 6种子初始化手眼 (NBV 前)
  B. NBV优化后最终手眼
  C. 球标定手眼 (基准)
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.sphere_validation import (
    SphereArtifact,
    validate_sphere_views,
    SphereValidationThresholds,
    transform_profile_to_base,
)

RUN = Path("/workspace/data/sphere_validation_runs/20260816_142845_sphere_20mm")
npz = np.load(str(RUN / "sphere_acquisition.npz"), allow_pickle=True)

ARTIFACT = SphereArtifact(
    artifact_id="sphere_20mm",
    model="D20GZ",
    diameter_m=0.020002,
    roundness_m=0.0007,
)

def load_handeye(path):
    d = json.load(open(path))
    return np.array(d["handeye"]["rotation"]), np.array(d["handeye"]["translation"])

final = json.load(open("/workspace/data/calibration_runs/20260817_134739/calibration_result.json"))
iters = final["simulation"]["iterations"]

handeyes = {
    "A_6seed_init": (
        np.array(iters[0]["handeye_rotation"]),
        np.array(iters[0]["handeye_translation_m"]),
        "6种子初始化 (12-DOF-V2, NBV前)",
    ),
    "B_6seed_plus_NBV": (
        np.array(final["handeye"]["rotation"]),
        np.array(final["handeye"]["translation"]),
        "6种子+NBV×6 最终 (12-DOF-V2)",
    ),
    "C_ball": (
        *load_handeye(str(RUN / "calibration_result_ball.json")),
        "球标定 (基准)",
    ),
}

selected_offsets = npz["selected_frame_offsets"]
selected_points = npz["selected_points_sensor_m"]
frame_pose_indices = npz["frame_pose_indices"]
flange_rots = npz["flange_rotations"]
flange_ts = npz["flange_translations_m"]

print(f"球数据: {len(frame_pose_indices)} 帧, 选中点 {len(selected_points)}")
print(f"球: D=20.002mm R=10.001mm\n")

summary = {}
for key, (R_fs, t_fs, desc) in handeyes.items():
    groups = []
    for fi in range(len(frame_pose_indices)):
        R_bf = flange_rots[fi]
        t_bf = flange_ts[fi]
        pts_s = selected_points[selected_offsets[fi]:selected_offsets[fi + 1]]
        if len(pts_s) == 0:
            continue
        groups.append(transform_profile_to_base(pts_s, R_bf, t_bf, R_fs, t_fs))
    report = validate_sphere_views(groups, ARTIFACT, thresholds=SphereValidationThresholds())
    fr = report["fixed_radius"]
    free = report.get("free_radius_diagnostic", {})
    summary[key] = {
        "desc": desc,
        "points": int(report["point_count"]),
        "fixed_rmse_mm": fr["all_points"]["rmse_mm"],
        "fixed_p95_mm": fr["all_points"]["p95_abs_mm"],
        "fixed_max_mm": fr["all_points"]["maximum_abs_mm"],
        "fixed_center_m": fr["center_base_m"],
        "free_rmse_mm": free.get("all_points", {}).get("rmse_mm"),
        "leave_one_pose": report.get("leave_one_pose_out", {}),
        "passed": report.get("passed"),
        "checks": report.get("checks"),
    }
    print(f"[{key}] {desc}")
    print(f"  点数 {summary[key]['points']}, 固定半径 R=10.001mm:")
    print(f"    RMSE = {fr['all_points']['rmse_mm']:.4f} mm")
    print(f"    P95  = {fr['all_points']['p95_abs_mm']:.4f} mm")
    print(f"    MAX  = {fr['all_points']['maximum_abs_mm']:.4f} mm")
    print(f"    center = {np.round(fr['center_base_m'], 4)}")
    if "all_points" in free:
        print(f"  自由半径: RMSE = {free['all_points']['rmse_mm']:.4f} mm")
    print()

print("=" * 60)
print("手眼对比 (固定半径球拟合 RMSE):")
for key in ["A_6seed_init", "B_6seed_plus_NBV", "C_ball"]:
    s = summary[key]
    print(f"  {key:20s} RMSE={s['fixed_rmse_mm']:.4f}mm  P95={s['fixed_p95_mm']:.4f}mm")
print()
if summary["B_6seed_plus_NBV"]["fixed_rmse_mm"] > 0:
    imp = summary["A_6seed_init"]["fixed_rmse_mm"] / summary["B_6seed_plus_NBV"]["fixed_rmse_mm"]
    print(f"NBV 提升: {summary['A_6seed_init']['fixed_rmse_mm']:.4f} -> {summary['B_6seed_plus_NBV']['fixed_rmse_mm']:.4f} mm ({imp:.2f}x)")
