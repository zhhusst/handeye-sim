#!/usr/bin/env python3
"""生成完整球验证报告 (markdown + json)"""
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

ARTIFACT = SphereArtifact(artifact_id="sphere_20mm", model="D20GZ", diameter_m=0.020002, roundness_m=0.0007)

def load_handeye(path):
    d = json.load(open(path))
    return np.array(d["handeye"]["rotation"]), np.array(d["handeye"]["translation"])

final = json.load(open("/workspace/data/calibration_runs/20260817_134739/calibration_result.json"))
iters = final["simulation"]["iterations"]

handeyes = {
    "A_6seed_init": (np.array(iters[0]["handeye_rotation"]), np.array(iters[0]["handeye_translation_m"]), "6种子初始化 (12-DOF-V2, NBV前)"),
    "B_6seed_plus_NBV": (np.array(final["handeye"]["rotation"]), np.array(final["handeye"]["translation"]), "6种子+NBV×6 最终 (12-DOF-V2)"),
    "C_ball": (*load_handeye(str(RUN / "calibration_result_ball.json")), "球标定 (基准)"),
}

selected_offsets = npz["selected_frame_offsets"]
selected_points = npz["selected_points_sensor_m"]
frame_pose_indices = npz["frame_pose_indices"]
flange_rots = npz["flange_rotations"]
flange_ts = npz["flange_translations_m"]

# 完整报告
full = {"schema_version": 1, "artifact": ARTIFACT.as_dict_mm(), "pose_count": len(np.unique(frame_pose_indices)),
        "point_count": int(len(selected_points)), "handeye_comparison": {}, "interpretation": ""}

for key, (R_fs, t_fs, desc) in handeyes.items():
    groups = []
    for fi in range(len(frame_pose_indices)):
        pts_s = selected_points[selected_offsets[fi]:selected_offsets[fi + 1]]
        if len(pts_s) == 0:
            continue
        groups.append(transform_profile_to_base(pts_s, flange_rots[fi], flange_ts[fi], R_fs, t_fs))
    report = validate_sphere_views(groups, ARTIFACT, thresholds=SphereValidationThresholds())
    fr = report["fixed_radius"]
    free = report.get("free_radius_diagnostic", {})
    full["handeye_comparison"][key] = {
        "description": desc,
        "handeye_rotation": R_fs.tolist(),
        "handeye_translation_m": t_fs.tolist(),
        "fixed_radius": {
            "rmse_mm": fr["all_points"]["rmse_mm"],
            "p95_abs_mm": fr["all_points"]["p95_abs_mm"],
            "median_abs_mm": fr["all_points"]["median_abs_mm"],
            "maximum_abs_mm": fr["all_points"]["maximum_abs_mm"],
            "signed_mean_mm": fr["all_points"]["signed_mean_mm"],
            "center_base_m": fr["center_base_m"],
            "inlier_count": fr.get("robust_inliers_diagnostic_only", {}).get("count"),
        },
        "free_radius_diagnostic": {
            "rmse_mm": free.get("all_points", {}).get("rmse_mm"),
            "radius_mm": 1000.0 * free.get("radius_m", 0.0) if free.get("radius_m") else None,
        },
        "leave_one_pose_out": report.get("leave_one_pose_out", {}),
        "passed": report.get("passed"),
    }

a = full["handeye_comparison"]["A_6seed_init"]["fixed_radius"]["rmse_mm"]
b = full["handeye_comparison"]["B_6seed_plus_NBV"]["fixed_radius"]["rmse_mm"]
c = full["handeye_comparison"]["C_ball"]["fixed_radius"]["rmse_mm"]
full["interpretation"] = (
    f"6种子+NBV 手眼球拟合 RMSE={b:.4f}mm, 相对6种子初始化 ({a:.4f}mm) 提升 {a/b:.2f}x; "
    f"球标定基准 RMSE={c:.4f}mm"
)

out = RUN / "sphere_report_6seed_plus_nbv.json"
json.dump(full, open(out, "w"), indent=2, ensure_ascii=False)
print("saved:", out)

# 打印汇总
print("\n=== 汇总 ===")
for key in ["A_6seed_init", "B_6seed_plus_NBV", "C_ball"]:
    s = full["handeye_comparison"][key]
    fr = s["fixed_radius"]
    free = s["free_radius_diagnostic"]
    lopo = s["leave_one_pose_out"]
    print(f"{key}: RMSE={fr['rmse_mm']:.4f}mm P95={fr['p95_abs_mm']:.4f}mm MAX={fr['maximum_abs_mm']:.4f}mm "
          f"free_rmse={free['rmse_mm']:.4f}mm" if free.get('rmse_mm') else f"{key}: RMSE={fr['rmse_mm']:.4f}mm")
    if lopo:
        print(f"   leave_one_pose: {json.dumps(lopo)[:200]}")
