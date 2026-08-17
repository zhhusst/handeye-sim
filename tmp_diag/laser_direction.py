#!/usr/bin/env python3
"""最终根因验证: 候选 sensor 的 laser 方向 vs 参考 sensor 的 laser 方向。

参考 sensor 的 laser 方向 = T_sensor 第2列 (sensor_z) 指向板。
候选 sensor 的 laser 方向 = sensor_transform 第2列。
对比: 如果参考 laser 指向 -Z (向下), 候选 laser 应也向下。
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform

def so3_log(R):
    cos_ang = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(cos_ang)
    if ang < 1e-10:
        return np.zeros(3)
    return 0.5 * ang / np.sin(ang) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])
n_board = R_board[:, 2]

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he
laser_ref = sensor_ref[:3, 2]   # sensor z = laser 方向
print(f"板法向 n: {np.round(n_board, 3)}")
print(f"参考 sensor z (laser): {np.round(laser_ref, 3)}")
print(f"  laser·n = {laser_ref @ n_board:.3f} (负=指向板)" )
print(f"  参考 laser 在板正侧还是负侧: {'负侧(指向板)' if laser_ref @ n_board < 0 else '正侧(背离板)'}")

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)
candidates = generate_candidates(
    estimate,
    edge_samples=4, edge_margin=0.04,
    alphas_deg=(31.0,), psis_deg=(0.0,),
    working_distances=(0.24,), profile_samples=40,
)
print("\n候选 sensor z (laser) 与板法向的点积:")
for c in candidates[:6]:
    laser_c = c.sensor_transform_nominal[:3, 2]
    dot = laser_c @ n_board
    print(f"  branch={c.branch:+d}: laser={np.round(laser_c,3)} laser·n={dot:.3f} "
          f"{'指向板' if dot < 0 else '背离板!!'}")
