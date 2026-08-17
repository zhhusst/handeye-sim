#!/usr/bin/env python3
"""决定性: 候选 sensor 姿态 vs 参考 sensor 姿态的旋转差。

如果候选 sensor 旋转差小 (<40°) 但 flange 差 177° => 变换问题
如果候选 sensor 旋转差本身就大 (~180°) => 候选生成姿态问题
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

def rot_dist(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb)-1)/2, -1, 1)))

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he
R_sensor_ref = sensor_ref[:3,:3]

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

# 贴参考候选
candidates = generate_candidates(
    estimate,
    edge_samples=4, edge_margin=0.04,
    alphas_deg=(31.0,), psis_deg=(0.0,),
    working_distances=(0.24,), profile_samples=40,
)
print(f"贴参考候选: {len(candidates)}")
for c in candidates[:4]:
    R_s = c.sensor_transform_nominal[:3,:3]
    t_s = c.sensor_transform_nominal[:3,3]
    print(f"  branch={c.branch:+d} sensor_dist={np.linalg.norm(t_s - sensor_ref[:3,3]):.3f} "
          f"sensor_rot_diff={rot_dist(R_s, R_sensor_ref):.1f}°")
