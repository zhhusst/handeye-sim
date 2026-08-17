#!/usr/bin/env python3
"""验证: 新参数 (alpha 25-38, dist 0.24-0.33) 下是否存在关节差<70°的候选。"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform
from fanuc_m20id25_support.fanuc_kinematic import inverse_kinematics_numeric

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

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
joints_ref = np.array(ref["joints"])
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

# 新参数
candidates = generate_candidates(
    estimate,
    edge_samples=4, edge_margin=0.01,
    alphas_deg=(25.0, 31.0, 37.0),
    psis_deg=(-15.0, 0.0, 15.0),
    working_distances=(0.24, 0.28, 0.32),
    profile_samples=40,
)
print(f"候选: {len(candidates)}")
deltas = []
best = None
for c in candidates:
    sols = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_ref)
    if len(sols) == 0:
        continue
    q = sols[0]
    delta = np.degrees(np.max(np.abs(q - joints_ref)))
    d_sensor = np.linalg.norm(c.sensor_transform_nominal[:3,3] - sensor_ref[:3,3])
    deltas.append(delta)
    if best is None or delta < best[0]:
        best = (delta, d_sensor, c.branch, c.a, c.b, np.degrees(c.alpha), np.degrees(c.psi), c.working_distance)
deltas = np.array(deltas)
print(f"IK有解: {len(deltas)}")
print(f"关节差 min={deltas.min():.1f}° med={np.median(deltas):.1f}° <70°: {(deltas<70).sum()}")
print(f"最佳: joint_delta={best[0]:.1f}° sensor_dist={best[1]:.3f} branch={best[2]:+d} "
      f"a={best[3]:.2f} b={best[4]:.2f} alpha={best[5]:.0f}° psi={best[6]:.0f}° dist={best[7]:.2f}")
