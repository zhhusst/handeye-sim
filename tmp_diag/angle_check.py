#!/usr/bin/env python3
"""验证: 检测模板 (25°斜线) vs 候选端点 (水平线) 的方向差异"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform

def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-10:
        return np.zeros(3)
    return 0.5 * a / np.sin(a) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])
u_hat = R_board[:, 0]
v_hat = R_board[:, 1]
seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he

# 1. 种子参考端点连线方向
eu_s = np.array(ref["endpoint_u_S"]); ev_s = np.array(ref["endpoint_v_S"])
vec_ref = eu_s - ev_s
print("种子参考端点: u=[%.3f, %.3f, %.3f] v=[%.3f, %.3f, %.3f]" % (*eu_s, *ev_s))
print("种子端点连线向量: [%.3f, %.3f, %.3f]" % (*vec_ref,))
ang_ref = np.degrees(np.arctan2(vec_ref[2], vec_ref[0]))
print("种子连线 x-z 角度: %.1f°" % ang_ref)

# 2. 候选 00369 端点连线方向
cands = generate_candidates(estimate, edge_samples=4, edge_margin=0.04,
    alphas_deg=(25.0,31.0,38.0), psis_deg=(-15.0,0.0,15.0),
    working_distances=(0.24,0.28,0.33), profile_samples=40,
    reference_sensor_transform=sensor_ref)
c369 = [c for c in cands if c.candidate_id == "candidate_00369"][0]
T_s = c369.sensor_transform_nominal
R_s = T_s[:3, :3]; t_s = T_s[:3, 3]
p_u = corner + c369.a * u_hat
p_v = corner + c369.b * v_hat
u_pred = R_s.T @ (p_u - t_s)
v_pred = R_s.T @ (p_v - t_s)
vec_c = u_pred - v_pred
print("\n候选端点: u=[%.3f, %.3f, %.3f] v=[%.3f, %.3f, %.3f]" % (*u_pred, *v_pred))
print("候选连线向量: [%.3f, %.3f, %.3f]" % (*vec_c,))
ang_c = np.degrees(np.arctan2(vec_c[2], vec_c[0]))
print("候选连线 x-z 角度: %.1f°" % ang_c)

# 3. 检测模板
angle_t = 25.0
print("\n检测模板连线 x-z 角度: %.1f° (固定配置)" % angle_t)
print("\n结论:")
print("  种子连线角 %.1f° vs 模板 %.1f°: 差 %.1f°" % (ang_ref, angle_t, abs(ang_ref - angle_t)))
print("  候选连线角 %.1f° vs 模板 %.1f°: 差 %.1f°" % (ang_c, angle_t, abs(ang_c - angle_t)))
print("  alignment_maximum_angle_difference_deg = 15.0")
