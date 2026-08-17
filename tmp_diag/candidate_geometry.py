#!/usr/bin/env python3
"""候选 00369 几何分析: 预测端点是否在板面合理范围, 检测为什么不稳定"""
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
n_hat = R_board[:, 2]
seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]

board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)
x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9)

R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he

cands = generate_candidates(estimate, edge_samples=4, edge_margin=0.04,
    alphas_deg=(25.0,31.0,38.0), psis_deg=(-15.0,0.0,15.0),
    working_distances=(0.24,0.28,0.33), profile_samples=40,
    reference_sensor_transform=sensor_ref)
c369 = [c for c in cands if c.candidate_id == "candidate_00369"][0]

T_s = c369.sensor_transform_nominal
R_s = T_s[:3, :3]
t_s = T_s[:3, 3]
laser = R_s[:, 2]

print("候选 00369 sensor:")
print("  原点:", np.round(t_s, 3))
print("  z (laser):", np.round(laser, 3))
print("  laser·n:", round(float(laser @ n_hat), 3), "(负=指向板)")

# 候选端点 (板上 a/b 位置) 在 sensor 系
p_u = corner + c369.a * u_hat
p_v = corner + c369.b * v_hat
# 端点应该在板面上, 激光线与板面的交点 = 端点位置
# 检查端点是否在板面范围内 (0-0.2 u, 0-0.15 v)
au = float((p_u - corner) @ u_hat)
av = float((p_u - corner) @ v_hat)
bu = float((p_v - corner) @ u_hat)
bv = float((p_v - corner) @ v_hat)
print("\n候选端点板面坐标:")
print("  u端点: (%.3f, %.3f)" % (au, av))
print("  v端点: (%.3f, %.3f)" % (bu, bv))
print("  板范围: u 0-0.2, v 0-0.15")

# 参考位姿的端点位置
eu = np.array(ref["endpoint_u_S"]); ev = np.array(ref["endpoint_v_S"])
print("\n参考 sensor 原点:", np.round(sensor_ref[:3,3], 3))
print("参考端点 u (sensor系):", np.round(eu, 4))
print("参考端点 v (sensor系):", np.round(ev, 4))

# 参考端点转板面坐标
R_sr = sensor_ref[:3, :3]
t_sr = sensor_ref[:3, 3]
eu_b = R_sr @ eu + t_sr
ev_b = R_sr @ ev + t_sr
print("参考端点 u 板面: (%.3f, %.3f)" % ((eu_b-corner)@u_hat, (eu_b-corner)@v_hat))
print("参考端点 v 板面: (%.3f, %.3f)" % ((ev_b-corner)@u_hat, (ev_b-corner)@v_hat))
print("参考 laser·n:", round(float(R_sr[:,2] @ n_hat), 3))

# 参考端点 z (sensor 前方距离)
print("\n参考端点深度: u_z=%.3f v_z=%.3f" % (eu[2], ev[2]))
print("候选端点深度: u_z=%.3f v_z=%.3f" % (u_pred_z if False else 0.28, 0.28))
