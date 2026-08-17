#!/usr/bin/env python3
"""验证候选 00369 预测端点误差来源:
对比 (a) 候选预测端点 (b) 种子参考实际端点, 在 sensor 系和板面系分别差多少。
并检查是不是手眼误差能解释。
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

# 1. 种子参考端点 (sensor系) -> 板面坐标
eu_s = np.array(ref["endpoint_u_S"]); ev_s = np.array(ref["endpoint_v_S"])
R_sr = sensor_ref[:3, :3]; t_sr = sensor_ref[:3, 3]
eu_b = R_sr @ eu_s + t_sr
ev_b = R_sr @ ev_s + t_sr
au = float((eu_b - corner) @ u_hat); av = float((eu_b - corner) @ v_hat)
bu = float((ev_b - corner) @ u_hat); bv = float((ev_b - corner) @ v_hat)
print("种子参考端点板面坐标: u=(%.3f, %.3f) v=(%.3f, %.3f)" % (au, av, bu, bv))
print("  (注意: u端点沿u轴 = (a, 0), v端点沿v轴 = (0, b))")

# 2. 候选 00369
cands = generate_candidates(estimate, edge_samples=4, edge_margin=0.04,
    alphas_deg=(25.0,31.0,38.0), psis_deg=(-15.0,0.0,15.0),
    working_distances=(0.24,0.28,0.33), profile_samples=40,
    reference_sensor_transform=sensor_ref)
c369 = [c for c in cands if c.candidate_id == "candidate_00369"][0]
print("候选 00369: a=%.3f b=%.3f" % (c369.a, c369.b))
# 候选预测端点板面坐标 (就是 a/b 位置)
print("候选预测端点板面坐标: u=(%.3f, 0) v=(0, %.3f)" % (c369.a, c369.b))

# 3. 关键: 候选传感器位姿的端点 -> sensor 系 (预测 guide)
T_s = c369.sensor_transform_nominal
R_s = T_s[:3, :3]; t_s = T_s[:3, 3]
p_u = corner + c369.a * u_hat
p_v = corner + c369.b * v_hat
u_pred = R_s.T @ (p_u - t_s)
v_pred = R_s.T @ (p_v - t_s)
print("\n候选预测端点 sensor 系:")
print("  u:", np.round(u_pred, 3))
print("  v:", np.round(v_pred, 3))

# 4. 对比检测节点的 guide (从诊断)
print("\n检测节点 guide (诊断):")
print("  first: [-36.25, 0, 263.1]")
print("  second: [36.25, 0, 296.9]")
print("  我的 u_pred:", np.round(u_pred, 3), "v_pred:", np.round(v_pred, 3))
print("  (guide 应该是预测端点, 对比看看是否一致)")

# 5. 真正的问题: 候选执行后, 实际角点在 sensor 系的什么位置?
# 如果传感器到达候选位姿, 板角点 (corner) 在 sensor 系:
corner_s = R_s.T @ (corner - t_s)
print("\n候选位姿下板角点 corner 在 sensor 系:", np.round(corner_s, 3))
# 板面端点 (a,b) 对应的物理角点?
# u 端点是板的一个物理角, v 端点是另一个物理角
# 实际角点: 候选端点 p_u, p_v 在 sensor 系
print("实际端点 p_u 在 sensor 系:", np.round(u_pred, 3))
print("实际端点 p_v 在 sensor 系:", np.round(v_pred, 3))
