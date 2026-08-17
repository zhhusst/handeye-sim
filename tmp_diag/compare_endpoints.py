#!/usr/bin/env python3
"""对比: 种子参考位姿实际端点 vs NBV 候选 00369 预测端点"""
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
seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]

# 种子的实际端点 (sensor系, 中位数)
eu = np.array(ref["endpoint_u_S"])
ev = np.array(ref["endpoint_v_S"])
print("=== 种子参考位姿实际端点 (sensor系) ===")
print("  u:", np.round(eu, 4))
print("  v:", np.round(ev, 4))
print("  帧数:", len(ref["frames"]), "batch:", ref["batch_diagnostics"])

# 逐帧端点分布
print("\n=== 种子逐帧端点 (检查一致性) ===")
for i, fr in enumerate(ref["frames"]):
    fu = fr.get("endpoint_u_S")
    fv = fr.get("endpoint_v_S")
    if fu is not None:
        du = np.linalg.norm(np.array(fu) - eu)
        dv = np.linalg.norm(np.array(fv) - ev)
        print(f"  frame {i}: u_dev={du*1000:.2f}mm v_dev={dv*1000:.2f}mm")

# 候选 00369 预测端点 (用简单方法: 直接用 sensor 变换和板交点)
print("\n=== 候选 00369 ===")
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
print("  参数: branch=%+d a=%.3f b=%.3f alpha=%.1f psi=%.0f dist=%.2f" % (
    c369.branch, c369.a, c369.b, np.degrees(c369.alpha), np.degrees(c369.psi), c369.working_distance))

# 候选的端点就是 a/b 决定的板面点 (在 sensor 系)
T_s = c369.sensor_transform_nominal
R_s = T_s[:3, :3]
# 板面点 u/v (候选 a,b 位置, 沿 u/v 方向)
p_u = corner + c369.a * R_board[:, 0]
p_v = corner + c369.b * R_board[:, 1]
u_pred = R_s.T @ (p_u - T_s[:3, 3])
v_pred = R_s.T @ (p_v - T_s[:3, 3])
print("  预测端点 u (sensor系):", np.round(u_pred, 4))
print("  预测端点 v (sensor系):", np.round(v_pred, 4))
print("\n  vs 种子实际端点差:")
print("  u 差:", np.round(u_pred - eu, 4), "norm=%.1fmm" % (1000*np.linalg.norm(u_pred-eu)))
print("  v 差:", np.round(v_pred - ev, 4), "norm=%.1fmm" % (1000*np.linalg.norm(v_pred-ev)))
