#!/usr/bin/env python3
"""精确定位: 候选 flange 为什么离参考 flange 159°?

逐步检查:
1. 参考位姿: sensor_transform 在基座系
2. 最近候选: sensor_transform, flange_transform
3. 对比 sensor 差异 vs flange 差异 -> 判断 handeye 应用问题
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.geometry import invert_transform, make_transform
from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf

# 真实求解结果
result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])

# 参考位姿 (种子 reference)
seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = None
for rec in seeds["seeds"]:
    if rec["label"] == "reference":
        ref = rec
        break
R_f = np.array(ref["R_BF"])
t_f = np.array(ref["t_BF"])
joints_ref = np.array(ref["joints"])

# 参考 sensor_transform (基座系): T_sensor_base = T_flange_base @ T_sensor_flange
T_he = make_transform(R_he, t_he)
T_fb = make_transform(R_f, t_f)
sensor_transform_ref = T_fb @ T_he
print("参考位姿:")
print(f"  joints: {np.round(np.degrees(joints_ref), 1)} deg")
print(f"  sensor 原点: {np.round(sensor_transform_ref[:3,3], 3)}")
print(f"  sensor z: {np.round(sensor_transform_ref[:3,3], 3)}")

# 生成候选 (用真实板)
from calibration_pipeline.models import BoardModel, CalibrationEstimate
board = BoardModel(corner=corner, rotation=R_board, length_u=0.2, length_v=0.15)

# x9 = so3_log(handeye) ++ t_he ++ so3_log(board_rot) (pipeline 的 state 格式)
def so3_log(R):
    cos_ang = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(cos_ang)
    if ang < 1e-10:
        return np.zeros(3)
    return 0.5 * ang / np.sin(ang) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

x9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_board)])
estimate = CalibrationEstimate(
    handeye_rotation=R_he, handeye_translation=t_he, board=board, x9=x9,
)
candidates = generate_candidates(
    estimate,
    edge_samples=4, edge_margin=0.04,
    alphas_deg=(20.0, 35.0, 50.0),
    psis_deg=(-15.0, 0.0, 15.0),
    working_distances=(0.33, 0.4, 0.5),
    profile_samples=40,
)
print(f"\n候选总数: {len(candidates)}")

# 找离参考 sensor 最近的候选
best = None
best_dist = float("inf")
for c in candidates:
    d = np.linalg.norm(c.sensor_transform_nominal[:3,3] - sensor_transform_ref[:3,3])
    if d < best_dist:
        best_dist = d
        best = c

print(f"\n最近候选 (sensor 距离 {best_dist:.3f} m):")
print(f"  a={best.a:.3f} b={best.b:.3f} alpha={np.degrees(best.alpha):.1f}° psi={np.degrees(best.psi):.1f}° dist={best.working_distance:.2f} branch={best.branch}")
print(f"  sensor 原点: {np.round(best.sensor_transform_nominal[:3,3], 3)}")
print(f"  参考 sensor 原点: {np.round(sensor_transform_ref[:3,3], 3)}")

# 参考 flange vs 候选 flange
T_flange_cmd = best.flange_transform_command
T_flange_ref = T_fb
print(f"\n参考 flange 原点: {np.round(T_flange_ref[:3,3], 3)}")
print(f"候选 flange 原点: {np.round(T_flange_cmd[:3,3], 3)}")
print(f"flange 原点差: {np.linalg.norm(T_flange_ref[:3,3] - T_flange_cmd[:3,3]):.3f} m")

# 旋转差
R_rel = T_flange_ref[:3,:3].T @ T_flange_cmd[:3,:3]
cos_ang = np.clip((np.trace(R_rel)-1)/2, -1, 1)
print(f"flange 旋转差: {np.degrees(np.arccos(cos_ang)):.1f}°")

# 候选的 IK (从参考关节出发)
from fanuc_m20id25_support.fanuc_kinematic import inverse_kinematics_numeric
sols = inverse_kinematics_numeric(T_flange_cmd, q_init=joints_ref)
print(f"\nIK 解数: {len(sols)}")
if len(sols):
    q = sols[0]
    print(f"IK 解: {np.round(np.degrees(q),1)} deg")
    print(f"参考关节: {np.round(np.degrees(joints_ref),1)} deg")
    print(f"关节差 max: {np.degrees(np.max(np.abs(q - joints_ref))):.1f}°")
