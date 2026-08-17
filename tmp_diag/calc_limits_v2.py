#!/usr/bin/env python3
"""从 NBV 5 后的位姿出发, 算所有候选的笛卡尔平移极值 (决定运动桥限制)"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform
from fanuc_m20id25_support.fanuc_kinematic import (
    forward_kinematics_urdf,
    inverse_kinematics_numeric,
)

def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-10:
        return np.zeros(3)
    return 0.5 * a / np.sin(a) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def rot_dist_deg(Ta, Tb):
    R = Ta[:3, :3].T @ Tb[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

# 用 NBV 5 提交后的手眼 (最新, 从 active_calibration 输出)
# 最后一次 NBV 更新: 手眼平移 [-24.548, -15.968, 374.006]
# 读最新 calibration_result
try:
    result = json.load(open("/workspace/data/calibration_runs/20260817_133558/calibration_result.json"))
    R_he = np.array(result["handeye"]["rotation"])
    t_he = np.array(result["handeye"]["translation"])
    print("用 20260817_133558 手眼:", t_he)
except Exception:
    result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
    R_he = np.array(result["handeye"]["rotation"])
    t_he = np.array(result["handeye"]["translation"])
    print("用 live2 手眼:", t_he)

corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])
u_hat = R_board[:, 0]

# 当前位姿: NBV 5 提交后, 假设机器人回到参考? 还是停在 NBV 5?
# 用种子参考位姿 (回退后的位姿, 日志显示 NBV5 提交后没有回退, 位置=NBV5目标)
# 保守: 用参考位姿作为当前 (机器人可能回参考)
seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
joints_cur = np.array(ref["joints"])

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

T_cur = forward_kinematics_urdf(joints_cur)
max_trans = 0; max_rot = 0; max_step = 0; max_dist = 0
worst = None
for c in cands:
    sols = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_cur)
    if len(sols) == 0:
        continue
    q = sols[0]
    step = np.degrees(np.max(np.abs(q - joints_cur)))
    dist = np.linalg.norm(q - joints_cur)
    if step > 70:
        continue
    T_t = forward_kinematics_urdf(q)
    tr = 1000 * np.linalg.norm(T_t[:3, 3] - T_cur[:3, 3])
    ro = rot_dist_deg(T_cur, T_t)
    if tr > max_trans:
        max_trans = tr; worst = (c.candidate_id, tr, ro, step)
    max_rot = max(max_rot, ro)
    max_step = max(max_step, step)
    max_dist = max(max_dist, dist)
print(f"<70° 候选 ({len(cands)} 总数, 通过 IK 的) 从参考位姿出发:")
print(f"  max joint_step: {max_step:.1f}°")
print(f"  max joint_dist: {max_dist:.3f} rad ({np.degrees(max_dist):.1f}°)")
print(f"  max 笛卡尔平移: {max_trans:.1f} mm")
print(f"  max 笛卡尔旋转: {max_rot:.1f}°")
print(f"  最坏候选: {worst}")
