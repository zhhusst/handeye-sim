#!/usr/bin/env python3
"""从任意位姿出发的候选极值分析:
模拟 NBV 5 后的位置 (用日志: NBV5 candidate_00251 的目标关节),
算从那里出发所有候选的笛卡尔平移/旋转极值。
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import generate_candidates
from calibration_pipeline.models import BoardModel, CalibrationEstimate
from calibration_pipeline.geometry import make_transform
from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf, inverse_kinematics_numeric

def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-10:
        return np.zeros(3)
    return 0.5 * a / np.sin(a) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def rot_dist_deg(Ta, Tb):
    R = Ta[:3, :3].T @ Tb[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

result = json.load(open("/workspace/data/calibration_runs/20260817_133558/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])

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

# 场景A: 从参考位姿出发
# 场景B: 从 NBV5 目标出发 (joint_distance=0.507rad 从参考, 但具体关节未知)
# 用参考位姿 + 一个偏移模拟: 取候选里 joint_distance~0.5 的目标
# 更实际: NBV5 的候选 00251 目标关节
c251 = [c for c in cands if c.candidate_id == "candidate_00251"]
if c251:
    sols = inverse_kinematics_numeric(c251[0].flange_transform_command, q_init=np.array(ref["joints"]))
    if len(sols):
        joints_nbv5 = sols[0]
        print("NBV5 目标关节:", np.round(np.degrees(joints_nbv5), 1))
        print("NBV5 joint_step: %.1f°" % np.degrees(np.max(np.abs(joints_nbv5 - np.array(ref["joints"])))))
        
        T_cur = forward_kinematics_urdf(joints_nbv5)
        max_trans = 0; max_rot = 0; max_step = 0; worst = None
        for c in cands:
            sols2 = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_nbv5)
            if len(sols2) == 0:
                continue
            q = sols2[0]
            step = np.degrees(np.max(np.abs(q - joints_nbv5)))
            if step > 70:
                continue
            T_t = forward_kinematics_urdf(q)
            tr = 1000 * np.linalg.norm(T_t[:3, 3] - T_cur[:3, 3])
            ro = rot_dist_deg(T_cur, T_t)
            if tr > max_trans:
                max_trans = tr; worst = (c.candidate_id, tr, ro, step)
            max_rot = max(max_rot, ro)
            max_step = max(max_step, step)
        print("\n从 NBV5 位姿出发, <70° 候选:")
        print("  max 笛卡尔平移: %.1f mm" % max_trans)
        print("  max 笛卡尔旋转: %.1f°" % max_rot)
        print("  max joint_step: %.1f°" % max_step)
        print("  最坏: %s" % (worst,))
