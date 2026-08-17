#!/usr/bin/env python3
"""分析 candidate_00344: 为什么平移 681mm"""
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
u_hat = R_board[:, 0]
v_hat = R_board[:, 1]

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

# 找 candidate_00344
c344 = [c for c in cands if c.candidate_id == "candidate_00344"]
if not c344:
    print("未找到 candidate_00344, 找 joint_distance 大的:")
    big = sorted(cands, key=lambda c: c.working_distance)[-3:]
    for c in big:
        print(" ", c.candidate_id, "wd=%.2f" % c.working_distance)
else:
    c = c344[0]
    print("候选 00344: branch=%+d a=%.3f b=%.3f alpha=%.1f psi=%.0f dist=%.2f" % (
        c.branch, c.a, c.b, np.degrees(c.alpha), np.degrees(c.psi), c.working_distance))
    T_s = c.sensor_transform_nominal
    # 当前 sensor 位姿
    T_cur_sensor = T_fb @ T_he
    print("当前 sensor 位姿:", np.round(T_cur_sensor[:3, 3], 3))
    print("候选 sensor 位姿:", np.round(T_s[:3, 3], 3))
    print("候选-当前 sensor 平移: %.1f mm" % (1000*np.linalg.norm(T_s[:3,3] - T_cur_sensor[:3,3])))
    sols = inverse_kinematics_numeric(c.flange_transform_command, q_init=joints_cur)
    if len(sols):
        q = sols[0]
        T_cur = forward_kinematics_urdf(joints_cur)
        T_t = forward_kinematics_urdf(q)
        print("IK 解: joint_step=%.1f° dist=%.3f rad" % (
            np.degrees(np.max(np.abs(q - joints_cur))), np.linalg.norm(q - joints_cur)))
        print("笛卡尔平移: %.1f mm" % (1000*np.linalg.norm(T_t[:3,3]-T_cur[:3,3])))
        print("笛卡尔旋转: %.1f°" % rot_dist_deg(T_cur, T_t))
