#!/usr/bin/env python3
"""终极检查: 用精确参考参数 (alpha=31.3, psi=9.6, dist=0.28, branch=+1) + 参考交点 a/b,
生成的候选 flange vs 参考 flange。"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.nbv.candidate_generator import _sensor_transform
from calibration_pipeline.geometry import make_transform, invert_transform
from fanuc_m20id25_support.fanuc_kinematic import inverse_kinematics_numeric

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])
u_hat = R_board[:, 0]; v_hat = R_board[:, 1]; n_hat = R_board[:, 2]

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
joints_ref = np.array(ref["joints"])
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor_ref = T_fb @ T_he

# 参考 laser 交点 (a=0.016, b=0.035)
hit = np.array([1.087, -0.306, 0.061])
a_ref, b_ref = 0.016, 0.035
point_u = corner + a_ref * u_hat
point_v = corner + b_ref * v_hat

# 用候选生成函数 (精确参考参数)
for branch in (1, -1):
    T_s = _sensor_transform(
        point_u, point_v, n_hat,
        np.deg2rad(31.3), np.deg2rad(9.6), 0.280, branch,
    )
    if T_s is None:
        print(f"branch={branch}: _sensor_transform None")
        continue
    print(f"\nbranch={branch:+d}:")
    print(f"  候选 sensor 原点: {np.round(T_s[:3,3], 3)}")
    print(f"  参考 sensor 原点: {np.round(sensor_ref[:3,3], 3)}")
    print(f"  sensor 原点差: {np.linalg.norm(T_s[:3,3]-sensor_ref[:3,3]):.3f}")
    R_rel = sensor_ref[:3,:3].T @ T_s[:3,:3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2,-1,1)))
    print(f"  sensor 旋转差: {ang:.1f}°")
    T_fl = T_s @ invert_transform(T_he)
    sols = inverse_kinematics_numeric(T_fl, q_init=joints_ref)
    if len(sols):
        q = sols[0]
        d = np.degrees(np.max(np.abs(q - joints_ref)))
        print(f"  IK 关节差: {d:.1f}°")
    else:
        print("  IK 无解")
