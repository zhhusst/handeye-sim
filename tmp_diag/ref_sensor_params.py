#!/usr/bin/env python3
"""提取参考位姿的 sensor 姿态参数 (alpha/psi/dist), 让候选围绕它生成。"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.geometry import make_transform

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
corner = np.array(result["board"]["corner"])
R_board = np.array(result["board"]["rotation"])
n = R_board[:, 2]
u_hat = R_board[:, 0]
v_hat = R_board[:, 1]

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor = T_fb @ T_he
R_s = sensor[:3, :3]
t_s = sensor[:3, 3]

# sensor 参数
laser = R_s[:, 2]          # sensor z = 测量方向
print(f"参考 sensor 原点: {np.round(t_s, 3)}")
print(f"参考 sensor z (laser): {np.round(laser, 3)}")
print(f"laser·n = {laser @ n:.3f} (负=指向板)")

# alpha = 入射角 = 板法向与 laser 的夹角 (laser 指向板, 所以用 -laser)
cos_alpha = (-laser) @ n
alpha_ref = np.degrees(np.arccos(np.clip(cos_alpha, -1, 1)))
print(f"参考入射角 alpha: {alpha_ref:.1f}°")

# 板面上 laser 投影方向 vs 板 u 方向 -> 决定扫描线角度
laser_proj = laser - n * (laser @ n)
laser_proj /= np.linalg.norm(laser_proj)
print(f"laser 板面投影: {np.round(laser_proj, 3)}")

# working distance = 传感器到板面的距离 (沿 laser 方向到板面)
# 板面: n·(p - corner) = 0
dist = n @ (t_s - corner) / np.abs(n @ laser)  # 沿 laser 到板面距离
# sensor 到板面的垂直距离:
perp_dist = abs(n @ (t_s - corner))
print(f"参考 sensor 到板面垂直距离: {perp_dist:.3f} m")
print(f"参考 working_distance(沿laser): {dist:.3f} m")

# sensor x 在板面投影方向 vs u -> psi 参数
sx = R_s[:, 0]
print(f"\n参考 sensor x: {np.round(sx, 3)}")
print(f"参考 sensor x·u = {sx @ u_hat:.3f}")
print(f"参考 sensor x·v = {sx @ v_hat:.3f}")

# 候选的扫描线在板面: line 方向由 (a,b) 决定, 是 u/v 的组合
# sensor_x_zero 应该沿 line 方向; 参考 sensor x 的板面投影:
sx_proj = sx - n * (sx @ n)
sx_proj /= np.linalg.norm(sx_proj)
ang_u = np.degrees(np.arctan2(sx_proj @ v_hat, sx_proj @ u_hat))
print(f"参考 sensor x 板面投影方向 (相对u的角度): {ang_u:.1f}°")
