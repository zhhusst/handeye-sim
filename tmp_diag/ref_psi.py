#!/usr/bin/env python3
"""计算参考位姿对应的 psi (绕 laser 轴旋转), 以及参考的完整 sensor 参数。"""
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

# 候选生成的 sensor 框架:
# laser = sensor z (指向板)
laser = R_s[:, 2]
print(f"参考 laser (sensor z): {np.round(laser, 3)}")
print(f"参考 laser·n: {laser @ n:.3f}")

# 候选 _sensor_transform 的中间量:
# line = 板上扫描线方向 = point_v - point_u (板面内)
# 从参考 sensor x 的板面投影反推 line 方向
sx = R_s[:, 0]
line_dir = sx - n * (sx @ n)
line_dir /= np.linalg.norm(line_dir)
print(f"\n参考 sensor x 板面投影 (≈line 方向): {np.round(line_dir, 3)}")

tangent = np.cross(n, line_dir)
tangent /= np.linalg.norm(tangent)
print(f"tangent_normal: {np.round(tangent, 3)}")

# alpha: laser 与 -n 的夹角 (候选 laser = -(cos(a)*n + branch*sin(a)*tangent))
cos_a = (-laser) @ n
alpha_ref = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
print(f"\n参考 alpha: {alpha_ref:.1f}°")

# branch: laser 在 tangent 方向的分量符号
laser_t = laser @ tangent
print(f"laser·tangent: {laser_t:.3f} (正=branch+1, 负=branch-1)")

# psi: sensor x 在 (line, z_zero) 平面的旋转角
# 候选无 psi 时 sensor_x_zero = line 方向
# 有 psi 时 sensor_x = cos(psi)*line + sin(psi)*z_zero
# z_zero = cross(line, laser) 方向
z_zero = np.cross(line_dir, laser)
z_zero /= np.linalg.norm(z_zero)
print(f"\nz_zero (psi=0 时的 sensor z): {np.round(z_zero, 3)}")
# sensor_x = cos(psi)*line + sin(psi)*z_zero
# 反解 psi: cos(psi) = sx·line, sin(psi) = sx·z_zero
cos_psi = float(sx @ line_dir)
sin_psi = float(sx @ z_zero)
psi_ref = np.degrees(np.arctan2(sin_psi, cos_psi))
print(f"参考 psi: {psi_ref:.1f}°")

# working distance: 沿 laser 到板面
d_wd = -(n @ (t_s - corner)) / (n @ laser)
print(f"参考 working_distance: {d_wd:.3f} m")
