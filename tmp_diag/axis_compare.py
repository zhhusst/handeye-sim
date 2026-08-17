#!/usr/bin/env python3
"""逐轴对比: 参考 sensor 的 x/y/z vs 候选生成逻辑的各轴。

候选 _sensor_transform:
  laser_normal = -(cos(a)*n + branch*sin(a)*tangent)   [指向板]
  sensor_z = laser
  sensor_x_zero = line (投影垂直 z)
  sensor_y_zero = cross(z, x_zero)
  sensor_x = cos(psi)*x_zero + sin(psi)*y_zero
  sensor_y = -sin(psi)*x_zero + cos(psi)*y_zero
"""
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
u_hat = R_board[:, 0]; v_hat = R_board[:, 1]

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor = T_fb @ T_he
R_s = sensor[:3, :3]

print("参考 sensor 轴:")
print(f"  x: {np.round(R_s[:,0], 3)}")
print(f"  y: {np.round(R_s[:,1], 3)}")
print(f"  z (laser): {np.round(R_s[:,2], 3)}")
print(f"  laser·n = {R_s[:,2] @ n:.3f}")

# 候选逻辑构建 (用参考参数)
laser = R_s[:, 2]
line_dir = R_s[:, 0] - n * (R_s[:, 0] @ n)
line_dir /= np.linalg.norm(line_dir)
tangent = np.cross(n, line_dir); tangent /= np.linalg.norm(tangent)

print(f"\n候选构建的中间量:")
print(f"  line 方向 (sensor x 板面投影): {np.round(line_dir, 3)}")
print(f"  tangent: {np.round(tangent, 3)}")
print(f"  laser·tangent: {laser @ tangent:.3f}")

# 候选 sensor_z = laser 方向
print(f"  候选 sensor_z (=laser): {np.round(laser, 3)}")
print(f"  参考 sensor_z: {np.round(R_s[:,2], 3)}")
print(f"  一致: {np.allclose(laser, R_s[:,2], atol=1e-3)}")

# 参考 sensor x 在垂直于 z 的平面内的分量
x_perp = R_s[:, 0] - laser * (laser @ R_s[:, 0])
print(f"\n参考 sensor x 垂直分量: {np.round(x_perp, 3)} (norm={np.linalg.norm(x_perp):.3f})")

# 关键: 参考 sensor 的 x 轴到底怎么来的?
print(f"\n参考 sensor x·line_dir = {R_s[:,0] @ line_dir:.3f}")
print(f"参考 sensor x·tangent = {R_s[:,0] @ tangent:.3f}")
print(f"参考 sensor y·line_dir = {R_s[:,1] @ line_dir:.3f}")
print(f"参考 sensor y·tangent = {R_s[:,1] @ tangent:.3f}")
