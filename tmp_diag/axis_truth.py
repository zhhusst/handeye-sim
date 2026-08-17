#!/usr/bin/env python3
"""验证核心假设: 候选 sensor 的 x/y 应该由 handeye 固定, 而不是 line。

方法: 参考位姿 sensor = T_flange @ T_he。候选生成的 sensor 应该保持
与 handeye 相同的 x/y/z 轴框架 (绕 laser 轴小幅旋转), 只移动位置。
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
R_board = np.array(result["board"]["rotation"])
n = R_board[:, 2]

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor = T_fb @ T_he
R_s = sensor[:3, :3]

# sensor 轴在基座系
sx_b = R_s[:, 0]
sy_b = R_s[:, 1]
sz_b = R_s[:, 2]  # laser
print("参考 sensor 轴 (基座系):")
print(f"  x: {np.round(sx_b, 3)}")
print(f"  y: {np.round(sy_b, 3)}")
print(f"  z (laser): {np.round(sz_b, 3)}")

# 候选生成的框架: line 是板面方向
# 但真实 sensor x 在基座系有 z 分量 0.411 (不沿板面)
# 检查: sensor x 与板面的夹角
line_b = sx_b - n * (sx_b @ n)
print(f"\n参考 sensor x 板面分量: {np.round(line_b, 3)} norm={np.linalg.norm(line_b):.3f}")
print(f"参考 sensor x 板面外分量 (n方向): {sx_b @ n:.3f} (非0=传感器x不在板面内)")

# 候选生成的 laser 是 sensor z, 一致
# 但候选 sensor x = cos(psi)*line + sin(psi)*cross(line, laser)
# 真实 sensor x 是固定安装方向 —— 与 line 无关
# => 候选生成应该用 handeye 固定的 x/y 框架, 而不是 line!
print(f"\n结论: 真实 sensor x 有板面外分量 {sx_b @ n:.2f}, 候选假设 x 沿板面(line).")
print(f"候选 sensor x 与真实差 ~{np.degrees(np.arccos(np.clip(np.dot(line_b, sx_b),-1,1))):.1f}°")
