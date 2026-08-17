#!/usr/bin/env python3
"""反解: 需要多大 psi 让候选 sensor x 与参考一致?

候选: sensor_x = cos(psi)*line + sin(psi)*z_zero, 其中 z_zero = cross(line, laser)
参考: sensor_x 已知。
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

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor = T_fb @ T_he
R_s = sensor[:3, :3]
laser = R_s[:, 2]

# line = sensor x 板面投影
line = R_s[:, 0] - n * (R_s[:, 0] @ n)
line /= np.linalg.norm(line)
z_zero = np.cross(line, laser)
z_zero /= np.linalg.norm(z_zero)

# 参考 sensor x 在 (line, z_zero) 基底的坐标
sx = R_s[:, 0]
cos_psi = float(sx @ line)
sin_psi = float(sx @ z_zero)
psi_est = np.degrees(np.arctan2(sin_psi, cos_psi))
print(f"line: {np.round(line,3)}")
print(f"z_zero: {np.round(z_zero,3)}")
print(f"参考 sensor x·line = {cos_psi:.3f}")
print(f"参考 sensor x·z_zero = {sin_psi:.3f}")
print(f"需要 psi = {psi_est:.1f}° (候选才能匹配参考 sensor x)")

# 用这个 psi 重建 sensor x/y
sx_rec = cos_psi * line + sin_psi * z_zero
sy_rec = np.cross(laser, sx_rec)  # 或 -sin*line + cos*z_zero
print(f"\n重建 sensor x: {np.round(sx_rec, 3)} (参考 {np.round(sx,3)})")
print(f"重建 x·参考x = {sx_rec @ sx:.4f}")
