#!/usr/bin/env python3
"""计算参考位姿 sensor 对应的板面 a/b 坐标 (候选生成应以它为中心)。"""
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
u_hat = R_board[:, 0]
v_hat = R_board[:, 1]
n_hat = R_board[:, 2]

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)
sensor = T_fb @ T_he
t_s = sensor[:3, 3]
R_s = sensor[:3, :3]

# sensor 的激光线在板面上的扫描位置
# laser 方向 (sensor z) 与板的交点:
laser = R_s[:, 2]
# 交点: t_s + d*laser 满足 n·(p-corner)=0
d = -(n_hat @ (t_s - corner)) / (n_hat @ laser)
hit = t_s + d * laser
print(f"参考 sensor 原点: {np.round(t_s, 3)}")
print(f"激光线与板交点: {np.round(hit, 3)}")
a_ref = float((hit - corner) @ u_hat)
b_ref = float((hit - corner) @ v_hat)
print(f"交点板面坐标: a={a_ref:.3f} (0-0.2), b={b_ref:.3f} (0-0.15)")
print(f"参考 sensor x 板面投影: {np.round(R_s[:,0] - n_hat*(R_s[:,0]@n_hat), 3)}")
print(f"sensor 到板垂直距离: {abs(n_hat @ (t_s - corner)):.3f}")
print(f"入射角: {np.degrees(np.arccos(np.clip(-laser@n_hat,-1,1))):.1f}°")
