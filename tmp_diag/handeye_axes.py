#!/usr/bin/env python3
"""分析真实 handeye 旋转矩阵: sensor 轴的真实约定 vs 候选生成假设。"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.geometry import make_transform

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])
print("真实 handeye rotation (flange -> sensor):")
print(np.round(R_he, 4))
print()
print("列 (sensor 轴在 flange 系):")
print(f"  x: {np.round(R_he[:,0], 3)}")
print(f"  y: {np.round(R_he[:,1], 3)}")
print(f"  z: {np.round(R_he[:,2], 3)}")

# 标定球手眼 (参考)
ball = json.load(open("/workspace/data/sphere_validation_runs/20260816_142845_sphere_20mm/calibration_result_ball.json"))
R_ball = np.array(ball["handeye"]["rotation"])
print("\n球标定 handeye rotation:")
print(np.round(R_ball, 4))
print(f"  列 z: {np.round(R_ball[:,2], 3)}")

# 6种子手眼在传感器 z 方向的含义: 传感器 z 应指向 laser (板方向)
# sensor z 在 flange 系 = R_he[:,2]
print(f"\nsensor z (flange系): {np.round(R_he[:,2], 3)}")
print(f"  norm: {np.linalg.norm(R_he[:,2]):.4f}")
