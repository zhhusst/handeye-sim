#!/usr/bin/env python3
"""决定性测试: 参考位姿的 sensor_transform -> flange_transform 转换一致性。

方法: 用参考位姿的 flange (T_fb) 和手眼 (T_he) 计算 sensor:
  T_sensor = T_fb @ T_he
然后反向: flange_recovered = T_sensor @ inv(T_he)
如果 flange_recovered == T_fb => 变换约定一致
如果差 159° => 候选生成用了不同的约定 (比如 inv(T_he) 反了)
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.geometry import invert_transform, make_transform

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)

T_sensor = T_fb @ T_he
print("T_sensor (参考):")
print(np.round(T_sensor, 4))
print()

# 方法1: 候选生成的方式 (flange = sensor @ inv(handeye))
T_flange_rec = T_sensor @ invert_transform(T_he)
print("反向 flange = sensor @ inv(handeye):")
print(np.round(T_flange_rec, 4))
print(f"与参考 flange 差: {np.linalg.norm(T_flange_rec[:3,3] - T_fb[:3,3]):.6f} m")
R_rel = T_fb[:3,:3].T @ T_flange_rec[:3,:3]
print(f"旋转差: {np.degrees(np.arccos(np.clip((np.trace(R_rel)-1)/2,-1,1))):.6f}°")
print()

# 方法2: 另一种约定 (flange = inv(handeye) @ sensor)
T_flange_alt = invert_transform(T_he) @ T_sensor
print("反向 flange = inv(handeye) @ sensor:")
print(np.round(T_flange_alt, 4))
print(f"与参考 flange 差: {np.linalg.norm(T_flange_alt[:3,3] - T_fb[:3,3]):.6f} m")
R_rel2 = T_fb[:3,:3].T @ T_flange_alt[:3,:3]
print(f"旋转差: {np.degrees(np.arccos(np.clip((np.trace(R_rel2)-1)/2,-1,1))):.6f}°")
