#!/usr/bin/env python3
"""终极验证: 用参考位姿的精确 sensor 姿态作为候选 -> IK 关节差应≈0。

如果修复正确, 把参考 sensor_transform 直接当候选, flange = sensor@inv(he),
IK 应该回到参考关节。
"""
import json
import sys
import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/ros2_ws/src/fanuc_m20id25_support")
from calibration_pipeline.geometry import make_transform, invert_transform
from fanuc_m20id25_support.fanuc_kinematic import inverse_kinematics_numeric

result = json.load(open("/workspace/data/calibration_runs/20260817_live2/calibration_result.json"))
R_he = np.array(result["handeye"]["rotation"])
t_he = np.array(result["handeye"]["translation"])

seeds = json.load(open("/workspace/data/calibration_runs/20260817_062447/seeds.json"))
ref = [r for r in seeds["seeds"] if r["label"] == "reference"][0]
joints_ref = np.array(ref["joints"])
R_f = np.array(ref["R_BF"]); t_f = np.array(ref["t_BF"])
T_fb = make_transform(R_f, t_f)
T_he = make_transform(R_he, t_he)

# 参考 sensor -> flange (候选的方式)
sensor = T_fb @ T_he
flange_rec = sensor @ invert_transform(T_he)
print("参考 flange -> 候选方式重建:")
print(f"  重建 flange 原点: {np.round(flange_rec[:3,3], 3)}")
print(f"  参考 flange 原点: {np.round(T_fb[:3,3], 3)}")
print(f"  原点差: {np.linalg.norm(flange_rec[:3,3]-T_fb[:3,3]):.2e} m")

# IK 应该回到参考关节
sols = inverse_kinematics_numeric(flange_rec, q_init=joints_ref)
if len(sols):
    q = sols[0]
    delta = np.degrees(np.max(np.abs(q - joints_ref)))
    print(f"  IK 关节差 max: {delta:.4f}°  (≈0 说明修复正确)")
else:
    print("  IK 无解!!")

# 现在: 模拟一个"微扰"候选: 在参考 sensor 位置附近小幅移动 (5cm, 5°)
print("\n=== 微扰测试 (5cm / 5°) ===")
for i in range(3):
    # 微扰 sensor 位置 +5cm 沿 x, 姿态绕某轴转 5°
    perturb = make_transform(
        np.array([[1,0,0],[0,1,0],[0,0,1.]], dtype=float),
        np.array([0.05*i, 0.0, 0.0]),
    )
    # 简化: 只平移
    sensor_p = sensor.copy()
    sensor_p[:3, 3] += np.array([0.05*i, 0.0, 0.0])
    flange_p = sensor_p @ invert_transform(T_he)
    sols = inverse_kinematics_numeric(flange_p, q_init=joints_ref)
    if len(sols):
        q = sols[0]
        delta = np.degrees(np.max(np.abs(q - joints_ref)))
        print(f"  +{0.05*i:.2f}m x: 关节差 max {delta:.1f}°")
    else:
        print(f"  +{0.05*i:.2f}m x: IK 无解")
