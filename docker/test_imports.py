#!/usr/bin/env python3
"""验证所有模块在 ROS2 环境下可正确导入"""
import sys
sys.path.insert(0, '/workspace/common')

from fov_geometry import generate_plane, generate_hand_eye_gt, compute_fov_plate_scanline
print("fov_geometry OK")

from handeye_sim_bridge.fanuc_transforms import pose_to_matrix_fanuc, matrix_to_quat
print("fanuc_transforms OK")

from handeye_sim_bridge.calib_visualization import make_plate_marker, make_fov_marker
print("calib_visualization OK")

from handeye_sim_bridge.bridge_publisher import CalibPublisher
print("bridge_publisher OK")

# 验证 rclpy
import rclpy
rclpy.init()
print("rclpy OK")

# 测试场景生成
import numpy as np
rng = np.random.default_rng(42)
C, n_B, u_B, v_B, w, h = generate_plane(rng)
X_gt = generate_hand_eye_gt(rng)
R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

print(f"场景: C={C}, n_B={n_B}")
print(f"板: {w*1000:.0f}x{h*1000:.0f}mm")

# 测试单帧扫描线
R_S = np.eye(3)
target = C + 0.1 * w * u_B + 0.1 * h * v_B
standoff = 0.5
t_i = target + standoff * n_B - R_S @ t_he
R_BS = R_S @ R_he
t_BS = t_i + R_S @ t_he

sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, w, h)
print(f"扫描线: {len(sl['scan_pts_S'])} 点, {len(sl['endpoints_S'])} 断点")

print("\n所有模块导入 + 核心几何验证通过!")
rclpy.shutdown()
