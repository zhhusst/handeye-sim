#!/usr/bin/env python3
"""
验证: 采集数据 + 边长约束 → 从单位阵出发 → t<0.1mm

流程:
  1. acquisition_sim.py 生成40帧采集数据
  2. 用边长约束公式 d_k² = s₁² + s₂² 加入残差
  3. 从单位阵出发 LM 优化
"""

import numpy as np, yaml, sys, os
sys.path.insert(0, '.')

from acquisition_sim import generate_linear_trajectory, collect_frames
from corner_scene import generate_corner_measurements, so3_log, so3_exp
from nbv_edge_plane import combined_solve_lm, combined_errors

# ============================================================================
# 1. 边长约束残差
# ============================================================================

def edge_distance_residuals(theta, poses, measurements):
    """边长约束残差。

    r_k = d_predicted² - d_measured²
    其中 d_predicted² 由激光平面几何和手眼 X 计算

    theta: [w_he(3), t_he(3), w_pl(3)]
    """
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    # 角点 C 在板坐标系的位置
    # 默认板坐标系原点在 C, 所以 C_plate = (0,0,0)
    # 在基坐标系: C_base 由 R_pl 和板尺寸决定
    #   C_base = (从 R_pl 的平移分量?) 不对...
    # C 是额外的参数, 不在 9-DOF 模型里!
    # 需要把 C 加进来吗? C 在板坐标是已知的=(0,0,0), 
    # 但在基坐标的位置未知...
    # 
    # 我们需要 C_base 或板原点。目前 R_pl 的平移在9-DOF里
    # 是不存在的——R_pl 只是旋转矩阵 [u_B, v_B, n_B]
    # 
    # 所以这个残差需要 C_base 作为额外参数!
    return None  # 暂停, 先讨论


# ============================================================================
# 2. 测试流程
# ============================================================================

if __name__ == '__main__':
    from reproduction_scene import generate_hand_eye_gt
    from corner_scene import generate_corner_plane

    rng = np.random.default_rng(42)
    C, n_B, u_B, v_B, _, _, w_m, h_m = generate_corner_plane(rng)
    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    scene = {
        'R_he': R_he, 't_he': t_he,
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
        'plate_w': 400, 'plate_h': 500, 'w': w_m, 'h': h_m,
    }

    # 生成采集数据
    print("生成采集轨迹...")
    poses, info = generate_linear_trajectory(scene, R_he, t_he, n_steps=40)
    print(f"  {info['n_frames']} 帧采集成功")

    # 生成测量
    meas = generate_corner_measurements(scene, poses, n_plane_pts=10, rng=rng, noise_sigma=0.055/1000)

    # 打印端点间距
    d_vals = []
    for m in meas:
        if m['valid_e1'] and m['valid_e2']:
            d = np.linalg.norm(m['p_S_e1'] - m['p_S_e2'])
            d_vals.append(d)
    print(f"  {len(d_vals)} 帧有双端点, 间距范围 {min(d_vals)*1000:.1f}~{max(d_vals)*1000:.1f}mm")

    # 中间结果等讨论
    print("\n需要讨论: C 的基坐标位置在 9-DOF 模型里没有被建模")
    print("边长约束需要知道板角点 C 在基坐标系的位置")
    print("当前 9-DOF 只有 R_pl = [u_B, v_B, n_B] (旋转)")
    print("没有平移参数来定位 C_base")
