"""
corner_scene.py — Num2 角点法标定场景生成
基于 RCIM_Zhong reproduction_scene.py 的 hand-eye GT、FANUC 运动学
严格按 CODE_REPORT.md §3.3 实现
"""

import numpy as np
from reproduction_scene import (
    generate_hand_eye_gt, rpy_to_matrix, rodrigues, make_transform,
    _HAVE_KINEMATICS
)


# ============================================================================
# SO(3) 工具
# ============================================================================

def so3_log(R):
    """SO(3) → so(3): 返回 ω (旋转向量, ||ω||=角度)"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-10:
        return np.zeros(3)
    omega_hat = (R - R.T) / (2 * np.sin(theta))
    return theta * np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])


def so3_exp(omega):
    """so(3) → SO(3): ω → R"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    axis = omega / theta
    return rodrigues(axis, theta)


# ============================================================================
# 1. 角点平板生成 (CODE_REPORT.md §2.2)
# ============================================================================

def generate_corner_plane(rng, plate_w=400, plate_h=500, alpha=np.pi/2):
    """生成带角点的平板, 对标 CODE_REPORT.md §2.2

    Returns: C, n_B, u_B, v_B, d_1, d_2, w_m, h_m
    """
    # 角点: 机器人前方 0.4-0.8m, 高度 0-0.1m
    C = np.array([rng.uniform(0.4, 0.8),
                  rng.uniform(-0.2, 0.2),
                  rng.uniform(0.0, 0.1)])

    # 法向量: 大致朝上 (±20°)
    ax = rng.uniform(-20, 20)
    ay = rng.uniform(-20, 20)
    R_pl = rpy_to_matrix(ax, ay, rng.uniform(-15, 15))
    n_B = R_pl @ np.array([0., 0., 1.])
    n_B /= np.linalg.norm(n_B)

    # 面内方向
    if abs(n_B[2]) < 0.9:
        u_B = np.cross(np.array([0., 0., 1.]), n_B)
    else:
        u_B = np.cross(np.array([0., 1., 0.]), n_B)
    u_B /= np.linalg.norm(u_B)
    v_B = np.cross(n_B, u_B)

    # 边方向 (CODE_REPORT.md §2.2)
    d_1 = u_B                                     # 边1方向
    d_2 = np.cos(alpha) * u_B + np.sin(alpha) * v_B  # 边2方向

    return C, n_B, u_B, v_B, d_1, d_2, plate_w / 1000.0, plate_h / 1000.0


# ============================================================================
# 2. 角点导向位姿生成 (CODE_REPORT.md §2.4)
# ============================================================================

def generate_corner_poses(rng, n_poses, C, n_B, u_B, v_B, R_he, t_he, alpha,
                          plate_w=400, plate_h=500):
    """生成传感器位姿 — CODE_REPORT.md §2.4 确定�性策略"""
    poses = []
    d_1, d_2 = u_B, np.cos(alpha) * u_B + np.sin(alpha) * v_B

    for i in range(n_poses):
        # 步骤1: 目标点在角点附近
        u_off = rng.uniform(0.03, 0.08)
        v_off = rng.uniform(0.03, 0.08)
        d_standoff = rng.uniform(0.35, 0.65)
        target = C + u_off * u_B + v_off * v_B

        # 步骤3: 传感器 Z 轴 = -n_B (指向平板)
        z_S = -n_B

        # 步骤4: R_align 满足 R_align @ R_he[:,2] = z_S
        v_from = R_he[:, 2]
        v_to = z_S
        cross_v = np.cross(v_from, v_to)
        dot_v = np.dot(v_from, v_to)
        if np.linalg.norm(cross_v) < 1e-10:
            R_align = np.eye(3) if dot_v > 0 else np.diag([1., -1., -1.])
        else:
            cross_v /= np.linalg.norm(cross_v)
            theta = np.arccos(np.clip(dot_v, -1, 1))
            K = np.array([[0, -cross_v[2], cross_v[1]],
                          [cross_v[2], 0, -cross_v[0]],
                          [-cross_v[1], cross_v[0], 0]])
            R_align = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

        # 步骤5: roll 使 x_S ≈ v_B, 加 ±15-30° 扰动覆盖边2
        x_desired = v_B
        x_default = R_align @ R_he[:, 0]
        cos_r = np.clip(np.dot(x_default, x_desired), -1, 1)
        sign_r = np.sign(np.dot(np.cross(x_default, x_desired), z_S) or 1.0)
        base_roll = sign_r * np.arccos(cos_r)
        roll = base_roll + rng.uniform(np.pi/12, np.pi/6) * rng.choice([-1, 1])

        zz = z_S
        Kr = np.array([[0, -zz[2], zz[1]], [zz[2], 0, -zz[0]], [-zz[1], zz[0], 0]])
        R_roll = np.eye(3) + np.sin(roll) * Kr + (1 - np.cos(roll)) * Kr @ Kr
        R_i = R_roll @ R_align

        # 步骤7: 位置补偿 t_he (CODE_REPORT.md: 关键!)
        t_i = target + (d_standoff - 0.05) * n_B - R_i @ t_he

        poses.append((R_i, t_i))

    return poses


# ============================================================================
# 3. 测量生成 (CODE_REPORT.md §2.5)
# ============================================================================

def generate_corner_measurements(scene, poses, n_plane_pts=20,
                                  rng=None, noise_sigma=0.0):
    """生成断点和平面点 — 使用统一 compute_fov_plate_scanline()

    流水线:
      Step 1: compute_fov_plate_scanline() → 扫描线段 + 断点分类
      Step 2: 从 endpoints_S 提取 e1/e2 断点 → p_S_e1, p_S_e2
      Step 3: 降采样 scan_pts_S → p_S_plane
      Step 4: 加噪声
    """
    from reproduction_scene import compute_fov_plate_scanline

    if rng is None:
        rng = np.random.default_rng(42)

    R_he, t_he = scene['R_he'], scene['t_he']
    C, alpha = scene['C'], scene['alpha']
    n_B, u_B, v_B = scene['n_B'], scene['u_B'], scene['v_B']
    pw, ph = scene['plate_w'] / 1000.0, scene['plate_h'] / 1000.0

    measurements = []

    for (R_i, t_i) in poses:
        R_BS = R_i @ R_he          # 传感器在 Base 系
        t_BS = t_i + R_i @ t_he

        # Step 1: 统一计算扫描线
        sl = compute_fov_plate_scanline(
            R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)

        # Step 2: 从端点提取断点
        p_S_e1 = np.zeros(3); valid_e1 = False
        p_S_e2 = np.zeros(3); valid_e2 = False
        for etype, pt_S in sl['endpoints_S']:
            if etype == 'e1':
                p_S_e1 = pt_S.copy()
                valid_e1 = True
            elif etype == 'e2':
                p_S_e2 = pt_S.copy()
                valid_e2 = True

        # Step 3: 平面点 (降采样)
        p_S_plane_raw = sl['scan_pts_S']
        if len(p_S_plane_raw) > 1 and sl['has_intersection']:
            if len(p_S_plane_raw) > n_plane_pts:
                idx = np.linspace(0, len(p_S_plane_raw)-1, n_plane_pts, dtype=int)
                p_S_plane_raw = p_S_plane_raw[idx]
            p_S_plane = p_S_plane_raw
        else:
            p_S_plane = np.zeros((0, 3))

        # Step 4: 加噪声
        if noise_sigma > 0:
            if valid_e1:
                p_S_e1[0] += rng.normal(0, noise_sigma)
                p_S_e1[2] += rng.normal(0, noise_sigma)
            if valid_e2:
                p_S_e2[0] += rng.normal(0, noise_sigma)
                p_S_e2[2] += rng.normal(0, noise_sigma)
            if len(p_S_plane) > 0:
                p_S_plane = p_S_plane.copy()
                for k in range(len(p_S_plane)):
                    p_S_plane[k, 0] += rng.normal(0, noise_sigma)
                    p_S_plane[k, 2] += rng.normal(0, noise_sigma)

        measurements.append({
            'p_S_e1': p_S_e1, 'p_S_e2': p_S_e2,
            'valid_e1': valid_e1, 'valid_e2': valid_e2,
            's_1': 0.0, 's_2': 0.0,  # 兼容旧字段
            'p_S_plane': p_S_plane,
        })

    return measurements


# ============================================================================
# 4. 主入口 (CODE_REPORT.md §3.3)
# ============================================================================

def generate_corner_scene(seed=42, n_poses=4, alpha=np.pi/2,
                           plate_w=400, plate_h=500,
                           n_plane_pts=20, noise_sigma=0.0):
    """Num2 角点法主场景生成器"""
    rng = np.random.default_rng(seed)

    # 手眼真值 (复用 RCIM_Zhong)
    np.random.seed(seed)
    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    # 角点平板
    C, n_B, u_B, v_B, d_1, d_2, w_m, h_m = generate_corner_plane(
        rng, plate_w, plate_h, alpha)

    scene = {
        'R_he': R_he, 't_he': t_he, 'X_gt': X_gt,
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'd_1': d_1, 'd_2': d_2, 'alpha': alpha,
        'plate_w': plate_w, 'plate_h': plate_h,
        'w': w_m, 'h': h_m,
    }

    # 位姿
    poses = generate_corner_poses(rng, n_poses, C, n_B, u_B, v_B,
                                  R_he, t_he, alpha, plate_w, plate_h)

    # 测量
    measurements = generate_corner_measurements(
        scene, poses, n_plane_pts, rng, noise_sigma)

    # 12-DOF 地真值向量
    w_he = so3_log(R_he)
    R_pl = np.column_stack([u_B, v_B, n_B])
    w_pl = so3_log(R_pl)
    theta_gt = np.concatenate([w_he, t_he, w_pl, C])

    return {
        'scene': scene,
        'poses': poses,
        'measurements': measurements,
        'theta_gt': theta_gt,
    }
