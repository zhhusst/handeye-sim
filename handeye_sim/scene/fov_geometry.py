#!/usr/bin/env python3
"""
scene/fov_geometry.py — 线激光 FOV 与平板的几何求交引擎

从 common/fov_geometry.py 提取，消除 SO(3) 工具重复。
纯 numpy，无 ROS 依赖。所有旋转函数统一使用 core/so3.py。
"""

import numpy as np
from handeye_sim.core.so3 import (so3_exp, so3_log, rpy_to_matrix,
                                    rot_x, rot_y, rot_z)


def make_transform(R, t):
    """4×4 齐次变换矩阵"""
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


# ═══════════════════════════════════════════════════════════════
# 标定板生成
# ═══════════════════════════════════════════════════════════════

def generate_hand_eye_gt(rng=None, rpy_range=(-30, 30), trans_range=(-0.2, 0.2)):
    """生成随机手眼真值 T_S_H (4×4)"""
    if rng is None:
        rng = np.random.default_rng()
    az = rng.uniform(*rpy_range)
    ax = rng.uniform(-10, 10)
    ay = rng.uniform(-10, 10)
    R = rpy_to_matrix(ax, ay, az)
    t = rng.uniform(*trans_range, 3)
    t[2] = abs(t[2])
    return make_transform(R, t)


def generate_plane(rng=None, plane_size=(0.4, 0.5)):
    """生成校准平板参数

    Returns: C, n_B, u_B, v_B, w, h
    """
    if rng is None:
        rng = np.random.default_rng()
    ax = rng.uniform(-10, 10)
    ay = rng.uniform(-10, 10)
    az = rng.uniform(-10, 10)
    R_plane = rpy_to_matrix(ax, ay, az)
    n_B = R_plane @ np.array([0., 0., 1.])
    n_B = n_B / np.linalg.norm(n_B)

    if abs(n_B[2]) < 0.9:
        u_B = np.cross(np.array([0., 0., 1.]), n_B)
    else:
        u_B = np.cross(np.array([0., 1., 0.]), n_B)
    u_B = u_B / np.linalg.norm(u_B)
    v_B = np.cross(n_B, u_B)

    w, h = plane_size
    C = np.array([rng.uniform(0.3, 0.6),
                  rng.uniform(-0.2, 0.2),
                  rng.uniform(0.0, 0.1)])
    return C, n_B, u_B, v_B, w, h


# ═══════════════════════════════════════════════════════════════
# FOV 三角与平板交线计算 (核心函数)
# ═══════════════════════════════════════════════════════════════

def _empty_result(with_line=False, P0=None, line_dir=None):
    r = {
        'scan_pts_B': np.zeros((0, 3)),
        'scan_pts_S': np.zeros((0, 3)),
        'endpoints_B': [],
        'endpoints_S': [],
        'has_intersection': False,
        'line_origin_B': np.zeros(3) if not with_line else P0,
        'line_dir': np.zeros(3) if not with_line else line_dir,
    }
    return r


def compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph,
                                half_fov_deg=15.0, min_range=0.27, max_range=0.82,
                                n_sample=500, half_span=0.8,
                                fov_corners_S=None):
    """计算 FOV 三角与平板的交线

    参数:
        R_BS, t_BS: 传感器在 Base 系位姿
        C, n_B, u_B, v_B: 板参数
        pw, ph: 板尺寸
        half_fov_deg: 半视场角
        min_range, max_range: 测量范围
        n_sample: 交线采样点数
        half_span: 搜索半跨度
        fov_corners_S: 4×3 FOV 校准角点 [c0,c1,c2,c3]

    返回:
        dict: scan_pts_B, scan_pts_S, endpoints_B, endpoints_S, has_intersection, ...
    """
    laser_normal = R_BS[:, 1]
    sensor_origin = t_BS

    line_dir = np.cross(laser_normal, n_B)
    dn = np.linalg.norm(line_dir)
    if dn < 1e-10:
        return _empty_result()
    line_dir /= dn

    A = np.vstack([laser_normal.reshape(1, 3), n_B.reshape(1, 3)])
    b = np.array([np.dot(laser_normal, sensor_origin), np.dot(n_B, C)])
    try:
        P0 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _empty_result()

    t_proj = np.dot(line_dir, C - P0)
    P0 = P0 + t_proj * line_dir

    R_SB = R_BS.T
    t_SB = -R_SB @ t_BS

    # FOV 裁剪
    if fov_corners_S is not None and len(fov_corners_S) >= 4:
        tip_x, tip_z = fov_corners_S[0][0], fov_corners_S[0][2]
        base_z = fov_corners_S[2][2]
        x_left_base = fov_corners_S[3][0]
        x_right_base = fov_corners_S[2][0]
        fov_range_z = base_z - tip_z
        use_calibrated_fov = True
    else:
        tip_x = tip_z = x_left_base = x_right_base = 0.0
        fov_range_z = 1.0
        tan_fov = np.tan(np.deg2rad(half_fov_deg))
        use_calibrated_fov = False

    def _in_fov(x, z):
        if use_calibrated_fov:
            if z < tip_z or z > base_z:
                return False
            t_frac = (z - tip_z) / fov_range_z
            x_left = tip_x + (x_left_base - tip_x) * t_frac
            x_right = tip_x + (x_right_base - tip_x) * t_frac
            return x_left - 1e-6 <= x <= x_right + 1e-6
        else:
            if z < min_range or z > max_range:
                return False
            return abs(x) <= z * tan_fov + 1e-6

    t_vals = np.linspace(-half_span, half_span, n_sample)
    valid = []
    for k, t_val in enumerate(t_vals):
        p_B = P0 + t_val * line_dir
        dp = p_B - C
        u = np.dot(dp, u_B); v = np.dot(dp, v_B)
        if u < -1e-6 or v < -1e-6 or u > pw + 1e-6 or v > ph + 1e-6:
            continue
        p_S = R_SB @ p_B + t_SB
        if not _in_fov(p_S[0], p_S[2]):
            continue
        valid.append(k)

    if len(valid) < 3:
        return _empty_result(with_line=True, P0=P0, line_dir=line_dir)

    segments = []
    seg_start = valid[0]
    for i in range(1, len(valid)):
        if valid[i] - valid[i-1] > 1:
            segments.append((seg_start, valid[i-1]))
            seg_start = valid[i]
    segments.append((seg_start, valid[-1]))
    best_seg = max(segments, key=lambda s: s[1] - s[0])
    seg_start, seg_end = best_seg

    n_scan = min(200, seg_end - seg_start + 1)
    idx_sample = np.linspace(seg_start, seg_end, n_scan, dtype=int)
    scan_pts_B = np.array([P0 + t_vals[i] * line_dir for i in idx_sample])
    scan_pts_S = np.array([R_SB @ p + t_SB for p in scan_pts_B])

    # 端点精确几何求交
    eps = 0.005
    endpoints_B = []
    endpoints_S = []

    denom_e1 = np.dot(laser_normal, u_B)
    if abs(denom_e1) > 1e-12:
        for (use_v, limit) in [(0.0, pw), (ph, pw)]:
            C_edge = C + use_v * v_B
            s_e1 = np.dot(laser_normal, sensor_origin - C_edge) / denom_e1
            if -eps <= s_e1 <= limit + eps:
                pB_e1 = C_edge + s_e1 * u_B
                pS_e1 = R_SB @ pB_e1 + t_SB
                if _in_fov(pS_e1[0], pS_e1[2]):
                    endpoints_B.append(('e1', pB_e1))
                    endpoints_S.append(('e1', pS_e1))

    denom_e2 = np.dot(laser_normal, v_B)
    if abs(denom_e2) > 1e-12:
        for (use_u, limit) in [(0.0, ph), (pw, ph)]:
            C_edge = C + use_u * u_B
            s_e2 = np.dot(laser_normal, sensor_origin - C_edge) / denom_e2
            if -eps <= s_e2 <= limit + eps:
                pB_e2 = C_edge + s_e2 * v_B
                pS_e2 = R_SB @ pB_e2 + t_SB
                if _in_fov(pS_e2[0], pS_e2[2]):
                    endpoints_B.append(('e2', pB_e2))
                    endpoints_S.append(('e2', pS_e2))

    return {
        'scan_pts_B': scan_pts_B,
        'scan_pts_S': scan_pts_S,
        'endpoints_B': endpoints_B,
        'endpoints_S': endpoints_S,
        'has_intersection': True,
        'line_origin_B': P0,
        'line_dir': line_dir,
    }


def compute_fov_triangle(R_BS, t_BS, half_fov_deg=15.0, max_range=0.82):
    """计算 FOV 三角在 Base 系的三个顶点"""
    tip = t_BS
    fov_rad = np.deg2rad(half_fov_deg)
    x_fov = max_range * np.tan(fov_rad)
    base_left = R_BS @ np.array([-x_fov, 0, max_range]) + t_BS
    base_right = R_BS @ np.array([x_fov, 0, max_range]) + t_BS
    return tip, base_left, base_right


# ═══════════════════════════════════════════════════════════════
# 位姿生成辅助
# ═══════════════════════════════════════════════════════════════

def build_R_edge(pitch_deg, yaw_deg, x_align, n_B, u_B, v_B):
    """构建使 FOV 穿过两条边的姿态"""
    z_S = -n_B
    R_yaw = np.eye(3)  # simplified: skip rodrigues, use core/so3
    # Reconstruct manually
    theta = np.deg2rad(yaw_deg)
    k = z_S / np.linalg.norm(z_S)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R_yaw = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

    x_S_aligned = R_yaw @ x_align
    x_S_aligned = x_S_aligned / np.linalg.norm(x_S_aligned)

    if abs(np.dot(x_S_aligned, z_S)) > 0.999:
        x_S_aligned = np.cross(z_S, np.array([1., 0., 0.]))
        if np.linalg.norm(x_S_aligned) < 1e-6:
            x_S_aligned = np.cross(z_S, np.array([0., 1., 0.]))
        x_S_aligned = x_S_aligned / np.linalg.norm(x_S_aligned)

    y_S = np.cross(z_S, x_S_aligned)
    y_S = y_S / np.linalg.norm(y_S)
    x_S_aligned = np.cross(y_S, z_S)

    # pitch
    theta_p = np.deg2rad(pitch_deg)
    k_p = x_S_aligned / np.linalg.norm(x_S_aligned)
    K_p = np.array([[0, -k_p[2], k_p[1]], [k_p[2], 0, -k_p[0]], [-k_p[1], k_p[0], 0]])
    R_pitch = np.eye(3) + np.sin(theta_p) * K_p + (1 - np.cos(theta_p)) * K_p @ K_p

    z_S_pitch = R_pitch @ z_S
    y_S_pitch = R_pitch @ y_S
    x_S_final = np.cross(y_S_pitch, z_S_pitch)
    x_S_final = x_S_final / np.linalg.norm(x_S_final)
    y_S_final = np.cross(z_S_pitch, x_S_final)
    y_S_final = y_S_final / np.linalg.norm(y_S_final)

    return np.column_stack([x_S_final, y_S_final, z_S_pitch])


def collect_frames(scene, poses):
    """对每个位姿计算 FOV 扫描线和断点"""
    C = scene['C']; n_B = scene['n_B']; u_B = scene['u_B']; v_B = scene['v_B']
    pw = scene['w']; ph = scene['h']
    R_he = scene['R_he']; t_he = scene['t_he']

    frames = []
    for k, (R_i, t_i) in enumerate(poses):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)
        endpoints = sl['endpoints_S']
        frame = {
            'has_intersection': sl['has_intersection'],
            'scan_pts_S': sl['scan_pts_S'],
            'endpoints_S': endpoints,
            'n_endpoints': len(endpoints),
            'valid': sl['has_intersection'] and len(endpoints) >= 2,
        }
        if len(endpoints) >= 2:
            pts = [pt for _, pt in endpoints]
            dists = [np.linalg.norm(pts[i] - pts[j])
                     for i in range(len(pts)) for j in range(i+1, len(pts))]
            frame['endpoint_dists'] = dists
            frame['physical_dist'] = max(dists) if dists else 0.0
        else:
            frame['endpoint_dists'] = []
            frame['physical_dist'] = 0.0
        frames.append(frame)
    return frames
