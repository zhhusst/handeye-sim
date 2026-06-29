"""
fov_geometry.py — 线激光 FOV 与平板的几何求交引擎

核心函数:
  compute_fov_plate_scanline() — FOV 三角与平板的交线计算
    1. 激光平面 ∩ 平板平面 → 交线 L
    2. 交线 L ∩ FOV 三角裁剪 → 有效线段
    3. 有效线段 ∩ 平板边界裁剪 → 最终扫描线
    4. 精确求交: 激光平面 ∩ 边1 和 边2 → 断点

SO(3) 工具:
  so3_exp(), so3_log() — 指数/对数映射
  rodrigues() — 罗德里格斯公式

无 ROS 依赖, 纯 numpy/scipy. 可作为离线脚本或 ROS 共享库。
"""

import numpy as np


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


def rodrigues(axis, angle_rad):
    """罗德里格斯公式: 绕任意轴旋转 angle_rad 弧度"""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])


# ============================================================================
# 基础几何
# ============================================================================

def rot_x(a_deg):
    a = np.deg2rad(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a_deg):
    a = np.deg2rad(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a_deg):
    a = np.deg2rad(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_matrix(rx_deg, ry_deg, rz_deg):
    """ZYX 欧拉角 → 旋转矩阵 R = Rx(rx)·Ry(ry)·Rz(rz)"""
    return rot_x(rx_deg) @ rot_y(ry_deg) @ rot_z(rz_deg)


def make_transform(R, t):
    """4×4 齐次变换矩阵"""
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


# ============================================================================
# 手眼真值 + 平板生成
# ============================================================================

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
    """生成校准平板参数"""
    if rng is None:
        rng = np.random.default_rng()
    n_B = np.array([0., 0., 1.])
    ax = rng.uniform(-10, 10)
    ay = rng.uniform(-10, 10)
    az = rng.uniform(-10, 10)
    R_plane = rpy_to_matrix(ax, ay, az)
    n_B = R_plane @ np.array([0., 0., 1.])
    n_B = n_B / np.linalg.norm(n_B)

    # 面内方向
    if abs(n_B[2]) < 0.9:
        u_B = np.cross(np.array([0., 0., 1.]), n_B)
    else:
        u_B = np.cross(np.array([0., 1., 0.]), n_B)
    u_B = u_B / np.linalg.norm(u_B)
    v_B = np.cross(n_B, u_B)

    # 角点位置（平板左下角, 机器人前方）
    C = np.array([rng.uniform(0.3, 0.6),
                  rng.uniform(-0.2, 0.2),
                  rng.uniform(0.0, 0.1)])

    w, h = plane_size
    return C, n_B, u_B, v_B, w, h


# ============================================================================
# FOV 三角与平板交线计算 <<< 核心函数
# ============================================================================

def compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph,
                                half_fov_deg=15.0, min_range=0.27, max_range=0.82,
                                n_sample=500, half_span=0.5):
    """统一计算 FOV 三角与平板的交线 — 动画可视化 + 测量生成共用

    参数:
        R_BS, t_BS: 传感器在 Base 系的位姿 (3×3, 3)
        C: 角点位置 (3)
        n_B: 平板法向量 (3)
        u_B, v_B: 平板坐标系基向量 (3, 3)
        pw, ph: 平板尺寸 (m)
        half_fov_deg: 半视场角, Gocator 2450 默认 15°
        min_range, max_range: 测量范围 (m), Gocator 2450 默认 0.27~0.82
        n_sample: 交线采样点数
        half_span: 交线搜索半跨度 (m)

    返回:
        dict with:
          'scan_pts_B': N×3 扫描点在 Base 系
          'scan_pts_S': N×3 扫描点在 Sensor 系
          'endpoints_B': [(edge_type, pt_3), ...]
          'endpoints_S': [(edge_type, pt_3), ...]
          'has_intersection': bool
          'line_origin_B': 交线上一点 (用于可视化)
          'line_dir': 交线方向 (用于可视化)
    """
    laser_normal = R_BS[:, 1]  # y_S — 激光平面法向量
    sensor_origin = t_BS

    # Step 1: 交线方向 = 激光平面法向量 × 板法向量
    line_dir = np.cross(laser_normal, n_B)
    dn = np.linalg.norm(line_dir)
    if dn < 1e-10:
        return _empty_result()
    line_dir /= dn

    # 交线上一点 P0: 两个平面方程联立
    A = np.vstack([laser_normal.reshape(1, 3), n_B.reshape(1, 3)])
    b = np.array([np.dot(laser_normal, sensor_origin), np.dot(n_B, C)])
    try:
        P0 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return _empty_result()

    # 投影 P0 到靠近 C 的位置
    t_proj = np.dot(line_dir, C - P0)
    P0 = P0 + t_proj * line_dir

    # Base → Sensor 变换
    R_SB = R_BS.T
    t_SB = -R_SB @ t_BS

    tan_fov = np.tan(np.deg2rad(half_fov_deg))
    t_vals = np.linspace(-half_span, half_span, n_sample)
    valid = []

    for k, t_val in enumerate(t_vals):
        p_B = P0 + t_val * line_dir

        # 裁剪: 平板边界
        dp = p_B - C
        u = np.dot(dp, u_B)
        v = np.dot(dp, v_B)
        if u < -1e-6 or v < -1e-6 or u > pw + 1e-6 or v > ph + 1e-6:
            continue

        # 裁剪: FOV 三角 + 测量范围
        p_S = R_SB @ p_B + t_SB
        z = p_S[2]
        x = p_S[0]
        if z < min_range or z > max_range:
            continue
        if abs(x) > z * tan_fov:
            continue

        valid.append(k)

    if len(valid) < 3:
        return _empty_result(with_line=True, P0=P0, line_dir=line_dir)

    # 找最长连续段
    segments = []
    seg_start = valid[0]
    for i in range(1, len(valid)):
        if valid[i] - valid[i-1] > 1:
            segments.append((seg_start, valid[i-1]))
            seg_start = valid[i]
    segments.append((seg_start, valid[-1]))
    best_seg = max(segments, key=lambda s: s[1] - s[0])
    seg_start, seg_end = best_seg

    # 均匀采样扫描点
    n_scan = min(200, seg_end - seg_start + 1)
    idx_sample = np.linspace(seg_start, seg_end, n_scan, dtype=int)
    scan_pts_B = np.array([P0 + t_vals[i] * line_dir for i in idx_sample])
    scan_pts_S = np.array([R_SB @ p + t_SB for p in scan_pts_B])

    # Step 4: 端点精确几何求交
    eps = 0.005
    endpoints_B = []
    endpoints_S = []

    # 边1: 沿 u_B 方向, C + s*u_B
    denom_e1 = np.dot(laser_normal, u_B)
    if abs(denom_e1) > 1e-12:
        s_e1 = np.dot(laser_normal, sensor_origin - C) / denom_e1
        if -eps <= s_e1 <= pw + eps:
            pB_e1 = C + s_e1 * u_B
            pS_e1 = R_SB @ pB_e1 + t_SB
            z_e1, x_e1 = pS_e1[2], pS_e1[0]
            if (min_range - eps <= z_e1 <= max_range + eps and
                    abs(x_e1) <= z_e1 * tan_fov + eps):
                endpoints_B.append(('e1', pB_e1))
                endpoints_S.append(('e1', pS_e1))

    # 边2: 沿 v_B 方向 (α=90°)
    denom_e2 = np.dot(laser_normal, v_B)
    if abs(denom_e2) > 1e-12:
        s_e2 = np.dot(laser_normal, sensor_origin - C) / denom_e2
        if -eps <= s_e2 <= ph + eps:
            pB_e2 = C + s_e2 * v_B
            pS_e2 = R_SB @ pB_e2 + t_SB
            z_e2, x_e2 = pS_e2[2], pS_e2[0]
            if (min_range - eps <= z_e2 <= max_range + eps and
                    abs(x_e2) <= z_e2 * tan_fov + eps):
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


def _empty_result(with_line=False, P0=None, line_dir=None):
    """返回空结果"""
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


# ============================================================================
# FOV 三角顶点计算 (用于可视化)
# ============================================================================

def compute_fov_triangle(R_BS, t_BS, half_fov_deg=15.0, max_range=0.82):
    """计算 FOV 三角在 Base 系中的三个顶点

    返回:
        tip: 传感器原点 (FOV 顶点)
        base_left, base_right: FOV 底部两端点
    """
    tip = t_BS
    fov_rad = np.deg2rad(half_fov_deg)
    x_fov = max_range * np.tan(fov_rad)
    base_left = R_BS @ np.array([-x_fov, 0, max_range]) + t_BS
    base_right = R_BS @ np.array([x_fov, 0, max_range]) + t_BS
    return tip, base_left, base_right


# ============================================================================
# 位姿生成: 过渡姿态 + _build_R_edge
# ============================================================================

def build_R_edge(pitch_deg, yaw_deg, x_align, n_B, u_B, v_B):
    """构建使 FOV 同时穿过两条边的姿态 (R_S = R_he @ R_search)

    策略: 传感器-Z指向板法线方向
          x_S 对齐到 x_align (使 FOV 三角穿边)
          pitch/yaw 在垂直方向扫描

    返回:
        R_i: 机器人末端姿态 (3×3), 即 R_B_H
    """
    z_S = -n_B  # 传感器指向平板

    # 先做 yaw: 绕 z_S 旋转
    R_yaw = rodrigues(z_S, np.deg2rad(yaw_deg))
    x_S_aligned = R_yaw @ x_align
    x_S_aligned = x_S_aligned / np.linalg.norm(x_S_aligned)

    # 如果 x_S_aligned 接近平行于 z_S, 取正交分量
    if abs(np.dot(x_S_aligned, z_S)) > 0.999:
        x_S_aligned = np.cross(z_S, np.array([1., 0., 0.]))
        if np.linalg.norm(x_S_aligned) < 1e-6:
            x_S_aligned = np.cross(z_S, np.array([0., 1., 0.]))
        x_S_aligned = x_S_aligned / np.linalg.norm(x_S_aligned)

    y_S = np.cross(z_S, x_S_aligned)
    y_S = y_S / np.linalg.norm(y_S)
    x_S_aligned = np.cross(y_S, z_S)  # 保证正交

    # pitch: 绕 x_S 旋转
    R_pitch = rodrigues(x_S_aligned, np.deg2rad(pitch_deg))
    z_S_pitch = R_pitch @ z_S
    y_S_pitch = R_pitch @ y_S
    x_S_final = np.cross(y_S_pitch, z_S_pitch)
    x_S_final = x_S_final / np.linalg.norm(x_S_final)
    y_S_final = np.cross(z_S_pitch, x_S_final)
    y_S_final = y_S_final / np.linalg.norm(y_S_final)

    R_S = np.column_stack([x_S_final, y_S_final, z_S_pitch])
    return R_S


# ============================================================================
# 连续采集轨迹生成
# ============================================================================

def generate_smooth_trajectory(scene, n_frames=40,
                                start_uv=(0.02, 0.02),
                                end_uv=(0.35, 0.35),
                                standoff_range=(0.40, 0.70)):
    """生成从角点向板中心的平滑轨迹

    每帧搜索姿态使 FOV 两边可见, 相邻帧姿态变化小 (平滑).

    返回:
        poses: [(R_i, t_i), ...] 共 n_frames 帧
    """
    C = scene['C']; n_B = scene['n_B']; u_B = scene['u_B']; v_B = scene['v_B']
    w_m = scene['w']; h_m = scene['h']
    R_he = scene['R_he']; t_he = scene['t_he']

    u_vals = np.linspace(start_uv[0], end_uv[0], n_frames)
    v_vals = np.linspace(start_uv[1], end_uv[1], n_frames)
    so_vals = np.linspace(standoff_range[0], standoff_range[1], n_frames)

    poses = []
    prev_R = None

    for k in range(n_frames):
        target = C + u_vals[k] * w_m * u_B + v_vals[k] * h_m * v_B
        standoff = so_vals[k]

        found = False
        # pitch/yaw 扫描范围
        pitches = list(range(-10, 11, 2))
        yaws = list(range(-15, 16, 3))

        # 如果有上一帧姿态, 优先在其附近搜索
        if prev_R is not None and k > 0:
            close_pitches = [-5, -3, 0, 3, 5]
            close_yaws = [-8, -5, -3, 0, 3, 5, 8]
            for align in [u_B, v_B]:
                for p in close_pitches:
                    for y in close_yaws:
                        R_S = build_R_edge(p, y, align, n_B, u_B, v_B)
                        sp = target + standoff * n_B
                        t_i = sp - R_S @ t_he
                        R_BS = R_S @ R_he
                        t_BS = t_i + R_S @ t_he
                        sl = compute_fov_plate_scanline(
                            R_BS, t_BS, C, n_B, u_B, v_B, w_m, h_m)
                        if sl['has_intersection'] and len(sl['scan_pts_S']) > 10:
                            eps = [e for e, _ in sl['endpoints_S']]
                            if 'e1' in eps and 'e2' in eps:
                                poses.append((R_S, t_i))
                                prev_R = R_S
                                found = True
                                break
                        if found: break
                    if found: break
                if found: break

        if not found:
            # 全局搜索
            for align in [u_B, v_B]:
                for p in pitches:
                    for y in yaws:
                        R_S = build_R_edge(p, y, align, n_B, u_B, v_B)
                        sp = target + standoff * n_B
                        t_i = sp - R_S @ t_he
                        R_BS = R_S @ R_he
                        t_BS = t_i + R_S @ t_he
                        sl = compute_fov_plate_scanline(
                            R_BS, t_BS, C, n_B, u_B, v_B, w_m, h_m)
                        if sl['has_intersection'] and len(sl['scan_pts_S']) > 10:
                            eps = [e for e, _ in sl['endpoints_S']]
                            if 'e1' in eps and 'e2' in eps:
                                poses.append((R_S, t_i))
                                prev_R = R_S
                                found = True
                                break
                        if found: break
                    if found: break
                if found: break

        if not found:
            # 实在找不到就跳过这帧, 后移一位
            pass

    return poses


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
            dists = []
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    dists.append(np.linalg.norm(pts[i] - pts[j]))
            frame['endpoint_dists'] = dists
            frame['physical_dist'] = max(dists) if dists else 0.0
        else:
            frame['endpoint_dists'] = []
            frame['physical_dist'] = 0.0

        frames.append(frame)

    return frames
