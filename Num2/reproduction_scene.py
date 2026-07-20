"""
================================================================================
reproduction_scene.py — 场景生成 (Zhong et al. 2025, RCIM)
================================================================================

严格复现论文 Section 3.1 的仿真场景生成协议。

论文流程:
  1. 随机生成手眼真值 T_S_H (Sensor→Hand, 待标定的变换)
  2. 随机生成校准平面 (法向量 n_B, 平面上一点 p_B0)
  3. 生成种子位姿, 然后用 Eq.(20) 随机生成 N 个测量位姿 T_B_H
  4. 对每个测量位姿, 用 Eq.(21) 模拟 2D 激光轮廓 p_S = (x, y, 0)
     → 激光平面(z=0)与校准平面的交线
  5. 按 Table 2 给位姿加噪声 (Eq.22), 模拟机器人定位误差
     → 标定接收噪声位姿 + 清洁扫描 (两者的不一致是真实挑战)

关键坐标系约定:
  B (Base)  : 机器人基座坐标系
  H (Hand)  : 机器人末端法兰坐标系
  S (Sensor): 2D激光传感器坐标系 (z=0 为激光平面)

关键变换:
  T_S_H (X_gt): Sensor→Hand, 手眼真值 (待标定的未知量)
  T_B_H       : Hand→Base, 机器人正运动学 (从控制器读取)
  T_B_S       : Sensor→Base, 传感器在世界系的位姿 = T_B_H · T_S_H

对应论文章节:
  Sec 2.1: 坐标系定义 (Eq.1-2)
  Sec 3.1: 仿真协议 (Eq.20-22, Table 2, Fig.4)
================================================================================
"""

import numpy as np
import math

# ============================================================================
# matplotlib 中文字体配置
# ============================================================================
import matplotlib
import matplotlib.pyplot as plt
# 使用系统已安装的 Noto Serif CJK 字体
matplotlib.rcParams['font.sans-serif'] = ['Noto Serif CJK JP', 'AR PL UKai CN',
                                           'AR PL UMing CN', 'Noto Sans CJK JP',
                                           'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# ============================================================================
# 可选：FANUC M-20iD/25 运动学模型（用于关节空间采样）
# ============================================================================
try:
    from fanuc_kinematic import forward_kinematics, inverse_kinematics, random_joints
    _HAVE_KINEMATICS = True
except ImportError:
    _HAVE_KINEMATICS = False

try:
    from fanuc_tool import get_T_B_TCP, simulate_contact_point
    _HAVE_TOOL = False  # 默认不用工具，由用户选择
except ImportError:
    pass

# ============================================================================
# 基础几何工具
# ============================================================================

def rot_x(a_deg):
    """绕X轴旋转 a_deg 度的旋转矩阵"""
    a = np.deg2rad(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(a_deg):
    """绕Y轴旋转"""
    a = np.deg2rad(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(a_deg):
    """绕Z轴旋转"""
    a = np.deg2rad(a_deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rpy_to_matrix(ax, ay, az):
    """ZYX欧拉角 → 旋转矩阵 R = Rx(ax)·Ry(ay)·Rz(az)
    
    论文 Eq.(20) 中 R_adjust 的实现方式。
    论文未指定旋转顺序, ZYX 是常见约定。
    """
    return rot_x(ax) @ rot_y(ay) @ rot_z(az)

def rodrigues(axis, angle_rad):
    """罗德里格斯公式: 绕任意轴旋转 angle_rad 弧度
    
    论文 Eq.(18): R_rand = Rot(ω_r, θ)
    用于 FSA 扰动生成和位姿噪声注入。
    
    参数:
        axis: 3D单位方向向量 (会被自动归一化)
        angle_rad: 旋转角度 (弧度)
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])

def make_transform(R, t):
    """构造 4×4 齐次变换矩阵 T = [R t; 0 1]"""
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T


# ============================================================================
# 1. 手眼真值生成
#    论文 Section 3.1: "the real values of the hand-eye matrix T_S_H0
#    and the target plane are determined first."
#    论文未指定 T_S_H 的具体范围, 这里用均匀分布模拟典型工业安装。
# ============================================================================

def generate_hand_eye_gt(rpy_range=(-30, 30), trans_range=(-0.2, 0.2)):
    """生成随机手眼真值 T_S_H (Sensor→Hand, 4×4)
    
    返回的 X_gt = T_S_H 满足: p_Hand = T_S_H · p_Sensor
    即传感器坐标系中的点变换到末端法兰坐标系。
    
    注意: 生成的 X_gt 保证传感器大致指向 +Z 方向（机器人的前方），
    这样在关节空间采样时激光更有可能照到平板。
    """
    # 旋转: 绕 Z 轴±30°, X/Y 轴小范围微调 → 传感器指向大致方向不变
    az = np.random.uniform(-30, 30)  # 绕Z轴旋转（传感器自身旋转，不影响指向）
    ax = np.random.uniform(-10, 10)  # 俯仰微调
    ay = np.random.uniform(-10, 10)  # 偏航微调
    R_base = rpy_to_matrix(ax, ay, az)
    # 传感器指向 +Z: 法兰Z轴到传感器Z轴的变换
    # 默认为传感器沿法兰 Z 轴向前安装
    R = R_base
    t = np.random.uniform(*trans_range, 3)
    # 保证传感器在法兰前方（Z正方向）
    t[2] = abs(t[2])  # 传感器沿法兰Z正方向安装
    return make_transform(R, t)


# ============================================================================
# 2. 平面生成
#    论文 Section 3: 单个校准平面, 在机器人前方任意朝向。
#    平面方程: n_B · p = d, 其中 d = n_B · p_B0
# ============================================================================

def generate_plane(plane_size=(0.4, 0.5), table_height=None):
    """生成校准平面

    参数:
        plane_size: (width, height) 平板尺寸 (m), 默认 (0.4, 0.5)
        table_height: 平板高度 (m), 默认 None(与 Sharifzadeh 一致 Z≈0)
                      运动学模式建议 0.3~0.5m (桌面高度)
    """
    if table_height is not None:
        p_B0 = np.array([1.0, 0.0, table_height])
    else:
        # 原: 机器人前方410mm, 右侧150mm
        p_B0 = np.array([0.41, 0.15, 0.0])
    # 微调 (模拟实际放置的微小偏差)
    p_B0 += np.random.uniform(-0.03, 0.03, 3)
    
    # 平面法向量: 大致朝上 (+Z, 像桌面上放置的标定板)
    # 随机微调 (±10°), 确保不是完全水平 (论文要求避免退化)
    ax = np.random.uniform(-10, 10)
    ay = np.random.uniform(-10, 10)
    az = np.random.uniform(-10, 10)
    R_plane = rpy_to_matrix(ax, ay, az)
    n_B = R_plane @ np.array([0., 0., 1.])  # 指向 +Z = 朝上
    n_B = n_B / np.linalg.norm(n_B)
    
    # 平面内方向: u沿宽度(400mm), v沿高度(500mm)
    if abs(n_B[2]) < 0.9:
        u_B = np.cross(np.array([0., 0., 1.]), n_B)
    else:
        u_B = np.cross(np.array([0., 1., 0.]), n_B)
    u_B = u_B / np.linalg.norm(u_B)
    v_B = np.cross(n_B, u_B)
    
    w, h = plane_size
    return n_B, p_B0, u_B, v_B, w, h


# ============================================================================
# 3. 测量位姿生成 (论文 Eq.20, Fig.4)
#
#    论文 Section 3.1:
#    "a group of measuring poses are generated by a random technique"
#    从种子位姿出发, 随机平移 + 随机调整姿态 →
#    模拟实际测量中机器人在平面附近的不同位置扫描。
#
#    Eq.(20):
#      T_B_H = [R_B_H_seed · R_adjust,  t_seed + l_rand · ω_t]
#              [         0,                         1        ]
#    其中:
#      l_rand ~ U(0, 150) mm  — 随机平移距离
#      ω_t                    — 随机平移方向 (单位向量)
#      R_adjust                — 随机姿态微调 (确保激光能照到平面)
# ============================================================================

def generate_seed_pose(p_B0, n_B, X_gt, d=0.8, theta_deg=20, target_center=None):
    """生成种子位姿 T_B_H_seed，考虑 X_gt

    Gocator 2450 参数: d=300mm, half_fov=15°, 测量范围 270~820mm
    
    关键设计: 绕平板面内随机方向倾斜传感器。
    通过随机化倾斜轴，使激光-平板交线能覆盖平板的不同区域。
    
    参数:
        target_center: 可选，在平板上的目标点（Base系3D坐标）。
                       若提供，传感器位于该点正上方 d 处；
                       否则传感器位于平板中心 p_B0 正上方。
    """
    theta = np.deg2rad(theta_deg)
    
    if target_center is None:
        target_center = p_B0

    # 平板内两个正交方向
    if abs(n_B[2]) < 0.9:
        u = np.cross(np.array([0., 0., 1.]), n_B)
    else:
        u = np.cross(np.array([0., 1., 0.]), n_B)
    u = u / np.linalg.norm(u)
    v = np.cross(n_B, u)  # v ≈ +Y 方向

    # -- 步骤1: 绕平板面内随机方向倾斜传感器 --
    # 在 u-v 平面内随机选一个倾斜轴
    tilt_angle = np.random.uniform(-np.pi, np.pi)
    tilt_axis = np.cos(tilt_angle) * u + np.sin(tilt_angle) * v
    tilt_axis = tilt_axis / np.linalg.norm(tilt_axis)
    
    R_tilt = rodrigues(tilt_axis, theta)
    z_S = R_tilt @ (-n_B)
    z_S = z_S / np.linalg.norm(z_S)

    # -- 步骤2: 构建传感器坐标系 --
    x_S = np.cross(tilt_axis, z_S)
    x_S = x_S / np.linalg.norm(x_S)
    y_S = np.cross(z_S, x_S)
    R_B_S = np.column_stack([x_S, y_S, z_S])

    # -- 步骤3: 传感器位置 (在 target_center 上方 d 米) --
    t_B_S = target_center + d * n_B

    # -- 步骤4: 反推手端位姿 --
    T_B_S = make_transform(R_B_S, t_B_S)
    X_gt_inv = np.linalg.inv(X_gt)
    seed_pose = T_B_S @ X_gt_inv

    return seed_pose


def is_point_on_plane(p_B, p_B0, u_B, v_B, w, h):
    """检查3D点 p_B 是否在平板矩形范围内
    
    平板由中心 p_B0, 宽度 w(沿u_B), 高度 h(沿v_B) 定义。
    返回 True 如果点在矩形内。
    """
    dp = p_B - p_B0
    du = np.dot(dp, u_B)
    dv = np.dot(dp, v_B)
    return abs(du) <= w/2 and abs(dv) <= h/2


def generate_measuring_poses(seed_pose, n_poses, X_gt, n_B, p_B0,
                              u_B, v_B, w, h, l_rand_max=0.08):
    """生成测量位姿和对应的2D激光轮廓 (论文 Eq.20 + Eq.21)
    
    新增约束 (基于 Sharifzadeh 2020 d/θ/β 参数思想):
      平移扰动限制在 ±l_rand_max (确保传感器仍在平板上方)
      姿态微调限制在 ±10° (确保激光线能照到平板)
      扫描线必须落在平板 400×500mm 矩形内 (过滤越界扫描)
    
    参数:
        u_B, v_B: 平面内正交方向 (沿宽度和高度)
        w, h: 平板宽度和高度 (m)
        l_rand_max: 最大平移扰动 (m), 减小到80mm以确保不过度偏离
    """
    t_seed = seed_pose[:3, 3]
    R_seed = seed_pose[:3, :3]
    
    poses = []
    points_2d = []
    
    for _ in range(n_poses * 3):  # 生成3倍候选, 过滤后仍能凑够
        # --- Eq.(20) 第一部分: 随机平移 ---
        l_rand = np.random.uniform(0, l_rand_max)
        omega_t = np.random.randn(3)
        omega_t = omega_t / np.linalg.norm(omega_t)
        t_B_H = t_seed + l_rand * omega_t
        
        # --- Eq.(20) 第二部分: 姿态微调 R_adjust (限制±10°) ---
        ax = np.random.uniform(-30, 30)
        ay = np.random.uniform(-30, 30)
        az = np.random.uniform(-30, 30)
        R_adjust = rpy_to_matrix(ax, ay, az)
        R_B_H = R_seed @ R_adjust
        
        T_B_H = make_transform(R_B_H, t_B_H)
        
        # --- Eq.(21): 模拟 2D 激光扫描 (带激光扇面和平板边界裁剪) ---
        pts_2d = simulate_laser_scan(T_B_H, X_gt, n_B, p_B0,
                                     u_B=u_B, v_B=v_B, w=w, h=h)
        
        if pts_2d is None or len(pts_2d) < 3:
            continue
        
        # --- 检查扫描线是否在平板边界内 (≥70%点即可) ---
        T_B_S = T_B_H @ X_gt
        n_inside = 0
        for p2 in pts_2d:
            p_S = np.array([p2[0], 0.0, p2[1]])  # (x, 0, z) 激光平面 y=0
            p_B = T_B_S[:3,:3] @ p_S + T_B_S[:3,3]
            if is_point_on_plane(p_B, p_B0, u_B, v_B, w, h):
                n_inside += 1
        
        if n_inside >= len(pts_2d) * 0.7:  # ≥70% 点在板内
            poses.append(T_B_H)
            points_2d.append(pts_2d)
            if len(poses) >= n_poses:
                break
    
    return poses, points_2d


# ============================================================================
# 4. 2D激光轮廓仿真 (论文 Eq.21)
#
#    论文 Section 3.1, Eq.(21):
#    p_S = T_S_H^{-1} · T_B_H^{-1} · p_B
#
#    其中 p_B 是激光平面与校准平面的交线上的3D点。
#
#    几何原理:
#      - 激光平面在传感器坐标系中为 z=0
#      - 传感器在世界系中的位姿为 T_B_S = T_B_H · X_gt
#      - 激光平面在世界系: 法向量=R_B_S[:,2](传感器Z轴), 过点 t_B_S
#      - 交线 = 激光平面 ∩ 校准平面
#      - 沿交线采样 → 3D点 p_B → 变换到传感器系 → (x,y,0)
# ============================================================================

def simulate_laser_scan(T_B_H, X_gt, n_B, p_B0, n_pts=50,
                         u_B=None, v_B=None, w=None, h=None,
                         half_fov_deg=15.0, min_range=0.27, max_range=0.82):
    """仿真单次 2D 激光扫描 (Gocator 2450 参数)

    Gocator 2450:
      - 安装静距离: 270mm → 测量起点 z=270mm in sensor
      - 测量范围: 550mm → 终点 z=820mm
      - 近端视野: 145mm → half_fov ≈ 15°
      - 远端视野: 425mm
      - 激光平面 = 传感器 XOZ 平面 (y=0 in sensor frame)
    """
    T_B_S = T_B_H @ X_gt
    R_B_S = T_B_S[:3, :3]
    t_B_S = T_B_S[:3, 3]
    y_S = R_B_S[:, 1]   # 激光平面法向量 (y=0)
    z_S = R_B_S[:, 2]   # 传感器光轴

    # 交线方向
    line_dir = np.cross(n_B, y_S)
    norm_l = np.linalg.norm(line_dir)
    if norm_l < 1e-6:
        return None
    line_dir = line_dir / norm_l

    # 锚点: 交线上一点 (lstsq 解欠定方程组, 同时满足两个平面方程)
    A_eq = np.vstack([n_B.reshape(1, 3), y_S.reshape(1, 3)])
    b_eq = np.array([n_B @ p_B0, y_S @ t_B_S])
    try:
        anchor_raw = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    # 沿交线方向投影到离 p_B0 最近 (使扫描线中心靠近平板中心)
    t_proj = np.dot(line_dir, p_B0 - anchor_raw)
    anchor = anchor_raw + t_proj * line_dir

    T_S_B = np.linalg.inv(T_B_S)
    half_fov = np.deg2rad(half_fov_deg)
    tan_fov = np.tan(half_fov)

    # 从锚点沿交线双向搜索至扇面边界
    n_candidates = 500
    half_span = 1.0
    t_vals = np.linspace(-half_span, half_span, n_candidates)
    pts_B = anchor + t_vals[:, None] * line_dir
    pts_S = (T_S_B[:3, :3] @ pts_B.T + T_S_B[:3, 3:4]).T

    valid = []
    for k in range(n_candidates):
        x, y, z = pts_S[k]
        p_B = pts_B[k]
        if z < min_range or z > max_range:
            continue
        if abs(x) > z * tan_fov:
            continue
        if u_B is not None:
            dp = p_B - p_B0
            du = np.dot(dp, u_B)
            dv = np.dot(dp, v_B)
            if abs(du) > w/2 or abs(dv) > h/2:
                continue
        valid.append(k)

    if len(valid) < 3:
        return None

    # 找最长连续段
    segments = []
    seg_start = valid[0]
    for i in range(1, len(valid)):
        if valid[i] - valid[i-1] > 1:
            segments.append((seg_start, valid[i-1]))
            seg_start = valid[i]
    segments.append((seg_start, valid[-1]))
    best_seg = max(segments, key=lambda s: s[1] - s[0])
    start, end = best_seg

    idx_sample = np.linspace(start, end, n_pts, dtype=int)
    pts_valid_B = pts_B[idx_sample]
    pts_valid_S = (T_S_B[:3, :3] @ pts_valid_B.T + T_S_B[:3, 3:4]).T

    return pts_valid_S[:, [0, 2]]


# ============================================================================
# 4.5 统一 FOV×平板交线计算 (动画 + 测量生成共用)
#
#    流水线:
#      Step 1: 激光平面 ∩ 板平面 → 无限交线 L
#      Step 2: 交线 L 依次过三个裁剪门:
#              a. FOV 三角内 (|x| ≤ z·tan(half_fov) in sensor frame)
#              b. 测量范围内 (min_range ≤ z ≤ max_range)
#              c. 平板边界内 (0 ≤ u ≤ pw, 0 ≤ v ≤ ph)
#      Step 3: 裁剪后的线段 = 扫描线
#      Step 4: 线段两端点 = 断点 (角点测量)
# ============================================================================

def compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph,
                                half_fov_deg=15.0, min_range=0.27, max_range=0.82,
                                n_sample=500, half_span=0.5):
    """统一计算 FOV 三角与平板的交线 — 动画可视化 + 测量生成共用

    Args:
        R_BS, t_BS: 传感器在 Base 系的位姿 (3×3, 3)
        C: 角点位置 (3)
        n_B: 平板法向量 (3)
        u_B, v_B: 平板坐标系基向量 (3, 3), 正交
        pw, ph: 平板沿 u_B 和 d_2 的尺寸 (m)
        half_fov_deg: 半视场角 (度), Gocator 2450 默认 15°
        min_range, max_range: 测量范围 (m), Gocator 2450 默认 0.27~0.82
        n_sample: 交线采样点数
        half_span: 交线搜索半跨度 (m)

    Returns:
        dict with:
          'scan_pts_B': N×3 扫描点在 Base 系 (若 N=0 则无有效交线)
          'scan_pts_S': N×3 扫描点在 Sensor 系
          'endpoints_B': [(pt_B, edge_type), ...] 断点在 Base 系, edge_type ∈ {'e1','e2'}
          'endpoints_S': [(pt_S, edge_type), ...] 断点在 Sensor 系
          'has_intersection': bool
    """
    # 传感器系: 激光平面法向量 = y_S (传感器 Y 轴), 传感器原点在 t_BS
    laser_normal = R_BS[:, 1]  # y_S
    sensor_origin = t_BS

    # Step 1: 交线方向 = 激光平面法向量 × 板法向量
    line_dir = np.cross(laser_normal, n_B)
    dn = np.linalg.norm(line_dir)
    if dn < 1e-10:
        return {'scan_pts_B': np.zeros((0, 3)), 'scan_pts_S': np.zeros((0, 3)),
                'endpoints_B': [], 'endpoints_S': [], 'has_intersection': False}
    line_dir /= dn

    # 交线上找一点 P0: 同时满足两个平面方程
    A = np.vstack([laser_normal.reshape(1, 3), n_B.reshape(1, 3)])
    b = np.array([np.dot(laser_normal, sensor_origin), np.dot(n_B, C)])
    try:
        P0 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {'scan_pts_B': np.zeros((0, 3)), 'scan_pts_S': np.zeros((0, 3)),
                'endpoints_B': [], 'endpoints_S': [], 'has_intersection': False}

    # 投影 P0 到靠近 C 的位置 (使扫描线中心靠近平板中心)
    t_proj = np.dot(line_dir, C - P0)
    P0 = P0 + t_proj * line_dir

    # T_S_B = inv(T_B_S), 用于 Base→Sensor 变换
    R_SB = R_BS.T
    t_SB = -R_SB @ t_BS

    # Step 2: 沿交线采样, 依次过三个裁剪门
    tan_fov = np.tan(np.deg2rad(half_fov_deg))
    t_vals = np.linspace(-half_span, half_span, n_sample)
    valid = []
    for k, t_val in enumerate(t_vals):
        p_B = P0 + t_val * line_dir

        # 裁剪门 c: 平板边界
        dp = p_B - C
        u = np.dot(dp, u_B)
        v = np.dot(dp, v_B)
        if u < -1e-6 or v < -1e-6 or u > pw + 1e-6 or v > ph + 1e-6:
            continue

        # 裁剪门 a+b: FOV 三角 + 测量范围 (在 Sensor 系检查)
        p_S = R_SB @ p_B + t_SB
        z = p_S[2]
        x = p_S[0]
        if z < min_range or z > max_range:
            continue
        if abs(x) > z * tan_fov:
            continue

        valid.append(k)

    if len(valid) < 3:
        return {'scan_pts_B': np.zeros((0, 3)), 'scan_pts_S': np.zeros((0, 3)),
                'endpoints_B': [], 'endpoints_S': [], 'has_intersection': False}

    # Step 3: 找最长连续段
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

    # Step 4: 端点识别 — 精确几何求交 (非采样逼近!)
    #   关键: 断点 = 激光平面 ∩ 板边缘的精确交点
    #   有效条件: 该交点 ∈ FOV三角 ∧ ∈ 测量范围 ∧ ∈ 板边界
    eps = 0.005  # 5mm
    endpoints_B = []
    endpoints_S = []

    # 边1: 沿 d_1=u_B 的射线, C + s*u_B
    denom_e1 = np.dot(laser_normal, u_B)
    if abs(denom_e1) > 1e-12:
        s_e1 = np.dot(laser_normal, sensor_origin - C) / denom_e1
        if -eps <= s_e1 <= pw + eps:
            pB_e1 = C + s_e1 * u_B
            pS_e1 = R_SB @ pB_e1 + t_SB
            z_e1, x_e1 = pS_e1[2], pS_e1[0]
            if min_range - eps <= z_e1 <= max_range + eps and abs(x_e1) <= z_e1 * tan_fov + eps:
                endpoints_B.append(('e1', pB_e1))
                endpoints_S.append(('e1', pS_e1))

    # 边2: 沿 d_2 的射线 (α=π/2 时为 v_B)
    d_2_edge = v_B  # Gocator 角点 α=90°, d_2 = v_B
    denom_e2 = np.dot(laser_normal, d_2_edge)
    if abs(denom_e2) > 1e-12:
        s_e2 = np.dot(laser_normal, sensor_origin - C) / denom_e2
        if -eps <= s_e2 <= ph + eps:
            pB_e2 = C + s_e2 * d_2_edge
            pS_e2 = R_SB @ pB_e2 + t_SB
            z_e2, x_e2 = pS_e2[2], pS_e2[0]
            if min_range - eps <= z_e2 <= max_range + eps and abs(x_e2) <= z_e2 * tan_fov + eps:
                endpoints_B.append(('e2', pB_e2))
                endpoints_S.append(('e2', pS_e2))

    return {
        'scan_pts_B': scan_pts_B,
        'scan_pts_S': scan_pts_S,
        'endpoints_B': endpoints_B,
        'endpoints_S': endpoints_S,
        'has_intersection': True,
    }


# ============================================================================
# 5. 完整场景生成 (论文 Section 3.1 完整协议)
#
#    论文仿真流程:
#      1. 确定手眼真值 T_S_H0 和平面参数 → 步骤1-2
#      2. 生成种子位姿, 用 Eq.(20) 产生 N 个随机测量位姿
#      3. 对每个位姿, Eq.(21) 计算理想传感器测量 p_S
#         → 这些是"清洁扫描"
#      4. Eq.(22) + Table 2 给机器人位姿加噪声
#         → 模拟机器人定位误差
#      5. 标定算法接收: 噪声位姿(T_B_H_err) + 清洁扫描(p_S)
#         → 位姿和扫描的噪声不一致, 模拟真实系统挑战
#
#    论文原文:
#    "only the robot positioning error is considered below,
#     and the involved poses with errors are generated by Eq.(22)"
#    → 只考虑机器人定位误差 (加到位姿上)
#    → 传感器测量从理想位姿计算 (无额外噪声)
# ============================================================================

def generate_calibration_scene(n_poses=120, noise_level=None, seed=None,
                               use_kinematics=False, joint_range_scale=0.85,
                               seed_d=0.3, seed_theta=20):
    """生成完整标定场景

    参数:
        n_poses: 测量位姿数量 (论文 Fig.7: 推荐≥120)
        noise_level: None(无噪声) / 'low' / 'medium' / 'high' (Table 2)
        seed: 随机种子
        use_kinematics: 若 True, 在关节空间采样+FK生成可达位姿; 
                        若 False (默认), 用原笛卡尔随机位姿
        joint_range_scale: 关节采样范围缩放因子 (0~1)
        seed_d: 种子位姿的传感器高度 (默认 0.3m, Gocator 2450 静距 270mm)
        seed_theta: 种子位姿的倾斜角度 (默认 20°)

    返回字典:
        X_gt:        手眼真值 T_S_H (4×4)
        poses_gt:    理想位姿列表 (无噪声)
        poses_noisy: 噪声位姿列表 (标定算法使用)
        scans_2d:    清洁 2D 扫描列表 (从理想位姿生成)
        n_B:         平面单位法向量 (Base系)
        p_B0:        平面上一点 (Base系)
        d:           平面距离 = n_B · p_B0
        noise_level: 噪声等级
        joints:      (仅 use_kinematics=True) 每个 pose 对应的关节角 (n_poses×6 弧度)
        u_B, v_B, w, h: 平板参数
    """
    if seed is not None:
        np.random.seed(seed)

    # 步骤1: 手眼真值 (论文: "real values ... determined first")
    X_gt = generate_hand_eye_gt()

    # 步骤2: 校准平面 (含物理尺寸和边界)
    # 运动学模式: 平板放在桌面高度 (0.35m), 更符合 FANUC 工作空间
    table_h = 0.35 if use_kinematics else None
    n_B, p_B0, u_B, v_B, w, h = generate_plane(table_height=table_h)
    d = n_B @ p_B0

    if use_kinematics and _HAVE_KINEMATICS:
        # ===== 方案A改进: 多种子分布生成 + IK 验证 =====
        # 每个位姿从平板上不同目标点生成种子，保证采集线均匀覆盖整板
        poses_clean = []
        joints_list = []
        scans_2d = []
        attempts = 0
        max_attempts = n_poses * 15
        l_rand_max = 0.08      # 平移扰动幅度
        theta_range = seed_theta  # 倾斜角度
        angle_range = 15       # 姿态随机扰动 (°)
        spread = 0.45          # 种子在平板上的散布范围比例
        
        while len(poses_clean) < n_poses and attempts < max_attempts:
            attempts += 1
            # 随机选平板上的目标点
            u_off = np.random.uniform(-w * spread, w * spread)
            v_off = np.random.uniform(-h * spread, h * spread)
            target = p_B0 + u_off * u_B + v_off * v_B
            # 生成该目标的种子位姿
            seed = generate_seed_pose(p_B0, n_B, X_gt, d=seed_d,
                                      theta_deg=theta_range, target_center=target)
            # Eq.20 扰动
            l_rand = np.random.uniform(0, l_rand_max)
            omega_t = np.random.randn(3)
            omega_t = omega_t / np.linalg.norm(omega_t)
            t_B_H = seed[:3, 3] + l_rand * omega_t
            ax, ay, az = np.random.uniform(-angle_range, angle_range, 3)
            R_adjust = rpy_to_matrix(ax, ay, az)
            R_B_H = seed[:3, :3] @ R_adjust
            T_B_H = make_transform(R_B_H, t_B_H)
            
            # IK → FK → scan
            sols = inverse_kinematics(T_B_H)
            if len(sols) == 0:
                continue
            j = sols[0]
            T_B_H_fk = forward_kinematics(j)
            sc = simulate_laser_scan(T_B_H_fk, X_gt, n_B, p_B0,
                                     u_B=u_B, v_B=v_B, w=w, h=h)
            if sc is None or len(sc) < 3:
                continue
            joints_list.append(j)
            poses_clean.append(T_B_H_fk)
            scans_2d.append(sc)

        result = {
            'X_gt': X_gt,
            'poses_gt': poses_clean,
            'scans_2d': scans_2d,
            'n_B': n_B, 'p_B0': p_B0, 'd': d,
            'u_B': u_B, 'v_B': v_B, 'w': w, 'h': h,
            'noise_level': noise_level,
            'joints': np.array(joints_list) if joints_list else np.array([]),
        }
    else:
        # ===== 原笛卡尔模式 (论文 Eq.20) =====
        seed_pose = generate_seed_pose(p_B0, n_B, X_gt)
        poses_clean, scans_2d = generate_measuring_poses(
            seed_pose, n_poses, X_gt, n_B, p_B0,
            u_B, v_B, w, h, l_rand_max=0.15
        )
        result = {
            'X_gt': X_gt,
            'poses_gt': poses_clean,
            'scans_2d': scans_2d,
            'n_B': n_B, 'p_B0': p_B0, 'd': d,
            'u_B': u_B, 'v_B': v_B, 'w': w, 'h': h,
            'noise_level': noise_level,
        }

    # 步骤5: 噪声注入 (Eq.22 + Table 2)
    poses_clean = result['poses_gt']
    if noise_level is None:
        poses_noisy = [p.copy() for p in poses_clean]
    else:
        noise_params = {
            # Table 2 参数: (θ_noise范围/度, l_noise范围/米)
            'low':    (0.025, 0.1e-3),    # ±0.025°,  ±0.1mm
            'medium': (0.1,   0.25e-3),    # ±0.1°,    ±0.25mm
            'high':   (0.25,  1.0e-3),     # ±0.25°,   ±1mm
        }
        poses_noisy = add_pose_noise(poses_clean, *noise_params[noise_level])
    
    result['poses_noisy'] = poses_noisy
    return result


# ============================================================================
# 6. 位姿噪声注入 (论文 Eq.22)
#
#    论文 Section 3.1, Eq.(22):
#      T_B_H_err = [R_noise · R_B_H,  t_B_H + t_noise]
#                  [      0,                 1        ]
#    其中:
#      R_noise = Rot(ω_r, θ_noise)     — 旋转噪声, θ_noise ~ U(-θ, +θ)
#      t_noise = l_noise · ω_t         — 平移噪声, l_noise ~ U(-l, +l)
#      ω_r, ω_t 为随机单位方向向量
# ============================================================================

def add_pose_noise(poses, theta_range_deg, l_range_m):
    """给位姿加噪声 (论文 Eq.22)
    
    参数:
        poses: 理想位姿列表
        theta_range_deg: 旋转噪声范围 (度), 如 Table 2 的 0.025/0.1/0.25
        l_range_m: 平移噪声范围 (米), 如 Table 2 的 0.1e-3/0.25e-3/1.0e-3
    
    返回:
        噪声位姿列表
    """
    noisy = []
    for T in poses:
        R, t = T[:3, :3].copy(), T[:3, 3].copy()
        
        # 旋转噪声: θ_noise ~ U(-θ_range, +θ_range)
        theta = np.random.uniform(-theta_range_deg, theta_range_deg)
        theta_rad = np.deg2rad(theta)
        omega_r = np.random.randn(3)
        omega_r = omega_r / np.linalg.norm(omega_r)
        R_noise = rodrigues(omega_r, theta_rad)
        
        # 平移噪声: l_noise ~ U(-l_range, +l_range)
        l_noise = np.random.uniform(-l_range_m, l_range_m)
        omega_t = np.random.randn(3)
        omega_t = omega_t / np.linalg.norm(omega_t)
        t_noise = t + l_noise * omega_t
        
        # Eq.(22): T_B_H_err
        T_noisy = make_transform(R_noise @ R, t_noise)
        noisy.append(T_noisy)
    
    return noisy


# ============================================================================
# 可视化 (检查场景是否正确)
# ============================================================================

def visualize_scene(scene, max_poses=20, figsize=(12, 10), show_robot=True):
    """3D可视化标定场景

    显示内容:
      - 校准平面 (蓝色半透明)
      - 平面法向量 (红色箭头)
      - 机器人基座坐标系 (原点)
      - 传感器位姿 (绿点) + 扫描线 (绿线, 在校准平面上)
      - 种子位姿 (大蓝点)

    参数:
        show_robot: 是否显示机械臂线段 (动画时设为 False 避免干扰)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.patches import FancyArrowPatch
        from mpl_toolkits.mplot3d import proj3d
    except ImportError:
        print("需要 matplotlib: pip install matplotlib")
        return
    
    X_gt = scene['X_gt']
    poses = scene['poses_gt'][:max_poses]
    scans = scene['scans_2d'][:max_poses]
    n_B = scene['n_B']
    p_B0 = scene['p_B0']
    d = scene['d']
    # 平板尺寸 (若场景包含则使用真实尺寸, 否则默认400×500mm)
    u_B = scene.get('u_B')
    v_B = scene.get('v_B')
    w = scene.get('w', 0.4)
    h_plane = scene.get('h', 0.5)
    
    # 若没有预设的u_B/v_B, 自动计算
    if u_B is None:
        if abs(n_B[2]) < 0.9:
            u_B = np.cross(np.array([0., 0., 1.]), n_B)
        else:
            u_B = np.cross(np.array([1., 0., 0.]), n_B)
        u_B = u_B / np.linalg.norm(u_B)
        v_B = np.cross(n_B, u_B)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # ---- 校准平面 (蓝色半透明矩形, 真实尺寸) ----
    half_w = w / 2
    half_h = h_plane / 2
    corners = np.array([
        p_B0 - half_w*u_B - half_h*v_B,
        p_B0 + half_w*u_B - half_h*v_B,
        p_B0 + half_w*u_B + half_h*v_B,
        p_B0 - half_w*u_B + half_h*v_B,
    ])
    
    tri1 = np.array([corners[0], corners[1], corners[2]])
    tri2 = np.array([corners[0], corners[2], corners[3]])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plane_poly = Poly3DCollection([tri1, tri2], alpha=0.25,
                                   facecolor='steelblue', edgecolor='navy',
                                   linewidth=1.5)
    ax.add_collection3d(plane_poly)
    
    # 平板边框文字标注
    ax.text(p_B0[0], p_B0[1], p_B0[2],
            f'{w*1000:.0f}×{h_plane*1000:.0f}mm', fontsize=9)
    
    # ---- 平面法向量箭头 (红色) ----
    arrow_len = 0.15
    ax.quiver(p_B0[0], p_B0[1], p_B0[2],
              n_B[0]*arrow_len, n_B[1]*arrow_len, n_B[2]*arrow_len,
              color='red', linewidth=3, arrow_length_ratio=0.2,
              label=f'平面法向量 n_B (d={d:.3f}m)')
    
    # ---- 机器人基座 (原点) ----
    ax.scatter([0], [0], [0], color='black', s=100, marker='s', label='Base (原点)')
    # 基座坐标系
    ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', alpha=0.5, linewidth=1)
    ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', alpha=0.5, linewidth=1)
    ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', alpha=0.5, linewidth=1)

    # ---- 机械臂简易线段模型 (仅运动学模式有 joint 信息) ----
    joints = scene.get('joints')
    if show_robot and joints is not None and len(joints) > 0 and len(poses) > 0:
        try:
            from fanuc_kinematic import M20ID_DH as _DH
            _dh = _DH
            # 用第一个位姿的关节角画机械臂
            j = joints[0]
            t = np.array(j, dtype=float).copy()
            diff_23 = t[1] + t[2]
            t[5] = -t[5]

            c1, s1 = math.cos(t[0]), math.sin(t[0])
            c2, s2 = math.cos(t[1]), math.sin(t[1])
            c23, s23 = math.cos(diff_23), math.sin(diff_23)

            # 各连杆变换矩阵 → 提取位置
            T_0_0 = np.eye(4)
            T_0_1 = np.array([[c1,-s1,0,0],[s1,c1,0,0],[0,0,1,_dh.d[0]],[0,0,0,1]])
            T_1_2 = np.array([[s2,c2,0,_dh.a[1]],[0,0,1,0],[c2,-s2,0,0],[0,0,0,1]])
            T_2_3 = np.array([[c23,-s23,0,_dh.a[2]],[-s23,-c23,0,0],[0,0,-1,0],[0,0,0,1]])

            T_B_1 = T_0_0 @ T_0_1
            T_B_2 = T_B_1 @ T_1_2
            T_B_3 = T_B_2 @ T_2_3

            # 连杆位置: base, J1, J2, J3(J4近似), flange
            link_positions = [
                np.array([0, 0, 0]),           # base
                T_B_1[:3, 3],                   # J1
                T_B_2[:3, 3],                   # J2
                T_B_3[:3, 3],                   # J3
                poses[0][:3, 3],                # flange (用第一个位姿的末端)
            ]
            # 连接线段
            link_segs = [(0,1), (1,2), (2,3), (3,4)]
            for idx_i, idx_j in link_segs:
                p_i, p_j = link_positions[idx_i], link_positions[idx_j]
                ax.plot([p_i[0], p_j[0]], [p_i[1], p_j[1]], [p_i[2], p_j[2]],
                        '-', color='gray', linewidth=3, alpha=0.8)
            # 关节球
            for pos in link_positions:
                ax.scatter(*pos, color='dimgray', s=40, marker='o')
            ax.scatter(*link_positions[0], color='black', s=60, marker='s')
            ax.text(link_positions[2][0], link_positions[2][1], link_positions[2][2],
                    'J3', fontsize=8, color='gray')
            # ---- 法兰末端 (红色大点) ----
            ax.scatter(*link_positions[4], color='red', s=80, marker='o', zorder=5,
                       label='法兰末端')
            ax.text(link_positions[4][0], link_positions[4][1], link_positions[4][2]+0.02,
                    'Flange', fontsize=8, color='red', ha='center')
        except Exception:
            pass  # 机械臂可视化失败不影响主图
    
    # ---- 传感器位姿和扫描线 ----
    n_scans_shown = 0
    for i, (T_B_H, pts_2d) in enumerate(zip(poses, scans)):
        if len(pts_2d) == 0:
            continue
        
        # 传感器在 Base 系中的位姿
        T_B_S = T_B_H @ X_gt
        sensor_pos = T_B_S[:3, 3]
        R_BS = T_B_S[:3, :3]
        
        # 激光扫描线在 Base 系中的 3D 点
        pts_B = []
        for p2 in pts_2d:
            p_S = np.array([p2[0], 0.0, p2[1]])  # 激光平面 y=0, 轮廓点 = (x, z)
            p_B = R_BS @ p_S + sensor_pos
            pts_B.append(p_B)
        pts_B = np.array(pts_B)
        
        # 每 3 条显示一条线 (避免太密)
        if i % max(1, len(poses)//15) == 0:
            ax.plot(pts_B[:, 0], pts_B[:, 1], pts_B[:, 2],
                    '-', color='forestgreen', linewidth=1.5, alpha=0.7)
            # 传感器位置
            ax.scatter(*sensor_pos, color='darkgreen', s=20, marker='o')
            n_scans_shown += 1
    
    # ---- 传感器坐标轴 (仅画第一个) ----
    T_B_S0 = poses[0] @ X_gt
    s0_pos = T_B_S0[:3, 3]
    R0 = T_B_S0[:3, :3]
    axis_len = 0.05
    colors_axis = ['r', 'g', 'b']
    labels_axis = ['X_S', 'Y_S', 'Z_S(光轴)']
    for k in range(3):
        ax.quiver(s0_pos[0], s0_pos[1], s0_pos[2],
                  R0[0,k]*axis_len, R0[1,k]*axis_len, R0[2,k]*axis_len,
                  color=colors_axis[k], linewidth=2, alpha=0.8)
        tip = s0_pos + R0[:, k] * axis_len * 1.3
        ax.text(tip[0], tip[1], tip[2], labels_axis[k],
                color=colors_axis[k], fontsize=9, weight='bold')

    # ---- 线激光传感器小盒子 (第一个位姿, Gocator 2450) ----
    # 传感器尺寸: 约 50×100×150mm (长×宽×高)
    box_half = np.array([0.025, 0.05, 0.075])  # 半尺寸
    corners_local = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],  # 底面
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],       # 顶面
    ]) * box_half
    # 变换到 base 系
    corners_base = R0 @ corners_local.T + s0_pos[:, None]
    corners_base = corners_base.T
    # 12 条边
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for ei, ej in edges:
        ax.plot([corners_base[ei,0], corners_base[ej,0]],
                [corners_base[ei,1], corners_base[ej,1]],
                [corners_base[ei,2], corners_base[ej,2]],
                '-', color='darkgreen', linewidth=2, alpha=0.9)
    # 中心点
    ax.scatter(*s0_pos, color='darkgreen', s=30, marker='o')
    # ---- 传感器原点 (橙色大点) ----
    ax.scatter(*s0_pos, color='orange', s=80, marker='o', zorder=5,
               label='传感器原点')
    ax.text(s0_pos[0], s0_pos[1], s0_pos[2]+0.02,
            'Sensor', fontsize=8, color='orange', ha='center')
    ax.text(s0_pos[0], s0_pos[1], s0_pos[2]-0.02,
            'Gocator 2450', fontsize=8, color='darkgreen', ha='center')
    
    # ---- 种子位姿 (d=300mm, theta=20deg, 绕v轴倾斜, Gocator 2450) ----
    seed_pos = p_B0 + 0.3 * n_B  # 传感器在平板上方300mm
    ax.scatter(*seed_pos, color='blue', s=80, marker='^',
               label='种子位姿 (d=300mm, Gocator 2450)')
    
    # ---- 激光平面三角形 (第一个位姿, 传感器 XOZ 平面) ----
    T0 = poses[0] @ X_gt
    s0_pos = T0[:3, 3]
    R0 = T0[:3, :3]
    x_S, y_S, z_S = R0[:, 0], R0[:, 1], R0[:, 2]
    
    # 三角形: 延伸到 Gocator 2450 测量范围远端 (820mm)
    half_fov_deg = 15
    half_fov = np.deg2rad(half_fov_deg)
    d_tri = 0.82  # 测量范围远端 820mm
    
    # 三角形顶点
    tri_apex = s0_pos
    tri_left = s0_pos + d_tri * (np.cos(half_fov) * z_S - np.sin(half_fov) * x_S)
    tri_right = s0_pos + d_tri * (np.cos(half_fov) * z_S + np.sin(half_fov) * x_S)
    
    # 画三角形边框
    tri_pts = np.array([tri_apex, tri_left, tri_right])
    ax.plot(tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2],
            '-', color='orange', linewidth=1.5, alpha=0.8)
    # 闭合三角形
    ax.plot([tri_apex[0], tri_right[0]], [tri_apex[1], tri_right[1]], [tri_apex[2], tri_right[2]],
            '-', color='orange', linewidth=1.5, alpha=0.8)
    # 填充半透明
    tri_face = Poly3DCollection([np.array([tri_apex, tri_left, tri_right])],
                                 alpha=0.15, facecolor='orange', edgecolor='none')
    ax.add_collection3d(tri_face)
    
    # 标注
    mid = (tri_left + tri_right) / 2
    ax.text(mid[0], mid[1], mid[2], '激光平面\n(XOZ)', color='orange',
            fontsize=9, ha='center', weight='bold')
    
    # 测量范围: 高亮 Gocator 2450 有效测量区域 (截锥体梯形)
    near_z = 0.27; far_z = 0.82
    tan_fov = np.tan(np.deg2rad(15))
    near_x = near_z * tan_fov
    far_x = far_z * tan_fov
    # 梯形四个顶点 (在传感器 XOZ 平面内)
    n_l = s0_pos + near_z * z_S - near_x * x_S  # 近端左
    n_r = s0_pos + near_z * z_S + near_x * x_S  # 近端右
    f_r = s0_pos + far_z * z_S + far_x * x_S    # 远端右
    f_l = s0_pos + far_z * z_S - far_x * x_S    # 远端左
    # 测量范围填充
    meas_poly = Poly3DCollection([np.array([n_l, n_r, f_r, f_l])],
                                  alpha=0.12, facecolor='lime', edgecolor='lime',
                                  linewidth=2)
    ax.add_collection3d(meas_poly)
    # 边框标注
    ax.plot([n_l[0], n_r[0]], [n_l[1], n_r[1]], [n_l[2], n_r[2]],
            '-', color='lime', linewidth=1.5, alpha=0.7, label='测量范围 (270~820mm)')
    ax.plot([f_l[0], f_r[0]], [f_l[1], f_r[1]], [f_l[2], f_r[2]],
            '-', color='lime', linewidth=1.5, alpha=0.7)
    ax.text(n_l[0]-0.02, n_l[1], n_l[2], '近端 270mm', color='lime', fontsize=8)
    ax.text(f_l[0]-0.02, f_l[1], f_l[2], '远端 820mm', color='lime', fontsize=8)
    
    # ---- 高亮第一个位姿的激光平面与校准平面的交线 ----
    pts_first = scans[0]  # 第一个位姿的2D扫描点 (x, z)
    pts_first_B = []
    for p2 in pts_first:
        p_S = np.array([p2[0], 0.0, p2[1]])  # (x, 0, z) in sensor
        p_B = R0 @ p_S + s0_pos
        pts_first_B.append(p_B)
    pts_first_B = np.array(pts_first_B)
    
    # 亮黄色粗线
    ax.plot(pts_first_B[:, 0], pts_first_B[:, 1], pts_first_B[:, 2],
            '-', color='gold', linewidth=4, alpha=1.0, zorder=10,
            label='交线 (第1个位姿)')
    # 端点加粗
    ax.scatter(pts_first_B[0, 0], pts_first_B[0, 1], pts_first_B[0, 2],
               color='gold', s=60, marker='o', zorder=10)
    ax.scatter(pts_first_B[-1, 0], pts_first_B[-1, 1], pts_first_B[-1, 2],
               color='gold', s=60, marker='o', zorder=10)
    
    # ---- 格式 ----
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'标定场景可视化 (显示{n_scans_shown}条扫描线/{len(poses)}个位姿)')
    ax.legend(loc='upper left', fontsize=8)
    
    # 等比例 (三个轴相同范围, 方便判断垂直/平行关系)
    all_pts_for_lim = [p_B0, seed_pos, np.zeros(3)]
    for T, s in zip(poses[:5], scans[:5]):
        T_BS = T @ X_gt
        all_pts_for_lim.append(T_BS[:3, 3])
    all_pts_arr = np.vstack(all_pts_for_lim)
    
    margin = 0.15
    x_range = all_pts_arr[:,0].ptp()  # max-min
    y_range = all_pts_arr[:,1].ptp()
    z_range = all_pts_arr[:,2].ptp()
    half_span = max(x_range, y_range, z_range) / 2.0 + margin
    
    x_mid = all_pts_arr[:,0].mean()
    y_mid = all_pts_arr[:,1].mean()
    z_mid = all_pts_arr[:,2].mean()
    
    ax.set_xlim(x_mid - half_span, x_mid + half_span)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_zlim(z_mid - half_span, z_mid + half_span)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 验证: 运行 `python reproduction_scene.py` 确认场景生成正确
# ============================================================================

def animate_robot_trajectory(scene):
    """动画展示机械臂依次运动到各采集位姿
    
    重写: 独立绘制全部内容, 不依赖 visualize_scene。
    每帧清除-重绘, 兼容 matplotlib 3.x 各后端。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from fanuc_kinematic import get_link_positions, forward_kinematics

    joints = scene.get('joints')
    if joints is None or len(joints) == 0:
        print("动画需要关节角信息 (use_kinematics=True)")
        return

    X_gt = scene['X_gt']
    poses = scene['poses_gt']
    scans = scene['scans_2d']
    n_B = scene['n_B']
    p_B0 = scene['p_B0']
    d = scene['d']

    u_B = scene.get('u_B')
    v_B = scene.get('v_B')
    w = scene.get('w', 0.4)
    h = scene.get('h', 0.5)
    if u_B is None:
        if abs(n_B[2]) < 0.9:
            u_B = np.cross(np.array([0., 0., 1.]), n_B)
        else:
            u_B = np.cross(np.array([1., 0., 0.]), n_B)
        u_B = u_B / np.linalg.norm(u_B)
        v_B = np.cross(n_B, u_B)

    # 预计算所有连杆位置
    all_links = [get_link_positions(j) for j in joints]

    # ===== 运动规划: 关节空间线性插值（含到达停顿） =====
    steps_per_move = 12  # 每两个位姿之间插入 12 帧
    interp_joints = []
    interp_pose_idx = []  # 记录每帧对应的原始位姿编号
    is_arrival = []       # True → 刚到达一个位姿（停顿帧）
    # 第 0 帧：初始位姿（当作一次"到达"）
    interp_joints.append(joints[0].copy())
    interp_pose_idx.append(0)
    is_arrival.append(True)
    for k in range(len(joints) - 1):
        # 运动帧：从 joints[k] 走向 joints[k+1]
        for t in np.linspace(0, 1, steps_per_move, endpoint=False):
            j_interp = joints[k] + t * (joints[k+1] - joints[k])
            interp_joints.append(j_interp)
            interp_pose_idx.append(k)  # 还没到 k+1，扫描线不提前出现
            is_arrival.append(False)
        # 停顿帧：到达 joints[k+1]，扫描线在此帧才出现
        interp_joints.append(joints[k+1].copy())
        interp_pose_idx.append(k + 1)
        is_arrival.append(True)

    all_links = [get_link_positions(j) for j in interp_joints]
    total_frames = len(all_links)
    n_poses_total = len(joints)
    segs = [(0,1),(1,2),(2,3),(3,4),(4,5)]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    half_w, half_h = w / 2, h / 2
    corners = np.array([
        p_B0 - half_w*u_B - half_h*v_B,
        p_B0 + half_w*u_B - half_h*v_B,
        p_B0 + half_w*u_B + half_h*v_B,
        p_B0 - half_w*u_B + half_h*v_B,
    ])

    n_poses_shown = len(poses)  # 显示所有采集线

    for frame_idx in range(total_frames + 1):
        ax.cla()  # 清除全部, 每帧重绘

        # 当前帧的 T_B_H (传感器位置由插值关节角决定)
        if frame_idx < total_frames:
            j_cur = interp_joints[frame_idx]
            T_B_H_cur = forward_kinematics(j_cur)
            pose_idx = interp_pose_idx[frame_idx]
        else:
            # 最后一帧: 显示"完成"
            pass

        # ===== 静态场景 (每帧重绘) =====
        # 校准平面
        tri1 = np.array([corners[0], corners[1], corners[2]])
        tri2 = np.array([corners[0], corners[2], corners[3]])
        ax.add_collection3d(Poly3DCollection(
            [tri1, tri2], alpha=0.25, facecolor='steelblue',
            edgecolor='navy', linewidth=1.5))
        ax.text(p_B0[0], p_B0[1], p_B0[2],
                f'{w*1000:.0f}×{h*1000:.0f}mm', fontsize=9)

        # 法向量
        arrow_len = 0.15
        ax.quiver(p_B0[0], p_B0[1], p_B0[2],
                  n_B[0]*arrow_len, n_B[1]*arrow_len, n_B[2]*arrow_len,
                  color='red', linewidth=3, arrow_length_ratio=0.2,
                  label=f'平面法向量 n_B (d={d:.3f}m)')

        # 基座
        ax.scatter([0], [0], [0], color='black', s=100, marker='s', label='Base')
        ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', alpha=0.5, linewidth=1)
        ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', alpha=0.5, linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', alpha=0.5, linewidth=1)

        # 扫描线 (仅到达位姿时出现新扫描)
        if frame_idx < total_frames:
            if is_arrival[frame_idx]:
                collected = pose_idx  # 到达 → 显示包含当前位姿的所有扫描
            else:
                collected = max(0, pose_idx)  # 运动中只显示已到达的位姿
        else:
            collected = n_poses_total - 1
        for i in range(min(collected + 1, n_poses_shown)):
            T_B_S = poses[i] @ X_gt
            R_BS, t_BS = T_B_S[:3, :3], T_B_S[:3, 3]
            if len(scans[i]) == 0:
                continue
            pts_B = np.array([R_BS @ np.array([p2[0], 0, p2[1]]) + t_BS for p2 in scans[i]])
            ax.plot(pts_B[:,0], pts_B[:,1], pts_B[:,2],
                    '-', color='forestgreen', linewidth=1.5, alpha=0.4)

        # 传感器坐标轴 (当前帧位姿)
        if frame_idx < total_frames:
            T_cur = T_B_H_cur @ X_gt
            s_pos, R_cur = T_cur[:3, 3], T_cur[:3, :3]

            # 坐标轴
            axis_len = 0.05
            for k, (c, lbl) in enumerate(zip(['r','g','b'], ['X_S','Y_S','Z_S'])):
                ax.quiver(s_pos[0], s_pos[1], s_pos[2],
                          R_cur[0,k]*axis_len, R_cur[1,k]*axis_len, R_cur[2,k]*axis_len,
                          color=c, linewidth=2, alpha=0.8)
                tip = s_pos + R_cur[:, k] * axis_len * 1.3
                ax.text(tip[0], tip[1], tip[2], lbl, color=c, fontsize=8)

            # 传感器小盒子
            box_half = np.array([0.025, 0.05, 0.075])
            corners_local = np.array([
                [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
            ]) * box_half
            corners_base = R_cur @ corners_local.T + s_pos[:, None]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for ei, ej in edges:
                ax.plot([corners_base[0,ei], corners_base[0,ej]],
                        [corners_base[1,ei], corners_base[1,ej]],
                        [corners_base[2,ei], corners_base[2,ej]],
                        '-', color='darkgreen', linewidth=2, alpha=0.9)
            ax.scatter(*s_pos, color='orange', s=80, marker='o', zorder=5, label='传感器原点')

            # ===== 当前帧激光平面 & 2D 扫描线 =====
            R_s = R_cur  # 传感器旋转矩阵
            t_s = s_pos
            z_s = R_s[:, 2]
            x_s = R_s[:, 0]

            # 激光平面: 橙色半透明三角形 (类 visualize_scene)
            half_fov = np.deg2rad(15)
            d_far = 0.82
            tri_apex = t_s
            tri_l = t_s + d_far * (np.cos(half_fov) * z_s - np.sin(half_fov) * x_s)
            tri_r = t_s + d_far * (np.cos(half_fov) * z_s + np.sin(half_fov) * x_s)
            tri_pts = np.array([tri_apex, tri_l, tri_r])
            ax.add_collection3d(Poly3DCollection(
                [tri_pts], alpha=0.12, facecolor='orange', edgecolor='orange',
                linewidth=1.5, label='激光平面'))
            ax.plot(tri_pts[:,0], tri_pts[:,1], tri_pts[:,2],
                    '-', color='orange', linewidth=1.5, alpha=0.8)

            # 2D 轮廓: 用对应的原始扫描线, 变换到当前传感器位姿
            scan_2d = scans[pose_idx]
            if len(scan_2d) > 0:
                scan_B = np.array([R_s @ np.array([p2[0], 0, p2[1]]) + t_s for p2 in scan_2d])
                ax.plot(scan_B[:,0], scan_B[:,1], scan_B[:,2],
                        '-', color='gold', linewidth=3, alpha=0.9, zorder=15,
                        label='激光轮廓线')

            # ===== 实时激光-平板交线（红色点线，每帧从当前位姿计算） =====
            try:
                live_scan_2d = simulate_laser_scan(
                    T_B_H_cur, X_gt, n_B, p_B0,
                    u_B=u_B, v_B=v_B, w=w, h=h,
                    half_fov_deg=15.0, min_range=0.27, max_range=0.82
                )
                if live_scan_2d is not None and len(live_scan_2d) > 2:
                    pts_3d = np.array([
                        R_s @ np.array([p[0], 0.0, p[1]]) + s_pos
                        for p in live_scan_2d
                    ])
                    # 降采样到 ~20 个点画点线图
                    step = max(1, len(pts_3d) // 20)
                    ax.plot(pts_3d[::step, 0], pts_3d[::step, 1], pts_3d[::step, 2],
                            ':', color='red', linewidth=2, alpha=0.8, zorder=12,
                            label='实时激光交线')
            except Exception:
                pass  # 计算失败时不显示，不影响动画

        # ===== 当前帧机械臂 =====
        if frame_idx < total_frames:
            links = all_links[frame_idx]
            for i, j in segs:
                ax.plot([links[i,0], links[j,0]],
                        [links[i,1], links[j,1]],
                        [links[i,2], links[j,2]],
                        '-', color='crimson', linewidth=4, alpha=0.9, zorder=20)
            for pi in range(6):
                ax.scatter(*links[pi], color='crimson', s=60, zorder=20)
            ax.scatter(*links[5], color='red', s=120, marker='o', zorder=21, label='法兰末端')
            ax.text(links[5][0], links[5][1], links[5][2]+0.02,
                    'Flange', fontsize=8, color='red', ha='center')

        # ===== 视角和标注 =====
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        if frame_idx < total_frames:
            is_pause = is_arrival[frame_idx]
            if is_pause:
                status = f'[POSE {pose_idx+1}/{n_poses_total}] Acquired -- pause to inspect'
            else:
                progress = int(pose_idx / (n_poses_total - 1) * 100) if n_poses_total > 1 else 100
                status = f'[MOVING to pose {pose_idx+2}/{n_poses_total}] Progress {progress}%'
        else:
            status = f'[DONE] All {n_poses_total} poses acquired'
        ax.set_title(f'机械臂运动轨迹 — {status}')
        ax.legend(loc='upper left', fontsize=8)

        # 坐标系范围 (固定, 避免跳动)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-0.2, 1.2)

        plt.draw()
        # 到达位姿时停顿 1 秒（观察扫描线），运动中快速过渡
        current_pause = 3 if (frame_idx < total_frames and is_arrival[frame_idx]) else 0.05
        plt.pause(current_pause)

    plt.ioff()
    plt.close(fig)  # 关闭动画窗口
    print("动画播放完成，显示静态场景...")
    visualize_scene(scene, max_poses=20, show_robot=True)
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("Zhong et al. 2025 完整标定仿真")
    print("=" * 60)
    print("流程: 数据采集动画 → 静态场景 → 标定求解 → 误差报告")
    print()
    
    np.random.seed(42)
    
    # ===== 1. 生成标定场景（含低噪声） =====
    scene = generate_calibration_scene(
        n_poses=120, seed=42, use_kinematics=True,noise_level='medium'
    )
    
    X_gt = scene['X_gt']
    n_B = scene['n_B']
    d = scene['d']
    
    print(f"手眼真值 T_S_H:")
    print(f"  R_gt:\n{np.round(X_gt[:3,:3], 3)}")
    print(f"  t_gt: {np.round(X_gt[:3,3]*1000, 1)} mm")
    print(f"平面: n_B={np.round(n_B, 3)}, d={d:.3f}m")
    print(f"有效位姿数: {len(scene['poses_gt'])}")
    print(f"噪声等级: {scene.get('noise_level', 'none')}")
    print()
    
    # 验证: 所有理想 3D 点都在校准平面上
    max_dist = 0
    for T_B_H, pts_2d in zip(scene['poses_gt'], scene['scans_2d']):
        T_B_S = T_B_H @ X_gt
        for p2 in pts_2d:
            p_S = np.array([p2[0], 0.0, p2[1]])
            p_B = T_B_S[:3,:3] @ p_S + T_B_S[:3, 3]
            dist = abs(n_B @ p_B - d)
            max_dist = max(max_dist, dist)
    print(f"扫描线平面一致性: {max_dist:.2e}m (应为~0)")
    print("场景生成验证通过" if max_dist < 1e-10 else "验证失败")
    print()
    
    # ===== 2. 播放采集动画 =====
    if 'joints' in scene and len(scene['joints']) > 0:
        print("正在播放机械臂采集动画...")
        animate_robot_trajectory(scene)
    else:
        visualize_scene(scene, max_poses=20)
        plt.show()
    
    # ===== 3. 标定求解（分阶段输出） =====
    print()
    print("=" * 60)
    print("开始标定求解...")
    print("=" * 60)
    
    from reproduction_calib import (
        fast_simulated_annealing, two_step_iterative, compute_errors
    )
    from reproduction_calib import compute_rotation_error, compute_translation_error
    
    # 初始猜测: 在真值上加小扰动 (~5° 旋转, ~20mm 平移)
    # 模拟"粗略标定"后已有大概估计的场景
    np.random.seed(seed=42)  # 不同的种子, 模拟不知情
    R_gt = X_gt[:3, :3]
    omega = np.random.randn(3)
    omega = omega / np.linalg.norm(omega)
    angle_pert = np.deg2rad(np.random.uniform(0, 5))   # 0-5° 扰动
    R_pert = rodrigues(omega, angle_pert)
    R_init = R_pert @ R_gt                               # 从真值旋转扰动
    t_init = X_gt[:3, 3] + np.random.uniform(-0.02, 0.02, 3)
    X_init = np.eye(4)
    X_init[:3, :3] = R_init
    X_init[:3, 3] = t_init
    
    print()
    print("─" * 50)
    print("阶段 0 | 初始猜测")
    print("─" * 50)
    rot0 = compute_rotation_error(X_init[:3,:3], X_gt[:3,:3])
    trans0 = compute_translation_error(X_init[:3,3], X_gt[:3,3])
    print(f"  旋转误差: {rot0:.4f}°")
    print(f"  平移误差: {trans0:.4f} mm")
    print()
    
    # ---- 阶段1: FSA 优化 ----
    print("─" * 50)
    print("阶段 1 | FSA 快速模拟退火...")
    print("─" * 50)
    X_fsa = fast_simulated_annealing(scene['poses_noisy'], scene['scans_2d'], X_init)
    rot1 = compute_rotation_error(X_fsa[:3,:3], X_gt[:3,:3])
    trans1 = compute_translation_error(X_fsa[:3,3], X_gt[:3,3])
    print(f"  FSA 后旋转误差: {rot1:.4f}°  (改善 {rot0-rot1:+.4f}°)")
    print(f"  FSA 后平移误差: {trans1:.4f} mm (改善 {trans0-trans1:+.2f} mm)")
    print()
    
    # ---- 阶段2: 两步迭代 ----
    print("─" * 50)
    print("阶段 2 | 两步迭代精确求解...")
    print("─" * 50)
    X_result, n_iter, converged = two_step_iterative(
        scene['poses_noisy'], scene['scans_2d'],
        X_fsa, n_iter_max=500
    )
    rot2 = compute_rotation_error(X_result[:3,:3], X_gt[:3,:3])
    trans2 = compute_translation_error(X_result[:3,3], X_gt[:3,3])
    status = "收敛" if converged else f"达上限({n_iter}次)"
    print(f"  两步迭代后旋转误差: {rot2:.4f}°  (再改善 {rot1-rot2:+.4f}°)")
    print(f"  两步迭代后平移误差: {trans2:.4f} mm (再改善 {trans1-trans2:+.2f} mm)")
    print()
    
    # ===== 4. 最终汇总 =====
    print()
    print("=" * 60)
    print("标定结果汇总")
    print("=" * 60)
    print(f"  迭代次数: {n_iter}")
    print(f"  收敛状态: {status}")
    print(f"  使用 FSA: 是")
    print(f"  位姿数: {len(scene['poses_noisy'])}")
    print()
    print(f"  {'':>12s} {'旋转误差':>10s}  {'平移误差':>10s}  {'改善贡献':>10s}")
    print(f"  {'─'*42}")
    print(f"  {'初始猜测':>12s}  {rot0:>8.4f}°  {trans0:>7.2f}mm  {'—':>10s}")
    print(f"  {'FSA之后':>12s}  {rot1:>8.4f}°  {trans1:>7.2f}mm  {abs(rot0-rot1):>6.2f}°/{abs(trans0-trans1):>5.1f}mm")
    print(f"  {'最终结果':>12s}  {rot2:>8.4f}°  {trans2:>7.2f}mm  {abs(rot1-rot2):>6.2f}°/{abs(trans1-trans2):>5.1f}mm")
    print()

    print(f"  R_est:\n{np.round(X_result[:3,:3], 4)}")
    print(f"  R_gt:\n{np.round(X_gt[:3,:3], 4)}")
    print(f"  t_est: {np.round(X_result[:3,3]*1000, 2)} mm")
    print(f"  t_gt:  {np.round(X_gt[:3,3]*1000, 2)} mm")
    print()

    np.random.seed(42)




# ============================================================================
# Num2 角点法采集动画
# ============================================================================

def animate_corner_acquisition(scene_data, steps_per_move=12):
    """Num2 角点法数据采集动画 (CODE_REPORT.md §2.7)"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from fanuc_kinematic import get_link_positions, forward_kinematics, inverse_kinematics

    scene = scene_data['scene']
    poses_raw = scene_data['poses']
    meas = scene_data['measurements']

    R_he, t_he = scene['R_he'], scene['t_he']
    C, alpha = scene['C'], scene['alpha']
    u_B, v_B, n_B = scene['u_B'], scene['v_B'], scene['n_B']
    d_1 = u_B
    d_2 = np.cos(alpha) * u_B + np.sin(alpha) * v_B
    pw = scene['plate_w'] / 1000.0
    ph = scene['plate_h'] / 1000.0
    n_poses_total = len(poses_raw)

    X_gt = np.eye(4); X_gt[:3, :3] = R_he; X_gt[:3, 3] = t_he

    # IK 求关节角: 正确的传感器位姿 → 法兰位姿
    joints_poses = []
    for R_i, t_i in poses_raw:
        # 传感器在 Base 系的位姿
        T_BS = np.eye(4)
        T_BS[:3, :3] = R_i @ R_he      # 传感器旋转
        T_BS[:3, 3] = t_i + R_i @ t_he # 传感器位置
        T_BH = T_BS @ np.linalg.inv(X_gt)  # 法兰位姿
        sols = inverse_kinematics(T_BH)
        joints_poses.append(sols[0] if len(sols) > 0 else np.deg2rad([0, 30, -60, 0, 30, 0]))

    # 关节插值
    interp_joints = [joints_poses[0].copy()]
    interp_pose_idx = [0]; is_arrival = [True]
    for k in range(len(joints_poses) - 1):
        for t in np.linspace(0, 1, steps_per_move, endpoint=False):
            interp_joints.append(joints_poses[k] + t * (joints_poses[k+1] - joints_poses[k]))
            interp_pose_idx.append(k); is_arrival.append(False)
        interp_joints.append(joints_poses[k+1].copy())
        interp_pose_idx.append(k+1); is_arrival.append(True)

    all_links = [get_link_positions(j) for j in interp_joints]
    total_frames = len(all_links)

    # --- 实时计算数据结构 (每到达一个位姿时填充) ---
    collected_scan_B = []   # 已采集的扫描线 (N×3 数组列表)
    collected_edge_B = []   # 已采集的断点 ([(etype, pt), ...] 列表)

    # 平板矩形 (C 为角点)
    corners = np.array([C, C + pw * d_1, C + pw * d_1 + ph * d_2, C + ph * d_2])
    tri1 = np.array([corners[0], corners[1], corners[2]])
    tri2 = np.array([corners[0], corners[2], corners[3]])

    # 传感器盒子
    box_half = np.array([0.025, 0.05, 0.075])
    box_local = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * box_half
    box_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    arm_segs = [(0,1),(1,2),(2,3),(3,4),(4,5)]

    all_lim = np.vstack([np.zeros((1,3)), C.reshape(1,3), np.array([t_i for _, t_i in poses_raw]), corners])
    half_span = max(all_lim[:,0].ptp(), all_lim[:,1].ptp(), all_lim[:,2].ptp()) / 2 + 0.3
    x_mid = all_lim[:,0].mean(); y_mid = all_lim[:,1].mean(); z_mid = all_lim[:,2].mean()

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    plt.show(block=False)  # 强制弹出窗口
    collected = 0

    for frame_idx in range(total_frames + 1):
        ax.cla()
        is_last = (frame_idx == total_frames)

        # 平板
        ax.add_collection3d(Poly3DCollection([tri1, tri2], alpha=0.25, facecolor='steelblue', edgecolor='navy', linewidth=1.5, zorder=1))
        # 角点 + 边方向
        ax.scatter(*C, color='gold', s=200, marker='*', zorder=10, edgecolors='darkorange', linewidths=1.5)
        elen = 0.12
        ax.quiver(C[0], C[1], C[2], d_1[0]*elen, d_1[1]*elen, d_1[2]*elen, color='darkblue', linewidth=3, arrow_length_ratio=0.2)
        ax.quiver(C[0], C[1], C[2], d_2[0]*elen, d_2[1]*elen, d_2[2]*elen, color='darkred', linewidth=3, arrow_length_ratio=0.2)
        ax.quiver(C[0], C[1], C[2], n_B[0]*0.15, n_B[1]*0.15, n_B[2]*0.15, color='red', linewidth=3, arrow_length_ratio=0.2)
        ax.scatter([0],[0],[0], color='black', s=100, marker='s', zorder=5)

        if not is_last:
            pose_idx = interp_pose_idx[frame_idx]

            T_B_H = forward_kinematics(interp_joints[frame_idx])
            s_pos = T_B_H[:3, :3] @ t_he + T_B_H[:3, 3]
            R_cur = T_B_H[:3, :3] @ R_he
            x_S, z_S = R_cur[:, 0], R_cur[:, 2]

            if is_arrival[frame_idx]:
                # ---- 统一计算 FOV×平板交线 + 断点 ----
                sl = compute_fov_plate_scanline(
                    R_cur, s_pos, C, n_B, u_B, v_B, pw, ph)
                if sl['has_intersection']:
                    collected_scan_B.append(sl['scan_pts_B'])
                    collected_edge_B.append(sl['endpoints_B'])
                else:
                    collected_scan_B.append(np.zeros((0, 3)))
                    collected_edge_B.append([])

                collected = pose_idx + 1

            # 已采集扫描线 (实时计算的)
            for si in range(min(collected, n_poses_total)):
                if si < len(collected_scan_B) and len(collected_scan_B[si]) > 0:
                    ax.plot(collected_scan_B[si][:,0], collected_scan_B[si][:,1],
                            collected_scan_B[si][:,2], '-', color='forestgreen',
                            linewidth=1.5, alpha=0.4)
                if si < len(collected_edge_B):
                    for etype, pt in collected_edge_B[si]:
                        ax.scatter(*pt, color='cyan' if etype=='e1' else 'magenta',
                                   s=40, marker='s' if etype=='e1' else 'D',
                                   zorder=11, alpha=0.5)

            # 传感器盒子
            cb = R_cur @ box_local.T + s_pos[:, None]; cb = cb.T
            for ei, ej in box_edges:
                ax.plot([cb[ei,0], cb[ej,0]], [cb[ei,1], cb[ej,1]], [cb[ei,2], cb[ej,2]], '-', color='darkgreen', linewidth=2, alpha=0.9)
            ax.scatter(*s_pos, color='orange', s=80, marker='o', zorder=5)

            # FOV 三角形
            hf = np.deg2rad(15); dt = 0.82
            ap = s_pos; tl = s_pos + dt * (np.cos(hf) * z_S - np.sin(hf) * x_S); tr = s_pos + dt * (np.cos(hf) * z_S + np.sin(hf) * x_S)
            for (a,b) in [(ap,tl),(ap,tr),(tl,tr)]:
                ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], '-', color='orange', linewidth=1.5, alpha=0.8)
            ax.add_collection3d(Poly3DCollection([np.array([ap, tl, tr])], alpha=0.15, facecolor='orange', edgecolor='none', zorder=2))

            # ---- FOV 三角与平板的实时交线 (统一函数) ----
            sl_viz = compute_fov_plate_scanline(
                R_cur, s_pos, C, n_B, u_B, v_B, pw, ph)
            if sl_viz['has_intersection']:
                pts = sl_viz['scan_pts_B']
                ax.plot(pts[:,0], pts[:,1], pts[:,2],
                        '-', color='red', linewidth=3, alpha=1.0, zorder=25,
                        label='FOV×Plate')
            # 标注交线状态
            in_plate = sl_viz['has_intersection']
            ax.text2D(0.02, 0.02, f"FOV in plate: {'YES' if in_plate else 'NO'}",
                      transform=ax.transAxes, fontsize=11,
                      color='red' if not in_plate else 'green',
                      weight='bold')

            # 测量范围梯形
            nz, fz = 0.27, 0.82; tf = np.tan(np.deg2rad(15))
            nl = s_pos + nz*z_S - nz*tf*x_S; nr = s_pos + nz*z_S + nz*tf*x_S
            fr = s_pos + fz*z_S + fz*tf*x_S; fl = s_pos + fz*z_S - fz*tf*x_S
            ax.add_collection3d(Poly3DCollection([np.array([nl, nr, fr, fl])], alpha=0.12, facecolor='lime', edgecolor='lime', linewidth=2, zorder=1))

            # 当前帧扫描线 + 断点 (实时计算的)
            if is_arrival[frame_idx] and pose_idx < len(collected_scan_B):
                if len(collected_scan_B[pose_idx]) > 0:
                    ax.plot(collected_scan_B[pose_idx][:,0],
                            collected_scan_B[pose_idx][:,1],
                            collected_scan_B[pose_idx][:,2],
                            '-', color='gold', linewidth=3, alpha=0.9, zorder=15)
                for etype, pt in collected_edge_B[pose_idx]:
                    ax.scatter(*pt, color='cyan' if etype=='e1' else 'magenta',
                               s=100, marker='s' if etype=='e1' else 'D',
                               zorder=20, edgecolors='black', linewidths=1)

        # 机械臂
        if not is_last:
            links = all_links[frame_idx]
            for i, j in arm_segs:
                ax.plot([links[i,0], links[j,0]], [links[i,1], links[j,1]], [links[i,2], links[j,2]], '-', color='crimson', linewidth=4, alpha=0.9, zorder=20)
            for pi in range(6):
                ax.scatter(*links[pi], color='crimson', s=60, zorder=20)
            ax.scatter(*links[5], color='red', s=120, marker='o', zorder=21)

        ax.set_xlim(x_mid - half_span, x_mid + half_span)
        ax.set_ylim(y_mid - half_span, y_mid + half_span)
        ax.set_zlim(z_mid - half_span, z_mid + half_span)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        if is_last: status = f'[DONE] {n_poses_total} poses'
        elif is_arrival[frame_idx]: status = f'[POSE {pose_idx+1}/{n_poses_total}]'
        else: status = f'[MOVE {pose_idx+2}/{n_poses_total}]'
        ax.set_title(f'Corner Method — {status} [终端按Enter继续]', fontsize=12, pad=12)
        plt.draw()
        plt.pause(0.1)  # 刷新窗口
        if not is_last and is_arrival[frame_idx]:
            input(f"  → 到达位姿 {pose_idx+1}/{n_poses_total}, 按 Enter 继续...")
        else:
            plt.pause(0.05)

    plt.ioff()
    plt.close(fig)
    print("动画播放完成")
