#!/usr/bin/env python3
"""
plane_calib.py — 平面约束手眼标定 + NBV 信息增益评估

核心:
  约束: n_B · (R_BH R_HS p_S + R_BH t_HS + t_BH) - d = 0
  未知参数: R_HS(3), t_HS(3), n_B(2), d(1) = 9 DOF
  NBV:  用 FIM 计算候选姿态的信息增益,选最优下一视角

Yang2023 NBV 框架 + Yang2022 约束方程思想 + 线激光适配
"""

import numpy as np
from fov_geometry import so3_exp, so3_log


# ============================================================
# 平面约束残差
# ============================================================

def _pack_plane_theta(w_he, t_he, theta_n, phi_n, d):
    """打包为 9 参数向量"""
    return np.array([*w_he, *t_he, theta_n, phi_n, d])


def _unpack_plane_theta(theta):
    """解包: w_he(3), t_he(3), theta_n(1), phi_n(1), d(1)"""
    w_he = theta[0:3]
    t_he = theta[3:6]
    theta_n = theta[6]
    phi_n = theta[7]
    d = theta[8]
    return w_he, t_he, theta_n, phi_n, d


def _n_from_angles(theta_n, phi_n):
    """球坐标 → 单位法向量"""
    st, ct = np.sin(theta_n), np.cos(theta_n)
    sp, cp = np.sin(phi_n), np.cos(phi_n)
    return np.array([st * cp, st * sp, ct])


def plane_residuals(theta, poses, scans):
    """
    平面约束残差向量

    theta: [w_he(3), t_he(3), theta_n(1), phi_n(1), d(1)] 共 9 DOF
    poses: [(R_BH_i, t_BH_i), ...]
    scans: [pts_S_i, ...]  每个位姿下的传感器系扫描点 (N_i × 3)

    返回: residuals (M × 1)  M = 所有扫描点的总数
    """
    w_he, t_he, theta_n, phi_n, d = _unpack_plane_theta(theta)
    R_he = so3_exp(w_he)
    n_B = _n_from_angles(theta_n, phi_n)

    residuals = []
    for (R_i, t_i), pts_S in zip(poses, scans):
        if len(pts_S) == 0:
            continue
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        pts_B = (R_BS @ pts_S.T).T + t_BS  # (N, 3)
        r_vals = pts_B @ n_B - d
        residuals.extend(r_vals.tolist())

    return np.array(residuals)


def plane_jacobian(theta, poses, scans, eps=1e-6):
    """数值 Jacobian"""
    r0 = plane_residuals(theta, poses, scans)
    n = len(theta)
    J = np.zeros((len(r0), n))
    for j in range(n):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        rp = plane_residuals(tp, poses, scans)
        rm = plane_residuals(tm, poses, scans)
        J[:, j] = (rp - rm) / (2 * eps)
    return J, r0


def plane_lm(theta_init, poses, scans, max_iter=200, tol=1e-12, lam0=1e-4):
    """LM 优化平面约束"""
    theta = theta_init.copy()
    lam = lam0
    r = plane_residuals(theta, poses, scans)
    best_cost = 0.5 * np.dot(r, r)

    for it in range(max_iter):
        J, r = plane_jacobian(theta, poses, scans)
        cost = 0.5 * np.dot(r, r)
        H = J.T @ J
        g = J.T @ r

        try:
            delta = -np.linalg.solve(H + lam * np.eye(len(theta)), g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        new_theta = theta + delta
        r_new = plane_residuals(new_theta, poses, scans)
        new_cost = 0.5 * np.dot(r_new, r_new)

        if new_cost < cost:
            theta = new_theta
            lam = max(lam / 3, 1e-12)
            if abs(cost - new_cost) < tol:
                break
        else:
            lam = min(lam * 3, 1e6)

    return theta


# ============================================================
# 初始化：从 bootstrap 数据估计板平面
# ============================================================

def init_plane_from_scans(poses, scans, R_he_nominal, t_he_nominal):
    """
    用名义手眼将扫描点转到基座标系 → PCA 拟合平面 → 初始化参数

    返回: theta_init(9) 或 None
    """
    all_pts_B = []
    for (R_i, t_i), pts_S in zip(poses, scans):
        if len(pts_S) < 5:
            continue
        R_BS = R_i @ R_he_nominal
        t_BS = t_i + R_i @ t_he_nominal
        pts_B = (R_BS @ pts_S.T).T + t_BS
        all_pts_B.append(pts_B)

    if not all_pts_B:
        return None

    all_pts = np.vstack(all_pts_B)
    centroid = np.mean(all_pts, axis=0)
    centered = all_pts - centroid

    # PCA: 最小特征值对应的特征向量 = 平面法向
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n_B = eigvecs[:, 0]  # 最小特征值
    d = np.dot(n_B, centroid)

    # 手眼参数初始化 = 名义值 (R_he_nominal 离真值 ~2.8°, 从这出发 LM 可收敛)
    w_he_init = so3_log(R_he_nominal)

    # 法向量 → 球坐标
    theta_n = np.arccos(np.clip(n_B[2], -1, 1))
    phi_n = np.arctan2(n_B[1], n_B[0])

    return _pack_plane_theta(w_he_init, t_he_nominal.copy(), theta_n, phi_n, d)


# ============================================================
# 多重随机重启求解
# ============================================================

def solve_plane_he(poses, scans, R_he_nominal, t_he_nominal,
                   n_restarts=20, seed=42, verbose=False):
    """
    平面约束手眼标定 (多重随机重启)

    返回: (R_he, t_he, info_dict)
    """
    theta_init = init_plane_from_scans(poses, scans, R_he_nominal, t_he_nominal)
    if theta_init is None:
        return None, None, {'error': 'init failed'}

    rng = np.random.RandomState(seed)
    best_cost, best_theta = float('inf'), None

    for trial in range(n_restarts):
        if trial == 0:
            theta = theta_init.copy()
        else:
            # 随机旋转 + 保持平面估计
            axis = rng.randn(3); axis /= np.linalg.norm(axis)
            angle = rng.uniform(0, np.pi)
            w_rand = axis * angle
            theta = theta_init.copy()
            theta[0:3] = w_rand

        theta_opt = plane_lm(theta, poses, scans)
        r = plane_residuals(theta_opt, poses, scans)
        cost = 0.5 * np.dot(r, r)

        if cost < best_cost:
            best_cost = cost
            best_theta = theta_opt

        if verbose and trial % 5 == 0:
            w = best_theta[0:3]
            ang = np.rad2deg(np.linalg.norm(w))
            print(f"  restart {trial}: cost={cost:.2e} R_angle={ang:.1f}°")

    w_he, t_he, tn, pn, d = _unpack_plane_theta(best_theta)
    R_he = so3_exp(w_he)
    n_B = _n_from_angles(tn, pn)

    info = {
        'best_cost': best_cost,
        'n_restarts': n_restarts,
        'R_he': R_he, 't_he': t_he,
        'n_B': n_B, 'd': d,
    }
    return R_he, t_he, info


# ============================================================
# NBV: Fisher Information Matrix & 信息增益
# ============================================================

def compute_fim(theta, poses, scans):
    """
    计算 Fisher Information Matrix (FIM) = J^T J
    用于评估参数估计的不确定性
    """
    J, _ = plane_jacobian(theta, poses, scans)
    return J.T @ J


def parameter_covariance(theta, poses, scans):
    """参数协方差 Σ = (J^T J)^{-1}"""
    FIM = compute_fim(theta, poses, scans)
    try:
        return np.linalg.inv(FIM)
    except np.linalg.LinAlgError:
        # FIM 奇异 → 加小正则化
        return np.linalg.inv(FIM + 1e-8 * np.eye(FIM.shape[0]))


def predict_info_gain(theta_current, scans_current, poses_current,
                       R_BH_new, t_BH_new, pts_S_predicted):
    """
    预测采集候选姿态后的信息增益

    theta_current: 当前最优参数 (9,)
    R_BH_new, t_BH_new: 候选机器人位姿
    pts_S_predicted: 预测的传感器系扫描点 (N × 3)

    返回: info_gain (标量) — 越大越好
    """
    # 当前 FIM 和协方差
    FIM_current = compute_fim(theta_current, poses_current, scans_current)
    try:
        Sigma_current = np.linalg.inv(FIM_current)
    except np.linalg.LinAlgError:
        Sigma_current = np.linalg.inv(FIM_current + 1e-8 * np.eye(9))

    # 预测新数据加入后的 FIM
    poses_new = poses_current + [(R_BH_new, t_BH_new)]
    scans_new = scans_current + [pts_S_predicted]
    FIM_new = compute_fim(theta_current, poses_new, scans_new)
    try:
        Sigma_new = np.linalg.inv(FIM_new)
    except np.linalg.LinAlgError:
        Sigma_new = np.linalg.inv(FIM_new + 1e-8 * np.eye(9))

    # 信息增益 = 对数行列式差 (entropy reduction)
    _, logdet_curr = np.linalg.slogdet(Sigma_current)
    _, logdet_new = np.linalg.slogdet(Sigma_new)

    return logdet_curr - logdet_new  # >0 means uncertainty reduced


# ============================================================
# NBV 候选姿态生成：在安全区域内采样
# ============================================================

def compute_laser_plate_intersection(plate_n_B, plate_d, plate_center,
                                      plate_w, plate_h,
                                      R_BS, t_BS, n_pts=50):
    """
    计算激光面与标定板的交线

    plate_n_B, plate_d: 板平面方程  n_B·p = d
    plate_center: 板中心 (基座标系)
    plate_w, plate_h: 板尺寸
    R_BS, t_BS: 传感器在基座标系的位姿
    n_pts: 沿交线采样点数

    返回:
      pts_S: 传感器系中的预测扫描点, 形状 (n_pts, 3) 或 None (不相交)
      segment_length: 交线段长度 (m)
    """
    # 激光面: 传感器 Y=0 平面 = 法向为 R_BS[:,1]
    n_laser = R_BS[:, 1]
    o_laser = t_BS

    # 检查是否平行
    dot_n = abs(np.dot(n_laser, plate_n_B))
    if dot_n > 0.999:  # 几乎平行 → 无交线或退化
        return None, 0.0

    # 激光面与板平面的交线方向
    line_dir = np.cross(n_laser, plate_n_B)
    line_dir /= np.linalg.norm(line_dir)

    # 交线上一点: 解 n_laser·p = n_laser·o_laser, n_B·p = d
    A = np.array([n_laser, plate_n_B])
    b = np.array([np.dot(n_laser, o_laser), plate_d])
    try:
        p0 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None, 0.0

    # 板的矩形边界（在板平面上）
    # 需要知道板的方向 u_B, v_B — 用 plate_center 和法向无法唯一确定
    # 近似: 用主成分方向或假设板是水平的
    # 简化: 用板中心 + 固定朝向 (从 plate_n_B 推导 u, v)
    if abs(plate_n_B[2]) > 0.9:
        u_B = np.array([1., 0., 0.])
        v_B = np.array([0., 1., 0.])
    else:
        u_B = np.cross(plate_n_B, np.array([0., 0., 1.]))
        u_B /= np.linalg.norm(u_B)
        v_B = np.cross(plate_n_B, u_B)

    # 交线与矩形边的交点
    corners = np.array([
        plate_center - plate_w/2 * u_B - plate_h/2 * v_B,
        plate_center + plate_w/2 * u_B - plate_h/2 * v_B,
        plate_center + plate_w/2 * u_B + plate_h/2 * v_B,
        plate_center - plate_w/2 * u_B + plate_h/2 * v_B,
    ])

    # 交线参数化 p(t) = p0 + t * line_dir
    # 对每条边求解交点参数 t
    t_vals = []
    edges = [(0,1), (1,2), (2,3), (3,0)]
    for i, j in edges:
        a, b = corners[i], corners[j]
        edge_vec = b - a
        # 交线与边 AB 的交点: p0 + t*ld = a + s*(b-a)
        # 2D 线性系统
        M = np.column_stack([line_dir[:2], -edge_vec[:2]])
        if abs(np.linalg.det(M[:2, :2])) < 1e-10:
            continue
        ts = np.linalg.solve(M[:2, :2], (a - p0)[:2])
        t, s = ts[0], ts[1]
        if 0 <= s <= 1:
            t_vals.append(t)

    if len(t_vals) < 2:
        return None, 0.0

    t_min, t_max = min(t_vals), max(t_vals)
    segment_length = abs(t_max - t_min) * np.linalg.norm(line_dir)

    if segment_length < 0.01:  # 交线太短
        return None, segment_length

    # 沿交线采样
    t_samples = np.linspace(t_min, t_max, n_pts)
    pts_B = p0 + np.outer(t_samples, line_dir)  # (n_pts, 3)

    # 转到传感器系
    pts_S = (R_BS.T @ (pts_B - t_BS).T).T
    # 传感器系中 Y 应为 0
    pts_S[:, 1] = 0.0

    return pts_S, segment_length


def generate_nbv_candidates(theta, plate_center, plate_w, plate_h,
                             n_candidates=50, rng=None):
    """
    在安全区域内生成 NBV 候选姿态

    安全区域: 传感器位姿使得激光面与板相交且交线 > 阈值

    返回: [(R_BH, t_BH, pts_S_predicted), ...]
    """
    if rng is None:
        rng = np.random.RandomState()

    w_he, t_he, tn, pn, d = _unpack_plane_theta(theta)
    R_he = so3_exp(w_he)
    n_B = _n_from_angles(tn, pn)

    candidates = []

    # 当前所有位姿的中心(用于生成候选平移范围)
    # 简化: 在板中心上方一定范围内采样
    for _ in range(n_candidates * 3):  # oversample 3x
        if len(candidates) >= n_candidates:
            break

        # 随机传感器朝向
        axis = rng.randn(3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(0, np.pi * 0.6)  # 限制在 0-108° 避免翻转
        R_BS_rot = so3_exp(axis * angle)

        # 传感器在基座标系的朝向
        # 从板法向约束推导: 传感器 Y 轴不平行于板法向即可
        # 简化: 在板中心上方球面上采样
        r_dist = rng.uniform(0.4, 0.8)  # 板到传感器距离 0.4-0.8m
        elev = rng.uniform(0.2, 1.2)  # 仰角
        azim = rng.uniform(0, 2 * np.pi)

        t_BS = plate_center + np.array([
            r_dist * np.cos(elev) * np.cos(azim),
            r_dist * np.cos(elev) * np.sin(azim),
            r_dist * np.sin(elev),
        ])

        # 假设传感器朝向大致指向板中心
        z_axis = plate_center - t_BS
        if np.linalg.norm(z_axis) < 1e-6:
            continue
        z_axis /= np.linalg.norm(z_axis)

        # 构建传感器朝向 (Y=0 平面法向)
        # 传感器 Y 轴应大致朝上 (或水平, 取决于安装)
        y_candidate = np.array([0., 0., 1.])  # 传感器朝天
        if abs(np.dot(y_candidate, z_axis)) > 0.95:
            y_candidate = np.array([1., 0., 0.])

        x_axis = np.cross(y_candidate, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R_BS = np.column_stack([x_axis, y_axis, z_axis])

        # 检查激光面与板是否相交
        pts_S, seg_len = compute_laser_plate_intersection(
            n_B, d, plate_center, plate_w, plate_h, R_BS, t_BS)

        if pts_S is None or seg_len < 0.05:
            continue

        # 传感器 → 法兰 (逆手眼)
        R_BH = R_BS @ R_he.T
        t_BH = t_BS - R_BH @ t_he

        candidates.append((R_BH, t_BH, pts_S))

    return candidates[:n_candidates]


def select_nbv(theta, poses_current, scans_current,
               plate_center, plate_w, plate_h,
               n_candidates=50, rng=None):
    """
    从候选姿态中选择信息增益最大的 NBV

    返回: (R_BH, t_BH, pts_S_predicted, info_gain)
    """
    candidates = generate_nbv_candidates(
        theta, plate_center, plate_w, plate_h, n_candidates, rng)

    if not candidates:
        return None

    best_gain = -1.0
    best = None

    for R_BH, t_BH, pts_S in candidates:
        gain = predict_info_gain(
            theta, scans_current, poses_current, R_BH, t_BH, pts_S)
        if gain > best_gain:
            best_gain = gain
            best = (R_BH, t_BH, pts_S, gain)

    return best


# ============================================================
# 工具: 误差计算
# ============================================================

def compute_plane_errors(R_he_est, t_he_est, n_B_est, d_est,
                          R_he_gt=None, t_he_gt=None,
                          n_B_gt=None, d_gt=None):
    """计算标定误差"""
    info = {}
    if R_he_gt is not None:
        dR = R_he_est.T @ R_he_gt
        ang_err = np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1))
        info['R_err_deg'] = float(np.rad2deg(ang_err))
    if t_he_gt is not None:
        info['t_err_mm'] = float(np.linalg.norm(t_he_est - t_he_gt) * 1000)
    return info
