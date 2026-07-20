#!/usr/bin/env python3
"""
candidate_generator.py — 角点观测约束下的候选位姿生成 (贡献3)

基于《最新思路》Sec 7:
  候选不是从六维位姿空间采样，而是从"激光线穿过两条相邻边"构造。

参数化: (s1, s2, α, ψ, h)
  s1, s2 ∈ [ρ, L-ρ]:  两条边上的截距 (m)
  α ∈ [α_min, α_max]:  激光平面与平板的二面角 (rad)
  ψ ∈ [0, 2π]:  传感器在激光平面内的姿态 (rad)
  h ∈ [h_min, h_max]:  传感器工作距离 (m)

链路:
  (a, b) → L_c → T_BS → T_BF → IK/FOV 检查 → F_k
"""

import numpy as np
from observation_model import so3_exp, so3_log, skew


def generate_line_targets(C, u_B, v_B, n_B, L1=0.4, L2=0.5,
                          rho=0.03, n_samples=(5, 5)):
    """在两条边上生成目标截距对 (a, b)

    a = C + s1*u_B  (边1交点)
    b = C + s2*v_B  (边2交点)

    Args:
        C: 角点坐标 (3,)
        u_B, v_B, n_B: 平板坐标系轴
        L1, L2: 两边长度 (m)
        rho: 安全裕度 (m)
        n_samples: (n_s1, n_s2) 每边的采样数

    Returns:
        lines: [(a, b), ...] 目标线段端点对
    """
    s1_vals = np.linspace(rho, L1 - rho, n_samples[0])
    s2_vals = np.linspace(rho, L2 - rho, n_samples[1])

    lines = []
    for s1 in s1_vals:
        for s2 in s2_vals:
            a = C + s1 * u_B
            b = C + s2 * v_B
            lines.append((a, b))
    return lines


def construct_sensor_pose(a, b, n_B, alpha, psi, h):
    """从目标激光线反求传感器位姿 T_BS

    目标线: L_c(λ) = a + λ(b-a), 方向 d_c = (b-a)/|b-a|
    激光平面法向: m_c = cos(α)·n_B + sin(α)·w_c,  w_c = n_B × d_c
    传感器坐标系构造: y_S = m_c (激光平面法向)
                      z_S = cos(ψ)·g + sin(ψ)·d_c,  g = y_S × d_c
                      x_S = y_S × z_S
    传感器原点: t_BS = p0 - h·z_S,  p0 = (a+b)/2

    Args:
        a, b: 两交点 (3,) each
        n_B: 平板法向 (3,)
        alpha: 二面角 (rad), 不能≈0 (近共面退化)
        psi: 面内姿态 (rad)
        h: 工作距离 (m)

    Returns:
        T_BS: 4×4 齐次变换
    """
    d_c = b - a
    d_len = np.linalg.norm(d_c)
    if d_len < 1e-6:
        return None
    d_c = d_c / d_len

    w_c = np.cross(n_B, d_c)
    w_norm = np.linalg.norm(w_c)
    if w_norm < 1e-6:
        return None
    w_c = w_c / w_norm

    # 激光平面法向
    m_c = np.cos(alpha) * n_B + np.sin(alpha) * w_c
    m_c = m_c / np.linalg.norm(m_c)

    # 传感器坐标系
    y_S = m_c  # 激光平面法向 = 传感器 y 轴
    g = np.cross(y_S, d_c)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-6:
        return None
    g = g / g_norm

    z_S = np.cos(psi) * g + np.sin(psi) * d_c
    z_S = z_S / np.linalg.norm(z_S)
    x_S = np.cross(y_S, z_S)
    x_S = x_S / np.linalg.norm(x_S)

    R_BS = np.column_stack([x_S, y_S, z_S])

    # 传感器原点: 目标线中点向后 h 沿 z_S
    p0 = (a + b) / 2.0
    t_BS = p0 - h * z_S

    T_BS = np.eye(4)
    T_BS[0:3, 0:3] = R_BS
    T_BS[0:3, 3] = t_BS
    return T_BS


def sensor_to_flange(T_BS, R_he_est, t_he_est):
    """从传感器位姿反求法兰位姿

    T_BF = T_BS · (F_T_S)^{-1}

    Args:
        T_BS: 4×4 传感器在基座标系的位姿
        R_he_est, t_he_est: 当前手眼估计

    Returns:
        T_BF: 4×4 法兰基准位姿
    """
    T_FS = np.eye(4)
    T_FS[0:3, 0:3] = R_he_est
    T_FS[0:3, 3] = t_he_est

    T_BF = T_BS @ np.linalg.inv(T_FS)
    return T_BF


def predict_measurement(T_BS, C, u_B, v_B, n_B,
                        n_plane_pts=30, noise_std=0.0):
    """预测候选位姿的传感器测量

    已知平板模型和传感器位姿，预测:
    - 两个边断点 e1_S, e2_S (传感器帧)
    - 平面轮廓点 p_S_plane (传感器帧)

    方法: 从平板边 C+s*u_B / C+s*v_B 逆变换到传感器帧，
    取 y_S≈0 的点即为激光平面与边的交点。

    Returns:
        meas: dict with p_S_e1, p_S_e2, p_S_plane, valid_e1, valid_e2
    """
    R_BS = T_BS[0:3, 0:3]
    t_BS = T_BS[0:3, 3]

    def edge_intersection(direction, d_max):
        """求边 direction 与激光平面 (y_S=0) 的交点"""
        # 在基座标系中, 边点: C + s*direction
        # 变换到传感器帧: q_S = R_BS^T (C + s*d - t_BS)
        # 要求 q_S[1] = 0
        # R_BS^T[1,:] @ (C - t_BS + s*d) = 0
        # s = -R_BS^T[1,:] @ (C - t_BS) / (R_BS^T[1,:] @ d)
        R_T = R_BS.T
        a = R_T[1, :] @ (C - t_BS)
        b_val = R_T[1, :] @ direction
        if abs(b_val) < 1e-10:
            return None
        s = -a / b_val
        if s < 0 or s > d_max:
            return None
        p_B = C + s * direction
        q_S = R_T @ (p_B - t_BS)
        return q_S

    e1_S = edge_intersection(u_B, 1.0)  # 用大的上界, 后面检查FOV
    e2_S = edge_intersection(v_B, 1.0)

    if e1_S is None or e2_S is None:
        return {'p_S_e1': None, 'p_S_e2': None, 'valid_e1': False,
                'valid_e2': False, 'p_S_plane': []}

    # FOV 检查 (Gocator 2450: x ∈ [-0.05, 0.07] m, z ∈ [0.05, 0.5] m)
    x_min, x_max = -0.06, 0.08
    z_min, z_max = 0.03, 0.55
    for q in [e1_S, e2_S]:
        if q[0] < x_min or q[0] > x_max or q[2] < z_min or q[2] > z_max:
            return {'p_S_e1': None, 'p_S_e2': None, 'valid_e1': False,
                    'valid_e2': False, 'p_S_plane': []}

    # 确保 e1 对应边1 (沿u), e2 对应边2 (沿v)
    # e1_S 是 u_B 边的交点, e2_S 是 v_B 边的交点

    # 平面点: 在 e1 和 e2 之间均匀采样
    plane_pts = []
    for k in range(n_plane_pts):
        alpha = k / (n_plane_pts - 1) if n_plane_pts > 1 else 0.5
        q_S = e1_S + alpha * (e2_S - e1_S)
        if noise_std > 0:
            q_S = q_S.copy()
            q_S[0] += np.random.normal(0, noise_std * 0.5e-3)
            q_S[2] += np.random.normal(0, noise_std * 1e-3)
        plane_pts.append(q_S)

    return {
        'p_S_e1': e1_S,
        'p_S_e2': e2_S,
        'valid_e1': True,
        'valid_e2': True,
        'p_S_plane': plane_pts,
    }


def generate_feasible_candidates(C, u_B, v_B, n_B,
                                  R_he_est, t_he_est,
                                  L1=0.4, L2=0.5, rho=0.03,
                                  n_line=(3, 3), n_alpha=3, n_psi=3, n_h=2,
                                  alpha_range=(0.15, 0.6),
                                  psi_range=(0, 2*np.pi),
                                  h_range=(0.15, 0.40)):
    """生成可行候选集合 F_k

    Args:
        C, u_B, v_B, n_B: 当前平板估计
        R_he_est, t_he_est: 当前手眼估计
        L1, L2, rho: 板尺寸和安全裕度
        n_line, n_alpha, n_psi, n_h: 各维度采样数
        alpha_range, psi_range, h_range: 参数范围

    Returns:
        candidates: [dict with 'T_BF', 'meas_pred', 'params', ...]
    """
    lines = generate_line_targets(C, u_B, v_B, n_B,
                                   L1, L2, rho, n_line)
    alpha_vals = np.linspace(*alpha_range, n_alpha)
    psi_vals = np.linspace(*psi_range, n_psi)
    h_vals = np.linspace(*h_range, n_h)

    candidates = []
    for a, b in lines:
        for alpha in alpha_vals:
            for psi in psi_vals:
                for h in h_vals:
                    T_BS = construct_sensor_pose(a, b, n_B, alpha, psi, h)
                    if T_BS is None:
                        continue

                    # 预测测量
                    meas_pred = predict_measurement(
                        T_BS, C, u_B, v_B, n_B,
                        n_plane_pts=20, noise_std=0.0)

                    if not (meas_pred['valid_e1'] and meas_pred['valid_e2']):
                        continue

                    # 反求法兰位姿
                    T_BF = sensor_to_flange(T_BS, R_he_est, t_he_est)
                    R_BF = T_BF[0:3, 0:3]
                    t_BF = T_BF[0:3, 3]

                    candidates.append({
                        'params': (a, b, alpha, psi, h),
                        'T_BS': T_BS,
                        'T_BF': T_BF,
                        'R_i': R_BF,
                        't_i': t_BF,
                        'meas_pred': meas_pred,
                    })

    return candidates


# ── 便捷接口 ────────────────────────────────────────────────

def generate_and_score(candidates, poses_current, meas_current,
                        theta_current, eps_R=0.001, eps_t=1.0):
    """对候选集评分并排序

    对每个候选:
    1. 预测 add 后联合信息 H_{k+c}
    2. Schur 边缘化得 H_X^{eff}
    3. 归一化
    4. 计算 D-opt 和 E-opt 增益

    Args:
        candidates: generate_feasible_candidates 的输出
        poses_current, meas_current: 已采集数据
        theta_current: 当前参数估计 (12,)
        eps_R, eps_t: 目标精度 (rad, mm)

    Returns:
        scored: 按 D-opt 增益降序排列的候选列表
    """
    from observation_model import (
        compute_jacobian, schur_handeye, normalize_hessian,
        d_optimal_gain, e_optimal_gain,
    )

    # 当前联合 Jacobian
    if len(poses_current) > 0:
        J_cur, _, _ = compute_jacobian(theta_current, poses_current, meas_current)
        H_bar_cur = normalize_hessian(schur_handeye(J_cur), eps_R, eps_t)
    else:
        H_bar_cur = np.zeros((6, 6))

    scored = []
    for c in candidates:
        # 构造候选位姿的测量数据
        poses_c = poses_current + [(c['R_i'], c['t_i'])]
        meas_c = meas_current + [c['meas_pred']]

        # 预测信息 (使用当前参数估计)
        J_c, _, info_c = compute_jacobian(theta_current, poses_c, meas_c)

        if J_c.shape[0] == 0:
            continue

        H_eff_c = schur_handeye(J_c)
        if H_eff_c is None:
            continue

        H_bar_c = normalize_hessian(H_eff_c, eps_R, eps_t)

        d_gain = d_optimal_gain(H_bar_cur, H_bar_c)
        e_gain = e_optimal_gain(H_bar_cur, H_bar_c)

        # 简单运动成本 (平移距离)
        if len(poses_current) > 0:
            t_last = poses_current[-1][1]
            move_cost = np.linalg.norm(c['t_i'] - t_last)
        else:
            move_cost = np.linalg.norm(c['t_i'])

        scored.append({
            **c,
            'd_gain': d_gain,
            'e_gain': e_gain,
            'move_cost': move_cost,
            'score': d_gain / max(move_cost, 1e-6),
        })

    scored.sort(key=lambda x: -x['score'])
    return scored
