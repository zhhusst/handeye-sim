#!/usr/bin/env python3
"""
observation_model.py — 标量残差 + 解析 Jacobian

基于《最新思路_完整理论推导与验证方案》
  - Sec 3: 统⼀标量残差模型 r = aᵀ(p - C),  a ∈ {n, v, u}
  - Sec 4: 解析 Jacobian Eq.(22)
  - 手眼旋转: 右扰动  R_X(δ) = R_X exp([δ]_×)
  - 平板旋转: 左扰动  R_P(δ) = exp([δ]_×) R_P

参数顺序: θ = [ω_X(3), t_X(3), ω_P(3), C(3)]^T  (12 DOF)
"""

import numpy as np


# ── SO(3) 基础工具 ──────────────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """反对称矩阵 [v]_×"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def so3_exp(w: np.ndarray) -> np.ndarray:
    """轴角 → SO(3)"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) → 轴角"""
    tr = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(tr)
    if abs(theta) < 1e-12:
        return np.zeros(3)
    return theta / (2 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def dexpm(w: np.ndarray) -> np.ndarray:
    """SO(3) 右 Jacobian (Dexp)

    将轴角增量 δw 映射到切空间增量 δφ:
      R(w + δw) ≈ R(w) exp(Dexp(w) δw)
    即: δφ = Dexp(w) δw

    Dexp(w) = I - (1-cosθ)/θ² [w]_× + (θ-sinθ)/θ³ [w]_×²
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = skew(w / theta)
    a = (1 - np.cos(theta)) / (theta * theta)
    b = (theta - np.sin(theta)) / (theta * theta * theta)
    return np.eye(3) - a * theta * K + b * theta * theta * (K @ K)


def dexpm_inv(w: np.ndarray) -> np.ndarray:
    """SO(3) 右 Jacobian 逆

    Dexp(w)^(-1) = I + (1/2)[w]_× + (1/θ² - (1+cosθ)/(2θ sinθ)) [w]_×²
    """
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    K = skew(w / theta)
    c = (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * theta * K + (1.0/(theta*theta) - c) * theta * theta * (K @ K)


# ── 参数解包 ────────────────────────────────────────────────

def unpack_params(theta: np.ndarray) -> dict:
    """解包 12 维参数向量

    theta = [w_X(3), t_X(3), w_P(3), C(3)]

    Returns:
        dict with R_X, t_X, R_P (=[u, v, n]), u, v, n, C
    """
    w_X, t_X, w_P, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_X = so3_exp(w_X)
    R_P = so3_exp(w_P)
    u, v, n = R_P[:, 0], R_P[:, 1], R_P[:, 2]
    return dict(R_X=R_X, t_X=t_X, R_P=R_P, u=u, v=v, n=n, C=C)


# ── 点变换 ──────────────────────────────────────────────────

def transform_point(q_S: np.ndarray, R_i: np.ndarray, t_i: np.ndarray,
                    R_X: np.ndarray, t_X: np.ndarray) -> np.ndarray:
    """传感器点 q_S → 基座标系 p_B.  Eq.(1)"""
    return R_i @ (R_X @ q_S + t_X) + t_i


# ── 残差计算 ────────────────────────────────────────────────

def compute_residuals(theta: np.ndarray, poses: list, meas: list):
    """计算全部标量残差

    Args:
        theta: [w_X(3), t_X(3), w_P(3), C(3)]
        poses: [(R_i, t_i), ...]  机器人法兰位姿
        meas:  [dict with 'p_S_plane', 'p_S_e1', 'p_S_e2',
                'valid_e1', 'valid_e2'], ...

    Returns:
        r:      残差向量
        r_plane: 平面残差索引列表
        r_e1:    edge 1 残差索引列表
        r_e2:    edge 2 残差索引列表
        info:   {'n_plane', 'n_e1', 'n_e2'}
    """
    p = unpack_params(theta)
    R_X, t_X, u, v, n, C = p['R_X'], p['t_X'], p['u'], p['v'], p['n'], p['C']

    residuals = []
    r_plane, r_e1, r_e2 = [], [], []

    for i, ((R_i, t_i), m) in enumerate(zip(poses, meas)):
        # 平面约束: nᵀ(p - C) = 0  Eq.(12)
        # 端点也加入平面约束 (文档 §3.2 建议)
        all_plane_pts = list(m.get('p_S_plane', []))
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            all_plane_pts.append(m['p_S_e1'])
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            all_plane_pts.append(m['p_S_e2'])

        for q_S in all_plane_pts:
            p_B = transform_point(np.asarray(q_S), R_i, t_i, R_X, t_X)
            r_plane.append(len(residuals))
            residuals.append(float(n @ (p_B - C)))

        # Edge 1 约束: vᵀ(p_e1 - C) = 0  Eq.(13)
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p_B = transform_point(np.asarray(m['p_S_e1']), R_i, t_i, R_X, t_X)
            r_e1.append(len(residuals))
            residuals.append(float(v @ (p_B - C)))

        # Edge 2 约束: uᵀ(p_e2 - C) = 0  Eq.(14)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p_B = transform_point(np.asarray(m['p_S_e2']), R_i, t_i, R_X, t_X)
            r_e2.append(len(residuals))
            residuals.append(float(u @ (p_B - C)))

    res = np.array(residuals)
    info = {'n_plane': len(r_plane), 'n_e1': len(r_e1), 'n_e2': len(r_e2)}
    return res, r_plane, r_e1, r_e2, info


# ── 解析 Jacobian Eq.(22) ──────────────────────────────────

def compute_jacobian(theta: np.ndarray, poses: list, meas: list):
    """计算解析 Jacobian

    Eq.(22): J(a, q) = [-aᵀ R_i R_X [q]_×,  aᵀ R_i,  aᵀ [p-C]_×,  -aᵀ]

    Returns:
        J: (M × 12) Jacobian 矩阵
        r: 残差向量
        info: 残差统计
    """
    r, r_plane, r_e1, r_e2, info = compute_residuals(theta, poses, meas)
    M = len(r)
    J = np.zeros((M, 12))
    p = unpack_params(theta)
    R_X, t_X, u, v, n, C = p['R_X'], p['t_X'], p['u'], p['v'], p['n'], p['C']
    w_X = theta[0:3]   # 手眼轴角 (用于 Dexp 校正)
    w_P = theta[6:9]   # 平板轴角 (用于 Dexp 校正)
    Dexp_X = dexpm(w_X)   # SO(3) 右 Jacobian
    Dexp_P = dexpm(w_P)

    row_idx = 0
    for i_pose, ((R_i, t_i), m) in enumerate(zip(poses, meas)):
        # 平面点: 与 compute_residuals 保持顺序一致
        # compute_residuals: plane_pts..., then e1, then e2
        for q_S in m.get('p_S_plane', []):
            q_S = np.asarray(q_S)
            p_B = transform_point(q_S, R_i, t_i, R_X, t_X)
            a = n
            J[row_idx, 0:3]  = (-a @ R_i @ R_X @ skew(q_S)) @ Dexp_X
            J[row_idx, 3:6]  = a @ R_i
            J[row_idx, 6:9]  = (a @ skew(p_B - C)) @ Dexp_P
            J[row_idx, 9:12] = -a
            row_idx += 1

        # e1 也加入平面约束 (compute_residuals 中在 plane 点之后)
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            q_S = np.asarray(m['p_S_e1'])
            p_B = transform_point(q_S, R_i, t_i, R_X, t_X)
            a = n
            J[row_idx, 0:3]  = (-a @ R_i @ R_X @ skew(q_S)) @ Dexp_X
            J[row_idx, 3:6]  = a @ R_i
            J[row_idx, 6:9]  = (a @ skew(p_B - C)) @ Dexp_P
            J[row_idx, 9:12] = -a
            row_idx += 1

        # e2 也加入平面约束
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            q_S = np.asarray(m['p_S_e2'])
            p_B = transform_point(q_S, R_i, t_i, R_X, t_X)
            a = n
            J[row_idx, 0:3]  = (-a @ R_i @ R_X @ skew(q_S)) @ Dexp_X
            J[row_idx, 3:6]  = a @ R_i
            J[row_idx, 6:9]  = (a @ skew(p_B - C)) @ Dexp_P
            J[row_idx, 9:12] = -a
            row_idx += 1

        # Edge 1 专有约束: a = v
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            q_S = np.asarray(m['p_S_e1'])
            p_B = transform_point(q_S, R_i, t_i, R_X, t_X)
            a = v
            J[row_idx, 0:3]  = (-a @ R_i @ R_X @ skew(q_S)) @ Dexp_X
            J[row_idx, 3:6]  = a @ R_i
            J[row_idx, 6:9]  = (a @ skew(p_B - C)) @ Dexp_P
            J[row_idx, 9:12] = -a
            row_idx += 1

        # Edge 2 专有约束: a = u
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            q_S = np.asarray(m['p_S_e2'])
            p_B = transform_point(q_S, R_i, t_i, R_X, t_X)
            a = u
            J[row_idx, 0:3]  = (-a @ R_i @ R_X @ skew(q_S)) @ Dexp_X
            J[row_idx, 3:6]  = a @ R_i
            J[row_idx, 6:9]  = (a @ skew(p_B - C)) @ Dexp_P
            J[row_idx, 9:12] = -a
            row_idx += 1

    assert row_idx == M, f"Jacobian row count mismatch: {row_idx} != {M}"
    return J, r, info


# ── 数值 Jacobian (用于验证) ────────────────────────────────

def compute_jacobian_numerical(theta: np.ndarray, poses: list, meas: list,
                                eps: float = 1e-6):
    """中心差分 Jacobian — 仅用于验证解析式"""
    r0, _, _, _, info0 = compute_residuals(theta, poses, meas)
    M = len(r0)
    J = np.zeros((M, 12))

    for k in range(12):
        step = np.zeros(12)
        step[k] = eps
        rp, _, _, _, _ = compute_residuals(theta + step, poses, meas)
        rm, _, _, _, _ = compute_residuals(theta - step, poses, meas)
        J[:, k] = (rp - rm) / (2 * eps)

    return J, r0, info0


# ── Schur 补: 边缘化平板参数 ────────────────────────────────

def schur_handeye(J: np.ndarray, W: np.ndarray = None) -> np.ndarray:
    """计算边缘化后的手眼有效信息矩阵 H_X^eff  Eq.(27)

    J 列顺序: [φ_X(3), t_X(3), φ_P(3), C(3)]
    H_X^eff = H_{XX} - H_{XΠ} H_{ΠΠ}^† H_{ΠX}

    Args:
        J: 全 Jacobian (M × 12)
        W: 可选对角权重矩阵 (M × M) 或 None (单位权重)

    Returns:
        H_eff: 6×6 手眼有效信息矩阵
    """
    if W is not None:
        Jw = W @ J
    else:
        Jw = J

    H = Jw.T @ Jw
    # 分块: XX(0:6,0:6), XΠ(0:6,6:12), ΠX(6:12,0:6), ΠΠ(6:12,6:12)
    H_XX = H[0:6, 0:6]
    H_XP = H[0:6, 6:12]
    H_PX = H[6:12, 0:6]
    H_PP = H[6:12, 6:12]

    # Schur complement with pseudo-inverse tolerance
    try:
        H_PP_inv = np.linalg.inv(H_PP)
    except np.linalg.LinAlgError:
        H_PP_inv = np.linalg.pinv(H_PP, hermitian=True)

    H_eff = H_XX - H_XP @ H_PP_inv @ H_PX
    # 对称化（消除数值不对称）
    H_eff = 0.5 * (H_eff + H_eff.T)
    return H_eff


# ── 归一化信息矩阵 Eq.(31) ──────────────────────────────────

def normalize_hessian(H_eff: np.ndarray, eps_R_rad: float,
                       eps_t_mm: float) -> np.ndarray:
    """目标精度归一化  H̄_X = D_Xᵀ H_eff D_X

    Args:
        H_eff:  6×6 手眼有效信息矩阵
        eps_R_rad: 期望旋转精度 (弧度)
        eps_t_mm:  期望平移精度 (毫米)

    Returns:
        H_bar: 归一化 6×6 矩阵
    """
    D = np.diag([eps_R_rad, eps_R_rad, eps_R_rad,
                  eps_t_mm,  eps_t_mm,  eps_t_mm])
    return D @ H_eff @ D


# ── 信息增益 ────────────────────────────────────────────────

def d_optimal_gain(H_bar_current: np.ndarray, H_bar_candidate: np.ndarray,
                    lambda_reg: float = 1e-6) -> float:
    """D-optimal 边际信息增益 Eq.(34)"""
    det_cur = np.linalg.det(H_bar_current + lambda_reg * np.eye(6))
    det_new = np.linalg.det(H_bar_current + H_bar_candidate + lambda_reg * np.eye(6))
    return np.log(det_new) - np.log(det_cur)


def e_optimal_gain(H_bar_current: np.ndarray, H_bar_new: np.ndarray) -> float:
    """E-optimal 弱方向增益 Eq.(35)"""
    eigs_cur = np.sort(np.linalg.eigvalsh(H_bar_current))
    eigs_new = np.sort(np.linalg.eigvalsh(H_bar_new))
    return eigs_new[0] - eigs_cur[0]


# ── 参数归一化 ──────────────────────────────────────────────

def params_to_SE3(theta: np.ndarray) -> tuple:
    """提取手眼和平板 SE(3)"""
    R_X = so3_exp(theta[0:3])
    t_X = theta[3:6]
    R_P = so3_exp(theta[6:9])
    C   = theta[9:12]
    return (R_X, t_X), (R_P, C)


# ── 误差计算 ────────────────────────────────────────────────

def compute_errors(theta_est: np.ndarray, theta_true: np.ndarray):
    """计算 R/t 误差

    Returns:
        R_err_deg, t_err_mm
    """
    (R_est, t_est), _ = params_to_SE3(theta_est)
    (R_true, t_true), _ = params_to_SE3(theta_true)
    tr = np.clip((np.trace(R_est.T @ R_true) - 1) / 2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(t_est - t_true) * 1000
    return R_err, t_err
