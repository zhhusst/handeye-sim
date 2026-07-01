"""
calib_solver.py — plane_edge_9dof 标定求解器

纯 numpy，无 ROS 依赖。可直接在容器内外使用。

依赖:
  - numpy
  - fov_geometry.py (在同一目录下) — 仅用于 so3_exp/so3_log

用法:
  from calib_solver import combined_solve_lm, CombinedResiduals
  
  theta_opt = combined_solve_lm(
      theta_init=np.zeros(9),
      poses=[(R_i, t_i), ...],  # robot hand poses
      measurements=[{'p_S_plane': ..., 'p_S_e1': ..., ...}, ...]
  )
"""

import numpy as np
from fov_geometry import so3_exp, so3_log


def combined_residuals(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """9-DOF: plane centered + edge collinearity
    
    Args:
        theta: [w_he(3), t_he(3), w_pl(3)]
        poses: [(R_i, t_i), ...] 机器人手部位姿
        meas: [{'p_S_plane': (N,3), 'p_S_e1': (3,) or None, 'p_S_e2': (3,) or None, 
                'valid_e1': bool, 'valid_e2': bool}, ...]
        w_plane, w_edge: 权重
    
    Returns:
        r: 残差向量
        mask: bool mask of valid residuals
        info: {'e1_pairs': n, 'e2_pairs': n, 'n_plane': n}
    """
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]
    
    plane_vals = []
    p_base_e1 = []
    p_base_e2 = []
    
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p_base_e1.append(R_BS @ m['p_S_e1'] + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p_base_e2.append(R_BS @ m['p_S_e2'] + t_BS)
        
        for p_S in m.get('p_S_plane', []):
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))
    
    # Plane: global centered (2D line scanner scan lines are 1D per pose,
    # per-pose centering would eliminate all plane information)
    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)
    
    residuals = []
    mask = []
    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)
    info = {'e1_pairs': 0, 'e2_pairs': 0, 'n_plane': len(plane_vals)}
    
    for v in plane_vals:
        residuals.append(v * wp)
        mask.append(True)
    
    # Edge 1 collinearity
    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        residuals.extend((r * we).tolist())
        mask.extend([True, True, True])
        info['e1_pairs'] += 1
    
    # Edge 2 collinearity
    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        residuals.extend((r * we).tolist())
        mask.extend([True, True, True])
        info['e2_pairs'] += 1
    
    return np.array(residuals), np.array(mask), info


def combined_cost(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    r, mask, _ = combined_residuals(theta, poses, meas, w_plane, w_edge)
    return 0.5 * np.dot(r[mask], r[mask])


def combined_jacobian(theta, poses, meas, w_plane=0.1, w_edge=1.0, eps=1e-6):
    """数值 Jacobian"""
    r0, mask, info = combined_residuals(theta, poses, meas, w_plane, w_edge)
    n_params = len(theta)
    J = np.zeros((len(r0), n_params))
    
    for j in range(n_params):
        theta_plus = theta.copy()
        theta_plus[j] += eps
        r_plus, _, _ = combined_residuals(theta_plus, poses, meas, w_plane, w_edge)
        
        theta_minus = theta.copy()
        theta_minus[j] -= eps
        r_minus, _, _ = combined_residuals(theta_minus, poses, meas, w_plane, w_edge)
        
        J[:, j] = (r_plus - r_minus) / (2 * eps)
    
    return J, r0, mask, info


def combined_solve_lm(theta_init, poses, meas, w_plane=0.1, w_edge=1.0,
                      max_iter=200, tol=1e-12, lam0=1e-6):
    """LM 求解器"""
    theta = theta_init.copy()
    lam = lam0
    
    for it in range(max_iter):
        J, r, mask, _ = combined_jacobian(theta, poses, meas, w_plane, w_edge)
        rv = r[mask]
        Jv = J[mask, :]
        
        if len(rv) == 0:
            break
        
        cost = 0.5 * np.dot(rv, rv)
        H = Jv.T @ Jv
        g = Jv.T @ rv
        
        try:
            delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        
        tn = theta + delta
        new_cost = combined_cost(tn, poses, meas, w_plane, w_edge)
        
        if new_cost < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - new_cost) < tol:
                break
        else:
            lam = min(lam * 3, 1e6)
    
    return theta


def compute_errors(theta_est, theta_gt):
    """计算 R/t 误差
    Returns:
        R_err_deg, t_err_mm
    """
    Re = so3_exp(theta_est[0:3])
    Rg = so3_exp(theta_gt[0:3])
    Rd = Re.T @ Rg
    tr = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(theta_est[3:6] - theta_gt[3:6]) * 1000
    return R_err, t_err


def compute_information_matrix(poses, meas, theta, w_plane=0.1, w_edge=1.0, eps=1e-6):
    """计算 Fisher 信息矩阵 I = J^T Σ^{-1} J
    
    Args:
        poses, meas: 标定数据
        theta: 当前参数估计
        w_plane, w_edge: 权重
    
    Returns:
        I: 9×9 信息矩阵
    """
    J, r, mask, _ = combined_jacobian(theta, poses, meas, w_plane, w_edge, eps)
    Jv = J[mask, :]
    # 假设单位权重协方差
    I = Jv.T @ Jv
    return I


def compute_d_optimality(I_current, J_cand):
    """计算候选位姿的 D-optimality 信息增益
    Args:
        I_current: 当前信息矩阵 (9×9)
        J_cand: 候选位姿的 Jacobian (m×9)
    Returns:
        D: det(I_new) / det(I_current), >1 表示增益
    """
    I_new = I_current + J_cand.T @ J_cand
    return np.linalg.det(I_new) / np.linalg.det(I_current)
