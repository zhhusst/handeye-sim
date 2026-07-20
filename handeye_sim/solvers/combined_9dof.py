#!/usr/bin/env python3
"""
solvers/combined_9dof.py — 9-DOF 平面+边 joint 标定

θ = [w_he(3), t_he(3), w_pl(3)] 共 9 参数
残差: plane centered + edge pairwise cross-product

从 common/calib_solver.py 和 Num2/nbv_edge_plane.py 提取，消除重复。
"""

import numpy as np
from handeye_sim.core.so3 import so3_exp, so3_log, skew
from handeye_sim.core.types import CalibResult


def combined_residuals(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """9-DOF joint residuals

    Args:
        theta: [w_he(3), t_he(3), w_pl(3)]
        poses: [(R_i, t_i), ...]
        meas: [{'valid_e1', 'p_S_e1', 'valid_e2', 'p_S_e2', 'p_S_plane'}, ...]

    Returns:
        r: 残差向量
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
            p_base_e1.append(R_BS @ np.asarray(m['p_S_e1']) + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p_base_e2.append(R_BS @ np.asarray(m['p_S_e2']) + t_BS)

        for p_S in m.get('p_S_plane', []):
            plane_vals.append(np.dot(n_B, R_BS @ np.asarray(p_S) + t_BS))

    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)

    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)
    residuals = list(plane_vals * wp)

    # Edge 1: pairwise cross-product with u_B
    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        residuals.extend((r * we).tolist())

    # Edge 2: pairwise cross-product with v_B
    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        residuals.extend((r * we).tolist())

    return np.array(residuals)


def combined_solve_lm(theta_init, poses, meas, w_plane=0.1, w_edge=1.0,
                       max_iter=100, verbose=False):
    """9-DOF LM 求解器

    Returns: theta_opt [w_he, t_he, w_pl]
    """
    theta = theta_init.copy().astype(float)
    lam = 1e-4
    eps = 1e-6

    for it in range(max_iter):
        r = combined_residuals(theta, poses, meas, w_plane, w_edge)
        cost = 0.5 * np.dot(r, r)
        if verbose and it % 20 == 0:
            print(f"  9dof iter {it}: cost={cost:.4e} λ={lam:.1e}")

        # Numerical Jacobian
        m_dim = len(r)
        J = np.zeros((m_dim, 9))
        for j in range(9):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += eps; tm[j] -= eps
            rp = combined_residuals(tp, poses, meas, w_plane, w_edge)
            rm = combined_residuals(tm, poses, meas, w_plane, w_edge)
            J[:, j] = (rp - rm) / (2 * eps)

        if m_dim < 9:
            break

        try:
            delta = -np.linalg.solve(J.T @ J + lam * np.eye(9), J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 10; continue

        tn = theta + delta
        rn = combined_residuals(tn, poses, meas, w_plane, w_edge)
        cn = 0.5 * np.dot(rn, rn)

        if cn < cost:
            theta = tn; lam = max(lam / 3, 1e-12)
        else:
            lam = min(lam * 3, 1e6)

        if abs(cost - cn) < 1e-12:
            break

    return theta


def calibrate_9dof(poses, meas, R_he_nom=None, t_he_nom=None) -> CalibResult:
    """运行 9-DOF 标定"""
    if R_he_nom is None:
        w_he_init = np.zeros(3)
    else:
        w_he_init = so3_log(np.asarray(R_he_nom))
    if t_he_nom is None:
        t_he_init = np.zeros(3)
    else:
        t_he_init = np.asarray(t_he_nom)

    theta_init = np.concatenate([w_he_init, t_he_init, np.zeros(3)])
    theta_opt = combined_solve_lm(theta_init, poses, meas)

    R_he = so3_exp(theta_opt[0:3])
    t_he = theta_opt[3:6]
    R_pl = so3_exp(theta_opt[6:9])
    cost = 0.5 * np.dot(combined_residuals(theta_opt, poses, meas),
                        combined_residuals(theta_opt, poses, meas))

    return CalibResult(
        method='9dof-combined',
        R_he=R_he, t_he=t_he, R_pl=R_pl, cost=cost,
        diagnostics={'w_he': theta_opt[0:3].tolist(), 'w_pl': theta_opt[6:9].tolist()},
    )
