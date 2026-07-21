#!/usr/bin/env python3
"""
solvers/hybrid_12dof.py — 混合 9→12-DOF 求解器

策略:
  1. 9-DOF pairwise (robust): 得 R_he, t_he, R_pl
  2. 用 R_he 投影所有 e1/e2 到 base 系, 分别拟合直线
  3. 两直线交点 = C (角点)
  4. 固定 R_pl 和 C, 6-DOF LM 精化 R_he + t_he
  5. (可选) 放开全部 12-DOF 做最终调整

优势: C 从 pairwise 边约束中解耦, 不拉偏 t_he
"""

import numpy as np
from handeye_sim.core.so3 import so3_exp, so3_log, skew
from handeye_sim.core.types import CalibResult
from handeye_sim.solvers.combined_9dof import combined_solve_lm, combined_residuals


def _he6_residuals(theta, poses, meas, R_pl, C):
    """6-DOF 残差: 固定 R_pl 和 C, 只优化 w_he, t_he"""
    w_he, t_he = theta[0:3], theta[3:6]
    R_he = so3_exp(w_he)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    r = []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he
        t_BS = np.asarray(t_i) + R_i @ t_he

        for pS in m.get('p_S_plane', []):
            pB = R_BS @ np.asarray(pS) + t_BS
            r.append(float(n_B @ (pB - C)))

        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            pB = R_BS @ np.asarray(m['p_S_e1']) + t_BS
            r.extend(np.cross(pB - C, u_B).tolist())

        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            pB = R_BS @ np.asarray(m['p_S_e2']) + t_BS
            r.extend(np.cross(pB - C, v_B).tolist())

    return np.array(r)


def _lm6(theta_init, poses, meas, R_pl, C, max_iter=100):
    """6-DOF LM: [w_he(3), t_he(3)]"""
    theta = theta_init.copy().astype(float)
    lam, eps = 1e-4, 1e-6

    for _ in range(max_iter):
        r = _he6_residuals(theta, poses, meas, R_pl, C)
        cost = 0.5 * np.dot(r, r)

        J = np.zeros((len(r), 6))
        for j in range(6):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += eps; tm[j] -= eps
            J[:, j] = (_he6_residuals(tp, poses, meas, R_pl, C) -
                       _he6_residuals(tm, poses, meas, R_pl, C)) / (2 * eps)

        try:
            delta = -np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 10; continue

        tn = theta + delta
        cn = 0.5 * np.dot(_he6_residuals(tn, poses, meas, R_pl, C),
                          _he6_residuals(tn, poses, meas, R_pl, C))
        if cn < cost: theta = tn; lam = max(lam/3, 1e-12)
        else: lam = min(lam*3, 1e6)
        if abs(cost - cn) < 1e-12: break

    return theta


def _fit_line_through_points(pts):
    """PCA 拟合 3D 点集 → 直线方向"""
    if len(pts) < 2:
        return np.array([1., 0., 0.]), np.zeros(3)
    pts = np.array(pts)
    c = np.mean(pts, axis=0)
    _, _, Vh = np.linalg.svd(pts - c, full_matrices=False)
    return Vh[0], c


def calibrate_hybrid(poses, meas, R_he_nom=None, t_he_nom=None,
                     refine_full_12dof=False, solver_cfg=None) -> CalibResult:
    """混合 9→12-DOF 标定

    Args:
        refine_full_12dof: 如果 True, 最后一步放开全部 12 DOF
    """
    # ── Step 1: 9-DOF pairwise (robust, 不依赖 C) ──
    R0 = np.asarray(R_he_nom) if R_he_nom is not None else np.eye(3)
    t0 = np.asarray(t_he_nom) if t_he_nom is not None else np.zeros(3)
    theta9_init = np.concatenate([so3_log(R0), t0, np.zeros(3)])
    theta9 = combined_solve_lm(theta9_init, poses, meas, max_iter=100)

    R_he_9 = so3_exp(theta9[0:3])
    t_he_9 = theta9[3:6]
    R_pl_9 = so3_exp(theta9[6:9])
    u_B, v_B, n_B = R_pl_9[:, 0], R_pl_9[:, 1], R_pl_9[:, 2]

    # ── Step 2: 投影边点到 base 系, 拟合直线, 交线算 C ──
    pB_e1, pB_e2 = [], []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he_9
        t_BS = np.asarray(t_i) + R_i @ t_he_9
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            pB_e1.append(R_BS @ np.asarray(m['p_S_e1']) + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            pB_e2.append(R_BS @ np.asarray(m['p_S_e2']) + t_BS)

    dir1, c1 = _fit_line_through_points(pB_e1)
    dir2, c2 = _fit_line_through_points(pB_e2)

    # 两条直线最近点 → 取中点作为 C (理论上的交点)
    # 构造法方程: dir1·s - dir2·t = c2 - c1
    A = np.column_stack([dir1, -dir2])
    b = c2 - c1
    try:
        st = np.linalg.lstsq(A, b, rcond=None)[0]
        C = (c1 + st[0] * dir1 + c2 + st[1] * dir2) / 2
    except:
        C = (c1 + c2) / 2

    # 正交化: 确保 u_B, v_B 跟上一步一致
    u_B_ref = dir1 / np.linalg.norm(dir1)
    n_ref = np.cross(u_B_ref, dir2)
    n_ref /= np.linalg.norm(n_ref)
    v_B_ref = np.cross(n_ref, u_B_ref)
    v_B_ref /= np.linalg.norm(v_B_ref)
    R_pl_ref = np.column_stack([u_B_ref, v_B_ref, n_ref])

    # ── Step 3: 6-DOF LM 精化 R_he + t_he (固定 R_pl 和 C) ──
    theta6_init = np.concatenate([so3_log(R_he_9), t_he_9])
    theta6 = _lm6(theta6_init, poses, meas, R_pl_ref, C, max_iter=200)

    R_he = so3_exp(theta6[0:3])
    t_he = theta6[3:6]

    # ── Step 4 (可选): 放开全部 12 DOF ──
    if refine_full_12dof:
        from handeye_sim.solvers.cross_12dof import solve_cross_12dof_lm
        theta12_init = np.concatenate([so3_log(R_he), t_he,
                                       so3_log(R_pl_ref), C])
        theta12 = solve_cross_12dof_lm(theta12_init, poses, meas, max_iter=200)
        R_he = so3_exp(theta12[0:3])
        t_he = theta12[3:6]
        R_pl_ref = so3_exp(theta12[6:9])
        C = theta12[9:12]

    # ── 评估 ──
    r_final = _he6_residuals(np.concatenate([so3_log(R_he), t_he]),
                             poses, meas, R_pl_ref, C)
    cost = 0.5 * np.dot(r_final, r_final)

    return CalibResult(
        method='hybrid',
        R_he=R_he, t_he=t_he, R_pl=R_pl_ref, C=C, cost=cost,
        diagnostics={
            'R_he_9dof': R_he_9.tolist(), 't_he_9dof': t_he_9.tolist(),
            'C': C.tolist(),
        },
    )
