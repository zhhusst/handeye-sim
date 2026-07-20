#!/usr/bin/env python3
"""
solvers/iterative_he.py — 交替 PCA→LM 手眼精化

从 Sim/common/calib_solver.py iterative_refine_he 提取。

策略:
  1. 用当前手眼估计投影点云 → 对平面点做 PCA 更新 R_pl
  2. 固定 R_pl 时 LM 优化 R_he, t_he
  3. 重复直到收敛
"""

import numpy as np
from handeye_sim.core.so3 import so3_exp, so3_log, skew
from handeye_sim.core.types import CalibResult


def _residuals_he_only(theta, poses, meas, R_pl):
    """固定 R_pl 时的手眼残差"""
    w_he, t_he = theta[0:3], theta[3:6]
    R_he = so3_exp(w_he)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    plane_vals = []
    p_base_e1, p_base_e2 = [], []

    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he
        t_BS = np.asarray(t_i) + R_i @ t_he

        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p_base_e1.append(R_BS @ np.asarray(m['p_S_e1']) + t_BS)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p_base_e2.append(R_BS @ np.asarray(m['p_S_e2']) + t_BS)
        for p_S in m.get('p_S_plane', []):
            plane_vals.append(np.dot(n_B, R_BS @ np.asarray(p_S) + t_BS))

    residuals = []
    if plane_vals:
        pv = np.array(plane_vals)
        pv = pv - np.mean(pv)
        residuals.extend((pv * 0.316).tolist())

    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        if len(r) >= 3:
            residuals.extend(r[:3].tolist())
    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        if len(r) >= 3:
            residuals.extend(r[:3].tolist())

    return np.array(residuals)


def iterative_refine_he(poses, meas, R_he_init, t_he_init, max_iter=5):
    """交替 PCA→LM 精化手眼外参

    Returns: (R_he, t_he, R_pl, n_B, n_iter)
    """
    R_he = np.asarray(R_he_init).copy()
    t_he = np.asarray(t_he_init).copy()
    R_pl = np.eye(3)

    for it in range(max_iter):
        # PCA: 投影所有平面点到 base 系，估计板法向
        all_pB = []
        pB_e1, pB_e2 = [], []
        for (R_i, t_i), m in zip(poses, meas):
            R_BS = R_i @ R_he
            t_BS = np.asarray(t_i) + R_i @ t_he
            for p_S in m.get('p_S_plane', []):
                all_pB.append(R_BS @ np.asarray(p_S) + t_BS)
            if m.get('valid_e1') and m.get('p_S_e1') is not None:
                pB_e1.append(R_BS @ np.asarray(m['p_S_e1']) + t_BS)
            if m.get('valid_e2') and m.get('p_S_e2') is not None:
                pB_e2.append(R_BS @ np.asarray(m['p_S_e2']) + t_BS)

        # Update R_pl from edge directions and plane normal
        def fit_dir(pts):
            if len(pts) < 2:
                return None
            U, S, Vh = np.linalg.svd(np.array(pts) - np.mean(pts, axis=0), full_matrices=False)
            return Vh[0]

        u_B = fit_dir(pB_e1)
        v_B = fit_dir(pB_e2)
        if u_B is not None and v_B is not None:
            n_B = np.cross(u_B, v_B)
            if np.linalg.norm(n_B) > 1e-6:
                n_B /= np.linalg.norm(n_B)
            v_B = np.cross(n_B, u_B)
            v_B /= np.linalg.norm(v_B)
            R_pl = np.column_stack([u_B, v_B, n_B])
        elif all_pB:
            _, _, Vt = np.linalg.svd(np.array(all_pB) - np.mean(all_pB, axis=0))
            n_B = Vt[2]
            n_B /= np.linalg.norm(n_B)
            u_B = np.array([1., 0., 0.]) if abs(n_B[2]) > 0.9 else np.cross(np.array([0.,0.,1.]), n_B)
            u_B /= np.linalg.norm(u_B)
            v_B = np.cross(n_B, u_B)
            v_B /= np.linalg.norm(v_B)
            R_pl = np.column_stack([u_B, v_B, n_B])

        n_B = R_pl[:, 2]

        # LM 优化 R_he, t_he
        w_he = so3_log(R_he)
        theta = np.concatenate([w_he, t_he])
        lam = 1e-4
        for _ in range(20):
            r = _residuals_he_only(theta, poses, meas, R_pl)
            if len(r) < 6:
                break
            cost = 0.5 * np.dot(r, r)
            eps = 1e-6
            J = np.zeros((len(r), 6))
            for j in range(6):
                tp, tm = theta.copy(), theta.copy()
                tp[j] += eps; tm[j] -= eps
                rp = _residuals_he_only(tp, poses, meas, R_pl)
                rm = _residuals_he_only(tm, poses, meas, R_pl)
                J[:, j] = (rp - rm) / (2 * eps)
            try:
                delta = -np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ r)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            tn = theta + delta
            rn = _residuals_he_only(tn, poses, meas, R_pl)
            cn = 0.5 * np.dot(rn, rn)
            if cn < cost:
                theta = tn; lam = max(lam / 3, 1e-12)
            else:
                lam = min(lam * 3, 1e6)
            if abs(cost - cn) < 1e-12:
                break
        R_he = so3_exp(theta[0:3])
        t_he = theta[3:6]

    return R_he, t_he, R_pl, n_B, it + 1


def calibrate_iterative(poses, meas, R_he_nom=None, t_he_nom=None) -> CalibResult:
    """交替精化标定"""
    if R_he_nom is None:
        R_he_nom = np.eye(3)
    if t_he_nom is None:
        t_he_nom = np.zeros(3)

    R_he, t_he, R_pl, n_B, nit = iterative_refine_he(
        poses, meas, R_he_nom, t_he_nom)

    return CalibResult(
        method='iterative',
        R_he=R_he, t_he=t_he, R_pl=R_pl,
        diagnostics={'n_iter': nit},
    )
