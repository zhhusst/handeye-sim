#!/usr/bin/env python3
"""
solvers/cross_12dof.py — 12-DOF C-anchored cross-product 标定

θ = [w_he(3), t_he(3), w_pl(3), C(3)] 共 12 参数
残差: cross(p_e1 - C, u_B) + cross(p_e2 - C, v_B) + n_B·(p_plane - C)

合并 calibrate.py 和 verify_solvers.py 的 C-anchored 版本。
"""

import numpy as np
from handeye_sim.core.so3 import so3_exp, so3_log, skew
from handeye_sim.core.types import CalibResult


def residuals_cross_12dof(theta, poses, meas):
    """C-anchored cross-product residuals

    theta = [w_he(3), t_he(3), w_pl(3), C(3)]
    """
    w_he, t_he, w_pl, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    residuals = []
    for (R_i, t_i), m in zip(poses, meas):
        # 平面约束: n_B · (p_B - C) = 0
        for pS in m.get('p_S_plane', []):
            pB = R_i @ (R_he @ np.asarray(pS) + t_he) + np.asarray(t_i)
            residuals.append(float(n_B @ (pB - C)))

        # 边1: cross(p_e1 - C, u_B) = 0
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            pB_e1 = R_i @ (R_he @ np.asarray(m['p_S_e1']) + t_he) + np.asarray(t_i)
            r = np.cross(pB_e1 - C, u_B)
            residuals.extend(r.tolist())

        # 边2: cross(p_e2 - C, v_B) = 0
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            pB_e2 = R_i @ (R_he @ np.asarray(m['p_S_e2']) + t_he) + np.asarray(t_i)
            r = np.cross(pB_e2 - C, v_B)
            residuals.extend(r.tolist())

    return np.array(residuals)


def solve_cross_12dof_lm(theta_init, poses, meas, max_iter=500):
    """LM for C-anchored cross-product"""
    theta = theta_init.copy().astype(float)
    lam = 1e-4

    for _ in range(max_iter):
        r = residuals_cross_12dof(theta, poses, meas)
        cost = 0.5 * np.dot(r, r)

        eps = 1e-6
        J = np.zeros((len(r), 12))
        for k in range(12):
            sp, sm = theta.copy(), theta.copy()
            sp[k] += eps; sm[k] -= eps
            rp = residuals_cross_12dof(sp, poses, meas)
            rm = residuals_cross_12dof(sm, poses, meas)
            J[:, k] = (rp - rm) / (2 * eps)

        try:
            delta = -np.linalg.solve(J.T @ J + lam * np.eye(12), J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 10; continue

        tn = theta + delta
        rn = residuals_cross_12dof(tn, poses, meas)
        cn = 0.5 * np.dot(rn, rn)

        if cn < cost:
            theta = tn; lam = max(lam / 3, 1e-12)
        else:
            lam = min(lam * 3, 1e6)

        if abs(cost - cn) < 1e-12:
            break

    return theta


def init_cross_12dof(poses, meas, R_he_nom=None, t_he_nom=None):
    """初始化 12-DOF: PCA 估计 u_B, v_B + 边线交点 C"""
    if R_he_nom is None:
        R_cf = np.eye(3)
    else:
        R_cf = np.asarray(R_he_nom)
    if t_he_nom is None:
        t_cf = np.zeros(3)
    else:
        t_cf = np.asarray(t_he_nom)

    # 按法兰朝向分组 (5° 阈值)
    groups = {}  # key: index of representative, value: list of indices
    group_reps = {}  # key: index, value: R_i matrix for that group
    for i, (R_i, _) in enumerate(poses):
        R_i = np.asarray(R_i)
        found = False
        for rep_idx, rep_R in group_reps.items():
            dR = rep_R.T @ R_i
            tr = np.clip((np.trace(dR) - 1) / 2, -1, 1)
            if np.rad2deg(np.arccos(tr)) < 5.0:
                groups[rep_idx].append(i)
                found = True
                break
        if not found:
            groups[i] = [i]
            group_reps[i] = R_i

    # 用名义手眼投影边点
    p_be1, p_be2 = [], []
    for indices in groups.values():
        for idx in indices:
            R_i, t_i = poses[idx]
            m = meas[idx]
            R_BS = R_i @ R_cf; t_BS = np.asarray(t_i) + R_i @ t_cf
            if m.get('valid_e1') and m.get('p_S_e1') is not None:
                p_be1.append(R_BS @ np.asarray(m['p_S_e1']) + t_BS)
            if m.get('valid_e2') and m.get('p_S_e2') is not None:
                p_be2.append(R_BS @ np.asarray(m['p_S_e2']) + t_BS)

    def fit_dir(pts):
        if len(pts) < 2:
            return None
        U, S, Vh = np.linalg.svd(np.array(pts) - np.mean(pts, axis=0), full_matrices=False)
        return Vh[0]

    u_B = fit_dir(p_be1)
    v_B = fit_dir(p_be2)
    n_B = np.cross(u_B, v_B)
    if np.linalg.norm(n_B) > 1e-6:
        n_B /= np.linalg.norm(n_B)
    else:
        n_B = np.array([0., 0., 1.])
    v_B = np.cross(n_B, u_B)
    v_B /= np.linalg.norm(v_B)
    R_pl = np.column_stack([u_B, v_B, n_B])

    # C: 边线交点
    p1r = np.mean(p_be1, axis=0) if p_be1 else np.zeros(3)
    p2r = np.mean(p_be2, axis=0) if p_be2 else np.zeros(3)
    K1, K2, K3 = skew(u_B), skew(v_B), n_B.reshape(1, 3)
    Ac = np.vstack([K1, K2, K3])
    bc = np.hstack([K1 @ p1r, K2 @ p2r, [np.dot(n_B, (p1r + p2r) / 2)]])
    C, _, _, _ = np.linalg.lstsq(Ac, bc, rcond=None)

    return np.concatenate([so3_log(R_cf), t_cf, so3_log(R_pl), C])


def calibrate_12dof_cross(poses, meas, R_he_nom=None, t_he_nom=None,
                           n_restarts=20, seed=42) -> CalibResult:
    """12-DOF C-anchored cross-product 标定 (多重随机重启)

    Returns: CalibResult
    """
    theta_nom = init_cross_12dof(poses, meas, R_he_nom, t_he_nom)
    rng = np.random.RandomState(seed)

    best_cost = float('inf')
    best_theta = None

    for trial in range(n_restarts + 1):
        if trial == 0:
            ti = theta_nom.copy()
        else:
            ti = theta_nom.copy()
            ax = rng.randn(3); ax /= np.linalg.norm(ax)
            ti[0:3] += ax * rng.uniform(0, 0.5)

        to = solve_cross_12dof_lm(ti, poses, meas, max_iter=200)
        cost = 0.5 * np.dot(residuals_cross_12dof(to, poses, meas),
                            residuals_cross_12dof(to, poses, meas))
        if cost < best_cost:
            best_cost = cost
            best_theta = to

    if best_theta is None:
        return CalibResult(method='12dof-cross', R_he=np.eye(3), t_he=np.zeros(3),
                           converged=False, cost=float('inf'))

    from handeye_sim.core.so3 import so3_exp
    R_he = so3_exp(best_theta[0:3])
    t_he = best_theta[3:6]
    R_pl = so3_exp(best_theta[6:9])
    C = best_theta[9:12]

    return CalibResult(
        method='12dof-cross',
        R_he=R_he, t_he=t_he, R_pl=R_pl, C=C, cost=best_cost,
        diagnostics={'theta': best_theta.tolist()},
    )
