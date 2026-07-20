#!/usr/bin/env python3
"""
legacy_solver.py — 旧标定方法迁移

从 Sim/common/calib_solver.py 和 Sim/Num2/ 迁移：
  - combined_solve_lm (9-DOF, plane centered + edge cross-product)
  - solve_12dof_with_restarts (12-DOF, C-anchored cross-product)
  - iterative_refine_he (交替 PCA→LM)

纯 numpy，无 ROS/仿真依赖，可直接在新方法目录中使用。
"""

import sys
import numpy as np

# 复用新方法的 SO(3) 工具
sys.path.insert(0, '/home/z/research_contact_handeye/verification/Sim/new_method')
from observation_model import so3_exp, so3_log, skew


# ══════════════════════════════════════════════════════════════
# 9-DOF: combined_solve_lm (plane centered + edge pairwise cross-product)
# ══════════════════════════════════════════════════════════════

def combined_residuals(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """9-DOF: plane centered + edge collinearity

    theta: [w_he(3), t_he(3), w_pl(3)]
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

    residuals = []
    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)

    for v_ in plane_vals:
        residuals.append(v_ * wp)

    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        residuals.extend((r * we).tolist())

    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        residuals.extend((r * we).tolist())

    return np.array(residuals)


def combined_cost(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    r = combined_residuals(theta, poses, meas, w_plane, w_edge)
    return 0.5 * np.dot(r, r)


def combined_jacobian(theta, poses, meas, w_plane=0.1, w_edge=1.0, eps=1e-6):
    """数值 Jacobian"""
    r0 = combined_residuals(theta, poses, meas, w_plane, w_edge)
    n_params = len(theta)
    J = np.zeros((len(r0), n_params))
    for j in range(n_params):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        rp = combined_residuals(tp, poses, meas, w_plane, w_edge)
        rm = combined_residuals(tm, poses, meas, w_plane, w_edge)
        J[:, j] = (rp - rm) / (2 * eps)
    return J, r0


def combined_solve_lm(theta_init, poses, meas, w_plane=0.1, w_edge=1.0,
                       max_iter=200, tol=1e-12, lam0=1e-6):
    """9-DOF LM"""
    theta = theta_init.copy()
    lam = lam0
    for it in range(max_iter):
        J, r = combined_jacobian(theta, poses, meas, w_plane, w_edge)
        cost = 0.5 * np.dot(r, r)
        H = J.T @ J + lam * np.eye(9)
        g = J.T @ r
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10; continue
        tn = theta + delta
        cn = combined_cost(tn, poses, meas, w_plane, w_edge)
        if cn < cost:
            theta = tn; lam = max(lam/3, 1e-12)
            if abs(cost-cn) < tol: break
        else:
            lam = min(lam*3, 1e6)
    return theta


# ══════════════════════════════════════════════════════════════
# 12-DOF: tilted_corner (cross-product edge, C-anchored)
# ══════════════════════════════════════════════════════════════

def residuals_12dof_cross(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """12-DOF cross-product residuals (C-anchored version from verify_solvers.py M1)

    theta: [w_he(3), t_he(3), w_pl(3), C(3)]
    """
    w_he, t_he, w_pl, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    residuals = []
    for (R_i, t_i), m in zip(poses, meas):
        # 平面约束
        for pS in m.get('p_S_plane', []):
            pB = R_i @ (R_he @ np.asarray(pS) + t_he) + t_i
            residuals.append(float(n_B @ (pB - C)))

        # 边1约束: cross(p_e1 - C, u_B)
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            pB_e1 = R_i @ (R_he @ np.asarray(m['p_S_e1']) + t_he) + t_i
            r = np.cross(pB_e1 - C, u_B)
            residuals.extend(r.tolist())

        # 边2约束: cross(p_e2 - C, v_B)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            pB_e2 = R_i @ (R_he @ np.asarray(m['p_S_e2']) + t_he) + t_i
            r = np.cross(pB_e2 - C, v_B)
            residuals.extend(r.tolist())

    return np.array(residuals)


def solve_12dof_lm(theta_init, poses, meas, max_iter=500, tol=1e-12):
    """12-DOF cross-product LM"""
    theta = theta_init.copy()
    lam = 1e-4
    eps = 1e-6
    for it in range(max_iter):
        r = residuals_12dof_cross(theta, poses, meas)
        cost = 0.5 * np.dot(r, r)
        J = np.zeros((len(r), 12))
        for k in range(12):
            sp = theta.copy(); sp[k] += eps
            sm = theta.copy(); sm[k] -= eps
            rp = residuals_12dof_cross(sp, poses, meas)
            rm = residuals_12dof_cross(sm, poses, meas)
            J[:, k] = (rp - rm) / (2 * eps)
        try:
            delta = -np.linalg.solve(J.T @ J + lam * np.eye(12), J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 10; continue
        tn = theta + delta
        cn = 0.5 * np.dot(residuals_12dof_cross(tn, poses, meas), residuals_12dof_cross(tn, poses, meas))
        if cn < cost:
            theta = tn; lam = max(lam/3, 1e-12)
            if abs(cost-cn) < tol: break
        else:
            lam = min(lam*3, 1e6)
    return theta


def init_12dof_cross(poses, meas):
    """初始化 12-DOF cross-product"""
    p1, p2 = [], []
    for (R_i, t_i), m in zip(poses, meas):
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p1.append(R_i @ np.asarray(m['p_S_e1']) + t_i)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p2.append(R_i @ np.asarray(m['p_S_e2']) + t_i)

    def fit_dir(pts):
        if len(pts) < 2: return np.array([1., 0., 0.])
        return np.linalg.svd(np.array(pts) - np.mean(pts, axis=0))[2][0]

    u_B = fit_dir(p1)
    v_B = fit_dir(p2)
    n_B = np.cross(u_B, v_B)
    if np.linalg.norm(n_B) > 1e-6: n_B /= np.linalg.norm(n_B)
    else: n_B = np.array([0., 0., 1.])
    v_B = np.cross(n_B, u_B); v_B /= np.linalg.norm(v_B)
    R_pl = np.column_stack([u_B, v_B, n_B])

    p1r = np.mean(p1, axis=0) if p1 else np.zeros(3)
    p2r = np.mean(p2, axis=0) if p2 else np.zeros(3)
    sk = lambda v_: np.array([[0,-v_[2],v_[1]],[v_[2],0,-v_[0]],[-v_[1],v_[0],0]])
    Ac = np.vstack([sk(u_B), sk(v_B), n_B.reshape(1,3)])
    bc = np.hstack([sk(u_B)@p1r, sk(v_B)@p2r, [np.dot(n_B, (p1r+p2r)/2)]])
    C, *_ = np.linalg.lstsq(Ac, bc, rcond=None)

    return np.concatenate([np.zeros(3), np.zeros(3), so3_log(R_pl), C])


def solve_12dof_cross_with_restarts(poses, meas, n_restarts=30, seed=42, verbose=False):
    """12-DOF cross-product 多重重启"""
    rng = np.random.RandomState(seed)
    best_cost = float('inf')
    best_theta = None

    theta_nom = init_12dof_cross(poses, meas)
    for trial in range(n_restarts + 1):
        if trial == 0:
            ti = theta_nom.copy()
        else:
            ti = theta_nom.copy()
            ax = rng.randn(3); ax /= np.linalg.norm(ax)
            ti[0:3] += ax * rng.uniform(0, np.pi)

        theta_opt = solve_12dof_lm(ti, poses, meas, max_iter=200)
        r = residuals_12dof_cross(theta_opt, poses, meas)
        cost = 0.5 * np.dot(r, r)
        if cost < best_cost:
            best_cost = cost; best_theta = theta_opt
        if verbose and trial % 10 == 0:
            print(f'  12-DOF restart {trial}: cost={cost:.2e}')

    return best_theta, {'best_cost': best_cost}


# ══════════════════════════════════════════════════════════════
# iterative_refine_he (交替 PCA→LM)
# ══════════════════════════════════════════════════════════════

def iterative_refine_he(poses, meas, R_he_init, t_he_init, max_iter=5, verbose=False):
    """Sharifzadeh 2020 式迭代"""
    R_he = R_he_init.copy()
    t_he = t_he_init.copy()

    for it in range(max_iter):
        all_pts = []
        for (R_i, t_i), m in zip(poses, meas):
            R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he
            for p_S in m.get('p_S_plane', []):
                all_pts.append(R_BS @ np.asarray(p_S) + t_BS)

        if len(all_pts) < 10: break
        all_pts = np.array(all_pts)
        c = all_pts.mean(axis=0)
        _, ev = np.linalg.eigh((all_pts - c).T @ (all_pts - c) / len(all_pts))
        n_B_new = ev[:, 0]; n_B_new /= np.linalg.norm(n_B_new)

        z_S_avg = np.mean([R_i @ R_he[:, 2] for R_i, _ in poses], axis=0)
        if np.dot(n_B_new, z_S_avg) > 0: n_B_new = -n_B_new

        u_B_new = np.array([1.,0.,0.]) if abs(n_B_new[2])<0.9 else np.array([0.,1.,0.])
        u_B_new -= np.dot(u_B_new, n_B_new)*n_B_new
        u_B_new /= np.linalg.norm(u_B_new)+1e-12
        v_B_new = np.cross(n_B_new, u_B_new)
        v_B_new /= np.linalg.norm(v_B_new)+1e-12
        R_pl_new = np.column_stack([u_B_new, v_B_new, n_B_new])

        theta_init = np.zeros(9)
        theta_init[0:3] = so3_log(R_he)
        theta_init[3:6] = t_he
        theta_init[6:9] = so3_log(R_pl_new)

        theta_opt = combined_solve_lm(theta_init, poses, meas, w_plane=0.1, w_edge=1.0, max_iter=100)

        R_he_new = so3_exp(theta_opt[0:3])
        t_he_new = theta_opt[3:6]

        dR = np.linalg.norm(so3_log(R_he_new.T @ R_he))
        dt = np.linalg.norm(t_he_new - t_he)
        if verbose:
            print(f'  iter {it}: |dR|={np.rad2deg(dR):.2f}° |dt|={dt*1000:.2f}mm')
        if dR < 1e-4 and dt < 1e-6:
            return R_he_new, t_he_new, R_pl_new, n_B_new, it+1
        R_he, t_he = R_he_new.copy(), t_he_new.copy()

    return R_he, t_he, R_pl_new, n_B_new, max_iter


# ══════════════════════════════════════════════════════════════
# 误差计算
# ══════════════════════════════════════════════════════════════

def compute_errors_legacy(theta_est, R_he_true, t_he_true):
    """计算 R/t 误差 (兼容 9-DOF 和 12-DOF)"""
    Re = so3_exp(theta_est[0:3])
    Rd = Re.T @ R_he_true
    tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(theta_est[3:6] - t_he_true) * 1000
    return R_err, t_err
