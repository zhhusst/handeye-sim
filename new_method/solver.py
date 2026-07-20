#!/usr/bin/env python3
"""
solver.py — 12-DOF LM 联合优化

基于《最新思路》Sec 10:
  - 在 SE(3)×SE(3) 上使用 LM
  - 解析 Jacobian
  - Schur 补边缘化
  - 多重重启 + 收敛域分析
"""

import numpy as np
from observation_model import (
    unpack_params, compute_residuals, compute_jacobian,
    schur_handeye, normalize_hessian, compute_errors,
    params_to_SE3, so3_exp, so3_log,
)


def solve_lm(theta_init: np.ndarray, poses: list, meas: list,
             max_iter: int = 200, tol: float = 1e-12,
             lam0: float = 1e-4, verbose: bool = False):
    """LM 求解器

    Args:
        theta_init: 12 维初始参数
        poses, meas: 标定数据
        max_iter, tol, lam0: LM 参数

    Returns:
        theta_opt: 优化后参数
        info: {'cost', 'n_iter', 'H_eff', 'sv_eff', 'converged'}
    """
    theta = theta_init.copy().astype(float)
    lam = lam0

    for it in range(max_iter):
        J, r, info = compute_jacobian(theta, poses, meas)
        cost = 0.5 * np.dot(r, r)

        H = J.T @ J
        g = J.T @ r

        try:
            delta = -np.linalg.solve(H + lam * np.eye(12), g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        theta_new = theta + delta
        r_new, _, _, _, _ = compute_residuals(theta_new, poses, meas)
        cost_new = 0.5 * np.dot(r_new, r_new)

        if cost_new < cost:
            theta = theta_new
            lam = max(lam / 3.0, 1e-12)
            if abs(cost - cost_new) < tol:
                if verbose:
                    print(f"  LM converged at iter {it}, cost={cost:.3e}")
                break
        else:
            lam = min(lam * 3.0, 1e6)

    H_eff = schur_handeye(J)
    sv_eff = np.sort(np.linalg.eigvalsh(H_eff)) if H_eff is not None else None

    return theta, {
        'cost': cost,
        'n_iter': it + 1,
        'H_eff': H_eff,
        'sv_eff': sv_eff,
        'converged': True,
    }


def solve_lm_with_restarts(poses: list, meas: list,
                           theta_nom: np.ndarray = None,
                           n_restarts: int = 30, seed: int = 42,
                           verbose: bool = False):
    """多重随机重启 LM

    第一个重启用名义值 (theta_nom 或其前3个分量为零)。
    后续用随机旋转向量。

    Args:
        poses, meas: 标定数据
        theta_nom: 可选名义参数 (12,)，用于第一个重启
        n_restarts: 随机重启次数
        seed: 随机种子

    Returns:
        best_theta, stats
    """
    rng = np.random.RandomState(seed)
    best_cost = float('inf')
    best_theta = None
    n_converged = 0
    total_trials = n_restarts + (1 if theta_nom is not None else 0)

    trials = []
    if theta_nom is not None:
        trials.append(('nominal', theta_nom.copy()))

    # 随机重启: 扰动手眼旋转在球面上均匀采样
    for _ in range(n_restarts):
        # 均匀随机旋转向量 (球面均匀)
        ax = rng.randn(3)
        ax /= np.linalg.norm(ax)
        angle = rng.uniform(0, np.pi)
        w_rand = ax * angle

        # 用平板初始化（平面点 PCA + 边方向）
        theta_init = init_from_data(poses, meas)
        theta_init[0:3] = w_rand
        # 保持 t_X 为零（让数据决定）
        theta_init[3:6] = np.zeros(3)
        trials.append(('random', theta_init))

    for label, ti in trials:
        try:
            theta_opt, info = solve_lm(ti, poses, meas, max_iter=100)
            r, _, _, _, _ = compute_residuals(theta_opt, poses, meas)
            cost = 0.5 * np.dot(r, r)

            if cost < best_cost:
                best_cost = cost
                best_theta = theta_opt

            if cost < 1e-3:
                n_converged += 1

            if verbose:
                flag = '✓' if cost < 1e-3 else ' '
                print(f"  [{flag}] {label}: cost={cost:.3e}")
        except Exception as e:
            if verbose:
                print(f"  [✗] {label}: {e}")

    stats = {
        'n_trials': total_trials,
        'n_converged': n_converged,
        'best_cost': best_cost,
        'convergence_rate': n_converged / max(total_trials, 1),
    }

    if best_theta is not None:
        # 最终 Jacobian 诊断
        J, r, _ = compute_jacobian(best_theta, poses, meas)
        H_eff = schur_handeye(J)
        sv = np.sort(np.linalg.eigvalsh(H_eff))
        stats.update({
            'rank_X': np.sum(sv > 1e-10),
            'sv_min': sv[0],
            'sv_max': sv[-1],
            'cond': sv[-1] / sv[0] if sv[0] > 1e-15 else float('inf'),
        })

    return best_theta, stats


def init_from_data(poses: list, meas: list) -> np.ndarray:
    """从数据初始化 12-DOF 参数

    1. 名义手眼: 零旋转 + 零平移
    2. 平板姿态: 平面点 PCA → n_B，边点 PCA → u_B
    3. 角点 C: 边线交点估计

    注意: 手眼旋转从零开始（后续由重启覆盖）
    """
    w_X = np.zeros(3)
    t_X = np.zeros(3)

    # ── 板法向: 投影全部平面点到基座标系 → PCA ──
    all_pts = []
    for (R_i, t_i), m in zip(poses, meas):
        for q_S in m.get('p_S_plane', []):
            all_pts.append(R_i @ np.asarray(q_S) + t_i)
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            all_pts.append(R_i @ np.asarray(m['p_S_e1']) + t_i)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            all_pts.append(R_i @ np.asarray(m['p_S_e2']) + t_i)

    if len(all_pts) < 3:
        return np.zeros(12)

    all_pts = np.array(all_pts)
    c = np.mean(all_pts, axis=0)
    _, ev = np.linalg.eigh((all_pts - c).T @ (all_pts - c) / len(all_pts))
    n_B = ev[:, 0]
    n_B /= np.linalg.norm(n_B)
    # 确保 n_B 指向传感器侧
    z_avg = np.mean([R_i[:, 2] for R_i, _ in poses], axis=0)
    if np.dot(n_B, z_avg) > 0:
        n_B = -n_B

    # ── 边方向 ──
    e1_pts, e2_pts = [], []
    for (R_i, t_i), m in zip(poses, meas):
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            e1_pts.append(R_i @ np.asarray(m['p_S_e1']) + t_i)
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            e2_pts.append(R_i @ np.asarray(m['p_S_e2']) + t_i)

    def fit_direction(pts):
        if len(pts) < 2:
            return None
        arr = np.array(pts)
        _, _, Vt = np.linalg.svd(arr - arr.mean(axis=0), full_matrices=False)
        return Vt[0] / np.linalg.norm(Vt[0])

    u_B = fit_direction(e1_pts)
    v_B = fit_direction(e2_pts)

    if u_B is None:
        u_B = np.cross(np.array([0., 0., 1.]), n_B)
        u_B /= max(np.linalg.norm(u_B), 1e-12)
    if v_B is None:
        v_B = np.cross(n_B, u_B)
        v_B /= max(np.linalg.norm(v_B), 1e-12)

    # 正交化: u_B 投影到平面 → v_B = n × u
    u_B -= np.dot(u_B, n_B) * n_B
    u_B /= np.linalg.norm(u_B)
    v_B = np.cross(n_B, u_B)
    v_B /= np.linalg.norm(v_B)

    R_P = np.column_stack([u_B, v_B, n_B])
    w_P = so3_log(R_P)

    # ── 角点 C: 边线交点 ──
    s = lambda v: np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    p1_mean = np.mean(e1_pts, axis=0) if e1_pts else np.zeros(3)
    p2_mean = np.mean(e2_pts, axis=0) if e2_pts else np.zeros(3)

    A = np.vstack([s(u_B), s(v_B), n_B.reshape(1, 3)])
    b_vec = np.hstack([s(u_B) @ p1_mean, s(v_B) @ p2_mean, [np.dot(n_B, (p1_mean + p2_mean) / 2)]])
    C, *_ = np.linalg.lstsq(A, b_vec, rcond=None)

    return np.concatenate([w_X, t_X, w_P, C])


# ── 便捷接口 ────────────────────────────────────────────────

def calibrate(poses: list, meas: list,
              R_he_init=None, t_he_init=None,
              verbose: bool = True):
    """一键标定 (单次求解, 不重启)

    Args:
        poses, meas: 标定数据
        R_he_init: 初始手眼旋转 (3×3), 默认用 init_from_data 估计
        t_he_init: 初始手眼平移 (3,),  默认用 init_from_data 估计

    Returns:
        theta_opt, stats
    """
    if verbose:
        n_e1 = sum(1 for m in meas if m.get('valid_e1'))
        n_e2 = sum(1 for m in meas if m.get('valid_e2'))
        n_plane = sum(len(m.get('p_S_plane', [])) for m in meas)
        print(f"数据: {len(poses)} 位姿, e1={n_e1}, e2={n_e2}, plane_pts={n_plane}")

    # 初始化
    theta_init = init_from_data(poses, meas)
    if R_he_init is not None:
        theta_init[0:3] = so3_log(R_he_init)
    if t_he_init is not None:
        theta_init[3:6] = t_he_init

    theta_opt, info = solve_lm(theta_init, poses, meas, max_iter=200, verbose=verbose)
    r_final, _, _, _, _ = compute_residuals(theta_opt, poses, meas)
    cost_final = 0.5 * np.dot(r_final, r_final)

    # Jacobian 诊断
    from observation_model import compute_jacobian, schur_handeye
    J, _, _ = compute_jacobian(theta_opt, poses, meas)
    H_eff = schur_handeye(J)
    sv = np.sort(np.linalg.eigvalsh(H_eff))

    stats = {
        'best_cost': cost_final,
        'n_iter': info['n_iter'],
        'converged': info['converged'],
        'rank_X': int(np.sum(sv > 1e-10)),
        'sv_min': sv[0],
        'sv_max': sv[-1],
        'cond': sv[-1] / sv[0] if sv[0] > 1e-15 else float('inf'),
    }

    return theta_opt, stats
