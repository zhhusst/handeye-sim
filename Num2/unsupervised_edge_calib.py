#!/usr/bin/env python3
"""
unsupervised_edge_calib.py — 点集聚类 + 联合优化

核心思想：不依赖 e1/e2 标签，让数据自动聚类。
方法：EM-like 迭代 — 聚类(E-step) → LM优化(M-step) → 重复。

约束：每个位姿的两个端点必须在不同边上（FOV三角形穿板自然保证）。
"""

import numpy as np
from corner_scene import so3_exp, so3_log
from nbv_edge_plane import (
    combined_residuals, combined_cost, combined_jacobian,
    combined_solve_lm, combined_errors
)


def cluster_endpoints(endpoints_B, u_B, v_B):
    """将端点分成两组，使组内共线残差最小。

    Args:
        endpoints_B: list of [p_i^A, p_i^B] for each pose
        u_B, v_B: 当前板方向估计 (3,), (3,)

    Returns:
        groups: {'u': [p0, p1, ...], 'v': [p0, p1, ...]}
        assignments: [int] 每个位姿的赋值 (0=p_A→u_B, 1=p_A→v_B)
    """
    n_poses = len(endpoints_B)
    best_cost = np.inf
    best_assign = None

    # 固定 pose 0 赋值 (u_B/v_B 对称, 只需试 2^(n-1) 种)
    for mask in range(1 << (n_poses - 1)):
        assign = [0]
        for i in range(1, n_poses):
            assign.append((mask >> (i - 1)) & 1)

        # 构建两组
        u_pts, v_pts = [], []
        for i in range(n_poses):
            if assign[i] == 0:
                u_pts.append(endpoints_B[i][0])
                v_pts.append(endpoints_B[i][1])
            else:
                u_pts.append(endpoints_B[i][1])
                v_pts.append(endpoints_B[i][0])

        # 组内共线代价
        cost = 0.0
        for i in range(len(u_pts)):
            for j in range(i + 1, len(u_pts)):
                cost += np.linalg.norm(np.cross(u_pts[j] - u_pts[i], u_B))
                cost += np.linalg.norm(np.cross(v_pts[j] - v_pts[i], v_B))

        if cost < best_cost:
            best_cost = cost
            best_assign = assign.copy()

    # 构建最佳分组
    u_pts, v_pts = [], []
    for i in range(n_poses):
        if best_assign[i] == 0:
            u_pts.append(endpoints_B[i][0])
            v_pts.append(endpoints_B[i][1])
        else:
            u_pts.append(endpoints_B[i][1])
            v_pts.append(endpoints_B[i][0])

    return {'u': u_pts, 'v': v_pts}, best_assign


def plane_only_residuals(theta, poses, measurements):
    """仅平面残差 (centered) — 用于初始化 R_pl"""
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    n_B = R_pl[:, 2]

    plane_vals = []
    for (R_i, t_i), m in zip(poses, measurements):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        for p_S in m['p_S_plane']:
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))

    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)
    return plane_vals


def plane_only_cost(theta, poses, measurements):
    r = plane_only_residuals(theta, poses, measurements)
    return 0.5 * np.dot(r, r) if len(r) > 0 else 1e30


def plane_only_jacobian(theta, poses, measurements, eps=1e-6):
    r0 = plane_only_residuals(theta, poses, measurements)
    J = np.zeros((len(r0), 9))
    for k in range(9):
        step = np.zeros(9); step[k] = eps
        rp = plane_only_residuals(theta + step, poses, measurements)
        rm = plane_only_residuals(theta - step, poses, measurements)
        J[:, k] = (rp - rm) / (2 * eps)
    return J, r0


def plane_only_solve_lm(theta_init, poses, measurements, max_iter=50, tol=1e-8, lam0=1e-4):
    theta = theta_init.copy()
    lam = lam0
    prev_cost = np.inf
    for it in range(max_iter):
        J, r = plane_only_jacobian(theta, poses, measurements)
        if len(r) == 0:
            break
        cost = 0.5 * np.dot(r, r)
        H = J.T @ J
        g = J.T @ r
        try:
            delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue
        tn = theta + delta
        cn = plane_only_cost(tn, poses, measurements)
        if cn < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - cn) < tol:
                break
            prev_cost = cost
        else:
            lam = min(lam * 3, 1e6)
    return theta


def unsupervised_edge_residuals(theta, poses, measurements, groups, w_plane=1.0, w_edge=1.0):
    """带分组的边缘共线残差 + 平面残差。

    Args:
        theta: 9-DOF [w_he(3), t_he(3), w_pl(3)]
        poses, measurements: 标准格式
        groups: {'u': [p_0, ...], 'v': [p_0, ...]} 聚类结果
    """
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    # 平面残差
    plane_vals = []
    for (R_i, t_i), m in zip(poses, measurements):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        for p_S in m['p_S_plane']:
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))

    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)

    # 边缘共线残差 — 基于聚类结果
    edge_res = []
    u_pts = groups['u']
    v_pts = groups['v']

    # u_B 组：相邻端点差应平行于 u_B
    for k in range(len(u_pts) - 1):
        r = np.cross(u_pts[k + 1] - u_pts[k], u_B)
        edge_res.extend(r.tolist())

    # v_B 组：相邻端点差应平行于 v_B
    for k in range(len(v_pts) - 1):
        r = np.cross(v_pts[k + 1] - v_pts[k], v_B)
        edge_res.extend(r.tolist())

    # 合并
    residuals = []
    mask = []
    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)

    for v in plane_vals:
        residuals.append(v * wp)
        mask.append(True)
    for v in edge_res:
        residuals.append(v * we)
        mask.append(True)

    return np.array(residuals), np.array(mask)


def unsupervised_cost(theta, poses, measurements, groups, w_plane=1.0, w_edge=1.0):
    r, m = unsupervised_edge_residuals(theta, poses, measurements, groups, w_plane, w_edge)
    rv = r[m]
    return 0.5 * np.dot(rv, rv) if len(rv) > 0 else 1e30


def unsupervised_jacobian(theta, poses, measurements, groups, w_plane=1.0, w_edge=1.0, eps=1e-6):
    r0, mask = unsupervised_edge_residuals(theta, poses, measurements, groups, w_plane, w_edge)
    J = np.zeros((len(r0), 9))
    for k in range(9):
        step = np.zeros(9); step[k] = eps
        rp, _ = unsupervised_edge_residuals(theta + step, poses, measurements, groups, w_plane, w_edge)
        rm, _ = unsupervised_edge_residuals(theta - step, poses, measurements, groups, w_plane, w_edge)
        J[:, k] = (rp - rm) / (2 * eps)
    return J, r0, mask


def unsupervised_solve_lm(theta_init, poses, measurements, groups, w_plane=1.0, w_edge=1.0,
                           max_iter=80, tol=1e-8, lam0=1e-4):
    theta = theta_init.copy()
    lam = lam0
    for it in range(max_iter):
        J, r, mask = unsupervised_jacobian(theta, poses, measurements, groups, w_plane, w_edge)
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
        cn = unsupervised_cost(tn, poses, measurements, groups, w_plane, w_edge)
        if cn < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - cn) < tol:
                break
        else:
            lam = min(lam * 3, 1e6)
    return theta


def build_endpoints_B(theta_current, poses, measurements):
    """从当前手眼估计计算所有端点在基座标系的位置。

    Returns:
        endpoints_B: list of [p_i^A, p_i^B] for each pose (未标记)
    """
    w_he, t_he, w_pl = theta_current[0:3], theta_current[3:6], theta_current[6:9]
    R_he = so3_exp(w_he)
    # w_pl not needed for endpoint transform, only R_he matters

    endpoints_B = []
    for (R_i, t_i), m in zip(poses, measurements):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        eps = []
        if m['valid_e1']:
            eps.append(R_BS @ m['p_S_e1'] + t_BS)
        if m['valid_e2']:
            eps.append(R_BS @ m['p_S_e2'] + t_BS)
        if len(eps) == 2:
            endpoints_B.append(tuple(eps))
    return endpoints_B


def init_from_endpoint_pca(poses, measurements):
    """用端点差分方向 PCA 初始化 u_B/v_B。

    原理：同一组端点之间的差分向量沿边方向。
    对所有端点做 PCA → 前两个主方向 = u_B, v_B (大致)。
    """
    all_diffs = []
    endpoints_B = build_endpoints_B(np.zeros(9), poses, measurements)
    if len(endpoints_B) < 2:
        return None

    # 收集所有端点之间的差分
    all_pts = []
    for ea, eb in endpoints_B:
        all_pts.append(ea)
        all_pts.append(eb)

    all_pts = np.array(all_pts)
    # PCA: 前两个主成分是 u_B, v_B 的近似方向
    center = np.mean(all_pts, axis=0)
    centered = all_pts - center
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt[0], Vt[1] 是板平面内两个主方向
    dir1 = Vt[0] / np.linalg.norm(Vt[0])
    dir2 = Vt[1] / np.linalg.norm(Vt[1])

    return dir1, dir2


def solve_unsupervised(theta_init, poses, measurements, n_em_iters=5,
                        w_plane=1.0, w_edge=1.0):
    """EM-like 联合优化。

    初始化:
      1. 平面 LM → 得到合理的 R_he, t_he, n_B
      2. 用平面 LM 的 R_he/t_he 计算端点 → PCA → u_B/v_B 方向
      3. 组合成完整初始 theta
    主循环:
      E-step: 用当前估计聚类端点
      M-step: 用聚类结果做 9-DOF LM
    """
    theta = theta_init.copy()

    # Phase 1: 平面 LM → 精化 R_he, t_he, n_B
    theta = plane_only_solve_lm(theta, poses, measurements, max_iter=50)

    # Phase 2: 用平面 LM 结果计算端点 → PCA → u_B/v_B
    # (关键：必须用合理的 R_he/t_he 算端点，否则 PCA 方向不对)
    endpoints_B_pl = build_endpoints_B(theta, poses, measurements)
    if len(endpoints_B_pl) >= 2:
        all_pts = []
        for ea, eb in endpoints_B_pl:
            all_pts.append(ea); all_pts.append(eb)
        all_pts = np.array(all_pts)
        center = np.mean(all_pts, axis=0)
        _, s_pca, Vt_pca = np.linalg.svd(all_pts - center, full_matrices=False)
        if s_pca[1] > 1e-6:
            dir1 = Vt_pca[0] / np.linalg.norm(Vt_pca[0])
            dir2 = Vt_pca[1] / np.linalg.norm(Vt_pca[1])
            n_B_pca = np.cross(dir1, dir2)
            n_B_pca /= np.linalg.norm(n_B_pca)
            # 保证 n_B 方向与平面 LM 结果一致
            w_pl = theta[6:9]
            R_pl_pl = so3_exp(w_pl)
            if np.dot(n_B_pca, R_pl_pl[:, 2]) < 0:
                n_B_pca = -n_B_pca
            u_B_pca = dir1
            v_B_pca = np.cross(n_B_pca, u_B_pca)
            v_B_pca /= np.linalg.norm(v_B_pca)
            R_pl_pca = np.column_stack([u_B_pca, v_B_pca, n_B_pca])
            theta[6:9] = so3_log(R_pl_pca)

    w_pl = theta[6:9]
    R_pl = so3_exp(w_pl)
    u_B, v_B = R_pl[:, 0], R_pl[:, 1]

    best_theta = theta.copy()
    best_cost = np.inf

    for em_it in range(n_em_iters):
        # E-step: 计算端点位置 → 聚类
        endpoints_B = build_endpoints_B(theta, poses, measurements)
        if len(endpoints_B) < 2:
            break
        groups, assignments = cluster_endpoints(endpoints_B, u_B, v_B)

        # M-step: LM 优化
        theta = unsupervised_solve_lm(
            theta, poses, measurements, groups,
            w_plane=w_plane, w_edge=w_edge,
            max_iter=80)

        cost = unsupervised_cost(theta, poses, measurements, groups, w_plane, w_edge)
        if cost < best_cost:
            best_cost = cost
            best_theta = theta.copy()

        # 更新方向
        w_pl = theta[6:9]
        R_pl = so3_exp(w_pl)
        u_B, v_B = R_pl[:, 0], R_pl[:, 1]

    # 最终用最优 theta 再聚类一次
    endpoints_B = build_endpoints_B(best_theta, poses, measurements)
    w_pl_b = best_theta[6:9]
    R_pl_b = so3_exp(w_pl_b)
    groups, assignments = cluster_endpoints(endpoints_B, R_pl_b[:, 0], R_pl_b[:, 1])

    return best_theta, groups, assignments


def unsupervised_errors(theta_est, theta_gt):
    """兼容 combined_errors 接口"""
    return combined_errors(theta_est, theta_gt)


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    import yaml
    from calibration_framework import (CalibrationEnvironment, generate_tilted_corner_poses,
                                        ResultReporter)
    from nbv_edge_plane import (generate_edge_candidates, select_diverse,
                                 combined_solve_lm, combined_errors)

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  无监督端点聚类 + 联合优化 验证")
    print("=" * 60)

    results_labeled = []
    results_unlabeled = []

    for trial in range(10):
        seed = 42 + trial * 137
        config['environment']['laser_noise'] = 0.0  # 零噪声
        env = CalibrationEnvironment(config, seed=seed)

        # 生成位姿 (tilted: FOV 同时穿两边)
        poses = generate_tilted_corner_poses(
            env.C, env.n_B, env.u_B, env.v_B, env.R_he, env.t_he,
            plate_w=400, plate_h=500, n_poses=6, rng=env.rng)

        if len(poses) < 4:
            continue

        meas = env.generate_measurements(poses, noise_sigma_mm=0)

        # ── 有标签基准 ──
        theta_init_l = env.theta_gt_9.copy()
        rng_l = np.random.default_rng(seed + 9999)
        theta_init_l[0:3] += rng_l.normal(0, 0.1, 3)
        theta_init_l[3:6] += rng_l.normal(0, 0.002, 3)
        theta_init_l[6:9] += rng_l.normal(0, 0.05, 3)
        theta_opt_l = combined_solve_lm(theta_init_l, poses, meas,
                                         w_plane=1.0, w_edge=1.0, max_iter=80)
        Re_l, te_l, _ = combined_errors(theta_opt_l, env.theta_gt_9)

        # ── 无监督 ──
        theta_init_u = env.theta_gt_9.copy()
        rng_u = np.random.default_rng(seed + 9999)
        theta_init_u[0:3] += rng_u.normal(0, 0.1, 3)
        theta_init_u[3:6] += rng_u.normal(0, 0.002, 3)
        theta_init_u[6:9] += rng_u.normal(0, 0.05, 3)

        theta_opt_u, groups, assign = solve_unsupervised(
            theta_init_u, poses, meas, n_em_iters=3)
        Re_u, te_u, _ = combined_errors(theta_opt_u, env.theta_gt_9)

        results_labeled.append((Re_l, te_l))
        results_unlabeled.append((Re_u, te_u))

        print(f"  trial {trial+1}: seed={seed}")
        print(f"    有标签: R={Re_l:.6f}°  t={te_l:.6f}mm")
        print(f"    无监督: R={Re_u:.6f}°  t={te_u:.6f}mm  "
              f"分组={assign}")

    # 汇总
    Rl = np.array([r[0] for r in results_labeled])
    Tl = np.array([r[1] for r in results_labeled])
    Ru = np.array([r[0] for r in results_unlabeled])
    Tu = np.array([r[1] for r in results_unlabeled])

    print(f"\n{'='*60}")
    print(f"  汇总 (10 trials, 零噪声)")
    print(f"  {'':>20} {'有标签':>12} {'无监督':>12}")
    print(f"  {'R median':>20} {np.median(Rl):12.6f}° {np.median(Ru):12.6f}°")
    print(f"  {'R mean':>20} {np.mean(Rl):12.6f}° {np.mean(Ru):12.6f}°")
    print(f"  {'R max':>20} {np.max(Rl):12.6f}° {np.max(Ru):12.6f}°")
    print(f"  {'t median':>20} {np.median(Tl):12.6f}mm {np.median(Tu):12.6f}mm")
    print(f"  {'R<0.01°':>20} {np.sum(Rl<0.01):12d} {np.sum(Ru<0.01):12d}")
