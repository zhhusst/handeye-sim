#!/usr/bin/env python3
"""
test_twostage.py — 两阶段 LM：先用 un-centered 粗锁，再用 centered 精化

Stage 1: un-centered 平面约束 + 边缘共线性 (9-DOF, 无 gauge 问题)
Stage 2: centered 平面约束 + 边缘共线性 (9-DOF, 消除 gauge)
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/common')
from calib_solver import (
    compute_errors, combined_cost
)
from fov_geometry import (
    so3_exp, so3_log, rodrigues, rpy_to_matrix,
    generate_plane, compute_fov_plate_scanline, build_R_edge,
)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ===== 场景 =====
rng = np.random.default_rng(20260630)

rpy_gt = np.array([0.485145, 0.160648, -1.509479])
R_he_gt = rpy_to_matrix(np.rad2deg(rpy_gt[0]),
                         np.rad2deg(rpy_gt[1]),
                         np.rad2deg(rpy_gt[2]))
t_he_gt = np.array([-0.011579, -0.004621, 0.359284])
w_he_gt = so3_log(R_he_gt)

C, n_B_raw, u_B_raw, v_B_raw, pw, ph = generate_plane(rng, (0.4, 0.5))
if n_B_raw[2] < 0:
    n_B_raw = -n_B_raw; u_B_raw = -u_B_raw; v_B_raw = -v_B_raw
n_B = n_B_raw; u_B = u_B_raw; v_B = v_B_raw

R_pl_gt = np.column_stack([u_B, v_B, n_B])
theta_gt = np.concatenate([w_he_gt, t_he_gt, so3_log(R_pl_gt)])

print("=" * 70)
print("场景")
print("=" * 70)
print(f"R_he (deg): {np.rad2deg(w_he_gt)}")
print(f"t_he (mm):  {t_he_gt * 1000}")
print(f"n_B:        {n_B}")

# ===== 数据生成（修正版）=====
def collect_labeled_data():
    poses = []
    meas = []
    for k in range(10):
        # 随机目标位置
        target_u = rng.uniform(0.02, 0.38)
        target_v = rng.uniform(0.02, 0.48)
        target = C + target_u * pw * u_B + target_v * ph * v_B
        standoff = rng.uniform(0.38, 0.58)
        sp = target + standoff * n_B
        
        pitch = rng.uniform(-8, 8)
        yaw = rng.uniform(-15, 15)
        R_i = build_R_edge(pitch, yaw, u_B, n_B, u_B, v_B)
        t_i = sp - R_i @ t_he_gt
        
        R_BS = R_i @ R_he_gt
        t_BS = t_i + R_i @ t_he_gt
        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)
        
        if not sl['has_intersection'] or len(sl['scan_pts_S']) < 5:
            continue
        if len(sl['endpoints_S']) < 1:
            continue
        
        eps = sl['endpoints_S']
        e1_pt = next((p for e, p in eps if e == 'e1'), None)
        e2_pt = next((p for e, p in eps if e == 'e2'), None)
        
        poses.append((R_i, t_i))
        meas.append({
            'p_S_plane': sl['scan_pts_S'],
            'p_S_e1': e1_pt,
            'p_S_e2': e2_pt,
            'valid_e1': e1_pt is not None,
            'valid_e2': e2_pt is not None,
        })
    return poses, meas

print("\n生成数据...")
poses, meas = collect_labeled_data()
n_e1 = sum(1 for m in meas if m['valid_e1'])
n_e2 = sum(1 for m in meas if m['valid_e2'])
print(f"总位姿: {len(poses)} (e1={n_e1}, e2={n_e2})")

# ===== 两阶段 LM =====
print("\n" + "=" * 70)
print("两阶段 LM")
print("=" * 70)

def combined_residuals_uncentered(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    """Un-centered 平面约束 + 边缘共线性 (9-DOF)
    
    区别: 平面约束不去均值，保留 t_he 信息
    """
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B_th = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]
    
    # 计算平板在基系中的位置 (centroid)
    # 注意: 这里的 n_B_th, u_B, v_B 在迭代中变化
    residuals = []
    mask = []
    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)
    info = {'n_plane': 0, 'n_e1': 0, 'n_e2': 0}
    
    # 平面约束 (un-centered): 所有点投影到法向，用同一个均值补偿
    all_d = []
    p_base_e1 = []
    p_base_e2 = []
    
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        
        if m['valid_e1'] and m['p_S_e1'] is not None:
            p_base_e1.append(R_BS @ m['p_S_e1'] + t_BS)
        if m['valid_e2'] and m['p_S_e2'] is not None:
            p_base_e2.append(R_BS @ m['p_S_e2'] + t_BS)
        
        for p_S in m['p_S_plane']:
            p_B = R_BS @ p_S + t_BS
            all_d.append(np.dot(n_B_th, p_B))
    
    if len(all_d) > 0:
        d_center = np.mean(all_d)
        for d in all_d:
            residuals.append((d - d_center) * wp)
            mask.append(True)
            info['n_plane'] += 1
    
    # 边缘共线性
    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        residuals.extend(r.tolist())
        mask.extend([True, True, True])
        info['n_e1'] += 1
    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        residuals.extend(r.tolist())
        mask.extend([True, True, True])
        info['n_e2'] += 1
    
    return np.array(residuals), np.array(mask), info


def combined_jacobian_num(theta, poses, meas, w_plane, w_edge, eps=1e-6):
    r0, mask, info = combined_residuals_uncentered(theta, poses, meas, w_plane, w_edge)
    n_params = len(theta)
    J = np.zeros((len(r0), n_params))
    for j in range(n_params):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        rp, _, _ = combined_residuals_uncentered(tp, poses, meas, w_plane, w_edge)
        rm, _, _ = combined_residuals_uncentered(tm, poses, meas, w_plane, w_edge)
        J[:, j] = (rp - rm) / (2 * eps)
    return J, r0, mask, info


def solve_lm_generic(theta_init, poses, meas, w_plane=0.1, w_edge=1.0,
                      max_iter=300, tol=1e-12, residual_fn=None, centered=True):
    """通用 LM 求解器，支持 centered / un-centered"""
    if residual_fn is None:
        from calib_solver import combined_residuals as res_fn
    else:
        res_fn = residual_fn
    
    theta = theta_init.copy()
    lam = 1e-6
    
    for it in range(max_iter):
        r, mask, info = res_fn(theta, poses, meas, w_plane, w_edge) \
            if not centered else resid_centered_wrapper(theta, poses, meas, w_plane, w_edge)
        
        J, _, _, _ = combined_jacobian_num(theta, poses, meas, w_plane, w_edge)
        
        rv = r[mask]
        Jv = J[mask]
        
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
        new_cost = 0.5 * np.dot(
            res_fn(tn, poses, meas, w_plane, w_edge)[0][mask],
            res_fn(tn, poses, meas, w_plane, w_edge)[0][mask])
        # Hmm, this recomputation is wasteful. Let me use the simpler approach.
        
        # Actually, let me just use the existing combined_residuals for un-centered
        # and centered
        # Let me use the standard one for un-centered
        new_r, new_mask, _ = res_fn(tn, poses, meas, w_plane, w_edge)
        new_cost = 0.5 * np.dot(new_r[new_mask], new_r[new_mask])
        
        if new_cost < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - new_cost) < tol:
                break
        else:
            lam = min(lam * 3, 1e6)
    
    return theta


# 包装 centered residual 函数
def resid_centered_wrapper(theta, poses, meas, w_plane, w_edge):
    from calib_solver import combined_residuals
    return combined_residuals(theta, poses, meas, w_plane, w_edge)


# 更简单的做法：直接用 combined_solve_lm 两次
from calib_solver import combined_solve_lm, combined_residuals as centered_residuals

# Stage 1: 用 un-centered 残差做初始化
# 修改: 把 centered 改成 un-centered 需要改 calib_solver
# 但更简单的方式：直接复制一份去掉 centered 的

def run_twostage():
    """两阶段 LM"""
    
    theta_init = np.zeros(9)
    
    # ---- Stage 1: un-centered LM (恢复 t_he) ----
    # 我们自己实现，使用不 centered 的残差
    theta_s1 = theta_init.copy()
    lam = 1e-6
    
    for it in range(300):
        r, mask, info = combined_residuals_uncentered(theta_s1, poses, meas)
        J, _, _, _ = combined_jacobian_num(theta_s1, poses, meas, 0.1, 1.0)
        rv = r[mask]
        Jv = J[mask]
        if len(rv) == 0:
            break
        
        cost = 0.5 * np.dot(rv, rv)
        H = Jv.T @ Jv
        g = Jv.T @ rv
        
        try:
            delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except:
            lam *= 10
            continue
        
        tn = theta_s1 + delta
        rn, mn, _ = combined_residuals_uncentered(tn, poses, meas)
        new_cost = 0.5 * np.dot(rn[mn], rn[mn])
        
        if new_cost < cost:
            theta_s1 = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - new_cost) < tol_val:
                break
        else:
            lam = min(lam * 3, 1e6)
    
    R_s1, t_s1 = compute_errors(theta_s1, theta_gt)
    print(f"\nStage 1 (un-centered): R={R_s1:.6f}deg  t={t_s1:.6f}mm")
    return theta_s1

tol_val = 1e-12

# 跑 Stage 1
theta_s1 = run_twostage()

# ---- Stage 2: centered LM (消除 gauge) ----
theta_s2 = combined_solve_lm(theta_s1, poses, meas,
                              w_plane=0.1, w_edge=1.0,
                              max_iter=300, tol=1e-12)
R_s2, t_s2 = compute_errors(theta_s2, theta_gt)
print(f"Stage 2 (centered):  R={R_s2:.6f}deg  t={t_s2:.6f}mm")

# ===== 对比 =====
print("\n" + "=" * 70)
print("对比")
print("=" * 70)

# 纯 centered (当前方法)
theta_centered = combined_solve_lm(np.zeros(9), poses, meas,
                                    w_plane=0.1, w_edge=1.0,
                                    max_iter=300, tol=1e-12)
R_c, t_c = compute_errors(theta_centered, theta_gt)
print(f"纯 centered (当前): t={t_c:.6f}mm  R={R_c:.6f}deg")

# 多重启
print("\n多重启 (10x random):")
best_R, best_t = 999, 999
for trial in range(10):
    th_rnd = np.concatenate([
        np.random.uniform(-0.8, 0.8, 3),
        np.random.uniform(-0.05, 0.05, 3),
        np.zeros(3),
    ])
    th_rnd_opt = combined_solve_lm(th_rnd, poses, meas,
                                    w_plane=0.1, w_edge=1.0,
                                    max_iter=300, tol=1e-12)
    R_e, t_e = compute_errors(th_rnd_opt, theta_gt)
    if t_e < best_t:
        best_R, best_t = R_e, t_e
print(f"最佳多重启: R={best_R:.6f}deg  t={best_t:.6f}mm")

# 两阶段 (本次方案)
print(f"\n两阶段 LM:   R={R_s2:.6f}deg  t={t_s2:.6f}mm")

# ===== 收敛盆测试 =====
print("\n" + "=" * 70)
print("两阶段 LM 收敛盆测试")
print("=" * 70)

for R_off_deg in [10, 20, 30, 45, 60, 90]:
    axis = np.array([1., 1., 1.]) / np.sqrt(3)
    R_off = rodrigues(axis, np.deg2rad(R_off_deg))
    w_off = so3_log(R_off)
    
    th_init = np.zeros(9)
    th_init[0:3] = w_off
    
    # Stage 1
    th_s1 = th_init.copy()
    lam = 1e-6
    for it in range(300):
        r, mask, _ = combined_residuals_uncentered(th_s1, poses, meas)
        J, _, _, _ = combined_jacobian_num(th_s1, poses, meas, 0.1, 1.0)
        rv = r[mask]; Jv = J[mask]
        cost = 0.5 * np.dot(rv, rv)
        H = Jv.T @ Jv; g = Jv.T @ rv
        try:
            delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except:
            lam *= 10; continue
        tn = th_s1 + delta
        rn, mn, _ = combined_residuals_uncentered(tn, poses, meas)
        nc = 0.5 * np.dot(rn[mn], rn[mn])
        if nc < cost:
            th_s1 = tn; lam = max(lam/3, 1e-12)
            if abs(cost - nc) < 1e-12: break
        else:
            lam = min(lam * 3, 1e6)
    
    th_s2 = combined_solve_lm(th_s1, poses, meas, 0.1, 1.0, 300, 1e-12)
    R_e, t_e = compute_errors(th_s2, theta_gt)
    ok = "OK" if (R_e < 0.1 and t_e < 1.0) else "FAIL"
    print(f"  +{R_off_deg:2d}deg: {ok}  R={R_e:.6f}deg  t={t_e:.6f}mm")

print("\n" + "=" * 70)
if t_s2 < 0.1:
    print("✅ 两阶段 LM 成功！")
else:
    print(f"❌ 两阶段 LM 未完全收敛 (t={t_s2:.4f}mm)")
