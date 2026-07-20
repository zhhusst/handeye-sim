#!/usr/bin/env python3
"""
test_plane_init.py — 平面约束闭式初值 → 9-DOF LM 精化

方案:
  1. R_s = I (名义值), 用板面点拟合平面 → n_B, C
  2. 用 un-centered 平面约束对 T_s 做线性 LS
  3. 用初值 [0, T_s, R_pl_approx] 跑 plane-edge 9-DOF LM

验证: 能否从零初值 -> 闭式初值 -> LM 精化到真值
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/common')
from calib_solver import (
    combined_solve_lm, combined_residuals, combined_jacobian,
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
print(f"平板尺寸: {pw:.2f} x {ph:.2f} m")

# ===== 生成数据 =====
def collect_data(n_per_edge=5):
    poses = []
    meas = []
    
    # 边1: 沿 v_B 方向对齐，看到 v=0 边
    for k in range(n_per_edge):
        frac = (k + 0.5) / n_per_edge
        target = C + frac * 0.3 * pw * u_B + 0.01 * ph * v_B
        standoff = rng.uniform(0.42, 0.52)
        sp = target + standoff * n_B
        
        best = None
        best_n = 0
        for pitch in np.linspace(-5, 5, 5):
            for yaw in np.linspace(-10, 10, 5):
                R_i = build_R_edge(pitch, yaw, v_B, n_B, u_B, v_B)
                t_i = sp - R_i @ t_he_gt
                R_BS = R_i @ R_he_gt
                t_BS = t_i + R_i @ t_he_gt
                sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)
                n_p = len(sl['scan_pts_S'])
                if n_p > best_n:
                    best_n = n_p
                    best = (R_i, t_i, sl)
        
        if best_n >= 5:
            R_i, t_i, sl = best
            poses.append((R_i, t_i))
            eps = sl['endpoints_S']
            e1_pt = next((p for e, p in eps if e == 'e1'), None)
            e2_pt = next((p for e, p in eps if e == 'e2'), None)
            meas.append({
                'p_S_plane': sl['scan_pts_S'],
                'p_S_e1': e1_pt, 'p_S_e2': e2_pt,
                'valid_e1': e1_pt is not None,
                'valid_e2': e2_pt is not None,
            })
    
    # 边2: 沿 u_B 方向对齐，看到 u=0 边
    for k in range(n_per_edge):
        frac = (k + 0.5) / n_per_edge
        target = C + 0.01 * pw * u_B + frac * 0.4 * ph * v_B
        standoff = rng.uniform(0.42, 0.52)
        sp = target + standoff * n_B
        
        best = None
        best_n = 0
        for pitch in np.linspace(-5, 5, 5):
            for yaw in np.linspace(-10, 10, 5):
                R_i = build_R_edge(pitch, yaw, u_B, n_B, u_B, v_B)
                t_i = sp - R_i @ t_he_gt
                R_BS = R_i @ R_he_gt
                t_BS = t_i + R_i @ t_he_gt
                sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)
                n_p = len(sl['scan_pts_S'])
                if n_p > best_n:
                    best_n = n_p
                    best = (R_i, t_i, sl)
        
        if best_n >= 5:
            R_i, t_i, sl = best
            poses.append((R_i, t_i))
            eps = sl['endpoints_S']
            e1_pt = next((p for e, p in eps if e == 'e1'), None)
            e2_pt = next((p for e, p in eps if e == 'e2'), None)
            meas.append({
                'p_S_plane': sl['scan_pts_S'],
                'p_S_e1': e1_pt, 'p_S_e2': e2_pt,
                'valid_e1': e1_pt is not None,
                'valid_e2': e2_pt is not None,
            })
    
    return poses, meas

print("\n" + "=" * 70)
print("生成数据")
print("=" * 70)
poses, meas = collect_data(5)
n_e1 = sum(1 for m in meas if m['valid_e1'])
n_e2 = sum(1 for m in meas if m['valid_e2'])
print(f"总位姿: {len(poses)} (e1={n_e1}, e2={n_e2})")

# ===== 闭式初始化 =====
print("\n" + "=" * 70)
print("闭式初始化: 平面约束线性 LS 解 T_s")
print("=" * 70)

def closed_form_init(poses, meas, fix_Rs=True):
    """闭式初始化
    
    1. 假设 R_s = I, T_s = 0
    2. 所有板面点投影到基系 → SVD 拟合平面 → n_B, centroid
    3. 用 un-centered 平面约束线性 LS 解 T_s
    4. 构造近似 R_pl
    """
    
    if fix_Rs:
        # Step 1: T_s = 0, R_s = I, 投影
        all_pts = []
        for (R_i, t_i), m in zip(poses, meas):
            for p_S in m['p_S_plane']:
                p_B = R_i @ p_S + t_i  # T_s=0
                all_pts.append(p_B)
        all_pts = np.array(all_pts)
        
        # Step 2: SVD 平面拟合
        centroid = np.mean(all_pts, axis=0)
        centered = all_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n_B_init = Vt[-1, :]
        n_B_init = n_B_init / np.linalg.norm(n_B_init)
        # 确保指向正向
        if n_B_init[2] < 0:
            n_B_init = -n_B_init
        
        d_init = np.dot(n_B_init, centroid)
        
        # Step 3: 线性 LS 解 T_s
        # n_B^T · (R_i · p_S + t_i + R_i · T_s) = d
        # n_B^T · R_i · T_s = d - n_B^T · R_i · p_S - n_B^T · t_i
        A_rows = []
        b_vals = []
        for (R_i, t_i), m in zip(poses, meas):
            for p_S in m['p_S_plane']:
                A_rows.append(n_B_init @ R_i)
                b_vals.append(d_init - n_B_init @ (R_i @ p_S + t_i))
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        t_he_init, *_ = np.linalg.lstsq(A, b, rcond=None)
        
        # Step 4: 用 T_s 重算平面
        all_pts2 = []
        for (R_i, t_i), m in zip(poses, meas):
            for p_S in m['p_S_plane']:
                p_B = R_i @ p_S + t_i + R_i @ t_he_init
                all_pts2.append(p_B)
        all_pts2 = np.array(all_pts2)
        centroid2 = np.mean(all_pts2, axis=0)
        centered2 = all_pts2 - centroid2
        _, _, Vt2 = np.linalg.svd(centered2, full_matrices=False)
        n_B_est = Vt2[-1, :]
        n_B_est = n_B_est / np.linalg.norm(n_B_est)
        if n_B_est[2] < 0:
            n_B_est = -n_B_est
        
        # Step 5: 构造 R_pl
        if abs(n_B_est[2]) < 0.9:
            u_B_est = np.cross(np.array([0., 0., 1.]), n_B_est)
        else:
            u_B_est = np.cross(np.array([0., 1., 0.]), n_B_est)
        u_B_est = u_B_est / np.linalg.norm(u_B_est)
        v_B_est = np.cross(n_B_est, u_B_est)
        R_pl_init = np.column_stack([u_B_est, v_B_est, n_B_est])
        w_pl_init = so3_log(R_pl_init)
        
        theta_init = np.concatenate([
            np.zeros(3),       # w_he = 0 (R_s = I)
            t_he_init,         # 平面 LS 估计
            w_pl_init,         # 平面拟合估计
        ])
        
        return theta_init, n_B_est, centroid2
    else:
        return np.zeros(9), None, None

theta_init, n_B_est, C_est = closed_form_init(poses, meas)

R_init_err, t_init_err = compute_errors(theta_init, theta_gt)
print(f"  初始 T_s (mm): {theta_init[3:6] * 1000}")
print(f"  真值 T_s (mm): {t_he_gt * 1000}")
print(f"  T_s 误差: {t_init_err:.4f} mm")
print(f"  R 误差: {R_init_err:.4f} deg")

# ===== LM 精化 =====
print("\n" + "=" * 70)
print("plane-edge 9-DOF LM 精化")
print("=" * 70)

# 从闭式初值
theta_opt = combined_solve_lm(
    theta_init, poses, meas,
    w_plane=0.1, w_edge=1.0,
    max_iter=300, tol=1e-12)
R_opt, t_opt = compute_errors(theta_opt, theta_gt)
print(f"  闭式初值+LM: R={R_opt:.8f} deg  t={t_opt:.8f} mm")

# 从零初值
theta_zero_opt = combined_solve_lm(
    np.zeros(9), poses, meas,
    w_plane=0.1, w_edge=1.0,
    max_iter=300, tol=1e-12)
R_zero, t_zero = compute_errors(theta_zero_opt, theta_gt)
print(f"  零初值+LM:   R={R_zero:.8f} deg  t={t_zero:.8f} mm")

# ===== 收敛盆测试 =====
print("\n" + "=" * 70)
print("收敛盆: 给 R_s 加偏差 + 闭式初始化 T_s")
print("=" * 70)

for R_off_deg in [10, 20, 30, 45, 60, 90]:
    axis = np.array([1., 1., 1.]) / np.sqrt(3)
    R_off = rodrigues(axis, np.deg2rad(R_off_deg))
    
    # 构造初值: R_s = R_off, T_s 从平面约束重算
    w_off = so3_log(R_off)
    n_B_off = n_B_est  # 用之前估计的平面
    
    # 重算 T_s
    A_rows = []
    b_vals = []
    for (R_i, t_i), m in zip(poses, meas):
        for p_S in m['p_S_plane']:
            A_rows.append(n_B_off @ (R_i @ R_off))
            b_vals.append(np.dot(n_B_off, C_est) - n_B_off @ (R_i @ R_off @ p_S + t_i))
    A = np.array(A_rows)
    b = np.array(b_vals)
    t_off = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # 初值
    theta_test = np.zeros(9)
    theta_test[0:3] = w_off
    theta_test[3:6] = t_off
    theta_test[6:9] = theta_init[6:9]  # 平板用同一估计
    
    th_opt = combined_solve_lm(theta_test, poses, meas,
                                w_plane=0.1, w_edge=1.0,
                                max_iter=300, tol=1e-12)
    R_e, t_e = compute_errors(th_opt, theta_gt)
    ok = "OK" if (R_e < 0.1 and t_e < 1.0) else "FAIL"
    print(f"  R+{R_off_deg:2d}deg: {ok}  R={R_e:.6f}deg  t={t_e:.6f}mm")

# ===== 最少位姿测试 =====
print("\n" + "=" * 70)
print("最少位姿测试")
print("=" * 70)

for n in [2, 3, 4, 5]:
    subset_p = poses[:n] + poses[5:5+n]
    subset_m = meas[:n] + meas[5:5+n]
    
    # 闭式初值
    th_i, _, _ = closed_form_init(subset_p, subset_m)
    try:
        th_o = combined_solve_lm(th_i, subset_p, subset_m,
                                  w_plane=0.1, w_edge=1.0,
                                  max_iter=300, tol=1e-12)
        R_e, t_e = compute_errors(th_o, theta_gt)
        ok = "OK" if (R_e < 0.1 and t_e < 1.0) else "FAIL"
        print(f"  {n}e1+{n}e2: {ok}  R={R_e:.6f}deg  t={t_e:.6f}mm")
    except Exception as e:
        print(f"  {n}e1+{n}e2: ERROR {e}")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
if t_opt < 0.01:
    print(f"  ✅ 闭式初值方案成功: t_opt={t_opt:.6f}mm")
else:
    print(f"  ❌ 闭式初值方案未收敛: t_opt={t_opt:.6f}mm")
print(f"  （对比零初值: t_zero={t_zero:.6f}mm）")
