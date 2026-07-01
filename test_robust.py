#!/usr/bin/env python3
"""
test_robust.py — 验证初值敏感性根因

核心假设: 问题的根因是位姿旋转变化不够 → t 收敛路径窄
测试: 增加位姿旋转变化幅度 → 看零初值能否收敛
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/common')
from calib_solver import combined_solve_lm, compute_errors
from fov_geometry import (
    so3_exp, so3_log, rodrigues, rpy_to_matrix,
    generate_plane, compute_fov_plate_scanline, build_R_edge,
)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

rng = np.random.default_rng(20260630)
rpy_gt = np.array([0.485145, 0.160648, -1.509479])
R_he_gt = rpy_to_matrix(np.rad2deg(rpy_gt[0]), np.rad2deg(rpy_gt[1]), np.rad2deg(rpy_gt[2]))
t_he_gt = np.array([-0.011579, -0.004621, 0.359284])
w_he_gt = so3_log(R_he_gt)
C, n_B_raw, u_B_raw, v_B_raw, pw, ph = generate_plane(rng, (0.4, 0.5))
if n_B_raw[2] < 0: n_B_raw = -n_B_raw; u_B_raw = -u_B_raw; v_B_raw = -v_B_raw
n_B = n_B_raw; u_B = u_B_raw; v_B = v_B_raw
theta_gt = np.concatenate([w_he_gt, t_he_gt, so3_log(np.column_stack([u_B, v_B, n_B]))])

def collect_data(n_poses, pitch_range=8, yaw_range=15):
    """生成位姿数据，可调旋转变化幅度"""
    poses, meas = [], []
    for k in range(n_poses):
        target = C + rng.uniform(0.02, 0.38) * pw * u_B + rng.uniform(0.02, 0.48) * ph * v_B
        sp = target + rng.uniform(0.38, 0.58) * n_B
        
        # 尝试不同的 align 方向
        best = None
        best_n = 0
        for align in [u_B, v_B, u_B+v_B]:
            for pitch in np.linspace(-pitch_range, pitch_range, 7):
                for yaw in np.linspace(-yaw_range, yaw_range, 7):
                    R_i = build_R_edge(pitch, yaw, align, n_B, u_B, v_B)
                    t_i = sp - R_i @ t_he_gt
                    sl = compute_fov_plate_scanline(
                        R_i @ R_he_gt, t_i + R_i @ t_he_gt,
                        C, n_B, u_B, v_B, pw, ph)
                    n_p = len(sl['scan_pts_S'])
                    if n_p > best_n:
                        best_n = n_p
                        best = (R_i, t_i, sl)
        
        if best_n >= 5 and best is not None:
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

# 测试不同旋转变化幅度
print("=" * 70)
print("旋转变化幅度 vs 零初值收敛性")
print("=" * 70)

for p_range, y_range in [(8, 15), (15, 30), (25, 45), (40, 60)]:
    poses, meas = collect_data(12, p_range, y_range)
    n_e1 = sum(1 for m in meas if m['valid_e1'])
    n_e2 = sum(1 for m in meas if m['valid_e2'])
    
    theta_opt = combined_solve_lm(np.zeros(9), poses, meas,
                                   w_plane=0.1, w_edge=1.0,
                                   max_iter=300, tol=1e-12)
    R_e, t_e = compute_errors(theta_opt, theta_gt)
    ok = "OK" if (R_e < 0.1 and t_e < 1.0) else "FAIL"
    print(f"  pitch±{p_range:2d} yaw±{y_range:2d}: {ok}  "
          f"R={R_e:.6f}deg  t={t_e:.6f}mm  "
          f"位姿={len(poses)} e1={n_e1} e2={n_e2}")

# 用大旋转变化，测试最少需要多少位姿
print("\n" + "=" * 70)
print("大旋转 (pitch±25, yaw±45): 最少位姿数")
print("=" * 70)

poses, meas = collect_data(14, 25, 45)
n_e1 = sum(1 for m in meas if m['valid_e1'])
n_e2 = sum(1 for m in meas if m['valid_e2'])
print(f"总位姿: {len(poses)} e1={n_e1} e2={n_e2}")

for n in [2, 3, 4, 5, 6, 7]:
    # 取 n 个 e1 位姿 + n 个 e2 位姿
    e1_idx = [i for i, m in enumerate(meas) if m['valid_e1']][:n]
    e2_idx = [i for i, m in enumerate(meas) if m['valid_e2']][:n]
    idx = list(set(e1_idx + e2_idx))
    if len(idx) < 4:
        continue
    
    sub_p = [poses[i] for i in idx]
    sub_m = [meas[i] for i in idx]
    
    try:
        th = combined_solve_lm(np.zeros(9), sub_p, sub_m,
                                w_plane=0.1, w_edge=1.0,
                                max_iter=300, tol=1e-12)
        R_e, t_e = compute_errors(th, theta_gt)
        ok = "OK" if (R_e < 0.1 and t_e < 1.0) else "FAIL"
        print(f"  {len(idx)} poses ({n}e1+{n}e2): {ok}  R={R_e:.6f}deg  t={t_e:.6f}mm")
    except Exception as e:
        print(f"  {n}e1+{n}e2: ERROR {e}")

# 直接零初值跑全部14个位姿
print("\n零初值 (全部位姿):")
th_all = combined_solve_lm(np.zeros(9), poses, meas,
                            w_plane=0.1, w_edge=1.0,
                            max_iter=300, tol=1e-12)
R_all, t_all = compute_errors(th_all, theta_gt)
print(f"  R={R_all:.8f}deg  t={t_all:.8f}mm")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)
if t_all < 0.1:
    print("✅ 大旋转变化 + 足够位姿 → 零初值也能收敛")
    print("   问题根因不是初值敏感，是位姿旋转变化不够")
else:
    print(f"❌ 仍未收敛 t={t_all:.6f}mm")
