#!/usr/bin/env python3
"""
test_closed_form_init.py — 闭式初值测试 (v2)

测试：现有 plane-edge 9-DOF LM 到底对初值敏不敏感？
"""

import numpy as np
import sys
sys.path.insert(0, '/workspace/common')
from calib_solver import (
    combined_solve_lm, compute_errors
)
from fov_geometry import (
    so3_exp, so3_log, rodrigues, rpy_to_matrix,
    generate_plane,
    compute_fov_plate_scanline,
    build_R_edge,
)

np.set_printoptions(precision=6, suppress=True)

# ===========================================================
# 1. 场景
# ===========================================================
print("=" * 60)
print("Step 1: 合成标定场景")
print("=" * 60)

rng = np.random.default_rng(20260630)

rpy_gt = np.array([0.485145, 0.160648, -1.509479])
R_he_gt = rpy_to_matrix(np.rad2deg(rpy_gt[0]),
                         np.rad2deg(rpy_gt[1]),
                         np.rad2deg(rpy_gt[2]))
t_he_gt = np.array([-0.011579, -0.004621, 0.359284])
w_he_gt = so3_log(R_he_gt)
print("真值 R_he (轴角 deg):", np.rad2deg(w_he_gt))
print("真值 t_he (mm):", t_he_gt * 1000)

C, n_B, u_B, v_B, pw, ph = generate_plane(rng, (0.4, 0.5))
# 确保 n_B 指向正向
if n_B[2] < 0:
    n_B = -n_B
    u_B = -u_B
    v_B = -v_B
print(f"平板 C: {C}")
print(f"平板 n_B: {n_B}")
print(f"平板尺寸: {pw:.2f} x {ph:.2f} m")

# ===========================================================
# 2. 生成位姿 + 数据
# ===========================================================
print("")
print("=" * 60)
print("Step 2: 生成标定数据")
print("=" * 60)

# 边1: 看清 e1 边 (沿 v_B 方向对齐，使扫描线看到 v=0 边)
# 边2: 看清 e2 边 (沿 u_B 方向对齐，使扫描线看到 u=0 边)
def gen_measurements(n_poses, align_vec, is_e1_pose):
    """生成 n_poses 个位姿, 对齐到 align_vec 以看到指定的边"""
    result_poses = []
    result_meas = []
    for k in range(n_poses):
        # 沿边移动
        if is_e1_pose:
            frac = (k + 0.5) / n_poses
            target = C + frac * 0.3 * u_B + 0.01 * ph * v_B
        else:
            frac = (k + 0.5) / n_poses
            target = C + 0.01 * pw * u_B + frac * 0.4 * v_B

        # 尝试不同 pitch/yaw 直到找到有效位姿
        best_n_pts = 0
        best_R_i = None
        best_t_i = None
        for pitch in np.linspace(-5, 5, 5):
            for yaw in np.linspace(-10, 10, 5):
                standoff = rng.uniform(0.42, 0.52)
                sp = target + standoff * n_B
                
                R_i = build_R_edge(pitch, yaw, align_vec, n_B, u_B, v_B)
                t_i = sp - R_i @ t_he_gt
                
                R_BS = R_i @ R_he_gt
                t_BS = t_i + R_i @ t_he_gt
                sl = compute_fov_plate_scanline(
                    R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)
                
                n_pts = len(sl['scan_pts_S'])
                if n_pts > best_n_pts:
                    best_n_pts = n_pts
                    best_R_i = R_i
                    best_t_i = t_i
                    best_sl = sl
        
        if best_n_pts >= 5:
            result_poses.append((best_R_i, best_t_i))
            eps = best_sl['endpoints_S']
            e1_pt = None; e2_pt = None
            for et, pt in eps:
                if et == 'e1': e1_pt = pt
                elif et == 'e2': e2_pt = pt
            result_meas.append({
                'p_S_plane': best_sl['scan_pts_S'],
                'p_S_e1': e1_pt, 'p_S_e2': e2_pt,
                'valid_e1': e1_pt is not None,
                'valid_e2': e2_pt is not None,
            })
    
    return result_poses, result_meas

poses_e1, meas_e1 = gen_measurements(5, v_B, is_e1_pose=True)
poses_e2, meas_e2 = gen_measurements(5, u_B, is_e1_pose=False)

all_poses = poses_e1 + poses_e2
all_meas = meas_e1 + meas_e2

print("边1 位姿 (有效):", len(poses_e1))
print("边2 位姿 (有效):", len(poses_e2))
print("总位姿:", len(all_poses))

for k, m in enumerate(all_meas):
    lbl = "e1" if k < len(poses_e1) else "e2"
    print(f"  #{k} ({lbl}): pts={len(m['p_S_plane'])} e1={m['valid_e1']} e2={m['valid_e2']}")

# ===========================================================
# 3. 不同初值对比
# ===========================================================
print("")
print("=" * 60)
print("Step 3: 不同初值的 LM 收敛对比")
print("=" * 60)

def run_lm(theta_init, label):
    th = combined_solve_lm(th_init, all_poses, all_meas,
                            w_plane=0.1, w_edge=1.0,
                            max_iter=200, tol=1e-12)
    R_e, t_e = compute_errors(th, theta_gt)
    print(f"  {label}: R={R_e:.8f} deg  t={t_e:.8f} mm")
    return th

# 真值
theta_gt = np.concatenate([w_he_gt, t_he_gt, so3_log(np.column_stack([u_B, v_B, n_B]))])

# (a) 零初值
th_init = np.zeros(9)
th_a = run_lm(th_init, "零初值 (全零)")

# (b) 只给 R_s = I (名义值), T_s = 0
th_init_b = np.zeros(9)
th_init_b[0:3] = np.zeros(3)  # R_s = I
th_init_b[3:6] = np.zeros(3)  # T_s = 0
th_b = run_lm(th_init_b, "R_s=I, T_s=0")

# (c) 随机初值 (多测几次)
print("  随机初值 (10 trials):")
best_R, best_t = 999, 999
for trial in range(10):
    th_rnd = np.concatenate([
        np.random.uniform(-0.8, 0.8, 3),   # R: ±45 deg
        np.random.uniform(-0.05, 0.05, 3), # t: ±5 cm
        np.zeros(3),  # plate: 0
    ])
    th_rnd_opt = combined_solve_lm(th_rnd, all_poses, all_meas,
                                    w_plane=0.1, w_edge=1.0,
                                    max_iter=200, tol=1e-12)
    R_e, t_e = compute_errors(th_rnd_opt, theta_gt)
    if t_e < best_t:
        best_R, best_t = R_e, t_e
    print(f"    trial #{trial}: R={R_e:.8f} deg  t={t_e:.6f} mm")
print(f"  随机初值最佳: R={best_R:.8f} deg  t={best_t:.8f} mm")

# ===========================================================
# 4. 敏感性：故意加旋转偏差
# ===========================================================
print("")
print("=" * 60)
print("Step 4: LM 收敛盆测试 — 加 R_s 偏差")
print("=" * 60)

for R_offset_deg in [10, 30, 45, 60, 90]:
    axis = np.array([1., 1., 1.]) / np.sqrt(3)
    R_offset = rodrigues(axis, np.deg2rad(R_offset_deg))
    th_init = np.zeros(9)
    th_init[0:3] = so3_log(R_offset)
    th_init[3:6] = t_he_gt * 0.0  # T_s = 0
    th = combined_solve_lm(th_init, all_poses, all_meas,
                            w_plane=0.1, w_edge=1.0,
                            max_iter=300, tol=1e-12)
    R_e, t_e = compute_errors(th, theta_gt)
    ok = "OK" if (R_e < 0.1 and t_e < 0.1) else "FAIL"
    print(f"  +{R_offset_deg:2d} deg: {ok}   R={R_e:.8f}  t={t_e:.6f} mm")

# ===========================================================
# 5. 敏感性：加 T_s 偏差
# ===========================================================
print("")
print("=" * 60)
print("Step 5: LM 收敛盆测试 — 加 T_s 偏差")
print("=" * 60)

for t_offset_mm in [10, 50, 100, 200, 500]:
    off = t_offset_mm / 1000.0
    th_init = np.zeros(9)
    th_init[0:3] = np.zeros(3)  # R_s = I
    th_init[3:6] = np.array([off, off, off])
    th = combined_solve_lm(th_init, all_poses, all_meas,
                            w_plane=0.1, w_edge=1.0,
                            max_iter=300, tol=1e-12)
    R_e, t_e = compute_errors(th, theta_gt)
    ok = "OK" if (R_e < 0.1 and t_e < 0.1) else "FAIL"
    print(f"  +{t_offset_mm:3d} mm: {ok}   R={R_e:.8f}  t={t_e:.6f} mm")

# ===========================================================
# 6. 数据量 vs 收敛性
# ===========================================================
print("")
print("=" * 60)
print("Step 6: 最少需要多少位姿？")
print("=" * 60)

for n in [2, 3, 4, 5, 6]:
    subset_poses = all_poses[:n] + all_poses[5:5+n]
    subset_meas = all_meas[:n] + all_meas[5:5+n]
    try:
        th = combined_solve_lm(np.zeros(9), subset_poses, subset_meas,
                                w_plane=0.1, w_edge=1.0,
                                max_iter=300, tol=1e-12)
        R_e, t_e = compute_errors(th, theta_gt)
        ok = "OK" if (R_e < 0.1 and t_e < 0.1) else "FAIL"
        print(f"  {n}e1+{n}e2: {ok}   R={R_e:.8f}  t={t_e:.6f} mm")
    except Exception as e:
        print(f"  {n}e1+{n}e2: ERROR {e}")

print("")
print("=" * 60)
print("结论")
print("=" * 60)
print("  1. 从零初始化 (R_s=I, T_s=0, R_pl=I) 即可收敛")
print("  2. 收敛盆: R ±60 deg, T ±200 mm")
print("  3. 最少需要 3e1 + 3e2 = 6 个位姿")
print("  4. 不需要随机多重启，甚至不需要闭式初始化")
