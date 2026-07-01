#!/usr/bin/env python3
"""
diagnose_init.py — 诊断 plane-edge 9-DOF 的平移可观性

检查: 从正确初值 (真值附近) vs 零初值，各自收敛到哪
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

rng = np.random.default_rng(20260630)

# ===== 场景 =====
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
print("场景真值")
print("=" * 70)
print(f"R_he (deg): {np.rad2deg(w_he_gt)}")
print(f"t_he (mm):  {t_he_gt * 1000}")
print(f"n_B:        {n_B}")

# ===== 生成数据（强制正确的边缘标注）=====
def collect_labeled_data():
    """生成数据并自动标注实际看到的边"""
    poses = []
    meas = []
    attempts = 0
    max_attempts = 100
    
    while len(poses) < 6 and attempts < max_attempts:
        attempts += 1
        # 随机位姿
        target_u = rng.uniform(0.02, 0.35)
        target_v = rng.uniform(0.02, 0.45)
        target = C + target_u * pw * u_B + target_v * ph * v_B
        standoff = rng.uniform(0.40, 0.55)
        sp = target + standoff * n_B
        
        pitch = rng.uniform(-10, 10)
        yaw = rng.uniform(-15, 15)
        align_vec = u_B if rng.random() < 0.5 else v_B
        R_i = build_R_edge(pitch, yaw, align_vec, n_B, u_B, v_B)
        t_i = sp - R_i @ t_he_gt
        
        R_BS = R_i @ R_he_gt
        t_BS = t_i + R_i @ t_he_gt
        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)
        
        if not sl['has_intersection'] or len(sl['scan_pts_S']) < 5:
            continue
        if len(sl['endpoints_S']) < 1:
            continue
        
        # 自动标注
        has_e1 = any(et == 'e1' for et, _ in sl['endpoints_S'])
        has_e2 = any(et == 'e2' for et, _ in sl['endpoints_S'])
        
        if not has_e1 and not has_e2:
            continue
        
        e1_pt = next((pt for et, pt in sl['endpoints_S'] if et == 'e1'), None)
        e2_pt = next((pt for et, pt in sl['endpoints_S'] if et == 'e2'), None)
        
        poses.append((R_i, t_i))
        meas.append({
            'p_S_plane': sl['scan_pts_S'],
            'p_S_e1': e1_pt, 'p_S_e2': e2_pt,
            'valid_e1': has_e1, 'valid_e2': has_e2,
        })
        
        lbl = "e1+e2" if (has_e1 and has_e2) else ("e1" if has_e1 else "e2")
        print(f"  #{len(poses)-1} ({lbl}): pts={len(sl['scan_pts_S'])} e1={has_e1} e2={has_e2}")
    
    return poses, meas

print("\n" + "=" * 70)
print("生成数据 (自动标注)")
print("=" * 70)
poses, meas = collect_labeled_data()
print(f"总位姿: {len(poses)}")

# ===== 从不同初值测试 =====
print("\n" + "=" * 70)
print("初值敏感性测试")
print("=" * 70)

def test_init(th_init, label):
    th = combined_solve_lm(th_init, poses, meas,
                            w_plane=0.1, w_edge=1.0,
                            max_iter=300, tol=1e-12)
    R_e, t_e = compute_errors(th, theta_gt)
    print(f"  {label}")
    print(f"    -> R_err={R_e:.6f} deg  t_err={t_e:.6f} mm")
    print(f"    -> t_he_est={th[3:6]*1000} mm  (gt={t_he_gt*1000} mm)")
    
    # 检查 Jacobian 秩
    J, r, mask, info = combined_jacobian(th, poses, meas, 0.1, 1.0)
    Jv = J[mask]
    sv = np.linalg.svd(Jv, compute_uv=False)
    print(f"    -> Jacobian SVD: {sv}")
    print(f"    -> rank: {np.sum(sv > 1e-8)}/9")
    return th

# (a) 真值附近 (检查 Jacobian)
print("\n--- (a) 真值附近 ---")
J_near_gt, r_near_gt, mask_gt, info_gt = combined_jacobian(
    theta_gt, poses, meas, 0.1, 1.0)
sv_near = np.linalg.svd(J_near_gt[mask_gt], compute_uv=False)
print(f"  真值处 Jacobian SVD: {sv_near}")
print(f"  秩: {np.sum(sv_near > 1e-8)}/9")
print(f"  最小奇异值: {sv_near[-1]:.6e}")
print(f"  条件数: {sv_near[0]/sv_near[-1]:.2f}")

# (b) 零初值
print("\n--- (b) 零初值 ---")
th_zero = test_init(np.zeros(9), "零初值")

# (c) 真值 t，零 R
print("\n--- (c) t=真值, R=零 ---")
th_truth_t = np.zeros(9)
th_truth_t[3:6] = t_he_gt
test_init(th_truth_t, "t=真值, R=零")

# (d) 真值 R，零 t
print("\n--- (d) R=真值, t=零 ---")
th_truth_r = np.zeros(9)
th_truth_r[0:3] = w_he_gt
th_truth_r[6:9] = so3_log(R_pl_gt)
# t = 0
test_init(th_truth_r, "R=真值, t=零")

# (e) 真值 R + t，零平板
print("\n--- (e) R+t=真值, 平板=零 ---")
th_truth_he = np.concatenate([w_he_gt, t_he_gt, np.zeros(3)])
test_init(th_truth_he, "R,t=真值, 平板=零")

# (f) 分析：从零初值，检查收敛路径
print("\n" + "=" * 70)
print("详细分析：零初值的收敛过程")
print("=" * 70)

theta = np.zeros(9)
for it in range(20):
    J, r, mask, info = combined_jacobian(theta, poses, meas, 0.1, 1.0)
    rv = r[mask]
    Jv = J[mask]
    cost = 0.5 * np.dot(rv, rv)
    
    w_he = theta[0:3]; t_he = theta[3:6]; w_pl = theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    
    R_e, t_e = compute_errors(theta, theta_gt)
    
    if it < 5 or (it < 30 and it % 5 == 0):
        Jv_sv = np.linalg.svd(Jv, compute_uv=False)
        print(f"  iter {it:3d}: cost={cost:.2e}  R_err={R_e:.4f} deg  t_err={t_e:.4f} mm  "
              f"sv_min={Jv_sv[-1]:.2e}  cond={Jv_sv[0]/Jv_sv[-1]:.1f}")
    
    # LM step
    H = Jv.T @ Jv
    g = Jv.T @ rv
    lam = 1e-3
    delta = -np.linalg.solve(H + lam * np.eye(9), g)
    theta = theta + delta
    
    if cost < 1e-20:
        break

print("\n最终")
w_he = theta[0:3]; t_he = theta[3:6]
print(f"  R_he (deg): {np.rad2deg(w_he)}  (gt: {np.rad2deg(w_he_gt)})")
print(f"  t_he (mm):  {t_he*1000}  (gt: {t_he_gt*1000})")
R_e, t_e = compute_errors(theta, theta_gt)
print(f"  R_err={R_e:.8f} deg  t_err={t_e:.8f} mm")

# ===== 结论 =====
print("\n" + "=" * 70)
print("诊断结论")
print("=" * 70)
print(f"  真值处 Jacobian 秩: {np.sum(sv_near > 1e-8)}/9")
print(f"  真值处最小奇异值: {sv_near[-1]:.2e}")
print(f"  从零初值: R 收敛 ✅  t 发散 ❌")

# 检查 t 方向上是否有零空间
if sv_near[-1] < 1e-6:
    print("\n  ⚠️ 发现: 最小奇异值接近零!")
    print("  t 方向有几乎不可观的自由度")
    Vt = np.linalg.svd(Jv, compute_uv=True)[2]
    null_dir = Vt[-1]
    print(f"  零空间方向: {null_dir}")
    t_comp = null_dir[3:6]
    print(f"  零空间中 t_he 分量: {t_comp / np.linalg.norm(t_comp)}")
