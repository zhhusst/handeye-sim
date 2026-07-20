"""
验证标定方法是否正确：使用 recorded_poses.json 手动数据
测试 M1(C-anchored cross-product) 和 M3(传感器帧预测+gauge固定)
"""
import json
import sys
import numpy as np
sys.path.insert(0, '/home/z/research_contact_handeye/verification/Sim/common')

from calib_solver import (
    solve_principle_12dof_with_restarts,
    solve_rs_xu2022,
    so3_log, so3_exp,
    residuals_principle_12dof,
    cost_principle_12dof,
    init_principle_12dof,
)
from fov_geometry import so3_exp as fov_so3_exp, so3_log as fov_so3_log

# ── 加载数据 ──────────────────────────────────────────
with open('/home/z/research_contact_handeye/verification/Sim/recorded_poses.json') as f:
    data = json.load(f)

poses_raw = data['poses']
R_he_gt = np.array(data['scene']['R_he_gt'])
t_he_gt = np.array(data['scene']['t_he_gt'])

# 构造 poses 和 meas (与 auto_calib_v2_node 格式一致)
poses = []
meas = []
for p in poses_raw:
    R_i = np.array(p['R_i'])
    t_i = np.array(p['t_i'])
    poses.append((R_i, t_i))
    m = {
        'p_S_e1': np.array(p['p_S_e1']) if p.get('valid_e1') else None,
        'valid_e1': p.get('valid_e1', False),
        'p_S_e2': np.array(p['p_S_e2']) if p.get('valid_e2') else None,
        'valid_e2': p.get('valid_e2', False),
        'p_S_plane': [np.array(pt) for pt in p['scan_pts_S']],
    }
    meas.append(m)

n = len(poses)
n_e1 = sum(1 for m in meas if m['valid_e1'])
n_e2 = sum(1 for m in meas if m['valid_e2'])
print(f"数据: {n} 位姿  e1={n_e1}/9  e2={n_e2}/9")

# ── 名义手眼 (模拟 auto_calib_v2 的扰动) ─────────────
# 用 ground truth + 扰动
rng = np.random.RandomState(42)
w_pert = rng.randn(3) * 0.05  # ~2.8°
R_he_nom = R_he_gt @ so3_exp(w_pert)
t_he_nom = t_he_gt + rng.randn(3) * 0.012  # ~12mm

R_err_nom = np.rad2deg(np.arccos(np.clip(
    (np.trace(R_he_nom.T @ R_he_gt) - 1) / 2, -1, 1)))
t_err_nom = np.linalg.norm(t_he_nom - t_he_gt) * 1000
print(f"名义手眼: R_err={R_err_nom:.2f}°  t_err={t_err_nom:.1f}mm")

# ── 计算 tilt 统计 ────────────────────────────────────
tilts = []
for R_i, t_i in poses:
    z_S = R_i @ R_he_nom[:, 2]
    tilt = np.rad2deg(np.arccos(np.clip(abs(np.dot(z_S, [0, 0, 1])), 0, 1)))
    tilts.append(tilt)
tilts = np.array(tilts)
print(f"Tilt: min={tilts.min():.1f}°  max={tilts.max():.1f}°  "
      f"mean={tilts.mean():.1f}°  std={tilts.std():.1f}°")

# ── Xu 2022 初值 ──────────────────────────────────────
R_he_xu = solve_rs_xu2022(poses, meas)
if R_he_xu is not None:
    w_he_xu = so3_log(R_he_xu)
    R_err_xu = np.rad2deg(np.arccos(np.clip(
        (np.trace(R_he_xu.T @ R_he_gt) - 1) / 2, -1, 1)))
    print(f"\nXu 2022 初值: R_err={R_err_xu:.4f}°")
else:
    w_he_xu = np.zeros(3)
    print("\nXu 2022 初值: 失败, 用零向量")

# ══════════════════════════════════════════════════════
# M3: 传感器帧预测 + gauge 固定 (auto_calib_v2 用的)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("M3: 传感器帧预测 + gauge 固定")

# 先看 init_principle_12dof 的 fix_C_proj
theta_nom, fix_C_proj = init_principle_12dof(poses, meas, R_he_nom, t_he_nom)
print(f"  init fix_C_proj = {fix_C_proj:.3f}m  (板估计高度)")

# 用地面真实验证 fix_C_proj
# n_B from initial estimate
n_B_init = so3_exp(theta_nom[6:9])[:, 2]
# 实际 C 在哪？用 GT 手眼反算
C_est_from_gt = []
for (R_i, t_i), m in zip(poses, meas):
    R_BS = R_i @ R_he_gt
    t_BS = t_i + R_i @ t_he_gt
    if m['valid_e1']:
        C_est_from_gt.append(R_BS @ m['p_S_e1'] + t_BS)
    if m['valid_e2']:
        C_est_from_gt.append(R_BS @ m['p_S_e2'] + t_BS)
C_gt_est = np.mean(C_est_from_gt, axis=0)
d_B_gt = np.dot(n_B_init, C_gt_est)
print(f"  GT估计板高度 d_B ≈ {d_B_gt:.3f}m")
print(f"  fix_C_proj vs d_B 偏差: {abs(fix_C_proj - d_B_gt)*1000:.1f}mm")

t3, i3 = solve_principle_12dof_with_restarts(
    poses, meas, R_he_nom, t_he_nom,
    n_restarts=20, verbose=False, w_he_init=w_he_xu)

if t3 is not None:
    R3 = so3_exp(t3[0:3])
    dR3 = R3.T @ R_he_gt
    re3 = np.rad2deg(np.arccos(np.clip(
        (np.trace(dR3) - 1) / 2, -1, 1)))
    te3 = np.linalg.norm(t3[3:6] - t_he_gt) * 1000
    C3 = t3[9:12]
    Ce3 = np.linalg.norm(C3 - C_gt_est) * 1000
    flag = '🎉' if re3 < 0.1 else '✅' if re3 < 1.0 else '⚠' if re3 < 5.0 else '❌'
    print(f"  {flag} M3: R_err={re3:.4f}°  t_err={te3:.2f}mm  "
          f"C_err={Ce3:.1f}mm  cost={i3['best_cost']:.2e}")
else:
    print("  ❌ M3 求解失败")

# ══════════════════════════════════════════════════════
# M1: C-anchored cross-product (verify_12dof.py 用的)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("M1: C-anchored cross-product (12-DOF, 无 gauge 固定)")

# 用 calibrate.py 的 residuals_12dof
try:
    from calib_solver import residuals_12dof as residuals_12dof_calib
except ImportError:
    # 如果有独立的 calibrate.py 版本
    sys.path.insert(0, '/home/z/research_contact_handeye/verification/Sim')
    # 不 import calibrate (有 main guard), 直接用 calib_solver 版本
    residuals_12dof_calib = None

# 用 calib_solver 自带的 residuals_12dof (pairwise 版——这个有 C gauge)
# 我们需要的是 calibrate.py 的 C-anchored 版本
# 直接在这里实现 C-anchored cross-product LM

def residuals_12dof_c_anchored(theta, poses, meas):
    """
    C-anchored cross-product residuals (M1).
    theta = [w_he(3), t_he(3), w_pl(3), C(3)]
    """
    w_he, t_he, w_pl, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    residuals = []
    for (R_i, t_i), m in zip(poses, meas):
        # 平面约束
        for pS in m.get('p_S_plane', []):
            pB = R_i @ (R_he @ pS + t_he) + t_i
            residuals.append(float(n_B @ (pB - C)))

        # 边1约束: cross(p_e1 - C, u_B)
        if m.get('valid_e1') and m['p_S_e1'] is not None:
            pB_e1 = R_i @ (R_he @ m['p_S_e1'] + t_he) + t_i
            r = np.cross(pB_e1 - C, u_B)
            residuals.extend(r.tolist())

        # 边2约束: cross(p_e2 - C, v_B)
        if m.get('valid_e2') and m['p_S_e2'] is not None:
            pB_e2 = R_i @ (R_he @ m['p_S_e2'] + t_he) + t_i
            r = np.cross(pB_e2 - C, v_B)
            residuals.extend(r.tolist())

    return np.array(residuals)


def lm_c_anchored(theta_init, poses, meas, max_iter=500):
    """LM for C-anchored cross-product (12-DOF, no gauge fixing)"""
    theta = theta_init.copy()
    lam = 1e-4

    for iteration in range(max_iter):
        r = residuals_12dof_c_anchored(theta, poses, meas)
        cost = 0.5 * np.dot(r, r)

        # 数值 Jacobian
        eps = 1e-6
        J = np.zeros((len(r), 12))
        for k in range(12):
            sp = theta.copy(); sp[k] += eps
            sm = theta.copy(); sm[k] -= eps
            rp = residuals_12dof_c_anchored(sp, poses, meas)
            rm = residuals_12dof_c_anchored(sm, poses, meas)
            J[:, k] = (rp - rm) / (2 * eps)

        # LM step
        try:
            delta = -np.linalg.solve(J.T @ J + lam * np.eye(12), J.T @ r)
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        tn = theta + delta
        rn = residuals_12dof_c_anchored(tn, poses, meas)
        cn = 0.5 * np.dot(rn, rn)

        if cn < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
        else:
            lam = min(lam * 3, 1e6)

        if abs(cost - cn) < 1e-12:
            break

    return theta, cost, iteration + 1


# 初始化 M1
# 用与 M3 相同的 init (PCA)
theta_nom_m1 = np.concatenate([
    so3_log(R_he_nom),      # w_he
    t_he_nom,                # t_he
    theta_nom[6:9],          # w_pl (复用 M3 init)
    theta_nom[9:12],         # C (复用 M3 init, 已经在板面上)
])

# 多重重启
rng_m1 = np.random.RandomState(42)
best_cost_m1 = float('inf')
best_theta_m1 = None
n_good_m1 = 0

for trial in range(21):  # Xu-init + 20 random
    if trial == 0:
        ti = theta_nom_m1.copy()
        if R_he_xu is not None:
            ti[0:3] = so3_log(R_he_xu)
    else:
        ti = theta_nom_m1.copy()
        ax = rng_m1.randn(3); ax /= np.linalg.norm(ax)
        ti[0:3] += ax * rng_m1.uniform(0, 0.5)

    to, cost, iters = lm_c_anchored(ti, poses, meas, max_iter=200)

    if cost < best_cost_m1:
        best_cost_m1 = cost
        best_theta_m1 = to

    if cost < 1e-4:
        n_good_m1 += 1

if best_theta_m1 is not None:
    R1 = so3_exp(best_theta_m1[0:3])
    dR1 = R1.T @ R_he_gt
    re1 = np.rad2deg(np.arccos(np.clip(
        (np.trace(dR1) - 1) / 2, -1, 1)))
    te1 = np.linalg.norm(best_theta_m1[3:6] - t_he_gt) * 1000
    C1 = best_theta_m1[9:12]
    Ce1 = np.linalg.norm(C1 - C_gt_est) * 1000

    # Jacobian 秩诊断
    r0 = residuals_12dof_c_anchored(best_theta_m1, poses, meas)
    eps_j = 1e-6
    J_diag = np.zeros((len(r0), 12))
    for k in range(12):
        sp = best_theta_m1.copy(); sp[k] += eps_j
        sm = best_theta_m1.copy(); sm[k] -= eps_j
        rp = residuals_12dof_c_anchored(sp, poses, meas)
        rm = residuals_12dof_c_anchored(sm, poses, meas)
        J_diag[:, k] = (rp - rm) / (2 * eps_j)
    _, sv, _ = np.linalg.svd(J_diag, full_matrices=False)
    cond = sv[0] / sv[-1] if sv[-1] > 1e-15 else float('inf')
    rank = sum(sv > 1e-10)

    flag = '🎉' if re1 < 0.1 else '✅' if re1 < 1.0 else '⚠' if re1 < 5.0 else '❌'
    print(f"  {flag} M1: R_err={re1:.4f}°  t_err={te1:.2f}mm  "
          f"C_err={Ce1:.1f}mm  cost={best_cost_m1:.2e}")
    print(f"      cond(J)={cond:.2e}  rank={rank}/12  "
          f"n_good={n_good_m1}/21")

    # 对比两种 C 估计
    print(f"      M1 C_est = [{C1[0]:.3f}, {C1[1]:.3f}, {C1[2]:.3f}]")
    print(f"      GT C_est = [{C_gt_est[0]:.3f}, {C_gt_est[1]:.3f}, {C_gt_est[2]:.3f}]")

    if re1 < 0.1:
        print("\n  ✅✅✅ M1 (C-anchored) 验证通过 —— 方法正确")
    elif re1 < 1.0:
        print("\n  ✅ M1 结果可用，误差 < 1°")
    else:
        print(f"\n  ❌ M1 结果差 (R_err={re1:.1f}°) —— 需要排查")
else:
    print("  ❌ M1 求解失败")

# ══════════════════════════════════════════════════════
# 总结
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("总结")
print(f"  名义手眼:    R_err={R_err_nom:.2f}°  t_err={t_err_nom:.1f}mm")
if t3 is not None:
    print(f"  M3 (gauge):  R_err={re3:.4f}°  t_err={te3:.2f}mm  "
          f"(fix_C_proj={fix_C_proj:.3f}m)")
if best_theta_m1 is not None:
    print(f"  M1 (anchored): R_err={re1:.4f}°  t_err={te1:.2f}mm  rank={rank}/12")
