#!/usr/bin/env python3
"""
verify_new_method.py — 新方法数值验证 (T1-T4)

基于《最新思路》Sec 13-14:
  T1: 零残差测试
  T2: 解析 Jacobian 有限差分检查
  T3: Schur 补等价性测试
  T4: 坐标变换一致性
"""

import sys
import json
import numpy as np

sys.path.insert(0, '/home/z/research_contact_handeye/verification/Sim/new_method')
from observation_model import (
    so3_exp, so3_log, skew,
    unpack_params, transform_point,
    compute_residuals, compute_jacobian, compute_jacobian_numerical,
    schur_handeye, normalize_hessian,
)
from solver import solve_lm, calibrate


# ══════════════════════════════════════════════════════════════
# 测试数据生成
# ══════════════════════════════════════════════════════════════

def generate_test_data(n_poses: int = 5, seed: int = 42,
                       noise_std: float = 0.0):
    """生成合成标定数据

    方法: 在平板坐标系生成点 → 变换到基座标 → 逆变换到传感器帧
    确保所有生成点精确满足几何约束。

    Returns:
        poses, meas, theta_true
    """
    rng = np.random.RandomState(seed)

    # 真实手眼
    w_X_true = rng.randn(3) * 0.3
    R_X_true = so3_exp(w_X_true)
    t_X_true = rng.randn(3) * 0.05  # 5cm

    # 真实平板
    w_P_true = rng.randn(3) * 0.2
    R_P_true = so3_exp(w_P_true)
    u_true, v_true, n_true = R_P_true[:, 0], R_P_true[:, 1], R_P_true[:, 2]
    C_true = rng.randn(3) * 0.5

    theta_true = np.concatenate([w_X_true, t_X_true, w_P_true, C_true])

    poses, meas = [], []
    for i in range(n_poses):
        # 法兰姿态: 多样化 tilt
        axis = rng.randn(3); axis /= np.linalg.norm(axis)
        angle = (0.2 + 0.5 * i / n_poses) * np.pi
        R_i = so3_exp(axis * angle)
        t_i = rng.randn(3) * 0.3

        poses.append((R_i, t_i))

        meas_i = _generate_measurement(
            R_i, t_i, R_X_true, t_X_true,
            u_true, v_true, n_true, C_true,
            rng, noise_std, plate_halfsize=0.15)
        meas.append(meas_i)

    return poses, meas, theta_true


def _generate_measurement(R_i, t_i, R_X, t_X, u, v, n, C, rng, noise_std, plate_halfsize):
    """生成单个位姿的测量数据

    策略: 在传感器帧中直接生成满足约束的点。
    
    1. 随机选传感器帧中的 x 坐标
    2. 解两个约束 (平面 + 某条边) → 得到 z 坐标  
    3. 验证 y_S=0 已满足
    
    对 edge 1: 约束为 n^T(p_B-C)=0 AND v^T(p_B-C)=0
    对 edge 2: 约束为 n^T(p_B-C)=0 AND u^T(p_B-C)=0
    
    p_B = R_i(R_X [x,0,z]^T + t_X) + t_i
    
    展开:
      n^T (R_i R_X[:,0]*x + R_i R_X[:,2]*z + R_i t_X + t_i - C) = 0
      v^T (R_i R_X[:,0]*x + R_i R_X[:,2]*z + R_i t_X + t_i - C) = 0  (edge 1)
    
    2 方程, 2 未知 (x,z) → 可解
    """
    RiRX = R_i @ R_X
    v0 = R_i @ t_X + t_i - C
    
    def solve_for_xz(axis_dir):
        """解线性系统得到 (x, z) 使得同时满足平面和边约束"""
        # 系数矩阵: [n^T  RiRX[:,0],  n^T  RiRX[:,2]]
        #           [axis^T RiRX[:,0], axis^T RiRX[:,2]]
        A = np.array([
            [n @ RiRX[:, 0], n @ RiRX[:, 2]],
            [axis_dir @ RiRX[:, 0], axis_dir @ RiRX[:, 2]],
        ])
        b_vec = np.array([-n @ v0, -axis_dir @ v0])
        
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-10:
            return None
        return np.linalg.solve(A, b_vec)
    
    # 找 edge 1 点
    e1_xz = solve_for_xz(v)  # edge 1: v^T(p-C)=0
    # 找 edge 2 点  
    e2_xz = solve_for_xz(u)  # edge 2: u^T(p-C)=0
    
    if e1_xz is None or e2_xz is None:
        return {'p_S_plane': [], 'p_S_e1': None, 'p_S_e2': None,
                'valid_e1': False, 'valid_e2': False}
    
    e1_S = np.array([e1_xz[0], 0.0, e1_xz[1]])
    e2_S = np.array([e2_xz[0], 0.0, e2_xz[1]])
    
    # 注意: 不要基于 x 坐标交换 e1/e2 — 它们对应不同的物理边
    # e1 在 edge 1 (沿 u) 上, e2 在 edge 2 (沿 v) 上
    # 残差计算依赖这个对应关系
    
    # 在 e1 和 e2 之间的线段上采样平面点
    n_plane = rng.randint(20, 50)
    plane_pts = []
    for _ in range(n_plane):
        alpha = rng.uniform(0, 1)
        q_S = e1_S + alpha * (e2_S - e1_S)
        if noise_std > 0:
            q_S = q_S.copy()
            q_S[0] += rng.normal(0, noise_std * 0.5e-3)
            q_S[2] += rng.normal(0, noise_std * 1e-3)
        plane_pts.append(q_S)
    
    if noise_std > 0:
        e1_S = e1_S.copy()
        e2_S = e2_S.copy()
        e1_S[0] += rng.normal(0, noise_std * 1e-3)
        e1_S[2] += rng.normal(0, noise_std * 2e-3)
        e2_S[0] += rng.normal(0, noise_std * 1e-3)
        e2_S[2] += rng.normal(0, noise_std * 2e-3)
    
    return {
        'p_S_plane': plane_pts,
        'p_S_e1': e1_S,
        'p_S_e2': e2_S,
        'valid_e1': True,
        'valid_e2': True,
    }


# ══════════════════════════════════════════════════════════════
# T1: 零残差测试
# ══════════════════════════════════════════════════════════════

def test_zero_residual():
    print("=" * 60)
    print("T1: 零残差测试")
    print("=" * 60)

    poses, meas, theta_true = generate_test_data(n_poses=5, seed=42, noise_std=0.0)
    r, _, _, _, info = compute_residuals(theta_true, poses, meas)
    r_max = np.max(np.abs(r))

    print(f"  残差数: {len(r)} (plane={info['n_plane']}, e1={info['n_e1']}, e2={info['n_e2']})")
    print(f"  max|r| = {r_max:.2e}")

    if r_max < 1e-10:
        print("  ✅ 零残差测试通过")
    elif r_max < 1e-8:
        print("  ⚠️  残差接近数值精度")
    else:
        print(f"  ❌ 零残差测试失败 (max|r|={r_max:.2e})")

    return r_max < 1e-8


# ══════════════════════════════════════════════════════════════
# T2: Jacobian 有限差分检查
# ══════════════════════════════════════════════════════════════

def test_jacobian_fd():
    print("\n" + "=" * 60)
    print("T2: 解析 Jacobian 有限差分检查")
    print("=" * 60)

    poses, meas, theta_true = generate_test_data(n_poses=5, seed=123, noise_std=0.0)

    # 在真值附近测试（扰动一点避免对称简化）
    theta_test = theta_true + np.random.RandomState(0).randn(12) * 0.01

    J_analytic, r, _ = compute_jacobian(theta_test, poses, meas)
    J_numeric, _, _ = compute_jacobian_numerical(theta_test, poses, meas, eps=1e-7)

    # 相对误差 Eq.(23)
    denom = max(1.0, np.linalg.norm(J_numeric, 'fro'))
    eps_J = np.linalg.norm(J_analytic - J_numeric, 'fro') / denom

    print(f"  ||J_analytic - J_numeric||_F / max(1,||J_numeric||_F) = {eps_J:.2e}")
    print(f"  J shape: {J_analytic.shape}")

    if eps_J < 1e-6:
        print("  ✅ Jacobian 有限差分检查通过")
    elif eps_J < 1e-4:
        print(f"  ⚠️  误差较大 ({eps_J:.2e}), 检查步长或解析式")
    else:
        print(f"  ❌ Jacobian 验证失败 ({eps_J:.2e})")

    # 逐列检查
    col_errs = []
    for k in range(12):
        col_norm = np.linalg.norm(J_numeric[:, k])
        if col_norm > 1e-10:
            col_err = np.linalg.norm(J_analytic[:, k] - J_numeric[:, k]) / col_norm
            col_errs.append((k, col_err))
    col_errs.sort(key=lambda x: -x[1])
    if col_errs:
        print(f"  最大列误差: 列{col_errs[0][0]} = {col_errs[0][1]:.2e}")
        if len(col_errs) > 1:
            print(f"  次大列误差: 列{col_errs[1][0]} = {col_errs[1][1]:.2e}")

    return eps_J < 1e-4


# ══════════════════════════════════════════════════════════════
# T3: Schur 补等价性测试
# ══════════════════════════════════════════════════════════════

def test_schur_equivalence():
    print("\n" + "=" * 60)
    print("T3: Schur 补等价性测试")
    print("=" * 60)

    poses, meas, theta_true = generate_test_data(n_poses=5, seed=456, noise_std=0.0)
    J, r, info = compute_jacobian(theta_true, poses, meas)

    # Schur 补
    H_eff = schur_handeye(J)
    assert H_eff.shape == (6, 6)

    # 随机扰动 δx
    rng = np.random.RandomState(0)
    J_X = J[:, 0:6]
    J_Pi = J[:, 6:12]

    n_test = 20
    max_rel_err = 0.0
    for _ in range(n_test):
        dx = rng.randn(6) * 0.01

        # q1 = dx^T H_eff dx
        q1 = dx @ H_eff @ dx

        # q2 = min_{dπ} ||J_X dx + J_Pi dπ||^2
        # Least squares: dπ = -J_Pi^† J_X dx
        dpi_opt, residuals, rank, sv = np.linalg.lstsq(J_Pi, -J_X @ dx, rcond=None)
        q2 = np.sum((J_X @ dx + J_Pi @ dpi_opt) ** 2)

        rel_err = abs(q1 - q2) / max(1.0, q2)
        max_rel_err = max(max_rel_err, rel_err)

    print(f"  {n_test} 次随机 δx 测试")
    print(f"  最大相对误差: {max_rel_err:.2e}")

    if max_rel_err < 1e-6:
        print("  ✅ Schur 补等价性测试通过")
    elif max_rel_err < 1e-4:
        print(f"  ⚠️  误差略大 ({max_rel_err:.2e})")
    else:
        print(f"  ❌ Schur 补测试失败 ({max_rel_err:.2e})")

    return max_rel_err < 1e-4


# ══════════════════════════════════════════════════════════════
# T4: 坐标变换一致性
# ══════════════════════════════════════════════════════════════

def test_coordinate_consistency():
    print("\n" + "=" * 60)
    print("T4: 坐标变换一致性")
    print("=" * 60)

    poses, meas, theta_true = generate_test_data(n_poses=5, seed=789, noise_std=0.0)

    # 原始残差
    r_orig, _, _, _, _ = compute_residuals(theta_true, poses, meas)
    cost_orig = 0.5 * np.dot(r_orig, r_orig)

    # 施加一个基座标系变换: T_B' = T_B * T_offset
    # 新基座标系中: R_i' = R_i @ R_offset^T, t_i' = t_i - R_i @ R_offset^T @ t_offset
    # 更简单的: 在基座标系施加刚体变换后，残差应不变
    w_offset = np.random.RandomState(1).randn(3) * 0.3
    R_offset = so3_exp(w_offset)
    t_offset = np.random.RandomState(2).randn(3) * 0.5

    # 变换 poses: p_B_new = R_offset @ p_B + t_offset
    # → p_B_new = R_offset (R_i (R_X q + t_X) + t_i) + t_offset
    # 这等价于 R_i_new = R_offset @ R_i, t_i_new = R_offset @ t_i + t_offset
    new_poses = [(R_offset @ R_i, R_offset @ t_i + t_offset) for R_i, t_i in poses]

    # 变换 theta: C_new = R_offset @ C + t_offset
    #             n_new = R_offset @ n, u_new = R_offset @ u, v_new = R_offset @ v
    #             手眼不变
    R_P = so3_exp(theta_true[6:9])
    R_P_new = R_offset @ R_P
    w_P_new = so3_log(R_P_new)
    C = theta_true[9:12]
    C_new = R_offset @ C + t_offset

    theta_new = theta_true.copy()
    theta_new[6:9] = w_P_new
    theta_new[9:12] = C_new
    # 手眼不变

    r_new, _, _, _, _ = compute_residuals(theta_new, new_poses, meas)
    cost_new = 0.5 * np.dot(r_new, r_new)

    rel_err = abs(cost_orig - cost_new) / max(1.0, cost_orig)

    print(f"  原始 cost: {cost_orig:.2e}")
    print(f"  变换 cost: {cost_new:.2e}")
    print(f"  相对误差:  {rel_err:.2e}")

    if rel_err < 1e-10:
        print("  ✅ 坐标变换一致性测试通过")
    elif rel_err < 1e-6:
        print("  ⚠️  微小差异 (数值精度)")
    else:
        print(f"  ❌ 坐标一致性失败 ({rel_err:.2e})")

    return rel_err < 1e-6


# ══════════════════════════════════════════════════════════════
# T5: 优化收敛测试
# ══════════════════════════════════════════════════════════════

def test_optimization():
    print("\n" + "=" * 60)
    print("T5: 优化收敛测试 (无噪声)")
    print("=" * 60)

    poses, meas, theta_true = generate_test_data(n_poses=5, seed=42, noise_std=0.0)

    # 从零初值出发
    theta_init = np.zeros(12)
    theta_opt, info = solve_lm(theta_init, poses, meas, max_iter=200, verbose=False)

    R_err, t_err = None, None
    from observation_model import compute_errors
    R_err, t_err = compute_errors(theta_opt, theta_true)

    print(f"  cost: {info['cost']:.2e}  n_iter: {info['n_iter']}")
    print(f"  R_err: {R_err:.4f}°  t_err: {t_err:.2f}mm")

    if R_err is not None and R_err < 0.01:
        print("  ✅ 优化收敛到真值")
    elif R_err is not None and R_err < 1.0:
        print("  ⚠️  接近真值但不精确")
    else:
        print(f"  ❌ 优化未收敛 (R_err={R_err:.4f}°)")

    return R_err is not None and R_err < 1.0


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("新方法数值验证 (T1-T5)")
    print("=" * 60)

    results = {}
    results['T1_zero_residual'] = test_zero_residual()
    results['T2_jacobian_fd'] = test_jacobian_fd()
    results['T3_schur'] = test_schur_equivalence()
    results['T4_coordinate'] = test_coordinate_consistency()
    results['T5_optimization'] = test_optimization()

    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  🎉 全部测试通过！")
    else:
        print("\n  ⚠️  部分测试未通过，需要修改")
