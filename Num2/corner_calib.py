"""
corner_calib.py — Num2 角点法标定求解器 (CODE_REPORT.md §2.6, §3.4)
12-DOF LM on manifold
"""

import numpy as np
from corner_scene import so3_exp, so3_log


# ============================================================================
# 残差计算 (CODE_REPORT.md §2.2)
# ============================================================================

def compute_residuals(theta, poses, measurements, alpha=np.pi/2):
    """计算所有残差

    θ = [ω_he(3), t_he(3), ω_pl(3), C(3)]

    残差:
      边1 (2D): 断点在传感器系 (x,z) 的预测误差
      边2 (2D): 同上
      平面 (1D): 变换后点到平板的距离 n_B·(p_B - C)

    Returns: residuals, mask
    """
    w_he, t_he, w_pl, C = (theta[0:3], theta[3:6],
                            theta[6:9], theta[9:12])
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]
    d_1 = u_B
    d_2 = np.cos(alpha) * u_B + np.sin(alpha) * v_B

    residuals = []
    mask = []

    for (R_i, t_i), m in zip(poses, measurements):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he

        # ---- 边1残差 (2D) ----
        if m['valid_e1']:
            n_laser = R_BS[:, 1]
            denom = np.dot(n_laser, d_1)
            if abs(denom) > 1e-12:
                s1 = np.dot(n_laser, t_BS - C) / denom
                p_B_pred = C + s1 * d_1
                p_S_pred = R_he.T @ (R_i.T @ (p_B_pred - t_i) - t_he)
                residuals.extend([p_S_pred[0] - m['p_S_e1'][0],
                                  p_S_pred[2] - m['p_S_e1'][2]])
                mask.extend([True, True])
            else:
                residuals.extend([0.0, 0.0])
                mask.extend([False, False])
        else:
            residuals.extend([0.0, 0.0])
            mask.extend([False, False])

        # ---- 边2残差 (2D) ----
        if m['valid_e2']:
            n_laser = R_BS[:, 1]
            denom = np.dot(n_laser, d_2)
            if abs(denom) > 1e-12:
                s2 = np.dot(n_laser, t_BS - C) / denom
                p_B_pred = C + s2 * d_2
                p_S_pred = R_he.T @ (R_i.T @ (p_B_pred - t_i) - t_he)
                residuals.extend([p_S_pred[0] - m['p_S_e2'][0],
                                  p_S_pred[2] - m['p_S_e2'][2]])
                mask.extend([True, True])
            else:
                residuals.extend([0.0, 0.0])
                mask.extend([False, False])
        else:
            residuals.extend([0.0, 0.0])
            mask.extend([False, False])

        # ---- 平面点残差 (1D each) ----
        for p_S in m['p_S_plane']:
            p_B = R_BS @ p_S + t_BS
            d_val = np.dot(n_B, p_B - C)
            residuals.append(d_val)
            mask.append(True)

    return np.array(residuals), np.array(mask)


def compute_cost(theta, poses, measurements, alpha=np.pi/2):
    """总代价 = 0.5 * Σ r²"""
    r, mask = compute_residuals(theta, poses, measurements, alpha)
    r_valid = r[mask]
    return 0.5 * np.dot(r_valid, r_valid) if len(r_valid) > 0 else 1e30


# ============================================================================
# 数值 Jacobian (中心差分, ε=1e-6)
# ============================================================================

def compute_jacobian_numerical(theta, poses, measurements, alpha=np.pi/2, eps=1e-6):
    """中心差分 Jacobian"""
    r0, mask = compute_residuals(theta, poses, measurements, alpha)
    n_res = len(r0)
    J = np.zeros((n_res, 12))

    for k in range(12):
        step = np.zeros(12)
        step[k] = eps
        r_plus, _ = compute_residuals(theta + step, poses, measurements, alpha)
        r_minus, _ = compute_residuals(theta - step, poses, measurements, alpha)
        J[:, k] = (r_plus - r_minus) / (2 * eps)

    return J, r0, mask


# ============================================================================
# LM 求解器 (CODE_REPORT.md §2.6)
# ============================================================================

def solve_lm(theta_init, poses, measurements, alpha=np.pi/2,
             max_iter=50, tol=1e-8, lam0=1e-4, verbose=False,
             fix_C_proj=None):
    """Levenberg-Marquardt on manifold — with gauge fixing

    Gauge fixing (optional): 固定 n_B·C = fix_C_proj
      这消除了 t_he ↔ C 耦合的 1-DOF 零空间。
      如果提供 fix_C_proj, 每次参数更新后会将 C 投影到 n_B·C = fix_C_proj。
    """
    theta = theta_init.copy()
    lam = lam0
    cost_history = []
    converged = False

    for iteration in range(max_iter):
        J, r, mask = compute_jacobian_numerical(theta, poses, measurements, alpha)
        r_valid = r[mask]
        J_valid = J[mask, :]

        if len(r_valid) == 0:
            if verbose:
                print(f"  iter {iteration}: no valid residuals")
            break

        cost = 0.5 * np.dot(r_valid, r_valid)
        cost_history.append(cost)

        if verbose and iteration % 5 == 0:
            print(f"  iter {iteration}: cost={cost:.6e}, λ={lam:.2e}")

        # Gauss-Newton 步
        H = J_valid.T @ J_valid
        g = J_valid.T @ r_valid
        H_damped = H + lam * np.eye(12)

        try:
            delta = -np.linalg.solve(H_damped, g)
        except np.linalg.LinAlgError:
            lam *= 10
            continue

        theta_new = theta + delta

        # Gauge 固定: n_B·C = constant (消除 t_he ↔ C 耦合)
        if fix_C_proj is not None:
            nB_new = so3_exp(theta_new[6:9])[:, 2]
            C_new = theta_new[9:12]
            current_proj = np.dot(nB_new, C_new)
            C_new = C_new - (current_proj - fix_C_proj) * nB_new
            theta_new[9:12] = C_new

        cost_new = compute_cost(theta_new, poses, measurements, alpha)

        if cost_new < cost:
            theta = theta_new
            lam = max(lam / 3, 1e-8)
            if abs(cost - cost_new) < tol:
                converged = True
                break
        else:
            lam = min(lam * 3, 1e6)
    else:
        converged = False

    return theta, converged, cost_history


# ============================================================================
# 误差评估 (CODE_REPORT.md §2.6)
# ============================================================================

def compute_errors(theta_est, theta_gt):
    """旋转误差 (°) 和平移误差 (mm)"""
    w_he_est, w_he_gt = theta_est[0:3], theta_gt[0:3]
    t_he_est, t_he_gt = theta_est[3:6], theta_gt[3:6]

    R_est = so3_exp(w_he_est)
    R_gt = so3_exp(w_he_gt)
    R_diff = R_est.T @ R_gt
    trace_val = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    R_error_deg = np.rad2deg(np.arccos(trace_val))

    t_error_mm = np.linalg.norm(t_he_est - t_he_gt) * 1000

    return R_error_deg, t_error_mm
