#!/usr/bin/env python3
"""
calibrate.py — 标定管线：闭式解 + 12-DOF LM 精化

用法: PYTHONPATH=./common python3 calibrate.py [data_file]
"""

import json, sys, os, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from calib_solver import solve_he_closed_form, solve_he_t_cf
from calib_solver import solve_12dof_with_restarts
from fov_geometry import so3_exp, so3_log


def load_data(data_file):
    with open(data_file) as f:
        data = json.load(f)
    raw = data['poses']

    has_Ri = 'R_i' in raw[0]
    poses, meas = [], []
    for p in raw:
        if has_Ri:
            R_i, t_i = np.array(p['R_i']), np.array(p['t_i'])
        else:
            R_i, t_i = np.array(p['R']), np.array(p['t'])
        poses.append((R_i, t_i))
        meas.append({
            'p_S_e1': np.array(p['p_S_e1']) if p.get('p_S_e1') else None,
            'p_S_e2': np.array(p['p_S_e2']) if p.get('p_S_e2') else None,
            'valid_e1': p.get('valid_e1', False),
            'valid_e2': p.get('valid_e2', False),
            'p_S_plane': p.get('scan_pts_S', []),
        })

    scene = data.get('scene', None)
    return poses, meas, scene


# ============================================================
# 12-DOF 残差与求解器
# ============================================================

def residuals_12dof(theta, poses, meas):
    """theta = [w_he(3), t_he(3), w_pl(3), C(3)]"""
    w_he, t_he = theta[0:3], theta[3:6]
    w_pl, C = theta[6:9], theta[9:12]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    residuals = []
    for (R_i, t_i), m in zip(poses, meas):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he
        if m.get('valid_e1') and m.get('p_S_e1') is not None:
            p_B = R_BS @ m['p_S_e1'] + t_BS
            residuals.extend(np.cross(p_B - C, u_B).tolist())
        if m.get('valid_e2') and m.get('p_S_e2') is not None:
            p_B = R_BS @ m['p_S_e2'] + t_BS
            residuals.extend(np.cross(p_B - C, v_B).tolist())
        for p_S in m.get('p_S_plane', []):
            p_B = R_BS @ p_S + t_BS
            residuals.append(float(n_B @ (p_B - C)))
    return np.array(residuals)


def solve_12dof_lm(theta_init, poses, meas, max_iter=500, tol=1e-12, verbose=False):
    theta = theta_init.copy()
    lam = 1e-4
    for it in range(max_iter):
        r = residuals_12dof(theta, poses, meas)
        if len(r) == 0:
            break
        cost = 0.5 * np.dot(r, r)
        J = np.zeros((len(r), 12))
        eps = 1e-6
        for j in range(12):
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            rp = residuals_12dof(tp, poses, meas)
            rm = residuals_12dof(tm, poses, meas)
            J[:, j] = (rp - rm) / (2 * eps)
        H = J.T @ J
        g = J.T @ r
        try:
            delta = -np.linalg.solve(H + lam * np.eye(12), g)
        except np.linalg.LinAlgError:
            lam *= 10; continue
        new_theta = theta + delta
        r_new = residuals_12dof(new_theta, poses, meas)
        new_cost = 0.5 * np.dot(r_new, r_new)
        if new_cost < cost:
            theta = new_theta
            lam = max(lam / 3, 1e-12)
            if verbose and it % 10 == 0:
                print(f"  iter {it}: cost={cost:.3e} λ={lam:.1e}")
            if abs(cost - new_cost) < tol:
                break
        else:
            lam = min(lam * 3, 1e6)
    return theta


def _group_poses_by_orientation(poses, meas, angle_threshold_deg=5.0):
    """按法兰朝向分组, 每组返回 (代表R, [(R_i,t_i,m), ...])"""
    groups = []
    used = set()
    for i, ((R_i, t_i), m) in enumerate(zip(poses, meas)):
        if i in used:
            continue
        group = [(R_i, t_i, m)]
        used.add(i)
        for j, ((R_j, t_j), m2) in enumerate(zip(poses, meas)):
            if j in used:
                continue
            # 两朝向夹角 < 阈值 → 同组
            dR = R_i.T @ R_j
            tr = np.clip((np.trace(dR) - 1) / 2, -1, 1)
            angle = np.rad2deg(np.arccos(tr))
            if angle < angle_threshold_deg:
                group.append((R_j, t_j, m2))
                used.add(j)
        groups.append((R_i, group))
    return groups


def init_12dof(poses, meas, R_he_nom=None, t_he_nom=None):
    """边线交点法初始化 12-DOF (使用名义手眼产线 C)
    
    闭式解 solve_he_closed_form / solve_he_t_cf 尚未实现，
    直接用名义值替代，C 通过边线交点计算。
    """
    if R_he_nom is None:
        R_cf, _ = solve_he_closed_form(poses, meas)
        if R_cf is None: R_cf = np.eye(3)
    else:
        R_cf = R_he_nom
    
    if t_he_nom is None:
        (_, t_cf), _ = solve_he_t_cf(poses, meas, R_cf_in=R_cf)
        if np.linalg.norm(t_cf) < 1e-6: t_cf = np.zeros(3)
    else:
        t_cf = t_he_nom

    # 估算边方向和 C
    groups = _group_poses_by_orientation(poses, meas, angle_threshold_deg=5.0)
    p_be1, p_be2 = [], []
    for R_b, group in groups:
        for R_i, t_i, m in group:
            R_BS = R_i @ R_cf; t_BS = t_i + R_i @ t_cf
            if m.get('valid_e1') and m.get('p_S_e1') is not None:
                p_be1.append(R_BS @ m['p_S_e1'] + t_BS)
            if m.get('valid_e2') and m.get('p_S_e2') is not None:
                p_be2.append(R_BS @ m['p_S_e2'] + t_BS)

    def fit_dir(pts):
        if len(pts) < 2: return np.array([1., 0., 0.])
        U, S, Vt = np.linalg.svd(np.array(pts) - np.mean(pts, axis=0))
        return Vt[0]

    u_B = fit_dir(p_be1)
    v_B = fit_dir(p_be2)
    n_B = np.cross(u_B, v_B)
    if np.linalg.norm(n_B) > 1e-6:
        n_B /= np.linalg.norm(n_B)
    else:
        n_B = np.array([0., 0., 1.])
    v_B = np.cross(n_B, u_B)
    v_B /= np.linalg.norm(v_B)
    R_pl = np.column_stack([u_B, v_B, n_B])

    # C: 边线交点
    def skew(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    p1r = np.mean(p_be1, axis=0) if p_be1 else np.zeros(3)
    p2r = np.mean(p_be2, axis=0) if p_be2 else np.zeros(3)
    Ac = np.vstack([skew(u_B), skew(v_B), n_B.reshape(1, 3)])
    bc = np.hstack([skew(u_B) @ p1r, skew(v_B) @ p2r, [np.dot(n_B, (p1r + p2r) / 2)]])
    C, _, _, _ = np.linalg.lstsq(Ac, bc, rcond=None)

    w_he = so3_log(R_cf)
    w_pl = so3_log(R_pl)
    return np.concatenate([w_he, t_cf, w_pl, C])


def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/recorded_poses.json')
    if not os.path.exists(data_file):
        print(f"[ERROR] 找不到 {data_file}")
        return 1

    poses, meas, scene = load_data(data_file)
    n_e1 = sum(1 for m in meas if m['valid_e1'])
    n_e2 = sum(1 for m in meas if m['valid_e2'])
    n_corner = sum(1 for m in meas if m['valid_e1'] and m['valid_e2'])
    print(f"加载 {len(poses)} 个位姿 (e1={n_e1}, e2={n_e2}, corner={n_corner})")

    # ===== 12-DOF 多重随机重启 =====
    print("\n===== 12-DOF (20 restarts) =====")
    theta_opt, rs_info = solve_12dof_with_restarts(poses, meas, n_restarts=20, verbose=True)

    R_opt = so3_exp(theta_opt[0:3])
    t_opt = theta_opt[3:6]
    R_pl_opt = so3_exp(theta_opt[6:9])
    C_opt = theta_opt[9:12]

    print(f"\n  {rs_info['n_good']}/{rs_info['n_tried']} restarts converged "
          f"(best cost={rs_info['best_cost']:.3e})")
    print(f"  R_he (axis-angle deg): {np.rad2deg(theta_opt[0:3])}")
    print(f"  t_he (mm): {t_opt * 1000}")
    print(f"  C (m): {C_opt.round(4)}")

    # ===== Jacobian SVD =====
    if '--svd' in sys.argv:
        print("\n===== Jacobian SVD =====")
        r0 = residuals_12dof(theta_opt, poses, meas)
        J = np.zeros((len(r0), 12))
        eps = 1e-6
        for j in range(12):
            tp = theta_opt.copy(); tp[j] += eps
            tm = theta_opt.copy(); tm[j] -= eps
            J[:, j] = (residuals_12dof(tp,poses,meas) - residuals_12dof(tm,poses,meas)) / (2*eps)
        S = np.linalg.svd(J, compute_uv=False)
        cond = S[0]/S[-1] if S[-1] > 1e-15 else float('inf')
        n_zero = int(np.sum(S < 1e-8))
        print(f"  cond(J)={cond:.2e}  rank={12-n_zero}/12  "
              f"σ_min={S[-1]:.2e}  {'⚠ GAUGE!' if n_zero>0 else '✓ 满秩'}")

    # ===== 真值对比 =====
    if scene:
        R_gt = np.array(scene['R_he_gt'])
        t_gt = np.array(scene['t_he_gt'])

        dR_opt = R_opt.T @ R_gt
        tr = np.clip((np.trace(dR_opt) - 1) / 2, -1, 1)
        R_opt_err = np.rad2deg(np.arccos(tr))
        t_opt_err = np.linalg.norm(t_opt - t_gt) * 1000

        print(f"\n与真值对比:")
        print(f"  12-DOF:  R={R_opt_err:.4f}°  t={t_opt_err:.2f} mm")

        if R_opt_err < 0.1 and t_opt_err < 1.0:
            print(f"\n  🎉 达标! R < 0.1°, t < 1mm")
        elif R_opt_err < 1.0:
            print(f"\n  ✅ R < 1°")
        else:
            print(f"\n  ⚠ 需检查: R 误差 > 1°")

    return 0


if __name__ == '__main__':
    sys.exit(main())
