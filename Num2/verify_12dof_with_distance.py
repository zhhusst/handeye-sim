#!/usr/bin/env python3
"""
verify_12dof_with_distance.py — 12-DOF + 边长约束 + 采集数据

未知数 12: w_he(3) + t_he(3) + w_pl(3) + C(3)

约束:
  1. 平面: n_B·p_B - mean = 0 (centered, 和之前一样)
  2. 边缘共线: cross(p_{k+1}-p_k, u_B/v_B) = 0 (每帧两个端点)
  3. ★ 边长: d_k² = s₁² + s₂² (新!)

流程:
  1. 生成采集轨迹 (40帧, 全部两边可见)
  2. 从单位阵出发 LM
  3. 对比: 有/无边长约束
"""

import numpy as np, yaml, sys, os
sys.path.insert(0, '.')
from corner_scene import so3_exp, so3_log
from reproduction_scene import compute_fov_plate_scanline

# ============================================================================
# 12-DOF 残差 + 边长约束
# ============================================================================

def residuals_12dof_with_distance(theta, poses, measurements, w_plane=1.0, w_edge=1.0, w_dist=1.0):
    """12-DOF 残差: 平面 + 边缘 + 边长"""

    w_he, t_he, w_pl, C = theta[0:3], theta[3:6], theta[6:9], theta[9:12]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    plane_vals = []
    edge_res = []
    dist_res = []

    # 按传感器帧 x 排序做硬赋值 (连续轨迹下 x 排序一致)
    edge0_S, edge1_S = [], []
    for m in measurements:
        pts = []
        if m['valid_e1']: pts.append(('e1', m['p_S_e1'].copy()))
        if m['valid_e2']: pts.append(('e2', m['p_S_e2'].copy()))
        if len(pts) >= 2:
            pts.sort(key=lambda p: p[1][0])
            edge0_S.append(pts[0][1]); edge1_S.append(pts[1][1])
        elif len(pts) == 1:
            edge0_S.append(pts[0][1]); edge1_S.append(None)
        else:
            edge0_S.append(None); edge1_S.append(None)

    # 变换到基坐标系
    edge0_B, edge1_B = [], []
    for k, ((R_i, t_i), m) in enumerate(zip(poses, measurements)):
        R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he

        if edge0_S[k] is not None:
            edge0_B.append(R_BS @ edge0_S[k] + t_BS)
        if edge1_S[k] is not None:
            edge1_B.append(R_BS @ edge1_S[k] + t_BS)

        # 平面点
        for p_S in m['p_S_plane']:
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))

    # 平面残差: centered
    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)

    # 边缘共线: edge0_B → u_B, edge1_B → v_B
    for pts in [edge0_B]:
        for k in range(len(pts) - 1):
            edge_res.extend(np.cross(pts[k+1] - pts[k], u_B).tolist())
    for pts in [edge1_B]:
        for k in range(len(pts) - 1):
            edge_res.extend(np.cross(pts[k+1] - pts[k], v_B).tolist())

    # 边长约束: d_k² = s₁² + s₂²
    for k, ((R_i, t_i), m) in enumerate(zip(poses, measurements)):
        if not (m['valid_e1'] and m['valid_e2']):
            continue

        R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he
        n_laser = R_BS[:, 1]  # 激光平面法向量
        s_pos = t_BS          # 传感器原点在基坐标系

        # 实测端点间距
        d_meas = np.linalg.norm(m['p_S_e1'] - m['p_S_e2'])

        # 预测: 激光平面与两条边的交点参数
        denom_u = np.dot(n_laser, u_B)
        denom_v = np.dot(n_laser, v_B)

        if abs(denom_u) < 1e-10 or abs(denom_v) < 1e-10:
            continue

        s1 = np.dot(n_laser, (s_pos - C)) / denom_u
        s2 = np.dot(n_laser, (s_pos - C)) / denom_v

        d_pred = np.sqrt(s1**2 + s2**2)

        # 残差: 预测间距 - 测量间距 (mm)
        dist_res.append((d_pred - d_meas) * 1000 * np.sqrt(w_dist))

    # 合并所有残差
    residuals = []
    mask = []
    wp, we = np.sqrt(w_plane), np.sqrt(w_edge)
    for v in plane_vals: residuals.append(v * wp); mask.append(True)
    for v in edge_res: residuals.append(v * we); mask.append(True)
    for v in dist_res: residuals.append(v); mask.append(True)

    info = {
        'n_plane': len(plane_vals),
        'n_edge': len(edge_res) // 3,
        'n_dist': len(dist_res),
    }

    return np.array(residuals), np.array(mask), info


def cost_12dof(theta, poses, meas, **kw):
    r, m, _ = residuals_12dof_with_distance(theta, poses, meas, **kw)
    rv = r[m.astype(bool)]
    return 0.5 * np.dot(rv, rv) if len(rv) > 0 else 1e30


def jacobian_12dof(theta, poses, meas, eps=1e-6, **kw):
    r0, mask0, info = residuals_12dof_with_distance(theta, poses, meas, **kw)
    n_fixed = len(r0)
    J = np.zeros((n_fixed, 12))
    for k in range(12):
        step = np.zeros(12); step[k] = eps
        rp, _, _ = residuals_12dof_with_distance(theta + step, poses, meas, **kw)
        rm, _, _ = residuals_12dof_with_distance(theta - step, poses, meas, **kw)
        # 填充较短的到固定长度 (用 0, 不影响梯度因为 mask 会去掉它们)
        if len(rp) < n_fixed: rp = np.pad(rp, (0, n_fixed - len(rp)))
        if len(rm) < n_fixed: rm = np.pad(rm, (0, n_fixed - len(rm)))
        if len(rp) > n_fixed: rp = rp[:n_fixed]
        if len(rm) > n_fixed: rm = rm[:n_fixed]
        J[:, k] = (rp - rm) / (2 * eps)
    return J, r0, mask0, info


def solve_lm_12dof(theta_init, poses, meas, w_plane=1.0, w_edge=1.0, w_dist=1.0,
                    max_iter=100, tol=1e-10, lam0=1e-4):
    theta = theta_init.copy()
    lam = lam0
    for it in range(max_iter):
        J, r, mask, info = jacobian_12dof(theta, poses, meas, w_plane=w_plane, w_edge=w_edge, w_dist=w_dist)
        rv = r[mask.astype(bool)]
        Jv = J[mask.astype(bool), :]
        if len(rv) == 0: break
        cost = 0.5 * np.dot(rv, rv)
        H = Jv.T @ Jv; g = Jv.T @ rv
        try:
            delta = -np.linalg.solve(H + lam * np.eye(12), g)
        except np.linalg.LinAlgError:
            lam *= 10; continue
        tn = theta + delta
        cn = cost_12dof(tn, poses, meas, w_plane=w_plane, w_edge=w_edge, w_dist=w_dist)
        if cn < cost:
            theta = tn
            lam = max(lam / 3, 1e-12)
            if abs(cost - cn) < tol: break
        else:
            lam = min(lam * 3, 1e6)
    return theta, info


def errors_12dof(theta_est, theta_gt_12dof):
    """手眼误差 + C 误差"""
    Re = so3_exp(theta_est[0:3])
    Rg = so3_exp(theta_gt_12dof[0:3])
    Rdiff = Re.T @ Rg
    tr = np.clip((np.trace(Rdiff) - 1) / 2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(theta_est[3:6] - theta_gt_12dof[3:6]) * 1000
    C_err = np.linalg.norm(theta_est[9:12] - theta_gt_12dof[9:12]) * 1000
    return R_err, t_err, C_err


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    from reproduction_scene import generate_hand_eye_gt
    from corner_scene import generate_corner_plane, generate_corner_measurements
    from acquisition_sim import generate_linear_trajectory

    print("=" * 70)
    print("  12-DOF + 边长约束 + 采集数据 验证")
    print("=" * 70)

    for noise_sigma_mm in [0, 0.055]:
        label = "零噪声" if noise_sigma_mm == 0 else f"σ={noise_sigma_mm}mm"
        trials_ok = 0
        all_R, all_t, all_C = [], [], []

        for trial in range(10):
            seed = 42 + trial * 137
            rng = np.random.default_rng(seed)
            np.random.seed(seed)

            # 场景
            C_true, n_B, u_B, v_B, _, _, w_m, h_m = generate_corner_plane(rng)
            X_gt = generate_hand_eye_gt()
            R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]
            R_pl = np.column_stack([u_B, v_B, n_B])

            scene = {
                'R_he': R_he, 't_he': t_he, 'C': C_true,
                'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
                'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
                'plate_w': 400, 'plate_h': 500, 'w': w_m, 'h': h_m,
            }

            # 完善 theta_gt_12dof
            theta_gt = np.concatenate([so3_log(R_he), t_he, so3_log(R_pl), C_true])

            # 生成轨迹
            poses, _ = generate_linear_trajectory(scene, R_he, t_he, n_steps=40)
            if len(poses) < 10: continue

            # 测量
            meas = generate_corner_measurements(scene, poses, n_plane_pts=10,
                                                 rng=rng, noise_sigma=noise_sigma_mm/1000)

            # 从单位阵出发
            theta_init = np.zeros(12)

            theta_opt, info = solve_lm_12dof(theta_init, poses, meas,
                                              w_plane=10.0, w_edge=1.0, w_dist=1.0/1000,
                                              max_iter=100)

            R_err, t_err, C_err = errors_12dof(theta_opt, theta_gt)

            all_R.append(R_err); all_t.append(t_err); all_C.append(C_err)
            if R_err < 0.1 and t_err < 0.1:
                trials_ok += 1

            if trial < 2:
                print(f"\n  seed={seed} ({label}):")
                print(f"    R={R_err:.4f}° t={t_err:.4f}mm C_err={C_err:.4f}mm")
                print(f"    info: plane={info['n_plane']} edge={info['n_edge']} dist={info['n_dist']}")

        R_arr = np.array(all_R); t_arr = np.array(all_t); C_arr = np.array(all_C)
        print(f"\n  [{label}] 汇总 ({len(all_R)} trials):")
        print(f"    R: median={np.median(R_arr):.4f}°  max={np.max(R_arr):.4f}°  "
              f"<0.1°={np.sum(R_arr<0.1)}/{len(all_R)}")
        print(f"    t: median={np.median(t_arr):.4f}mm  max={np.max(t_arr):.4f}mm  "
              f"<0.1mm={np.sum(t_arr<0.1)}/{len(all_R)}")
        print(f"    C: median={np.median(C_arr):.4f}mm")
        print(f"    R<0.1° & t<0.1mm: {trials_ok}/{len(all_R)}")
