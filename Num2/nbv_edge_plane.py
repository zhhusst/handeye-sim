#!/usr/bin/env python3
"""
nbv_edge_plane.py — 几何 NBV + 边缘共线 + 平面联合标定

策略:
  - 3 个 x_S=u_B 候选 (看 e2) + 3 个 x_S=v_B 候选 (看 e1)
  - 各自内部按几何多样性选
  - 求解器: 平面(降权, 板不平3mm) + 边缘共线(正权, 激光0.1mm)
"""

import numpy as np
from corner_scene import so3_exp, so3_log
from reproduction_scene import compute_fov_plate_scanline, generate_hand_eye_gt
from corner_scene import generate_corner_plane, generate_corner_measurements


# ============================================================================
# Combined solver: plane (centered, weighted) + edge collinearity
# ============================================================================

def combined_residuals(theta, poses, measurements, w_plane=0.1, w_edge=1.0):
    """9-DOF: plane centered + edge collinearity (not centered)"""
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    plane_vals = []
    edge_res = []
    edge_mask = []
    e1_pairs = 0
    e2_pairs = 0

    # Collect edge breakpoints in base frame
    p_base_e1 = []
    p_base_e2 = []

    for idx, ((R_i, t_i), m) in enumerate(zip(poses, measurements)):
        R_BS = R_i @ R_he
        t_BS = t_i + R_i @ t_he

        if m['valid_e1']:
            p_base_e1.append(R_BS @ m['p_S_e1'] + t_BS)
        if m['valid_e2']:
            p_base_e2.append(R_BS @ m['p_S_e2'] + t_BS)

        for p_S in m['p_S_plane']:
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))

    # Plane: global center
    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0:
        plane_vals = plane_vals - np.mean(plane_vals)

    # Edge 1 collinearity
    for k in range(len(p_base_e1) - 1):
        r = np.cross(p_base_e1[k+1] - p_base_e1[k], u_B)
        edge_res.extend(r.tolist())
        edge_mask.extend([True, True, True])
        e1_pairs += 1

    # Edge 2 collinearity
    for k in range(len(p_base_e2) - 1):
        r = np.cross(p_base_e2[k+1] - p_base_e2[k], v_B)
        edge_res.extend(r.tolist())
        edge_mask.extend([True, True, True])
        e2_pairs += 1

    # Combine
    residuals = []
    mask = []
    wp = np.sqrt(w_plane)
    we = np.sqrt(w_edge)

    for v in plane_vals:
        residuals.append(v * wp)
        mask.append(True)
    for v, m in zip(edge_res, edge_mask):
        residuals.append(v * we)
        mask.append(m)

    info = {'n_plane': len(plane_vals), 'e1_pairs': e1_pairs, 'e2_pairs': e2_pairs}
    return np.array(residuals), np.array(mask), info


def combined_cost(theta, poses, meas, w_plane=0.1, w_edge=1.0):
    r, m, _ = combined_residuals(theta, poses, meas, w_plane, w_edge)
    rv = r[m]
    return 0.5 * np.dot(rv, rv) if len(rv) > 0 else 1e30


def combined_jacobian(theta, poses, meas, w_plane=0.1, w_edge=1.0, eps=1e-6):
    r0, mask, info = combined_residuals(theta, poses, meas, w_plane, w_edge)
    J = np.zeros((len(r0), 9))
    for k in range(9):
        step = np.zeros(9); step[k] = eps
        rp, _, _ = combined_residuals(theta + step, poses, meas, w_plane, w_edge)
        rm, _, _ = combined_residuals(theta - step, poses, meas, w_plane, w_edge)
        J[:, k] = (rp - rm) / (2 * eps)
    return J, r0, mask, info


def combined_solve_lm(theta_init, poses, meas, w_plane=0.1, w_edge=1.0,
                       max_iter=100, tol=1e-10, lam0=1e-6):
    theta = theta_init.copy()
    lam = lam0
    for it in range(max_iter):
        J, r, mask, _ = combined_jacobian(theta, poses, meas, w_plane, w_edge)
        rv = r[mask]; Jv = J[mask, :]
        if len(rv) == 0: break
        cost = 0.5 * np.dot(rv, rv)
        H = Jv.T @ Jv; g = Jv.T @ rv
        try: delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except np.linalg.LinAlgError: lam *= 10; continue
        tn = theta + delta
        if combined_cost(tn, poses, meas, w_plane, w_edge) < cost:
            theta = tn; lam = max(lam/3, 1e-12)
            if abs(cost - combined_cost(tn, poses, meas, w_plane, w_edge)) < tol: break
        else: lam = min(lam*3, 1e6)
    return theta


def combined_errors(theta_est, theta_gt):
    Re = so3_exp(theta_est[0:3]); Rg = so3_exp(theta_gt[0:3])
    Rd = Re.T @ Rg; tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(theta_est[3:6] - theta_gt[3:6]) * 1000
    n_B = so3_exp(theta_gt[6:9])[:, 2]
    dt = theta_est[3:6] - theta_gt[3:6]
    t_inp = np.linalg.norm(dt - np.dot(dt, n_B) * n_B) * 1000
    return R_err, t_err, t_inp


# ============================================================================
# Candidate generation with edge visibility
# ============================================================================

def _build_R_edge(pitch_deg, yaw_deg, x_align, n_B, u_B, v_B):
    """x_S = x_align, z_S = compound tilt"""
    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    Ku = np.array([[0,-u_B[2],u_B[1]],[u_B[2],0,-u_B[0]],[-u_B[1],u_B[0],0]])
    Kv = np.array([[0,-v_B[2],v_B[1]],[v_B[2],0,-v_B[0]],[-v_B[1],v_B[0],0]])
    Rp = np.eye(3)+np.sin(p)*Ku+(1-np.cos(p))*Ku@Ku
    Ry = np.eye(3)+np.sin(y)*Kv+(1-np.cos(y))*Kv@Kv
    z_S = Ry @ Rp @ (-n_B)
    xd = x_align
    yd = np.cross(z_S, xd); nrm = np.linalg.norm(yd)
    if nrm < 1e-8: yd = v_B
    else: yd /= nrm
    xd = np.cross(yd, z_S)
    return np.column_stack([xd, yd, z_S])


def generate_edge_candidates(scene, R_he_est, t_he_est, edge='both', n_grid=6):
    """生成看得到边的候选位姿

    edge='e2': x_S=u_B → 看 e2
    edge='e1': x_S=v_B → 看 e1
    edge='both': 各半
    """
    C, n_B, u_B, v_B = scene['C'], scene['n_B'], scene['u_B'], scene['v_B']
    w_m, h_m = scene['w'], scene['h']

    pitch_range = np.linspace(-25, 25, n_grid)
    yaw_range = np.linspace(-10, 10, max(3, n_grid//2))

    candidates_e1 = []
    candidates_e2 = []

    for p_d in pitch_range:
        for y_d in yaw_range:
            # e2 candidates: x_S = u_B
            R_i2 = _build_R_edge(p_d, y_d, u_B, n_B, u_B, v_B)
            R_BS2 = R_i2 @ R_he_est
            # e1 candidates: x_S = v_B
            R_i1 = _build_R_edge(p_d, y_d, v_B, n_B, u_B, v_B)
            R_BS1 = R_i1 @ R_he_est

            # Try different positions along the edge
            for u_f in [0.08, 0.5] if edge in ('e2', 'both') else [0.5]:
                for v_f in np.linspace(0.1, 0.9, 4):
                    target = C + u_f * w_m * u_B + v_f * h_m * v_B
                    for s in [0.35, 0.45]:
                        for hs in [1.0, 1.3, 1.6, 2.0, 3.0]:
                            # e2
                            if edge in ('e2', 'both'):
                                t_i2 = target + s * hs * n_B - R_i2 @ t_he_est
                                t_BS2 = t_i2 + R_i2 @ t_he_est
                                sl2 = compute_fov_plate_scanline(
                                    R_BS2, t_BS2, C, n_B, u_B, v_B, w_m, h_m)
                                if sl2['has_intersection'] and len(sl2['scan_pts_S']) > 5:
                                    eps2 = [e for e, _ in sl2['endpoints_S']]
                                    if 'e2' in eps2:
                                        candidates_e2.append({
                                            'R_i': R_i2, 't_i': t_i2,
                                            'pitch': p_d, 'yaw': y_d,
                                            'edge': 'e2',
                                        })
                                        break
                            # e1
                            if edge in ('e1', 'both'):
                                t_i1 = target + s * hs * n_B - R_i1 @ t_he_est
                                t_BS1 = t_i1 + R_i1 @ t_he_est
                                sl1 = compute_fov_plate_scanline(
                                    R_BS1, t_BS1, C, n_B, u_B, v_B, w_m, h_m)
                                if sl1['has_intersection'] and len(sl1['scan_pts_S']) > 5:
                                    eps1 = [e for e, _ in sl1['endpoints_S']]
                                    if 'e1' in eps1:
                                        candidates_e1.append({
                                            'R_i': R_i1, 't_i': t_i1,
                                            'pitch': p_d, 'yaw': y_d,
                                            'edge': 'e1',
                                        })
                                        break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

    return candidates_e1, candidates_e2


def select_diverse(candidates, n_select):
    """从候选中选 n_select 个多样化位姿"""
    if len(candidates) <= n_select:
        return candidates

    def ang_dist(a, b):
        Rd = a['R_i'].T @ b['R_i']
        tr = np.clip((np.trace(Rd)-1)/2, -1, 1)
        return np.arccos(tr)

    def pos_dist(a, b):
        return np.linalg.norm(a['t_i'] - b['t_i']) / 0.5

    # 贪心
    selected = [max(range(len(candidates)),
                    key=lambda i: abs(candidates[i]['pitch']) + abs(candidates[i]['yaw']))]
    for _ in range(n_select - 1):
        best_i, best_s = None, -1
        for i in range(len(candidates)):
            if i in selected: continue
            md = min(ang_dist(candidates[i], candidates[j]) + 0.3*pos_dist(candidates[i], candidates[j])
                     for j in selected)
            if md > best_s: best_s = md; best_i = i
        selected.append(best_i)
    return [candidates[i] for i in selected]


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("几何 NBV + 边缘共线联合标定")
    print(f"  激光 σ=0.055mm, 板不平=3mm (平面降权 w_plane=0.1)")
    print("=" * 60)

    R_all, t_all, tp_all = [], [], []
    e1_vis_all, e2_vis_all = [], []

    for trial in range(30):
        seed = 42 + trial * 137
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        X_gt = generate_hand_eye_gt()
        R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]
        C, n_B, u_B, v_B, _, _, w_m, h_m = generate_corner_plane(rng)

        scene = {
            'R_he': R_he, 't_he': t_he,
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
            'plate_w': w_m * 1000, 'plate_h': h_m * 1000,
            'w': w_m, 'h': h_m,
        }

        cands_e1, cands_e2 = generate_edge_candidates(scene, R_he, t_he, 'both', 6)

        # 选 3 e1 + 3 e2
        sel_e1 = select_diverse(cands_e1, 3)
        sel_e2 = select_diverse(cands_e2, 3)
        selected = sel_e1 + sel_e2
        poses = [(c['R_i'], c['t_i']) for c in selected]

        # Count edge visibility
        e1v = 0; e2v = 0
        for c in selected:
            if c['edge'] == 'e1': e1v += 1
            if c['edge'] == 'e2': e2v += 1

        meas = generate_corner_measurements(
            scene, poses, n_plane_pts=30, rng=rng, noise_sigma=0.055/1000)

        R_pl = np.column_stack([u_B, v_B, n_B])
        tg = np.concatenate([so3_log(R_he), t_he, so3_log(R_pl)])

        best_Re = np.inf
        for _ in range(3):
            ti = tg.copy()
            ti[0:3] += rng.normal(0, 0.1, 3)
            ti[3:6] += rng.normal(0, 0.002, 3)
            ti[6:9] += rng.normal(0, 0.05, 3)
            to = combined_solve_lm(ti, poses, meas, w_plane=0.1, w_edge=1.0,
                                    max_iter=100)
            Re, te, tipe = combined_errors(to, tg)
            if Re < best_Re: best_Re = Re; best_te = te; best_tipe = tipe

        R_all.append(best_Re); t_all.append(best_te); tp_all.append(best_tipe)
        e1_vis_all.append(e1v); e2_vis_all.append(e2v)

    R_a = np.array(R_all); t_a = np.array(t_all); tp_a = np.array(tp_all)

    print(f"\n  {'':>12} {'平面+边缘':>12}")
    print(f"  {'R median':>12} {np.median(R_a):12.4f}°")
    print(f"  {'R mean':>12} {np.mean(R_a):12.4f}°")
    print(f"  {'R max':>12} {np.max(R_a):12.4f}°")
    print(f"  {'t median':>12} {np.median(t_a):12.4f}mm")
    print(f"  {'t_inp median':>12} {np.median(tp_a):12.4f}mm")
    print(f"  {'R<0.05°':>12} {np.sum(R_a < 0.05):12d}")
    print(f"  {'R<0.10°':>12} {np.sum(R_a < 0.10):12d}")
    print(f"  {'t_inp<0.2mm':>12} {np.sum(tp_a < 0.2):12d}")
    print(f"  {'avg e1,e2':>12} {np.mean(e1_vis_all):11.1f},{np.mean(e2_vis_all):.1f}")
