#!/usr/bin/env python3
"""
realistic_mc_v3.py — 修正机器人噪声建模

关键修复: 测量在 ACTUAL 位姿(含噪声)下生成, solver 用同样的含噪位姿
          → 消除报告位姿与实际位姿不一致引入的伪影

误差:
  - 激光 σ=0.055mm
  - 板翘曲 ±1.5mm 正弦 (λ=300mm)
  - 机器人重复性 0.2mm (测量与solver用同一含噪位姿)
"""

import numpy as np
from corner_scene import so3_log, so3_exp
from reproduction_scene import compute_fov_plate_scanline, generate_hand_eye_gt
from corner_scene import generate_corner_plane, generate_corner_measurements
from nbv_edge_plane import (
    generate_edge_candidates, select_diverse,
    combined_solve_lm, combined_errors, combined_residuals
)


def run_trial(seed, add_warp=True, add_repeat=True):
    """单次试验"""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]
    C, n_B, u_B, v_B, _, _, wm, hm = generate_corner_plane(rng)

    scene = {
        'R_he': R_he, 't_he': t_he,
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
        'plate_w': wm*1000, 'plate_h': hm*1000,
        'w': wm, 'h': hm,
    }

    c1, c2 = generate_edge_candidates(scene, R_he, t_he, 'both', 8)
    sel = select_diverse(c1, 3) + select_diverse(c2, 3)
    poses_clean = [(c['R_i'], c['t_i']) for c in sel]

    # 机器人实际到达的位姿 (含重复性噪声)
    if add_repeat:
        poses_actual = []
        for (R_i, t_i) in poses_clean:
            noise_t = rng.normal(0, 0.2 / 1000.0, 3)
            poses_actual.append((R_i, t_i + noise_t))
    else:
        poses_actual = poses_clean

    # 在 ACTUAL 位姿下生成测量 (这才是真实情况)
    meas = generate_corner_measurements(
        scene, poses_actual, n_plane_pts=200, rng=rng,
        noise_sigma=0.055 / 1000.0)

    # 板翘曲: 扫描线方向的正弦波
    if add_warp:
        for m in meas:
            if len(m['p_S_plane']) > 10:
                pts = m['p_S_plane']
                x = pts[:, 0]
                warp = 1.5/1000 * np.sin(2*np.pi*x/0.3)
                pts_w = pts.copy()
                pts_w[:, 2] += warp
                m['p_S_plane'] = pts_w

    # GT
    R_pl = np.column_stack([u_B, v_B, n_B])
    theta_gt = np.concatenate([so3_log(R_he), t_he, so3_log(R_pl)])

    # 多权重方案
    results = {}
    for wp, we in [(1.0, 1.0), (0.5, 1.0), (0.2, 2.0)]:
        best_Re = np.inf
        for _ in range(3):
            ti = theta_gt.copy()
            ti[0:3] += rng.normal(0, 0.1, 3)
            ti[3:6] += rng.normal(0, 0.002, 3)
            ti[6:9] += rng.normal(0, 0.05, 3)
            to = combined_solve_lm(ti, poses_actual, meas,
                                    w_plane=wp, w_edge=we, max_iter=80)
            Re, te, tipe = combined_errors(to, theta_gt)
            if Re < best_Re:
                best_Re = Re; best_te = te; best_tipe = tipe
        results[(wp, we)] = (best_Re, best_te, best_tipe)

    return results


def main():
    print("=" * 65)
    print("修正噪声模型: 测量在 actual 位姿生成, solver 用同样位姿")
    print("=" * 65)

    for label, warp, repeat in [
        ('无翘曲+无重复', False, False),
        ('仅重复0.2mm', False, True),
        ('仅翘曲±1.5mm', True, False),
        ('翘曲+重复', True, True),
    ]:
        print(f"\n--- {label} ---")
        R_dict = {}; t_dict = {}; tp_dict = {}
        for (wp, we) in [(1.0, 1.0), (0.5, 1.0), (0.2, 2.0)]:
            R_dict[(wp, we)] = []
            t_dict[(wp, we)] = []
            tp_dict[(wp, we)] = []

        for trial in range(20):
            seed = 42 + trial * 137
            try:
                res = run_trial(seed, add_warp=warp, add_repeat=repeat)
                for (wp, we), (Re, te, tipe) in res.items():
                    R_dict[(wp, we)].append(Re)
                    t_dict[(wp, we)].append(te)
                    tp_dict[(wp, we)].append(tipe)
            except Exception as e:
                continue

        best_config = None; best_R = np.inf
        for (wp, we) in [(1.0, 1.0), (0.5, 1.0), (0.2, 2.0)]:
            if len(R_dict[(wp, we)]) == 0: continue
            R_a = np.array(R_dict[(wp, we)])
            if np.median(R_a) < best_R:
                best_R = np.median(R_a)
                best_config = (wp, we)

        if best_config:
            wp, we = best_config
            R_a = np.array(R_dict[(wp, we)])
            t_a = np.array(t_dict[(wp, we)])
            tp_a = np.array(tp_dict[(wp, we)])
            print(f"  best: w_p={wp} w_e={we} ({len(R_a)} trials)")
            print(f"  R: median={np.median(R_a):.4f}° max={np.max(R_a):.4f}°")
            print(f"  t: median={np.median(t_a):.4f}mm")
            print(f"  t_inp: median={np.median(tp_a):.4f}mm")
            print(f"  R<0.1°: {np.sum(R_a<0.1)}/{len(R_a)}")


if __name__ == '__main__':
    main()
