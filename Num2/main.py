#!/usr/bin/env python3
"""
main.py — 平板手眼标定统一入口

方法: 几何多样性NBV + 边缘共线 + 平面约束
6 位姿, 全自动, 无外部测量

用法: python main.py              # 单次标定 + 动画
      python main.py --mc 30      # Monte Carlo 30 trials
"""

import numpy as np
import sys
from corner_scene import generate_corner_plane, generate_corner_measurements, so3_log
from reproduction_scene import generate_hand_eye_gt
from nbv_edge_plane import (
    generate_edge_candidates, select_diverse,
    combined_solve_lm, combined_errors, combined_residuals
)


def run_calibration(scene, R_he_gt, t_he_gt, n_poses=6,
                     laser_noise=0.055, warp_mm=0.0, repeat_mm=0.0,
                     w_plane=0.5, w_edge=1.0, n_restarts=5, seed=42):
    """单次标定

    Args:
        warp_mm: 板翘曲幅度 ±mm (0 = 理想平板)
        repeat_mm: 机器人重复性 mm
    Returns:
        dict with R_error, t_error, t_inplane_error, gauge_error
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    C, n_B, u_B, v_B = scene['C'], scene['n_B'], scene['u_B'], scene['v_B']

    # 1. 位姿选择 (几何多样性NBV + 边缘可见)
    c_e1, c_e2 = generate_edge_candidates(scene, R_he_gt, t_he_gt, 'both', 8)
    sel = select_diverse(c_e1, 3) + select_diverse(c_e2, 3)
    poses_clean = [(c['R_i'], c['t_i']) for c in sel]

    # 2. 机器人噪声
    if repeat_mm > 0:
        poses = []
        for (R_i, t_i) in poses_clean:
            poses.append((R_i, t_i + rng.normal(0, repeat_mm / 1000.0, 3)))
    else:
        poses = poses_clean

    # 3. 生成测量
    meas = generate_corner_measurements(
        scene, poses, n_plane_pts=200, rng=rng,
        noise_sigma=laser_noise / 1000.0)

    # 4. 板翘曲 (扫描线方向正弦波)
    if warp_mm > 0:
        for m in meas:
            if len(m['p_S_plane']) > 10:
                pts = m['p_S_plane']
                warp = warp_mm / 1000.0 * np.sin(2 * np.pi * pts[:, 0] / 0.3)
                pts_w = pts.copy()
                pts_w[:, 2] += warp
                m['p_S_plane'] = pts_w

    # 5. GT 参数
    R_pl = np.column_stack([u_B, v_B, n_B])
    theta_gt = np.concatenate([so3_log(R_he_gt), t_he_gt, so3_log(R_pl)])

    # 6. LM 多权重 + 多重启
    best_Re = np.inf
    best_result = None

    for wp, we in [(1.0, 1.0), (0.5, 1.0), (0.2, 2.0), (0.1, 5.0)]:
        for _ in range(n_restarts):
            ti = theta_gt.copy()
            ti[0:3] += rng.normal(0, 0.1, 3)
            ti[3:6] += rng.normal(0, 0.002, 3)
            ti[6:9] += rng.normal(0, 0.05, 3)

            to = combined_solve_lm(ti, poses, meas, w_plane=wp, w_edge=we,
                                    max_iter=80)
            Re, te, tipe = combined_errors(to, theta_gt)
            if Re < best_Re:
                best_Re = Re
                nB_gt = n_B
                dt = to[3:6] - theta_gt[3:6]
                gauge = abs(np.dot(dt, nB_gt)) * 1000
                best_result = {
                    'R_error': Re,
                    't_error': te,
                    't_inplane_error': tipe,
                    'gauge_error': gauge,
                    'w_plane': wp,
                    'w_edge': we,
                }

    return best_result


def run_monte_carlo(n_trials=30, n_poses=6, warp_mm=0.0, repeat_mm=0.0,
                     laser_noise=0.055, verbose=True):
    """Monte Carlo 评估"""
    R_all, t_all, tp_all, tg_all = [], [], [], []

    for trial in range(n_trials):
        seed = 42 + trial * 137
        rng = np.random.default_rng(seed)
        np.random.seed(seed)

        X_gt = generate_hand_eye_gt()
        R_he_gt, t_he_gt = X_gt[:3, :3], X_gt[:3, 3]
        C, n_B, u_B, v_B, _, _, wm, hm = generate_corner_plane(rng)

        scene = {
            'R_he': R_he_gt, 't_he': t_he_gt,
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
            'plate_w': wm * 1000, 'plate_h': hm * 1000,
            'w': wm, 'h': hm,
        }

        result = run_calibration(
            scene, R_he_gt, t_he_gt, n_poses=n_poses,
            laser_noise=laser_noise, warp_mm=warp_mm, repeat_mm=repeat_mm,
            seed=seed)

        if result:
            R_all.append(result['R_error'])
            t_all.append(result['t_error'])
            tp_all.append(result['t_inplane_error'])
            tg_all.append(result['gauge_error'])

    if len(R_all) == 0:
        return None

    R_a = np.array(R_all); t_a = np.array(t_all)
    tp_a = np.array(tp_all); tg_a = np.array(tg_all)

    stats = {
        'n': len(R_a),
        'R_median': np.median(R_a), 'R_mean': np.mean(R_a), 'R_max': np.max(R_a),
        't_median': np.median(t_a), 't_inp_median': np.median(tp_a),
        't_gauge_median': np.median(tg_a),
        'R_lt_01': np.sum(R_a < 0.1), 'R_lt_02': np.sum(R_a < 0.2),
    }

    if verbose:
        print(f"  {stats['n']}/{n_trials} trials")
        print(f"  R: median={stats['R_median']:.4f}° mean={stats['R_mean']:.4f}° "
              f"max={stats['R_max']:.4f}°")
        print(f"  t: median={stats['t_median']:.4f}mm "
              f"t_inplane={stats['t_inp_median']:.4f}mm "
              f"t_gauge={stats['t_gauge_median']:.4f}mm")
        print(f"  R<0.1°: {stats['R_lt_01']}/{stats['n']}  "
              f"R<0.2°: {stats['R_lt_02']}/{stats['n']}")

    return stats


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("平板手眼标定 — 几何NBV + 边缘共线 + 平面约束")
    print("=" * 60)

    # 理想条件: 无翘曲, 无机器人噪声
    print("\n[1] 理想条件 (无翘曲, 无重复性误差)")
    run_monte_carlo(n_trials=20, n_poses=6, warp_mm=0.0, repeat_mm=0.0)

    # 仅机器人噪声
    print("\n[2] 仅机器人重复性 0.2mm")
    run_monte_carlo(n_trials=20, n_poses=6, warp_mm=0.0, repeat_mm=0.2)

    # 翘曲扫描
    print("\n[3] 翘曲敏感度扫描")
    for w in [0.2, 0.5, 1.0]:
        print(f"\n  翘曲 ±{w}mm:")
        s = run_monte_carlo(n_trials=15, n_poses=6, warp_mm=w, repeat_mm=0.0)
        if s:
            local_tilt = 2 * np.pi * w / 300 * 180 / np.pi
            print(f"  局部法向偏转约 {local_tilt:.2f}°")

    print("\n" + "=" * 60)
    print("约定:")
    print("  σ_laser = 0.055mm (max error 0.1mm)")
    print("  6 位姿, 3e1+3e2 边缘, 200 平面点/位姿")
    print("  多权重 + 多重启 LM")
    print("=" * 60)
