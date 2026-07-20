#!/usr/bin/env python3
"""
run_all.py — 全方法对比 + 初始误差容忍度统计

用法:
  cd Sim/new_method && python3 run_all.py
"""

import sys, json, time, numpy as np

sys.path.insert(0, '/home/z/research_contact_handeye/verification/Sim/new_method')

from observation_model import so3_exp, so3_log, params_to_SE3, compute_errors
from solver import calibrate as new_calibrate
from legacy_solver import (
    combined_solve_lm, combined_residuals,
    solve_12dof_cross_with_restarts, residuals_12dof_cross,
    iterative_refine_he, compute_errors_legacy,
)

SEP = "=" * 60


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    R_he_gt = np.array(data['scene']['R_he_gt'])
    t_he_gt = np.array(data['scene']['t_he_gt'])

    poses, meas = [], []
    for p in data['poses']:
        R_i = np.array(p['R_i']); t_i = np.array(p['t_i'])
        poses.append((R_i, t_i))
        meas.append({
            'p_S_e1': np.array(p['p_S_e1']) if p.get('valid_e1') else None,
            'valid_e1': p.get('valid_e1', False),
            'p_S_e2': np.array(p['p_S_e2']) if p.get('valid_e2') else None,
            'valid_e2': p.get('valid_e2', False),
            'p_S_plane': [np.array(pt) for pt in p['scan_pts_S']],
        })

    return poses, meas, R_he_gt, t_he_gt


def run_comparison(poses, meas, R_gt, t_gt, R_init, t_init):
    """运行四种方法对比"""
    results = {}
    times = {}

    # ── Method 1: 新方法 ──
    t0 = time.time()
    theta_new, stats_new = new_calibrate(
        poses, meas, R_he_init=R_init, t_he_init=t_init, verbose=False)
    times['new_scalar'] = time.time() - t0
    (R_new, t_new), _ = params_to_SE3(theta_new)
    dR = R_new.T @ R_gt
    tr = np.clip((np.trace(dR)-1)/2, -1, 1)
    results['new_scalar'] = (np.rad2deg(np.arccos(tr)),
                              np.linalg.norm(t_new - t_gt) * 1000)

    # ── Method 2: 旧 9-DOF ──
    t0 = time.time()
    theta9_init = np.zeros(9)
    theta9_init[0:3] = so3_log(R_init)
    theta9_init[3:6] = t_init
    theta9 = combined_solve_lm(theta9_init, poses, meas, w_plane=0.1, w_edge=1.0, max_iter=200)
    times['old_9dof'] = time.time() - t0
    results['old_9dof'] = compute_errors_legacy(theta9, R_gt, t_gt)

    # ── Method 3: 旧 12-DOF cross-product ──
    t0 = time.time()
    theta12_init = np.zeros(12)
    theta12_init[0:3] = so3_log(R_init)
    theta12_init[3:6] = t_init
    from legacy_solver import init_12dof_cross
    theta12_init[6:12] = init_12dof_cross(poses, meas)[6:12]
    theta12 = solve_12dof_cross_with_restarts_core(theta12_init, poses, meas)
    times['old_12dof'] = time.time() - t0
    results['old_12dof'] = compute_errors_legacy(theta12, R_gt, t_gt)

    # ── Method 4: iterative_refine ──
    t0 = time.time()
    R_ref, t_ref, _, _, _ = iterative_refine_he(
        poses, meas, R_init, t_init, max_iter=5, verbose=False)
    times['iter_refine'] = time.time() - t0
    dR_ref = R_ref.T @ R_gt
    tr = np.clip((np.trace(dR_ref)-1)/2, -1, 1)
    results['iter_refine'] = (np.rad2deg(np.arccos(tr)),
                               np.linalg.norm(t_ref - t_gt) * 1000)

    return results, times


def solve_12dof_cross_with_restarts_core(theta_init, poses, meas):
    """旧 12-DOF 单次求解 (不重启)"""
    from legacy_solver import solve_12dof_lm
    return solve_12dof_lm(theta_init, poses, meas, max_iter=200)


def main():
    data_path = '/home/z/research_contact_handeye/verification/Sim/recorded_poses.json'
    poses, meas, R_gt, t_gt = load_data(data_path)

    n = len(poses)
    n_e1 = sum(1 for m in meas if m['valid_e1'])
    n_e2 = sum(1 for m in meas if m['valid_e2'])
    print('数据: {} 位姿, e1={}, e2={}'.format(n, n_e1, n_e2))

    # ════════════════════════════════════════════════════════
    # Part A: 名义初值单次求解对比
    # ════════════════════════════════════════════════════════
    rng = np.random.RandomState(0)
    ax = rng.randn(3); ax /= np.linalg.norm(ax)
    perturb_deg = 10.0  # 初始扰动 10°
    R_init = R_gt @ so3_exp(ax * np.deg2rad(perturb_deg))
    t_init = t_gt + rng.randn(3) * 0.01

    dR_nom = R_init.T @ R_gt
    tr = np.clip((np.trace(dR_nom)-1)/2, -1, 1)
    init_err = np.rad2deg(np.arccos(tr))
    print('初始扰动: R_err={:.1f}deg, t_err={:.1f}mm\n'.format(
        init_err, np.linalg.norm(t_init-t_gt)*1000))

    print(SEP)
    print('单次求解对比 (无重启, 初值={:.0f}deg扰动)'.format(perturb_deg))
    print(SEP)
    results, times = run_comparison(poses, meas, R_gt, t_gt, R_init, t_init)

    for name in ['new_scalar', 'old_9dof', 'old_12dof', 'iter_refine']:
        re, te = results[name]
        print('  {:<20} R_err={:8.4f}deg  t_err={:8.3f}mm  ({:.1f}s)'.format(
            name, re, te, times[name]))

    # ════════════════════════════════════════════════════════
    # Part B: 初始误差容忍度统计
    # ════════════════════════════════════════════════════════
    print('\n' + SEP)
    print('初始误差容忍度统计 (每种扰动角度测10次)')
    print(SEP)

    perturb_list = [1, 3, 5, 10, 20, 30, 50, 90, 120, 150, 179]
    n_trials = 10
    seed_base = 100

    print('{:>6s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}'.format(
        'perturb', 'new_scalar', 'old_9dof', 'old_12dof', 'iter_refine'))
    print('-' * 60)

    for perturb_deg in perturb_list:
        row = []
        for seed_offset in range(n_trials):
            rng = np.random.RandomState(seed_base + seed_offset)
            ax = rng.randn(3); ax /= np.linalg.norm(ax)
            R_init_i = R_gt @ so3_exp(ax * np.deg2rad(perturb_deg))
            t_init_i = t_gt + rng.randn(3) * 0.01

            res_i, _ = run_comparison(poses, meas, R_gt, t_gt, R_init_i, t_init_i)
            row.append(res_i)

        # 统计各方法的成功率 (R_err < 1.0°)
        stats_line = '{:>6d}°'.format(perturb_deg)
        for mi, name in enumerate(['new_scalar', 'old_9dof', 'old_12dof', 'iter_refine']):
            success = sum(1 for r in row if r[name][0] < 1.0)
            stats_line += '  {:>5d}/{:d} ({:>3.0f}%)'.format(
                success, n_trials, success/n_trials*100)
        print(stats_line)

    print('\n收敛判据: R_err < 1.0°')

    return 0


if __name__ == '__main__':
    sys.exit(main())
