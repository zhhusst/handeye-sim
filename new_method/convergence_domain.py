#!/usr/bin/env python3
"""
convergence_domain.py — 旧方法初值容忍度扫描 (R扰动 × t扰动)

回答: 9-DOF 和 12-DOF 分别能接受多大误差的手眼初值？
"""

import sys, json, time, numpy as np

sys.path.insert(0, '/home/z/research_contact_handeye/verification/Sim/new_method')
from observation_model import so3_exp, so3_log
from legacy_solver import (
    combined_solve_lm, combined_residuals, compute_errors_legacy,
    solve_12dof_lm, residuals_12dof_cross, init_12dof_cross,
)


def load_data(path):
    with open(path) as f:
        data = json.load(f)
    R_gt = np.array(data['scene']['R_he_gt'])
    t_gt = np.array(data['scene']['t_he_gt'])
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
    return poses, meas, R_gt, t_gt


def test_one(poses, meas, R_gt, t_gt, R_err_deg, t_err_mm, seed):
    """单次测试：给定 R/t 扰动幅度，随机方向，测两种方法是否收敛"""
    rng = np.random.RandomState(seed)

    # 随机旋转方向
    ax = rng.randn(3); ax /= np.linalg.norm(ax)
    R_init = R_gt @ so3_exp(ax * np.deg2rad(R_err_deg))

    # 随机平移方向
    t_dir = rng.randn(3); t_dir /= np.linalg.norm(t_dir)
    t_init = t_gt + t_dir * (t_err_mm / 1000.0)

    results = {}

    # ── 9-DOF ──
    try:
        theta9_init = np.zeros(9)
        theta9_init[0:3] = so3_log(R_init)
        theta9_init[3:6] = t_init
        theta9 = combined_solve_lm(theta9_init, poses, meas, w_plane=0.1, w_edge=1.0, max_iter=200)
        R_err_9, t_err_9 = compute_errors_legacy(theta9, R_gt, t_gt)
        results['9dof'] = (R_err_9 < 1.0, float(R_err_9), float(t_err_9))
    except Exception as e:
        results['9dof'] = (False, 999.0, 999.0)

    # ── 12-DOF ──
    try:
        theta12_init = init_12dof_cross(poses, meas)
        theta12_init[0:3] = so3_log(R_init)
        theta12_init[3:6] = t_init
        theta12 = solve_12dof_lm(theta12_init, poses, meas, max_iter=200)
        R_err_12, t_err_12 = compute_errors_legacy(theta12, R_gt, t_gt)
        results['12dof'] = (R_err_12 < 1.0, float(R_err_12), float(t_err_12))
    except Exception as e:
        results['12dof'] = (False, 999.0, 999.0)

    return results


def main():
    data_path = '/home/z/research_contact_handeye/verification/Sim/recorded_poses.json'
    poses, meas, R_gt, t_gt = load_data(data_path)
    print(f'数据: {len(poses)} 位姿, e1={sum(1 for m in meas if m["valid_e1"])}, e2={sum(1 for m in meas if m["valid_e2"])}')

    # 扫描网格 (精简版, 聚焦关键区域)
    R_errs = [1, 3, 5, 10, 20, 30, 50, 90, 150]    # deg
    t_errs = [0, 10, 50, 100, 500]                    # mm
    n_trials = 5

    print('\n收敛域扫描 (收敛判据: R_err < 1.0°)')
    print('=' * 80)
    header = '{:>5s}'.format('R\\t')
    for te in t_errs:
        header += ' {:>7d}mm'.format(te)
    print(header)
    print('-' * 80)

    all_results = {}

    for re in R_errs:
        row_9 = '{:>5d}°'.format(re)
        row_12 = '{:>5d}°'.format(re)
        for te in t_errs:
            n9, n12 = 0, 0
            med_r9, med_r12 = [], []
            for trial in range(n_trials):
                res = test_one(poses, meas, R_gt, t_gt, re, te, seed=1000*re + 100*te + trial)
                if res['9dof'][0]: n9 += 1
                if res['12dof'][0]: n12 += 1
                if res['9dof'][1] < 999: med_r9.append(res['9dof'][1])
                if res['12dof'][1] < 999: med_r12.append(res['12dof'][1])

            bar_9 = '█' * (n9 // 2) if n9 >= 10 else ('▄' * n9 if n9 > 0 else '·')
            bar_12 = '█' * (n12 // 2) if n12 >= 10 else ('▄' * n12 if n12 > 0 else '·')
            row_9 += ' {:>2d}/{} '.format(n9, n_trials)
            row_12 += ' {:>2d}/{} '.format(n12, n_trials)

            all_results[(re, te)] = {
                '9dof_success': n9, '12dof_success': n12,
                '9dof_med_r': np.median(med_r9) if med_r9 else 999,
                '12dof_med_r': np.median(med_r12) if med_r12 else 999,
            }

        print(row_9 + '  ← 9-DOF')
        print(row_12 + '  ← 12-DOF')
        print()

    # 汇总: 找 100% 成功边界
    print('=' * 80)
    print('100% 成功率边界:')
    print('  9-DOF:  R≤{}°, t≤{}mm'.format(
        max([re for (re, te), v in all_results.items() if v['9dof_success'] == 10] or [0]),
        max([te for (re, te), v in all_results.items() if v['9dof_success'] == 10 and re <= 30] or [0])))
    print('  12-DOF: R≤{}°, t≤{}mm'.format(
        max([re for (re, te), v in all_results.items() if v['12dof_success'] == 10] or [0]),
        max([te for (re, te), v in all_results.items() if v['12dof_success'] == 10 and re <= 30] or [0])))

    return 0


if __name__ == '__main__':
    sys.exit(main())
