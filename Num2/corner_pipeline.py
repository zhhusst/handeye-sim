"""
corner_pipeline.py — Num2 角点法标定管道 (CODE_REPORT.md §3.3)
"""

import numpy as np
from corner_scene import generate_corner_scene
from corner_calib import compute_cost, compute_errors, solve_lm


def run_single_calibration(seed=42, n_poses=20, alpha=np.pi/2,
                            noise_sigma_mm=0.0, perturbation_scale=0.3,
                            show_acquisition=True, verbose=True):
    """单次角点法标定"""
    # 场景生成
    res = generate_corner_scene(seed=seed, n_poses=n_poses, alpha=alpha,
                                noise_sigma=noise_sigma_mm / 1000.0)
    scene, poses, meas, theta_gt = (res['scene'], res['poses'],
                                     res['measurements'], res['theta_gt'])

    # 可视化
    if show_acquisition:
        try:
            from reproduction_scene import animate_corner_acquisition
            import matplotlib.pyplot as plt
            plt.ion()  # 确保交互模式
            animate_corner_acquisition(res)
            plt.ioff()
        except Exception as e:
            import traceback
            print(f"(动画出错: {e})")
            traceback.print_exc()

    # 扰动初始值
    rng = np.random.default_rng(seed * 2 + 1)
    theta_init = theta_gt.copy()
    theta_init[0:3] += rng.normal(0, perturbation_scale, 3)
    theta_init[3:6] += rng.normal(0, perturbation_scale * 0.02, 3)
    theta_init[6:9] += rng.normal(0, perturbation_scale * 0.5, 3)
    theta_init[9:12] += rng.normal(0, perturbation_scale * 0.01, 3)

    # Gauge 固定: n_B·C
    from corner_calib import so3_exp
    n_B_scene = scene['n_B']
    C_scene = scene['C']
    fix_C_proj = np.dot(n_B_scene, C_scene)
    # 也修复初始 C 的投影
    nB_init = so3_exp(theta_init[6:9])[:, 2]
    C_init = theta_init[9:12]
    theta_init[9:12] = C_init - (np.dot(nB_init, C_init) - fix_C_proj) * nB_init

    # LM 求解 (带 gauge 固定)
    theta_opt, converged, cost_history = solve_lm(
        theta_init, poses, meas, alpha=alpha,
        verbose=verbose, max_iter=50,
        fix_C_proj=fix_C_proj)

    R_err, t_err = compute_errors(theta_opt, theta_gt)

    if verbose:
        print(f"\n结果: converged={converged}, R_err={R_err:.4f}°, "
              f"t_err={t_err:.4f}mm")

    return {
        'converged': converged,
        'R_error': R_err,
        't_error': t_err,
        'cost_history': cost_history,
    }


def run_monte_carlo(n_trials=50, n_poses=20, alpha=np.pi/2,
                     noise_sigma_mm=0.0, perturbation_scale=0.3,
                     verbose=True):
    """Monte Carlo 标定精度评估"""
    R_errors, t_errors = [], []
    converged_count = 0

    for trial in range(n_trials):
        seed = 42 + trial * 137
        result = run_single_calibration(
            seed=seed, n_poses=n_poses, alpha=alpha,
            noise_sigma_mm=noise_sigma_mm,
            perturbation_scale=perturbation_scale,
            show_acquisition=False, verbose=False)

        if result['converged']:
            converged_count += 1
            R_errors.append(result['R_error'])
            t_errors.append(result['t_error'])

    R_arr = np.array(R_errors)
    t_arr = np.array(t_errors)

    stats = {
        'n_trials': n_trials,
        'converged': converged_count,
        'converged_rate': converged_count / n_trials * 100,
        'R_median': np.median(R_arr) if len(R_arr) else np.nan,
        't_median': np.median(t_arr) if len(t_arr) else np.nan,
    }

    if verbose:
        print(f"\nMC ({n_trials} trials): "
              f"收敛率={stats['converged_rate']:.0f}%, "
              f"R_median={stats['R_median']:.4f}°, "
              f"t_median={stats['t_median']:.4f}mm")

    return stats
