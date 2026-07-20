#!/usr/bin/env python3
"""
test_tilted_corner.py — 验证假说: 只要FOV三角形同时穿过两个边(z_S不必垂直), gauge自然消失

对比:
  A) 旧角点法: z_S = -n_B (强制垂直) → gauge存在
  B) 倾斜角点法: z_S 倾斜但两边都可见 → gauge消失?
"""

import numpy as np
from corner_scene import (
    generate_corner_scene, generate_corner_plane, generate_corner_measurements,
    so3_exp, so3_log
)
from corner_calib import compute_jacobian_numerical, compute_cost, solve_lm, compute_errors
from reproduction_scene import generate_hand_eye_gt, compute_fov_plate_scanline
from nbv_edge_plane import _build_R_edge


# ============================================================================
# 倾斜角点位姿生成: z_S 可变, 但保证两边都可见
# ============================================================================

def generate_tilted_corner_poses(C, n_B, u_B, v_B, R_he, t_he,
                                  plate_w=400, plate_h=500,
                                  pitch_range=(-30, 30), yaw_range=(-15, 15),
                                  n_poses=8, rng=None):
    """生成 z_S 倾斜但能看到角点两边的位姿

    策略: 在 pitch/yaw 网格中搜索, 用 compute_fov_plate_scanline 验证两边可见
    """
    if rng is None:
        rng = np.random.default_rng(42)

    w_m, h_m = plate_w / 1000.0, plate_h / 1000.0
    pitches = np.linspace(pitch_range[0], pitch_range[1], 8)
    yaws = np.linspace(yaw_range[0], yaw_range[1], 5)

    candidates = []
    MAX_CANDIDATES = 200  # 找到足够候选就停

    # 尝试两种 x_align: u_B 和 v_B, 增加方向多样性
    for x_align in [u_B, v_B]:
        for pitch in pitches:
            for yaw in yaws:
                if len(candidates) >= MAX_CANDIDATES:
                    break
                R_i = _build_R_edge(pitch, yaw, x_align, n_B, u_B, v_B)

                # 传感器在基座标系的朝向
                R_BS = R_i @ R_he
                z_S = R_BS[:, 2]

                # 目标点: 角点附近, 传感器在 n_B 方向拉开距离
                for u_off in np.linspace(0.01, 0.10, 3):
                    for v_off in np.linspace(0.01, 0.10, 3):
                        target = C + u_off * w_m * u_B + v_off * h_m * v_B
                        for standoff_h in [1.5, 2.2, 3.0]:
                            if len(candidates) >= MAX_CANDIDATES:
                                break
                            standoff_m = 0.35 * standoff_h
                            t_i = target + standoff_m * n_B - R_i @ t_he
                            t_BS = t_i + R_i @ t_he

                            sl = compute_fov_plate_scanline(
                                R_BS, t_BS, C, n_B, u_B, v_B, w_m, h_m)

                            if not sl['has_intersection']:
                                continue
                            if len(sl['scan_pts_S']) < 10:
                                continue

                            eps = [e for e, _ in sl['endpoints_S']]
                            if 'e1' in eps and 'e2' in eps:
                                z_dev = np.rad2deg(np.arccos(
                                    np.clip(np.dot(z_S, -n_B), -1, 1)))
                                candidates.append({
                                    'R_i': R_i, 't_i': t_i,
                                    'pitch': pitch, 'yaw': yaw,
                                    'z_dev': z_dev,
                                    'n_pts': len(sl['scan_pts_S']),
                                })
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
            if len(candidates) >= MAX_CANDIDATES:
                break
        if len(candidates) >= MAX_CANDIDATES:
            break

    if len(candidates) == 0:
        return []

    # 按 z_dev 排序, 选最多样化的
    candidates.sort(key=lambda c: c['z_dev'])

    if len(candidates) <= n_poses:
        return [(c['R_i'], c['t_i']) for c in candidates], candidates

    # 贪心选多样化的
    selected_idx = [0]  # z_dev 最小的(最接近垂直)
    for _ in range(n_poses - 1):
        best_i, best_dist = None, -1
        for i in range(len(candidates)):
            if i in selected_idx:
                continue
            min_d = min(
                np.arccos(np.clip(
                    (np.trace(candidates[i]['R_i'].T @ candidates[j]['R_i']) - 1) / 2,
                    -1, 1))
                for j in selected_idx
            )
            if min_d > best_dist:
                best_dist = min_d
                best_i = i
        if best_i is not None:
            selected_idx.append(best_i)

    sel = [candidates[i] for i in sorted(selected_idx)]
    return [(c['R_i'], c['t_i']) for c in sel], sel


# ============================================================================
# 测试
# ============================================================================

def test_one_scenario(seed=42, n_poses=6, alpha=np.pi/2,
                       noise_sigma_mm=0.0, label="", verbose=True):
    """单次测试"""

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # 手眼真值
    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    # 角点平板
    C, n_B, u_B, v_B, d_1, d_2, w_m, h_m = generate_corner_plane(rng)
    R_pl = np.column_stack([u_B, v_B, n_B])

    scene = {
        'R_he': R_he, 't_he': t_he, 'X_gt': X_gt,
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'd_1': d_1, 'd_2': d_2, 'alpha': alpha,
        'plate_w': w_m * 1000, 'plate_h': h_m * 1000,
        'w': w_m, 'h': h_m,
    }

    # 倾斜位姿
    pose_tuples, candidates = generate_tilted_corner_poses(
        C, n_B, u_B, v_B, R_he, t_he,
        plate_w=w_m*1000, plate_h=h_m*1000,
        n_poses=n_poses, rng=rng)

    if len(pose_tuples) == 0:
        print(f"  {label}: 找不到足够位姿")
        return None

    poses = pose_tuples

    # 测量 (零噪声)
    meas = generate_corner_measurements(
        scene, poses, n_plane_pts=30, rng=rng,
        noise_sigma=noise_sigma_mm / 1000.0)

    # 12-DOF GT
    w_he = so3_log(R_he)
    w_pl = so3_log(R_pl)
    theta_gt = np.concatenate([w_he, t_he, w_pl, C])

    # ====== SVD 分析 ======
    J, r, mask = compute_jacobian_numerical(theta_gt, poses, meas, alpha)
    J_valid = J[mask, :]
    U, s, Vt = np.linalg.svd(J_valid, full_matrices=False)

    cond_num = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
    gauge_exist = s[-1] / s[0] < 1e-6

    # ====== 零噪声 LM (无 gauge fixing) ======
    rng2 = np.random.default_rng(seed * 3 + 7)
    theta_init = theta_gt.copy()
    theta_init[0:3] += rng2.normal(0, 0.1, 3)
    theta_init[3:6] += rng2.normal(0, 0.002, 3)
    theta_init[6:9] += rng2.normal(0, 0.05, 3)
    theta_init[9:12] += rng2.normal(0, 0.001, 3)

    theta_opt, converged, _ = solve_lm(
        theta_init, poses, meas, alpha=alpha,
        verbose=False, max_iter=80,
        fix_C_proj=None)  # 不固定 gauge!

    R_err, t_err = compute_errors(theta_opt, theta_gt)

    # ====== z_S 偏离角度 ======
    z_devs = [np.rad2deg(np.arccos(np.clip(
        np.dot(R_i @ R_he[:, 2], -n_B), -1, 1)))
              for (R_i, _) in poses]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {'='*60}")
        print(f"  位姿数: {len(poses)}")
        print(f"  z_S 偏离 -n_B: min={min(z_devs):.1f}°  max={max(z_devs):.1f}°  "
              f"mean={np.mean(z_devs):.1f}°  std={np.std(z_devs):.1f}°")
        print(f"  cond(J): {cond_num:.2e}")
        print(f"  σ_min/σ_max: {s[-1]/s[0]:.2e}")
        print(f"  Gauge: {'✗ 存在' if gauge_exist else '✓ 消失'}")
        print(f"  奇异值: ", end="")
        for sv in s:
            print(f"{sv:.2e}  ", end="")
        print()
        print(f"  Vt[-1,:] (零空间方向):")
        labels = ['w_he_x','w_he_y','w_he_z','t_he_x','t_he_y','t_he_z',
                   'w_pl_x','w_pl_y','w_pl_z','C_x','C_y','C_z']
        for j, (lbl, v) in enumerate(zip(labels, Vt[-1, :])):
            bar = '█' * int(abs(v) * 40)
            print(f"    {lbl}: {v:+.4f}  {bar}")
        print(f"  零噪声 LM (无 gauge fixing):")
        print(f"    converged = {converged}")
        print(f"    R_err     = {R_err:.6f}°")
        print(f"    t_err     = {t_err:.6f}mm")

    return {
        'label': label,
        'n_poses': len(poses),
        'z_dev_mean': np.mean(z_devs),
        'z_dev_std': np.std(z_devs),
        'z_dev_max': max(z_devs),
        'cond_num': cond_num,
        'sigma_min_ratio': s[-1] / s[0],
        'gauge': gauge_exist,
        'converged': converged,
        'R_err': R_err,
        't_err': t_err,
        'singular_values': s,
        'Vt_last': Vt[-1, :],
    }


if __name__ == '__main__':
    print("=" * 70)
    print("验证: 倾斜角点法 (z_S ≠ -n_B, 两边都可见) — gauge 是否消失?")
    print("=" * 70)

    results = []

    # 测试多组随机场景
    for seed in [42, 99, 137, 200, 300]:
        r = test_one_scenario(seed=seed, n_poses=8, noise_sigma_mm=0.0,
                               label=f"seed={seed}")
        if r:
            results.append(r)

    # 汇总
    if results:
        print(f"\n{'='*70}")
        print("汇总")
        print(f"{'='*70}")
        print(f"  {'场景':<12} {'z_dev°':>8} {'cond(J)':>10} {'gauge':>6} {'R_err°':>10} {'t_errmm':>10}")
        print(f"  {'-'*60}")
        for r in results:
            print(f"  {r['label']:<12} {r['z_dev_mean']:>7.1f}±{r['z_dev_std']:.0f}  "
                  f"{r['cond_num']:>10.2e}  {'✗' if r['gauge'] else '✓':>6}  "
                  f"{r['R_err']:>10.6f}  {r['t_err']:>10.6f}")

        conds = [r['cond_num'] for r in results]
        gauges = [r['gauge'] for r in results]
        R_errs = [r['R_err'] for r in results]
        print(f"\n  cond(J) 范围: {min(conds):.2e} ~ {max(conds):.2e}")
        print(f"  gauge 消失率: {sum(1 for g in gauges if not g)}/{len(gauges)}")
        print(f"  R=0 收敛率: {sum(1 for e in R_errs if e < 0.001)}/{len(R_errs)}")
