#!/usr/bin/env python3
"""
slam_calib.py — SLAM式连续采集手眼标定 v2

核心改动: 用传感器帧 x 坐标排序端点 (x较小→edge0, x较大→edge1)
           → 赋值完全独立于 R_he → 从单位阵开始优化

原理:
  FOV三角形从角点出发, 沿板面滑动, 20+帧连续采集。
  每帧扫描线从左到右穿过两条边 → 左端点和右端点天然可区分。
  左端点始终在边0上, 右端点始终在边1上 (传感器不翻转前提)。
"""

import numpy as np
from corner_scene import so3_exp, so3_log


# ============================================================================
# 1. SLAM 轨迹生成
# ============================================================================

def generate_slam_trajectory(scene, R_he, t_he, n_frames=30,
                               start_uv=(0.05, 0.05), end_uv=(0.45, 0.40),
                               tilt_range_deg=25.0, rng=None):
    """平滑轨迹: 每帧验证 FOV 同时穿过两边, 只保留有效帧。"""
    if rng is None:
        rng = np.random.default_rng(42)
    C, n_B, u_B, v_B = scene['C'], scene['n_B'], scene['u_B'], scene['v_B']
    w_m, h_m = scene['w'], scene['h']

    from reproduction_scene import compute_fov_plate_scanline

    # 密集采样位置, 每个位置试多种朝向, 找两边都可见的
    u_vals = np.linspace(start_uv[0], end_uv[0], n_frames * 3)
    v_vals = np.linspace(start_uv[1], end_uv[1], n_frames * 3)

    poses, z_devs = [], []
    last_tilt_sign = 1

    for k in range(len(u_vals)):
        target = C + u_vals[k] * w_m * u_B + v_vals[k] * h_m * v_B

        # 对于每个位置, 搜索 tilt 角度和 standoff 使两边可见
        found = False
        for tilt_mag in [15, 20, 25]:
            for sign in [1, -1]:
                tilt_rad = np.deg2rad(tilt_mag * sign)
                for x_S in [v_B, u_B]:
                    Kx = np.array([[0, -x_S[2], x_S[1]], [x_S[2], 0, -x_S[0]], [-x_S[1], x_S[0], 0]])
                    R_tilt = np.eye(3) + np.sin(tilt_rad) * Kx + (1 - np.cos(tilt_rad)) * Kx @ Kx

                    z_S = R_tilt @ (-n_B); z_S /= np.linalg.norm(z_S)
                    y_S = np.cross(z_S, x_S)
                    if np.linalg.norm(y_S) < 1e-8: continue
                    y_S /= np.linalg.norm(y_S)
                    x_S = np.cross(y_S, z_S)
                    R_S = np.column_stack([x_S, y_S, z_S])
                    R_i = R_S @ R_he.T

                    for standoff in [0.55, 0.75, 1.0]:
                        sensor_pos = target + standoff * n_B
                        t_i = sensor_pos - R_i @ t_he
                        R_BS = R_i @ R_he
                        t_BS = t_i + R_i @ t_he

                        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, w_m, h_m)
                        if not sl['has_intersection'] or len(sl['scan_pts_S']) < 10:
                            continue
                        eps = [e for e, _ in sl['endpoints_S']]
                        if 'e1' in eps and 'e2' in eps:
                            z_dev = np.rad2deg(np.arccos(np.clip(np.dot(z_S, -n_B), -1, 1)))
                            poses.append((R_i, t_i))
                            z_devs.append(z_dev)
                            found = True
                            break
                    if found: break
                if found: break
            if found: break

        if found and len(poses) >= n_frames:
            break

    return poses, {'z_devs': z_devs, 'n_frames': len(poses)}


# ============================================================================
# 2. SLAM 残差 — 传感器帧 x 排序硬赋值
# ============================================================================

def assign_endpoints_by_scan_order(measurements):
    """传感器帧 x 坐标排序: x较小→edge0, x较大→edge1。完全独立于 R_he。"""
    edge0_S, edge1_S = [], []
    for m in measurements:
        pts = []
        if m['valid_e1']: pts.append(m['p_S_e1'].copy())
        if m['valid_e2']: pts.append(m['p_S_e2'].copy())
        if len(pts) < 2: continue
        pts.sort(key=lambda p: p[0])  # 按 x 排序
        edge0_S.append(pts[0]); edge1_S.append(pts[1])
    return edge0_S, edge1_S


def slam_residuals(theta, poses, measurements, w_plane=1.0, w_edge=1.0):
    """SLAM残差: 平面centered + 边缘共线(传感器帧x排序赋值)"""
    w_he, t_he, w_pl = theta[0:3], theta[3:6], theta[6:9]
    R_he = so3_exp(w_he)
    R_pl = so3_exp(w_pl)
    u_B, v_B, n_B = R_pl[:, 0], R_pl[:, 1], R_pl[:, 2]

    edge0_S, edge1_S = assign_endpoints_by_scan_order(measurements)

    edge0_B, edge1_B, plane_vals = [], [], []
    for k, ((R_i, t_i), m) in enumerate(zip(poses, measurements)):
        R_BS = R_i @ R_he; t_BS = t_i + R_i @ t_he
        if k < len(edge0_S): edge0_B.append(R_BS @ edge0_S[k] + t_BS)
        if k < len(edge1_S): edge1_B.append(R_BS @ edge1_S[k] + t_BS)
        for p_S in m['p_S_plane']:
            plane_vals.append(np.dot(n_B, R_BS @ p_S + t_BS))

    plane_vals = np.array(plane_vals)
    if len(plane_vals) > 0: plane_vals = plane_vals - np.mean(plane_vals)

    edge_res = []
    for k in range(len(edge0_B) - 1):
        edge_res.extend(np.cross(edge0_B[k + 1] - edge0_B[k], u_B).tolist())
    for k in range(len(edge1_B) - 1):
        edge_res.extend(np.cross(edge1_B[k + 1] - edge1_B[k], v_B).tolist())

    residuals, mask = [], []
    wp, we = np.sqrt(w_plane), np.sqrt(w_edge)
    for v in plane_vals: residuals.append(v * wp); mask.append(True)
    for v in edge_res: residuals.append(v * we); mask.append(True)

    return np.array(residuals), np.array(mask)


def slam_cost(theta, poses, measurements, w_plane=1.0, w_edge=1.0):
    r, m = slam_residuals(theta, poses, measurements, w_plane, w_edge)
    rv = r[m.astype(bool)]; return 0.5 * np.dot(rv, rv) if len(rv) > 0 else 1e30


def slam_jacobian(theta, poses, measurements, w_plane=1.0, w_edge=1.0, eps=1e-6):
    r0, mask = slam_residuals(theta, poses, measurements, w_plane, w_edge)
    J = np.zeros((len(r0), 9))
    for k in range(9):
        step = np.zeros(9); step[k] = eps
        rp, _ = slam_residuals(theta + step, poses, measurements, w_plane, w_edge)
        rm, _ = slam_residuals(theta - step, poses, measurements, w_plane, w_edge)
        J[:, k] = (rp - rm) / (2 * eps)
    return J, r0, mask


def slam_solve_lm(theta_init, poses, measurements, w_plane=1.0, w_edge=1.0,
                   max_iter=100, tol=1e-10, lam0=1e-4):
    theta = theta_init.copy(); lam = lam0
    for it in range(max_iter):
        J, r, mask = slam_jacobian(theta, poses, measurements, w_plane, w_edge)
        rv, Jv = r[mask.astype(bool)], J[mask.astype(bool), :]
        if len(rv) == 0: break
        cost = 0.5 * np.dot(rv, rv)
        H, g = Jv.T @ Jv, Jv.T @ rv
        try: delta = -np.linalg.solve(H + lam * np.eye(9), g)
        except np.linalg.LinAlgError: lam *= 10; continue
        tn = theta + delta
        cn = slam_cost(tn, poses, measurements, w_plane, w_edge)
        if cn < cost:
            theta = tn; lam = max(lam / 3, 1e-12)
            if abs(cost - cn) < tol: break
        else: lam = min(lam * 3, 1e6)
    return theta


# ============================================================================
# 3. 完整求解 (含粗搜索初始化)
# ============================================================================

def _sample_rotations(n_theta=18, n_phi=12, n_angle=8):
    rotations = []
    for theta in np.linspace(0, np.pi, n_theta):
        for phi in np.linspace(0, 2 * np.pi, int(n_phi * np.sin(theta) + 1)):
            for angle in np.linspace(0, 2 * np.pi, n_angle, endpoint=False):
                axis = np.array([np.sin(theta) * np.cos(phi),
                                 np.sin(theta) * np.sin(phi), np.cos(theta)])
                rotations.append(angle * axis)
    return rotations[::3]


def coarse_init_rotation(poses, measurements, n_samples=500):
    rng = np.random.default_rng(42)
    all_rots = _sample_rotations(18, 12)
    if len(all_rots) > n_samples:
        idx = rng.choice(len(all_rots), n_samples, replace=False)
        sampled = [all_rots[i] for i in idx]
    else: sampled = all_rots

    best_cost, best_w_he = np.inf, np.zeros(3)
    for w_he in sampled:
        theta_try = np.zeros(9); theta_try[0:3] = w_he
        cost = slam_cost(theta_try, poses, measurements, w_plane=0.0, w_edge=1.0)
        if cost < best_cost and cost > 1e-12:
            best_cost, best_w_he = cost, w_he.copy()
    return best_w_he, best_cost


def slam_solve_with_init(poses, measurements, w_plane=1.0, w_edge=1.0,
                          use_coarse=True, max_iter=80):
    if use_coarse:
        w_he_init, _ = coarse_init_rotation(poses, measurements, n_samples=1000)
        theta = np.zeros(9); theta[0:3] = w_he_init
    else: theta = np.zeros(9)

    from unsupervised_edge_calib import plane_only_solve_lm
    theta = plane_only_solve_lm(theta, poses, measurements, max_iter=40)
    theta = slam_solve_lm(theta, poses, measurements,
                           w_plane=w_plane, w_edge=w_edge, max_iter=max_iter)
    return theta


def slam_errors(theta_est, theta_gt):
    Re_est = so3_exp(theta_est[0:3]); Re_gt = so3_exp(theta_gt[0:3])
    Rdiff = Re_est.T @ Re_gt
    tr = np.clip((np.trace(Rdiff) - 1) / 2, -1, 1)
    R_err = np.rad2deg(np.arccos(tr))
    t_err = np.linalg.norm(theta_est[3:6] - theta_gt[3:6]) * 1000
    return R_err, t_err


# ============================================================================
# 4. 测试
# ============================================================================

if __name__ == '__main__':
    import yaml
    from calibration_framework import CalibrationEnvironment
    from corner_scene import generate_corner_measurements

    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  SLAM 连续采集手眼标定 v2 (传感器帧x排序)")
    print("=" * 60)

    results_zero = []; results_noise = []

    for trial_num in range(10):
        seed = 42 + trial_num
        config['environment']['laser_noise'] = 0.0
        env = CalibrationEnvironment(config, seed=seed)
        scene = env.scene
        rng = np.random.default_rng(seed)

        poses, traj = generate_slam_trajectory(scene, env.R_he, env.t_he,
                                                n_frames=30, tilt_range_deg=25.0, rng=rng)

        meas_zero = generate_corner_measurements(scene, poses, n_plane_pts=15, noise_sigma=0.0)
        theta_opt = slam_solve_with_init(poses, meas_zero, w_plane=1.0, w_edge=1.0, max_iter=80)
        Rz, tz = slam_errors(theta_opt, env.theta_gt_9)
        results_zero.append((Rz, tz))

        meas_noisy = generate_corner_measurements(scene, poses, n_plane_pts=15, noise_sigma=0.055 / 1000)
        theta_opt_n = slam_solve_with_init(poses, meas_noisy, w_plane=1.0, w_edge=1.0, max_iter=80)
        Rn, tn = slam_errors(theta_opt_n, env.theta_gt_9)
        results_noise.append((Rn, tn))

        if trial_num < 3:
            print(f"\n  trial {trial_num+1} (seed={seed}):")
            print(f"    零噪声: R={Rz:.6f}°, t={tz:.6f}mm")
            print(f"    σ=0.055: R={Rn:.6f}°, t={tn:.6f}mm")
            print(f"    轨迹={traj['n_frames']}帧, tilt={np.mean(traj['z_devs']):.1f}°±{np.std(traj['z_devs']):.1f}°")

    Rz_a = np.array([r[0] for r in results_zero])
    Tz_a = np.array([r[1] for r in results_zero])
    Rn_a = np.array([r[0] for r in results_noise])
    Tn_a = np.array([r[1] for r in results_noise])

    print(f"\n{'='*60}")
    print(f"  汇总 (10 trials × 24帧)")
    print(f"  {'':>20} {'零噪声':>12} {'σ=0.055mm':>12}")
    print(f"  {'R median':>20} {np.median(Rz_a):12.6f}° {np.median(Rn_a):12.6f}°")
    print(f"  {'R mean':>20} {np.mean(Rz_a):12.6f}° {np.mean(Rn_a):12.6f}°")
    print(f"  {'R max':>20} {np.max(Rz_a):12.6f}° {np.max(Rn_a):12.6f}°")
    print(f"  {'t median':>20} {np.median(Tz_a):12.6f}mm {np.median(Tn_a):12.6f}mm")
    print(f"  {'R<0.1°':>20} {np.sum(Rz_a<0.1):12d} {np.sum(Rn_a<0.1):12d}")
    print(f"  {'R<0.05°':>20} {np.sum(Rz_a<0.05):12d} {np.sum(Rn_a<0.05):12d}")
