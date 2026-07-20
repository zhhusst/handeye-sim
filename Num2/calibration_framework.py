#!/usr/bin/env python3
"""
calibration_framework.py — Num2 手眼标定统一框架

提供:
  CalibrationEnvironment  — 共享仿真环境 (场景生成 + 噪声 + 测量)
  PoseCollector           — 位姿采集 (动画 / NBV / 网格 / 角点导向)
  SolverDispatcher        — 方法分发 (12-DOF / 9-DOF / 倾斜角点)
  ResultReporter          — 结果汇总报告

用法:
  from calibration_framework import CalibrationFramework
  fw = CalibrationFramework('config.yaml')
  result = fw.run()
  fw.report(result)
"""

import numpy as np
import sys
import os
import yaml

# 确保可以导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corner_scene import (
    generate_corner_scene, generate_corner_plane,
    generate_corner_measurements, so3_exp, so3_log
)
from corner_calib import (
    compute_residuals, compute_cost, compute_jacobian_numerical,
    solve_lm, compute_errors
)
from reproduction_scene import generate_hand_eye_gt, compute_fov_plate_scanline
from nbv_edge_plane import (
    _build_R_edge, generate_edge_candidates, select_diverse,
    combined_residuals, combined_cost, combined_jacobian,
    combined_solve_lm, combined_errors
)


# ============================================================================
# 1. CalibrationEnvironment — 共享仿真环境
# ============================================================================

class CalibrationEnvironment:
    """统一的仿真环境: 场景生成 + 噪声施加 + 测量生成"""

    def __init__(self, config, seed=42):
        self.config = config['environment']
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # 生成场景
        self._build_scene()

    def _build_scene(self):
        """构建完整仿真场景"""
        c = self.config
        rng = self.rng

        # 手眼真值
        X_gt = generate_hand_eye_gt()
        R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

        # 角点平板
        alpha = np.deg2rad(c.get('corner_angle', 90))
        pw, ph = c['plate_width'], c['plate_height']
        C, n_B, u_B, v_B, d_1, d_2, w_m, h_m = generate_corner_plane(
            rng, plate_w=pw, plate_h=ph, alpha=alpha)

        R_pl = np.column_stack([u_B, v_B, n_B])

        # 12-DOF GT 向量
        w_he = so3_log(R_he)
        w_pl = so3_log(R_pl)
        theta_gt = np.concatenate([w_he, t_he, w_pl, C])

        # 9-DOF GT 向量
        theta_gt_9 = np.concatenate([w_he, t_he, w_pl])

        self.X_gt = X_gt
        self.R_he = R_he
        self.t_he = t_he
        self.C = C
        self.n_B = n_B
        self.u_B = u_B
        self.v_B = v_B
        self.d_1 = d_1
        self.d_2 = d_2
        self.w_m = w_m
        self.h_m = h_m
        self.alpha = alpha
        self.R_pl = R_pl
        self.theta_gt = theta_gt
        self.theta_gt_9 = theta_gt_9

        # 场景 dict (兼容旧接口)
        self.scene = {
            'R_he': R_he, 't_he': t_he, 'X_gt': X_gt,
            'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
            'd_1': d_1, 'd_2': d_2, 'alpha': alpha,
            'plate_w': pw, 'plate_h': ph,
            'w': w_m, 'h': h_m,
        }

    def generate_measurements(self, poses, noise_sigma_mm=None):
        """从位姿列表生成测量 (施加噪声)"""
        if noise_sigma_mm is None:
            noise_sigma_mm = self.config.get('laser_noise', 0.055)

        meas = generate_corner_measurements(
            self.scene, poses,
            n_plane_pts=30,
            rng=self.rng,
            noise_sigma=noise_sigma_mm / 1000.0)

        # 板翘曲
        warp_mm = self.config.get('warp', 0.0)
        if warp_mm > 0:
            self._apply_warp(meas, warp_mm)

        return meas

    def _apply_warp(self, measurements, warp_mm):
        """施加板翘曲 (正弦波)"""
        for m in measurements:
            if len(m['p_S_plane']) > 10:
                pts = m['p_S_plane']
                warp = warp_mm / 1000.0 * np.sin(2 * np.pi * pts[:, 0] / 0.3)
                pts[:, 2] += warp

    def apply_robot_noise(self, poses, repeat_mm=None):
        """对位姿施加机器人重复性噪声"""
        if repeat_mm is None:
            repeat_mm = self.config.get('repeat_noise', 0.0)
        if repeat_mm <= 0:
            return poses

        noisy = []
        for R_i, t_i in poses:
            noisy.append((R_i, t_i + self.rng.normal(0, repeat_mm / 1000.0, 3)))
        return noisy

    def compute_z_deviation(self, poses):
        """计算每个位姿 z_S 与 -n_B 的偏离角度"""
        devs = []
        for R_i, _ in poses:
            z_S = R_i @ self.R_he[:, 2]
            dev = np.rad2deg(np.arccos(np.clip(np.dot(z_S, -self.n_B), -1, 1)))
            devs.append(dev)
        return np.array(devs)


# ============================================================================
# 2. PoseCollector — 位姿采集
# ============================================================================

class PoseCollector:
    """位姿采集器: 支持动画交互 / NBV 自动搜索 / 网格搜索 / 角点导向"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.pc_config = config.get('pose_collection', {})
        self.n_poses = self.pc_config.get('n_poses', 6)
        self.mode = self.pc_config.get('mode', 'auto_grid')

    def collect(self):
        """根据配置选择采集模式"""
        mode = self.mode
        if mode == 'animation':
            return self._collect_animation()
        elif mode == 'auto_nbv':
            return self._collect_nbv()
        elif mode == 'auto_grid':
            return self._collect_grid()
        elif mode == 'auto_corner':
            return self._collect_corner()
        else:
            raise ValueError(f"未知采集模式: {mode}")

    def _collect_animation(self):
        """动画交互采集"""
        from reproduction_scene import animate_corner_acquisition

        # 先自动生成一批候选位姿
        poses, candidates = self._collect_grid(return_candidates=True)
        if len(poses) == 0:
            print("⚠ 网格搜索未找到可用位姿，使用角点导向生成")
            poses = self._collect_corner()

        # 构建动画数据
        res = {
            'scene': self.env.scene,
            'poses': poses,
            'measurements': self.env.generate_measurements(poses, noise_sigma_mm=0),
            'theta_gt': self.env.theta_gt,
        }

        print(f"\n  🎬 启动动态采集动画 ({len(poses)} 个位姿)...")
        print(f"     按 Enter 在每个位姿处暂停，观察 FOV 是否穿过两边")
        print(f"     关闭窗口继续标定")

        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            plt.ion()
            animate_corner_acquisition(res)
            plt.ioff()
        except Exception as e:
            import traceback
            print(f"  (动画出错: {e})")
            traceback.print_exc()

        return poses

    def _collect_nbv(self):
        """NBV 自动候选搜索 (9-DOF 模式)"""
        nbv_cfg = self.pc_config.get('nbv', {})
        edge_mode = nbv_cfg.get('edge_mode', 'both')

        c_e1, c_e2 = generate_edge_candidates(
            self.env.scene, self.env.R_he, self.env.t_he,
            edge=edge_mode, n_grid=6)

        sel_e1 = select_diverse(c_e1, self.n_poses // 2)
        sel_e2 = select_diverse(c_e2, self.n_poses // 2)
        selected = sel_e1 + sel_e2

        return [(c['R_i'], c['t_i']) for c in selected]

    def _collect_grid(self, return_candidates=False):
        """网格搜索: 倾斜位姿 + 两边都可见"""
        return generate_tilted_corner_poses(
            self.env.C, self.env.n_B, self.env.u_B, self.env.v_B,
            self.env.R_he, self.env.t_he,
            plate_w=self.env.config['plate_width'],
            plate_h=self.env.config['plate_height'],
            pitch_range=tuple(self.pc_config.get('grid', {}).get('pitch_range', [-30, 30])),
            yaw_range=tuple(self.pc_config.get('grid', {}).get('yaw_range', [-15, 15])),
            n_poses=self.n_poses,
            rng=self.env.rng,
            return_candidates=return_candidates)

    def _collect_corner(self):
        """角点导向位姿 (强制垂直, 仅用于对比)"""
        from corner_scene import generate_corner_poses
        return generate_corner_poses(
            self.env.rng, self.n_poses,
            self.env.C, self.env.n_B, self.env.u_B, self.env.v_B,
            self.env.R_he, self.env.t_he, self.env.alpha,
            plate_w=self.env.config['plate_width'],
            plate_h=self.env.config['plate_height'])


# ============================================================================
# 3. SolverDispatcher — 方法分发
# ============================================================================

class SolverDispatcher:
    """标定方法分发器"""

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.solver_cfg = config.get('solver', {})

    def solve(self, poses, measurements, method):
        """根据方法名分发求解"""
        if method == 'corner_12dof':
            return self._solve_corner_12dof(poses, measurements)
        elif method == 'plane_edge_9dof':
            return self._solve_plane_edge_9dof(poses, measurements)
        elif method == 'tilted_corner':
            return self._solve_tilted_corner(poses, measurements)
        else:
            raise ValueError(f"未知方法: {method}")

    def _solve_corner_12dof(self, poses, measurements):
        """12-DOF 角点法 (强制垂直模式, 需 gauge fixing)"""
        gauge_fix = self.solver_cfg.get('gauge_fix', False)
        fix_C_proj = None
        if gauge_fix:
            fix_C_proj = np.dot(self.env.n_B, self.env.C)

        theta_init = self._perturb_init(n_dof=12)
        if gauge_fix:
            # 修复初始 C 投影
            nB_init = so3_exp(theta_init[6:9])[:, 2]
            C_init = theta_init[9:12]
            theta_init[9:12] = C_init - (np.dot(nB_init, C_init) - fix_C_proj) * nB_init

        theta_opt, converged, history = solve_lm(
            theta_init, poses, measurements,
            alpha=self.env.alpha,
            max_iter=self.solver_cfg.get('max_iter', 50),
            tol=self.solver_cfg.get('tol', 1e-8),
            lam0=self.solver_cfg.get('lam0', 1e-4),
            fix_C_proj=fix_C_proj,
            verbose=False)

        R_err, t_err = compute_errors(theta_opt, self.env.theta_gt)

        return {
            'theta_opt': theta_opt,
            'converged': converged,
            'R_error': R_err,
            't_error': t_err,
            'cost_history': history,
            'n_iter': len(history),
        }

    def _solve_tilted_corner(self, poses, measurements):
        """倾斜角点法: 12-DOF, 无 gauge fixing"""
        theta_init = self._perturb_init(n_dof=12)

        theta_opt, converged, history = solve_lm(
            theta_init, poses, measurements,
            alpha=self.env.alpha,
            max_iter=self.solver_cfg.get('max_iter', 80),
            tol=self.solver_cfg.get('tol', 1e-8),
            lam0=self.solver_cfg.get('lam0', 1e-4),
            fix_C_proj=None,
            verbose=False)

        R_err, t_err = compute_errors(theta_opt, self.env.theta_gt)

        return {
            'theta_opt': theta_opt,
            'converged': converged,
            'R_error': R_err,
            't_error': t_err,
            'cost_history': history,
            'n_iter': len(history),
        }

    def _solve_plane_edge_9dof(self, poses, measurements):
        """9-DOF 平面+边缘联合 (多权重 + 多重启)"""
        best_result = None
        best_Re = np.inf
        rng = self.env.rng

        for wp, we in self.solver_cfg.get('weight_schemes', [(1.0, 1.0)]):
            for _ in range(self.solver_cfg.get('n_restarts', 3)):
                ti = self.env.theta_gt_9.copy()
                ti[0:3] += rng.normal(0, 0.1, 3)
                ti[3:6] += rng.normal(0, 0.002, 3)
                ti[6:9] += rng.normal(0, 0.05, 3)

                to = combined_solve_lm(
                    ti, poses, measurements,
                    w_plane=wp, w_edge=we,
                    max_iter=self.solver_cfg.get('max_iter', 100))

                Re, te, tipe = combined_errors(to, self.env.theta_gt_9)
                if Re < best_Re:
                    best_Re = Re
                    dt = to[3:6] - self.env.theta_gt_9[3:6]
                    gauge = abs(np.dot(dt, self.env.n_B)) * 1000
                    best_result = {
                        'theta_opt': to,
                        'converged': True,
                        'R_error': Re,
                        't_error': te,
                        't_inplane_error': tipe,
                        'gauge_error': gauge,
                    }

        return best_result

    def _perturb_init(self, n_dof=12):
        """扰动 GT 生成初始值"""
        scale = self.solver_cfg.get('perturbation_scale', 0.3)
        rng = self.env.rng

        if n_dof == 12:
            ti = self.env.theta_gt.copy()
            ti[0:3] += rng.normal(0, scale, 3)
            ti[3:6] += rng.normal(0, scale * 0.02, 3)
            ti[6:9] += rng.normal(0, scale * 0.5, 3)
            ti[9:12] += rng.normal(0, scale * 0.01, 3)
            return ti
        else:
            ti = self.env.theta_gt_9.copy()
            ti[0:3] += rng.normal(0, scale, 3)
            ti[3:6] += rng.normal(0, scale * 0.02, 3)
            ti[6:9] += rng.normal(0, scale * 0.5, 3)
            return ti

    def svd_analysis(self, poses, measurements):
        """SVD 分析: 条件数, 奇异值, 零空间"""
        J, r, mask = compute_jacobian_numerical(
            self.env.theta_gt, poses, measurements, self.env.alpha)
        J_valid = J[mask, :]
        U, s, Vt = np.linalg.svd(J_valid, full_matrices=False)

        cond_num = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
        gauge_exist = s[-1] / s[0] < 1e-6

        return {
            'cond_num': cond_num,
            'singular_values': s,
            'sigma_min_ratio': s[-1] / s[0],
            'gauge_exist': gauge_exist,
            'Vt_last': Vt[-1, :],
            'n_residuals': len(r),
            'n_valid': np.sum(mask),
        }


# ============================================================================
# 4. ResultReporter — 结果汇总
# ============================================================================

class ResultReporter:
    """格式化输出标定结果"""

    @staticmethod
    def report_single(result, svd_info, pose_stats, method, verbose=True):
        """单次标定报告"""
        if not verbose:
            return

        print(f"\n{'='*60}")
        print(f"  方法: {method}")
        print(f"  {'='*60}")

        if pose_stats:
            print(f"  位姿数: {pose_stats['n_poses']}")
            z = pose_stats['z_dev']
            print(f"  z_S 偏离 -n_B: min={z['min']:.1f}°  max={z['max']:.1f}°  "
                  f"mean={z['mean']:.1f}°  std={z['std']:.1f}°")

        if svd_info:
            print(f"  cond(J): {svd_info['cond_num']:.2e}")
            print(f"  σ_min/σ_max: {svd_info['sigma_min_ratio']:.2e}")
            print(f"  Gauge: {'✗ 存在' if svd_info['gauge_exist'] else '✓ 消失'}")
            print(f"  残差: {svd_info['n_valid']} valid / {svd_info['n_residuals']} total")

        if result:
            print(f"  收敛: {'✓' if result.get('converged') else '✗'}")
            print(f"  R_err: {result['R_error']:.6f}°")
            print(f"  t_err: {result['t_error']:.6f}mm")
            extra = []
            if 't_inplane_error' in result:
                extra.append(f"t_inp={result['t_inplane_error']:.4f}mm")
            if 'gauge_error' in result:
                extra.append(f"t_gauge={result['gauge_error']:.4f}mm")
            if extra:
                print(f"  ({', '.join(extra)})")

    @staticmethod
    def report_mc(stats_list, method):
        """Monte Carlo 汇总报告"""
        if not stats_list:
            return None

        R_arr = np.array([s['R_error'] for s in stats_list if s])
        t_arr = np.array([s['t_error'] for s in stats_list if s])
        conv = sum(1 for s in stats_list if s and s.get('converged', True))

        summary = {
            'n_trials': len(stats_list),
            'converged': conv,
            'R_median': np.median(R_arr) if len(R_arr) else np.nan,
            'R_mean': np.mean(R_arr) if len(R_arr) else np.nan,
            'R_max': np.max(R_arr) if len(R_arr) else np.nan,
            't_median': np.median(t_arr) if len(t_arr) else np.nan,
            'R_lt_01': np.sum(R_arr < 0.1) if len(R_arr) else 0,
            'R_lt_005': np.sum(R_arr < 0.05) if len(R_arr) else 0,
            'R_zero': np.sum(R_arr < 0.001) if len(R_arr) else 0,
        }

        print(f"\n{'='*60}")
        print(f"  Monte Carlo 汇总 ({method})")
        print(f"  {'='*60}")
        print(f"  Trials: {summary['n_trials']}, 收敛: {summary['converged']}")
        print(f"  R: median={summary['R_median']:.4f}°  mean={summary['R_mean']:.4f}°  "
              f"max={summary['R_max']:.4f}°")
        print(f"  t: median={summary['t_median']:.4f}mm")
        print(f"  R<0.1°: {summary['R_lt_01']}/{summary['n_trials']}  "
              f"R<0.05°: {summary['R_lt_005']}/{summary['n_trials']}  "
              f"R≈0: {summary['R_zero']}/{summary['n_trials']}")

        return summary


# ============================================================================
# 5. CalibrationFramework — 顶层协调器
# ============================================================================

class CalibrationFramework:
    """统一标定框架顶层"""

    # 方法 → 默认采集模式映射
    METHOD_POSE_MODE = {
        'corner_12dof': 'auto_corner',    # 强制垂直 (对比用)
        'plane_edge_9dof': 'auto_nbv',    # NBV 候选
        'tilted_corner': 'auto_grid',     # 倾斜网格
    }

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.method = self.config.get('method', 'tilted_corner')
        self.verbose = self.config.get('output', {}).get('verbose', True)

        # 方法未显式覆盖采集模式时，使用默认映射
        if 'pose_collection' not in self.config:
            self.config['pose_collection'] = {}
        if 'mode' not in self.config['pose_collection']:
            default_mode = self.METHOD_POSE_MODE.get(self.method, 'auto_grid')
            self.config['pose_collection']['mode'] = default_mode

    def run_single(self, seed=42):
        """运行单次标定"""
        env = CalibrationEnvironment(self.config, seed=seed)
        collector = PoseCollector(env, self.config)
        solver = SolverDispatcher(env, self.config)
        reporter = ResultReporter()

        # 1. 采集位姿
        poses = collector.collect()
        if len(poses) == 0:
            print("❌ 未找到可用位姿")
            return None

        # 2. 施加机器人噪声
        poses = env.apply_robot_noise(poses)

        # 3. 生成测量
        measurements = env.generate_measurements(poses)

        # 4. SVD 分析 (零噪声 Jacobian)
        svd_info = None
        if self.config.get('output', {}).get('show_svd', True):
            meas_zero = env.generate_measurements(poses, noise_sigma_mm=0)
            svd_info = solver.svd_analysis(poses, meas_zero)

        # 5. 求解
        result = solver.solve(poses, measurements, self.method)

        # 6. 位姿统计
        pose_stats = None
        if self.config.get('output', {}).get('show_pose_stats', True):
            z_devs = env.compute_z_deviation(poses)
            pose_stats = {
                'n_poses': len(poses),
                'z_dev': {
                    'min': float(np.min(z_devs)),
                    'max': float(np.max(z_devs)),
                    'mean': float(np.mean(z_devs)),
                    'std': float(np.std(z_devs)),
                }
            }

        # 7. 报告
        reporter.report_single(result, svd_info, pose_stats, self.method, self.verbose)

        return {
            'result': result,
            'svd_info': svd_info,
            'pose_stats': pose_stats,
            'poses': poses,
        }

    def run_mc(self, n_trials=30):
        """运行 Monte Carlo"""
        all_results = []
        methods = (['corner_12dof', 'plane_edge_9dof', 'tilted_corner']
                   if self.method == 'all' else [self.method])

        for method in methods:
            self.method = method
            # 切换采集模式
            default_mode = self.METHOD_POSE_MODE.get(method, 'auto_grid')
            self.config.setdefault('pose_collection', {})['mode'] = default_mode

            stats_list = []
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  方法: {method} ({n_trials} trials)")
                print(f"  {'='*60}")

            for trial in range(n_trials):
                seed = 42 + trial * 137
                out = self.run_single(seed=seed)
                if out and out['result']:
                    stats_list.append(out['result'])

            summary = ResultReporter.report_mc(stats_list, method)
            if summary:
                all_results.append({'method': method, 'summary': summary})

        return all_results


# ============================================================================
# 6. 倾斜位姿生成 (从 test_tilted_corner.py 提取)
# ============================================================================

def generate_tilted_corner_poses(C, n_B, u_B, v_B, R_he, t_he,
                                  plate_w=400, plate_h=500,
                                  pitch_range=(-30, 30), yaw_range=(-15, 15),
                                  n_poses=8, rng=None, return_candidates=False):
    """生成 z_S 倾斜但能看到角点两边的位姿"""
    if rng is None:
        rng = np.random.default_rng(42)

    w_m, h_m = plate_w / 1000.0, plate_h / 1000.0
    pitches = np.linspace(pitch_range[0], pitch_range[1], 8)
    yaws = np.linspace(yaw_range[0], yaw_range[1], 5)
    candidates = []
    MAX_CANDIDATES = 200

    for x_align in [u_B, v_B]:
        for pitch in pitches:
            for yaw in yaws:
                if len(candidates) >= MAX_CANDIDATES:
                    break
                R_i = _build_R_edge(pitch, yaw, x_align, n_B, u_B, v_B)
                R_BS = R_i @ R_he
                z_S = R_BS[:, 2]

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
                            if not sl['has_intersection'] or len(sl['scan_pts_S']) < 10:
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
        if return_candidates:
            return [], []
        return []

    candidates.sort(key=lambda c: c['z_dev'])

    if len(candidates) <= n_poses:
        selected = candidates
    else:
        selected_idx = [0]
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
        selected = [candidates[i] for i in sorted(selected_idx)]

    poses = [(c['R_i'], c['t_i']) for c in selected]
    if return_candidates:
        return poses, selected
    return poses
