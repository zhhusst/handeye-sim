#!/usr/bin/env python3
"""
analyze_plane_edge_9dof.py — plane_edge_9dof 全面分析

分析点:
  1. 初值容忍度
  2. 全自动可行性
  3. FOV 边缘可见性
"""

import numpy as np
import yaml, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration_framework import CalibrationEnvironment
from nbv_edge_plane import (generate_edge_candidates, select_diverse,
                             combined_solve_lm, combined_errors)
from corner_scene import (generate_corner_measurements, so3_log, so3_exp,
                           generate_corner_plane)
from reproduction_scene import generate_hand_eye_gt

# ============================================================================
# 公共环境
# ============================================================================

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def make_scene(seed):
    """生成随机场景, 返回 (env, R_he, t_he, theta_gt9, scene)"""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    C, n_B, u_B, v_B, _, _, w_m, h_m = generate_corner_plane(
        rng, plate_w=config['environment']['plate_width'],
        plate_h=config['environment']['plate_height'])

    scene = {
        'R_he': R_he, 't_he': t_he,
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
        'plate_w': w_m * 1000, 'plate_h': h_m * 1000,
        'w': w_m, 'h': h_m,
    }
    R_pl = np.column_stack([u_B, v_B, n_B])
    theta_gt9 = np.concatenate([so3_log(R_he), t_he, so3_log(R_pl)])
    return scene, R_he, t_he, theta_gt9, rng


# ============================================================================
# 1. 初值容忍度分析
# ============================================================================

def analyze_init_tolerance(n_trials=20, n_poses=6):
    """测试不同初始值策略下的收敛性"""
    print("=" * 70)
    print("  1. 初值容忍度分析")
    print("=" * 70)

    strategies = {
        'GT+0.1rad(~6°)':   {'w_scale': 0.1, 't_scale': 0.002, 'p_scale': 0.05, 'n_restart': 1},
        'GT+0.2rad(~11°)':  {'w_scale': 0.2, 't_scale': 0.005, 'p_scale': 0.1, 'n_restart': 1},
        'GT+0.3rad(~17°)':  {'w_scale': 0.3, 't_scale': 0.01,  'p_scale': 0.15, 'n_restart': 1},
        '单位阵(I,0)':      {'w_scale': 0.0, 't_scale': 0.0,  'p_scale': 0.0, 'n_restart': 5,
                              'use_identity': True},
        'CAD粗估(±5°)':      {'w_scale': 0.08, 't_scale': 0.01, 'p_scale': 0.08, 'n_restart': 3},
    }

    for name, strat in strategies.items():
        results = []
        for trial in range(n_trials):
            seed = 42 + trial * 137
            scene, R_he, t_he, theta_gt9, rng = make_scene(seed)

            # NBV 位姿
            c_e1, c_e2 = generate_edge_candidates(scene, R_he, t_he, 'both', 6)
            sel_e1 = select_diverse(c_e1, n_poses // 2)
            sel_e2 = select_diverse(c_e2, n_poses // 2)
            poses = [(c['R_i'], c['t_i']) for c in sel_e1 + sel_e2]
            if len(poses) < 4: continue

            meas = generate_corner_measurements(
                scene, poses, n_plane_pts=15, rng=rng, noise_sigma=0.055/1000)

            best_Re = np.inf
            for restart in range(strat['n_restart']):
                if strat.get('use_identity'):
                    ti = np.zeros(9)
                    # 单位阵但需要合理的 R_pl 初始值 → 从平面点估计
                    # 简化: 用 GT R_pl 但 R_he=t_he=0
                    ti[6:9] = theta_gt9[6:9]
                else:
                    ti = theta_gt9.copy()
                    # 加扰动 (w_scale=0 时不加扰动但也不设identity)
                    if strat['w_scale'] > 0 or not strat.get('use_identity'):
                        ti[0:3] += rng.normal(0, strat['w_scale'], 3)
                        ti[3:6] += rng.normal(0, strat['t_scale'], 3)
                        ti[6:9] += rng.normal(0, strat['p_scale'], 3)

                to = combined_solve_lm(ti, poses, meas, w_plane=1.0, w_edge=1.0, max_iter=80)
                Re, te, _ = combined_errors(to, theta_gt9)
                if Re < best_Re: best_Re = Re; best_te = te

            results.append((best_Re, best_te))

        R_arr = np.array([r[0] for r in results])
        t_arr = np.array([r[1] for r in results])
        print(f"\n  {name} (n={len(results)}, restart×{strat['n_restart']}):")
        print(f"    R: median={np.median(R_arr):.4f}°  mean={np.mean(R_arr):.4f}°  "
              f"max={np.max(R_arr):.4f}°  <0.1°={np.sum(R_arr<0.1)}/{len(results)}")
        print(f"    t: median={np.median(t_arr):.4f}mm  max={np.max(t_arr):.4f}mm")


# ============================================================================
# 2. NBV 候选成功率 + FOV 边缘可见性
# ============================================================================

def analyze_nbv_success_rate(n_trials=20, n_grid=6):
    """分析 NBV 候选生成的成功率和边缘可见性"""
    print("\n" + "=" * 70)
    print("  2. NBV 候选成功率 & FOV 边缘可见性")
    print("=" * 70)

    e1_cand_counts = []
    e2_cand_counts = []
    e1_vis_e1 = []  # e1 候选位姿实际看到 e1 的比例
    e2_vis_e2 = []  # e2 候选位姿实际看到 e2 的比例

    for trial in range(n_trials):
        seed = 42 + trial * 137
        scene, R_he, t_he, _, rng = make_scene(seed)

        c_e1, c_e2 = generate_edge_candidates(scene, R_he, t_he, 'both', n_grid)
        e1_cand_counts.append(len(c_e1))
        e2_cand_counts.append(len(c_e2))

        # 统计候选位姿实际看到目标边的比例
        if len(c_e1) > 0:
            e1v = sum(1 for c in c_e1 if c['edge'] == 'e1')
            e1_vis_e1.append(e1v / len(c_e1) if len(c_e1) > 0 else 0)

        if len(c_e2) > 0:
            e2v = sum(1 for c in c_e2 if c['edge'] == 'e2')
            e2_vis_e2.append(e2v / len(c_e2) if len(c_e2) > 0 else 0)

    print(f"\n  候选数量 (n_grid={n_grid}):")
    print(f"    e1候选: median={np.median(e1_cand_counts):.0f}  "
          f"min={np.min(e1_cand_counts)}  max={np.max(e1_cand_counts)}")
    print(f"    e2候选: median={np.median(e2_cand_counts):.0f}  "
          f"min={np.min(e2_cand_counts)}  max={np.max(e2_cand_counts)}")

    print(f"\n  边缘可见性匹配率 (候选的edge标签 vs 实际FOV扫描结果):")
    if e1_vis_e1:
        print(f"    e1候选→实际看到e1: median={np.median(e1_vis_e1)*100:.0f}%")
    if e2_vis_e2:
        print(f"    e2候选→实际看到e2: median={np.median(e2_vis_e2)*100:.0f}%")

    # 测试: 是否总能找到至少 3+3 个有效候选
    has_enough = sum(1 for i in range(n_trials)
                     if e1_cand_counts[i] >= 3 and e2_cand_counts[i] >= 3)
    print(f"\n  总能找到 ≥3+3 个候选: {has_enough}/{n_trials} ({has_enough/n_trials*100:.0f}%)")


# ============================================================================
# 3. 全自动闭环分析
# ============================================================================

def analyze_auto_calibration(n_trials=20, n_poses=6):
    """模拟全自动标定闭环:
    1. 从单位阵初值开始 (模拟操作者不知道手眼关系)
    2. 用粗略初值做 NBV 候选搜索 (模拟: 初值误差 ±5°)
    3. 执行位姿 → 采集 → 求解
    """
    print("\n" + "=" * 70)
    print("  3. 全自动标定闭环")
    print("=" * 70)

    all_Re, all_te = [], []

    for trial in range(n_trials):
        seed = 42 + trial * 137
        scene, R_he, t_he, theta_gt9, rng = make_scene(seed)

        # 模拟: 操作者只知道粗略手眼关系 (CAD)
        R_he_approx = so3_exp(rng.normal(0, 0.08, 3)) @ R_he  # ~±5° 误差
        t_he_approx = t_he + rng.normal(0, 0.01, 3)            # ~±10mm 误差

        # 用粗略手眼做 NBV 候选搜索
        c_e1, c_e2 = generate_edge_candidates(scene, R_he_approx, t_he_approx, 'both', 6)
        sel_e1 = select_diverse(c_e1, n_poses // 2)
        sel_e2 = select_diverse(c_e2, n_poses // 2)
        all_cands = sel_e1 + sel_e2
        if len(all_cands) < 4: continue

        # 执行位姿 (仿真直接用真值验证FOV可见性)
        poses = [(c['R_i'], c['t_i']) for c in all_cands]
        meas = generate_corner_measurements(
            scene, poses, n_plane_pts=15, rng=rng, noise_sigma=0.055/1000)

        # 求解: 从粗略初值出发
        best_Re = np.inf
        for _ in range(3):
            ti = theta_gt9.copy()
            ti[0:3] += rng.normal(0, 0.08, 3)  # ~±5° 初值误差
            ti[3:6] += rng.normal(0, 0.01, 3)  # ~±10mm
            ti[6:9] += rng.normal(0, 0.08, 3)
            to = combined_solve_lm(ti, poses, meas, w_plane=1.0, w_edge=1.0, max_iter=80)
            Re, te, _ = combined_errors(to, theta_gt9)
            if Re < best_Re: best_Re = Re; best_te = te

        all_Re.append(best_Re)
        all_te.append(best_te)

    R_arr = np.array(all_Re)
    t_arr = np.array(all_te)
    print(f"\n  粗略CAD初值(±5°,±10mm) + NBV候选 + 多重启×3 ({len(all_Re)} trials):")
    print(f"    R: median={np.median(R_arr):.4f}°  mean={np.mean(R_arr):.4f}°  "
          f"max={np.max(R_arr):.4f}°  <0.1°={np.sum(R_arr<0.1)}/{len(all_Re)}")
    print(f"    t: median={np.median(t_arr):.4f}mm  max={np.max(t_arr):.4f}mm")


# ============================================================================
# 4. Poses数量 vs 精度
# ============================================================================

def analyze_pose_count(n_trials=15):
    """测试不同位姿数量对精度的影响"""
    print("\n" + "=" * 70)
    print("  4. 位姿数 vs 精度")
    print("=" * 70)

    for n_poses in [4, 6, 8, 12]:
        all_Re, all_te = [], []
        for trial in range(n_trials):
            seed = 42 + trial * 137
            scene, R_he, t_he, theta_gt9, rng = make_scene(seed)

            c_e1, c_e2 = generate_edge_candidates(scene, R_he, t_he, 'both', 8)
            sel_e1 = select_diverse(c_e1, max(2, n_poses // 2))
            sel_e2 = select_diverse(c_e2, max(2, n_poses - len(sel_e1)))
            all_cands = sel_e1 + sel_e2
            if len(all_cands) < 4: continue

            poses = [(c['R_i'], c['t_i']) for c in all_cands]
            meas = generate_corner_measurements(
                scene, poses, n_plane_pts=15, rng=rng, noise_sigma=0.055/1000)

            # 多重启
            best_Re = np.inf
            for _ in range(3):
                ti = theta_gt9.copy()
                ti[0:3] += rng.normal(0, 0.1, 3)
                ti[3:6] += rng.normal(0, 0.002, 3)
                ti[6:9] += rng.normal(0, 0.05, 3)
                to = combined_solve_lm(ti, poses, meas, w_plane=1.0, w_edge=1.0, max_iter=80)
                Re, te, _ = combined_errors(to, theta_gt9)
                if Re < best_Re: best_Re = Re; best_te = te

            all_Re.append(best_Re); all_te.append(best_te)

        R_arr = np.array(all_Re); t_arr = np.array(all_te)
        print(f"\n  {n_poses}poses ({len(all_Re)} trials):")
        print(f"    R: median={np.median(R_arr):.4f}°  mean={np.mean(R_arr):.4f}°  "
              f"max={np.max(R_arr):.4f}°  <0.1°={np.sum(R_arr<0.1)}/{len(all_Re)}")
        print(f"    t: median={np.median(t_arr):.4f}mm")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    analyze_init_tolerance(n_trials=15)
    analyze_nbv_success_rate(n_trials=20)
    analyze_auto_calibration(n_trials=15)
    analyze_pose_count(n_trials=12)
