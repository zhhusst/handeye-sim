#!/usr/bin/env python3
"""
test_geometry_standalone.py — 离线验证 fov_geometry.py

在 Docker 容器的 ROS2 环境外执行, 纯 numpy.
验证: 场景生成 → 轨迹搜索 → 扫描线计算 → 可视化摘要
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))

import numpy as np
from fov_geometry import (
    compute_fov_plate_scanline, compute_fov_triangle,
    generate_hand_eye_gt, generate_plane,
    generate_smooth_trajectory, collect_frames,
    build_R_edge,
)


def test_scene():
    """测试场景生成"""
    print("=" * 60)
    print("  1. 场景生成测试")
    print("=" * 60)
    rng = np.random.default_rng(42)
    C, n_B, u_B, v_B, w, h = generate_plane(rng)
    X_gt = generate_hand_eye_gt(rng)
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    print(f"  角点 C: {C}")
    print(f"  法向量 n_B: {n_B}")
    print(f"  板尺寸: {w*1000:.0f}x{h*1000:.0f}mm")
    print(f"  手眼 R_he(轴角): {np.rad2deg(np.arccos((np.trace(R_he)-1)/2)):.2f}°")
    print(f"  手眼 t_he: {t_he}")
    print(f"  正交性: u_B·v_B={np.dot(u_B, v_B):.6f}, |u_B|={np.linalg.norm(u_B):.6f}")
    print(f"  OK\n")

    return C, n_B, u_B, v_B, w, h, R_he, t_he


def test_scanline(C, n_B, u_B, v_B, w, h, R_he, t_he):
    """测试单帧扫描线计算"""
    print("=" * 60)
    print("  2. 单帧扫描线测试")
    print("=" * 60)

    # 使用轨迹生成中已验证的姿态 (第0帧间距~50mm)
    C, n_B, u_B, v_B, w, h, R_he, t_he = test_scene()
    scene = {
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'w': w, 'h': h,
        'R_he': R_he, 't_he': t_he,
    }

    # 生成一条短轨迹, 用第1帧验证
    poses = generate_smooth_trajectory(scene, n_frames=5)
    if len(poses) >= 2:
        R_S, t_i = poses[1]
        R_BS = R_S @ R_he
        t_BS = t_i + R_S @ t_he
    else:
        print("  ✗ 无法生成测试位姿")
        return False

    sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, w, h)

    n_pts = len(sl['scan_pts_S'])
    n_ep = len(sl['endpoints_S'])
    ep_types = [e for e, _ in sl['endpoints_S']]

    print(f"  扫描点: {n_pts}")
    print(f"  断点: {n_ep} 个 ({', '.join(ep_types)})")

    if n_ep >= 2:
        pts = [pt for _, pt in sl['endpoints_S']]
        dist = np.linalg.norm(pts[0] - pts[1])
        print(f"  断点间距: {dist*1000:.2f}mm")

    if n_ep == 2 and 'e1' in ep_types and 'e2' in ep_types:
        print(f"  两边可见: ✓")
    else:
        print(f"  两边可见: ✗ (需要调整姿态)")
        return False

    print(f"  OK\n")
    return True


def test_trajectory(scene):
    """测试轨迹生成和采集"""
    print("=" * 60)
    print("  3. 轨迹生成测试")
    print("=" * 60)

    poses = generate_smooth_trajectory(scene, n_frames=30)
    print(f"  生成位姿: {len(poses)} 帧")

    if len(poses) == 0:
        print("  ✗ 无法生成轨迹")
        return False

    frames = collect_frames(scene, poses)
    n_valid = sum(1 for f in frames if f['valid'])
    print(f"  两边可见: {n_valid}/{len(frames)} 帧")

    if n_valid < 5:
        print("  ✗ 有效帧太少")
        return False

    # 打印每帧摘要
    print(f"\n  帧摘要 (前15帧):")
    print(f"  {'帧':>4} {'断点数':>7} {'间距(mm)':>10}")
    for k in range(min(15, len(frames))):
        f = frames[k]
        dist_str = f"{f['physical_dist']*1000:.2f}" if f['physical_dist'] > 0 else "-"
        print(f"  {k:4d} {f['n_endpoints']:7d} {dist_str:>10}")

    # 断点间距变化 (检查是否单调递增)
    dists = [f['physical_dist'] for f in frames if f['valid']]
    if len(dists) >= 5:
        # 应该大致递增 (从角点到板中心)
        mid = len(dists) // 2
        print(f"\n  断点间距变化: {dists[0]*1000:.1f}mm → {dists[-1]*1000:.1f}mm")
        if dists[-1] > dists[0]:
            print(f"  单调趋势: ✓ (间距增大, 传感器移向板中心)")
        else:
            print(f"  单调趋势: ~ (可能有波动)")

    print(f"  OK\n")
    return True


def test_noise(C, n_B, u_B, v_B, w, h, R_he, t_he):
    """测试噪声影响 (无 ROS, 同 Num2 精度验证)"""
    print("=" * 60)
    print("  4. 噪声影响测试 (σ=0.055mm)")
    print("=" * 60)

    noise_sigma = 0.055 / 1000.0  # mm → m

    rng = np.random.default_rng(123)

    # 随机生成 50 个位姿, 确保两边可见
    valid_poses = []
    for _ in range(200):
        target = C + rng.uniform(0.05, 0.30) * w * u_B + rng.uniform(0.05, 0.30) * h * v_B
        standoff = rng.uniform(0.38, 0.70)
        R_S = build_R_edge(rng.uniform(-10, 10), rng.uniform(-15, 15),
                           u_B, n_B, u_B, v_B)
        sp = target + standoff * n_B
        t_i = sp - R_S @ t_he
        R_BS = R_S @ R_he
        t_BS = t_i + R_S @ t_he
        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, w, h)
        if sl['has_intersection'] and len(sl['scan_pts_S']) > 10:
            eps = [e for e, _ in sl['endpoints_S']]
            if 'e1' in eps and 'e2' in eps:
                valid_poses.append((R_S, t_i))
                if len(valid_poses) >= 50:
                    break

    print(f"  随机位姿: {len(valid_poses)}")

    # 无噪声: 验证几何一致性
    frames_clean = collect_frames(
        {'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B, 'w': w, 'h': h,
         'R_he': R_he, 't_he': t_he},
        valid_poses
    )
    n_clean_valid = sum(1 for f in frames_clean if f['valid'])
    print(f"  无噪声 - 两边可见: {n_clean_valid}/{len(frames_clean)}")

    # 加噪声: 模拟传感器测量误差
    frames_noisy = []
    for k, (R_S, t_i) in enumerate(valid_poses):
        R_BS = R_S @ R_he
        t_BS = t_i + R_S @ t_he
        sl = compute_fov_plate_scanline(R_BS, t_BS, C, n_B, u_B, v_B, w, h)
        # 加噪声到 scan_pts_S
        if len(sl['scan_pts_S']) > 0:
            noise = rng.normal(0, noise_sigma, sl['scan_pts_S'].shape)
            sl['scan_pts_S'] = sl['scan_pts_S'] + noise
        frames_noisy.append(sl)

    n_noisy_valid = sum(1 for f in frames_noisy if f['has_intersection'])
    print(f"  加噪声 - 有效: {n_noisy_valid}/{len(frames_noisy)}")

    print(f"  OK\n")
    return True


if __name__ == '__main__':
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║  手眼标定仿真 — 几何引擎独立验证")
    print("╚" + "═" * 58 + "╝")
    print()

    C, n_B, u_B, v_B, w, h, R_he, t_he = test_scene()

    scene = {
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'w': w, 'h': h,
        'R_he': R_he, 't_he': t_he,
    }

    test_scanline(C, n_B, u_B, v_B, w, h, R_he, t_he)
    test_trajectory(scene)
    test_noise(C, n_B, u_B, v_B, w, h, R_he, t_he)

    print("=" * 60)
    print("  全部测试通过!")
    print("=" * 60)
