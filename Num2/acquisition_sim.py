#!/usr/bin/env python3
"""
acquisition_sim.py — 角点->板中心直线运动采集仿真

采集方案:
  1. 机器人从板外侧开始（FOV在板外）
  2. 沿 n_B 方向匀速向板推进
  3. FOV先碰到角点（单断点）
  4. 继续推进 → FOV同时穿过两边（双断点）
  5. 断点间距随推进增大
  6. 记录每一帧的传感器数据

输出: 完整的时间序列数据，用于后续手眼标定
"""

import numpy as np
from corner_scene import generate_corner_plane, generate_corner_measurements
from reproduction_scene import compute_fov_plate_scanline, generate_hand_eye_gt


def generate_linear_trajectory(scene, R_he, t_he, n_steps=40,
                                start_uv=(0.02, 0.02), end_uv=(0.40, 0.35)):
    """从板角向中心运动，每帧用 _build_R_edge 搜索确保两边可见。

    策略: 对每个位置，搜索 pitch/yaw/x_align 组合使FOV同时穿过两边
    """
    C = scene['C']; n_B = scene['n_B']; u_B = scene['u_B']; v_B = scene['v_B']
    w_m = scene['w']; h_m = scene['h']

    from reproduction_scene import compute_fov_plate_scanline
    from nbv_edge_plane import _build_R_edge

    u_vals = np.linspace(start_uv[0], end_uv[0], n_steps * 2)
    v_vals = np.linspace(start_uv[1], end_uv[1], n_steps * 2)

    poses = []; z_devs = []
    pitches = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    yaws = [-15, -10, -5, 0, 5, 10, 15]
    standoffs = [0.40, 0.55, 0.70, 0.90]

    for k in range(len(u_vals)):
        target = C + u_vals[k] * w_m * u_B + v_vals[k] * h_m * v_B
        found = False
        for pitch in pitches:
            for yaw in yaws:
                for x_align in [u_B, v_B]:
                    R_i = _build_R_edge(pitch, yaw, x_align, n_B, u_B, v_B)
                    for so in standoffs:
                        sp = target + so * n_B
                        t_i = sp - R_i @ t_he
                        sl = compute_fov_plate_scanline(
                            R_i@R_he, t_i+R_i@t_he, C, n_B, u_B, v_B, w_m, h_m)
                        if sl['has_intersection'] and len(sl['scan_pts_S']) > 10:
                            eps = [e for e, _ in sl['endpoints_S']]
                            if 'e1' in eps and 'e2' in eps:
                                z_dev = np.rad2deg(np.arccos(
                                    np.clip(np.dot(R_i@R_he[:,2], -n_B), -1, 1)))
                                poses.append((R_i, t_i))
                                z_devs.append(z_dev)
                                found = True
                                break
                        if found: break
                    if found: break
                if found: break
            if found: break
        if found and len(poses) >= n_steps:
            break

    return poses, {'z_devs': z_devs, 'n_frames': len(poses)}


def collect_frames(scene, poses, n_plane_pts=10):
    """对每个位姿计算 FOV 扫描线和断点。

    Returns:
        frames: [Frame[0], ..., Frame[n-1]]
        Frame = {
            'has_intersection': bool,
            'scan_pts_S': (N, 3) 或 empty,
            'endpoints_S': [(etype, 3), ...] 或 [],
            'n_endpoints': int,
            'endpoint_dists': [d1, d2, ...] or [],
            'valid': bool,  # 是否有2个断点
        }
    """
    C = scene['C']; n_B = scene['n_B']; u_B = scene['u_B']; v_B = scene['v_B']
    pw = scene['plate_w'] / 1000.0
    ph = scene['plate_h'] / 1000.0

    frames = []
    for k, (R_i, t_i) in enumerate(poses):
        R_BS = R_i @ scene['R_he']
        t_BS = t_i + R_i @ scene['t_he']

        sl = compute_fov_plate_scanline(
            R_BS, t_BS, C, n_B, u_B, v_B, pw, ph)

        frame = {
            'has_intersection': sl['has_intersection'],
            'scan_pts_S': sl['scan_pts_S'],
            'endpoints_S': sl['endpoints_S'],
            'n_endpoints': len(sl['endpoints_S']),
            'valid': sl['has_intersection'] and len(sl['endpoints_S']) >= 2,
        }

        # 端点间距
        if len(sl['endpoints_S']) >= 2:
            pts = [pt for _, pt in sl['endpoints_S']]
            dists = []
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    dists.append(np.linalg.norm(pts[i] - pts[j]))
            frame['endpoint_dists'] = dists
            frame['physical_dist'] = max(dists) if dists else 0.0
        else:
            frame['endpoint_dists'] = []
            frame['physical_dist'] = 0.0

        frames.append(frame)

    return frames


def print_acquisition_summary(scene, poses, frames):
    """打印采集过程摘要"""
    C = scene['C']

    print(f"  起始传感器位置: {poses[0][0]@scene['t_he'] + poses[0][1]}")
    print(f"  结束传感器位置: {poses[-1][0]@scene['t_he'] + poses[-1][1]}")
    print(f"  角点 C: {C}")
    print(f"  总帧数: {len(poses)}")

    # 统计
    n_hit = sum(1 for f in frames if f['has_intersection'])
    n_1ep = sum(1 for f in frames if f['n_endpoints'] == 1)
    n_2ep = sum(1 for f in frames if f['n_endpoints'] >= 2)
    print(f"\n  击中板: {n_hit}/{len(frames)}")
    print(f"  单断点(角点): {n_1ep}")
    print(f"  双断点(两边): {n_2ep}")

    if n_2ep > 0:
        first_2ep = next(i for i, f in enumerate(frames) if f['n_endpoints'] >= 2)
        last_2ep = max(i for i, f in enumerate(frames) if f['n_endpoints'] >= 2)
        dist_range = [frames[k]['physical_dist'] for k in range(first_2ep, last_2ep+1)
                      if frames[k]['valid']]
        if dist_range:
            print(f"\n  双断点帧范围: {first_2ep} ~ {last_2ep}")
            print(f"  断点间距范围: {min(dist_range)*1000:.3f}~{max(dist_range)*1000:.3f}mm")

    return {
        'n_hit': n_hit, 'n_1ep': n_1ep, 'n_2ep': n_2ep,
    }


# ============================================================================
# 可视化
# ============================================================================

def animate_acquisition(scene, poses, frames):
    """动画展示采集过程。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    C = scene['C']; n_B = scene['n_B']; u_B = scene['u_B']; v_B = scene['v_B']
    pw = scene['plate_w'] / 1000.0; ph = scene['plate_h'] / 1000.0
    R_he = scene['R_he']; t_he = scene['t_he']

    d_1 = u_B
    d_2 = v_B  # α=90°

    corners = np.array([C, C + pw * d_1, C + pw * d_1 + ph * d_2, C + ph * d_2])
    tri1 = np.array([corners[0], corners[1], corners[2]])
    tri2 = np.array([corners[0], corners[2], corners[3]])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    plt.ion()
    for k in range(len(poses)):
        ax.cla()

        # 平板
        ax.add_collection3d(Poly3DCollection([tri1, tri2],
                            alpha=0.25, facecolor='steelblue',
                            edgecolor='navy', linewidth=1.5))

        # 角点
        ax.scatter(*C, color='gold', s=200, marker='*', zorder=10,
                   edgecolors='darkorange', linewidths=1.5)

        # 边方向
        elen = 0.12
        ax.quiver(C[0], C[1], C[2], d_1[0]*elen, d_1[1]*elen, d_1[2]*elen,
                  color='blue', linewidth=3, arrow_length_ratio=0.2)
        ax.quiver(C[0], C[1], C[2], d_2[0]*elen, d_2[1]*elen, d_2[2]*elen,
                  color='red', linewidth=3, arrow_length_ratio=0.2)

        # 基座原点
        ax.scatter(0, 0, 0, color='black', s=100, marker='s', zorder=5)

        # 传感器位置和FOV
        R_i, t_i = poses[k]
        s_pos = t_i + R_i @ t_he
        R_BS = R_i @ R_he

        # 传感器盒子
        box_half = np.array([0.02, 0.03, 0.05])
        box_local = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                              [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * box_half
        box_world = R_BS @ box_local.T + s_pos.reshape(3, 1)
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot3D(box_world[0,e], box_world[1,e], box_world[2,e],
                      color='green', linewidth=1.5)

        # FOV 三角
        fov_deg = 15
        fov_rad = np.deg2rad(fov_deg)
        zs, range_max = s_pos[2] + 0.15, 0.7
        x_fov = range_max * np.tan(fov_rad)
        fov_tip = s_pos
        fov_b1 = R_BS @ np.array([-x_fov, 0, range_max]) + s_pos
        fov_b2 = R_BS @ np.array([x_fov, 0, range_max]) + s_pos
        ax.plot3D([fov_tip[0], fov_b1[0]], [fov_tip[1], fov_b1[1]],
                  [fov_tip[2], fov_b1[2]], 'g--', alpha=0.5)
        ax.plot3D([fov_tip[0], fov_b2[0]], [fov_tip[1], fov_b2[1]],
                  [fov_tip[2], fov_b2[2]], 'g--', alpha=0.5)
        ax.plot3D([fov_b1[0], fov_b2[0]], [fov_b1[1], fov_b2[1]],
                  [fov_b1[2], fov_b2[2]], 'g-', alpha=0.5)

        # 扫描线和断点
        f = frames[k]
        if f['has_intersection'] and len(f['scan_pts_S']) > 0:
            scan_B = np.array([R_BS @ p + s_pos for p in f['scan_pts_S']])
            ax.scatter(scan_B[:,0], scan_B[:,1], scan_B[:,2],
                      color='cyan', s=2, alpha=0.7)

        for etype, pt_S in f['endpoints_S']:
            pt_B = R_BS @ pt_S + s_pos
            color = 'blue' if etype == 'e1' else 'red'
            ax.scatter(*pt_B, color=color, s=80, marker='o',
                      edgecolors='black', linewidths=1)

        # 视图设置
        all_pts = np.vstack([s_pos, C, corners])
        center = all_pts.mean(axis=0)
        span = max(0.5, all_pts[:, 0].ptp(), all_pts[:, 1].ptp())
        ax.set_xlim(center[0]-span, center[0]+span)
        ax.set_ylim(center[1]-span, center[1]+span)
        ax.set_zlim(min(0, center[2]-span/2), center[2]+span/2)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f"帧 {k}/{len(poses)}  "
                     f"断点数={f['n_endpoints']}  "
                     f"间距={f['physical_dist']*1000:.1f}mm")
        ax.view_init(elev=30, azim=-60)

        plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  角点->板中心直线运动采集仿真")
    print("=" * 60)

    # 生成场景
    rng = np.random.default_rng(42)
    C, n_B, u_B, v_B, _, _, w_m, h_m = generate_corner_plane(rng)
    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    scene = {
        'R_he': R_he, 't_he': t_he,
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'd_1': u_B, 'd_2': v_B, 'alpha': np.pi/2,
        'plate_w': 400, 'plate_h': 500,
        'w': w_m, 'h': h_m,
    }

    from corner_scene import so3_log
    print(f"\n  场景:")
    print(f"  角点 C: {C}")
    print(f"  n_B: {n_B}")
    print(f"  板尺寸: 400×500mm")

    # 生成轨迹
    from corner_scene import so3_log
    poses, info = generate_linear_trajectory(scene, R_he, t_he,
                                              n_steps=40)

    # 采集帧
    frames = collect_frames(scene, poses)

    # 打印摘要
    stats = print_acquisition_summary(scene, poses, frames)

    # 演示: 打印到第5帧看看
    print(f"\n  每帧摘要 (前20帧):")
    print(f"  {'帧':>4} {'击板':>5} {'断点数':>7} {'间距(mm)':>10}")
    for k in range(min(20, len(frames))):
        f = frames[k]
        dist_str = f"{f['physical_dist']*1000:.2f}" if f['physical_dist'] > 0 else "-"
        print(f"  {k:4d} {'✓' if f['has_intersection'] else '✗':>5} "
              f"{f['n_endpoints']:7d} {dist_str:>10}")

    # 可视化 (注释掉，运行时代码打开)
    # animate_acquisition(scene, poses, frames)
