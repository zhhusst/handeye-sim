"""
golden_pose_search.py — 黄金位姿搜索
目标: 找到 FOV 三角同时跨两条边的传感器位姿

搜索空间: 传感器在平板 UV 坐标 + 高度 + roll 角
验证: compute_fov_plate_scanline() → 检查 e1 和 e2 同时有效
"""

import numpy as np
from reproduction_scene import compute_fov_plate_scanline, rodrigues


def search_golden_poses(C, n_B, u_B, v_B, pw=0.4, ph=0.5,
                         u_range=(0.02, 0.20), v_range=(0.02, 0.20),
                         h_range=(0.25, 0.55), n_roll=36,
                         n_u=10, n_v=10, n_h=10,
                         half_fov=15.0, min_range=0.27, max_range=0.82):
    """网格搜索: 传感器位姿 → 检查是否跨两边

    传感器坐标系定义:
      z_S: 传感器光轴 (指向平板 = -n_B)
      x_S: 激光线方向, 绕 z_S 旋转 roll 角
      y_S: z_S × x_S (激光平面法向量)

    Args:
        C, n_B, u_B, v_B: 平板参数
        pw, ph: 平板尺寸 (m)
        u_range, v_range: 传感器在平板 UV 坐标搜索范围 (m)
        h_range: 传感器高度搜索范围 (m, 沿 n_B)
        n_roll: roll 角离散点数
        n_u, n_v, n_h: 空间离散点数

    Returns:
        list of dict: 每个包含 {'R_BS','t_BS','u_s','v_s','h_s','roll_deg',
                               'scan_u_range','scan_v_range'}
    """
    golden = []

    u_vals = np.linspace(u_range[0], u_range[1], n_u)
    v_vals = np.linspace(v_range[0], v_range[1], n_v)
    h_vals = np.linspace(h_range[0], h_range[1], n_h)
    roll_vals = np.linspace(0, np.pi, n_roll)  # 0 to 180°

    total = n_u * n_v * n_h * n_roll
    count = 0

    for u_s in u_vals:
        for v_s in v_vals:
            for h_s in h_vals:
                # 传感器位置
                t_BS = C + u_s * u_B + v_s * v_B + h_s * n_B

                for roll in roll_vals:
                    count += 1

                    # 传感器朝向:
                    # z_S = -n_B (指向平板)
                    z_S = -n_B

                    # x_S 初始方向 = u_B, 绕 z_S 转 roll
                    x_S_0 = u_B
                    K_z = np.array([[0, -z_S[2], z_S[1]],
                                    [z_S[2], 0, -z_S[0]],
                                    [-z_S[1], z_S[0], 0]])
                    R_roll = np.eye(3) + np.sin(roll) * K_z + (1 - np.cos(roll)) * K_z @ K_z
                    x_S = R_roll @ x_S_0

                    # y_S = z_S × x_S
                    y_S = np.cross(z_S, x_S)
                    y_S /= np.linalg.norm(y_S)

                    R_BS = np.column_stack([x_S, y_S, z_S])

                    # 验证
                    sl = compute_fov_plate_scanline(
                        R_BS, t_BS, C, n_B, u_B, v_B, pw, ph,
                        half_fov_deg=half_fov, min_range=min_range, max_range=max_range)

                    has_e1 = any(et == 'e1' for et, _ in sl['endpoints_B'])
                    has_e2 = any(et == 'e2' for et, _ in sl['endpoints_B'])

                    if has_e1 and has_e2:
                        # 计算扫描线在平板上的 UV 范围
                        if sl['has_intersection']:
                            pts_B = sl['scan_pts_B']
                            uu = [np.dot(p - C, u_B) for p in pts_B]
                            vv = [np.dot(p - C, v_B) for p in pts_B]
                            scan_u_range = (min(uu) * 1000, max(uu) * 1000)
                            scan_v_range = (min(vv) * 1000, max(vv) * 1000)
                        else:
                            scan_u_range = scan_v_range = (0, 0)

                        golden.append({
                            'R_BS': R_BS,
                            't_BS': t_BS,
                            'u_s': u_s * 1000,  # mm
                            'v_s': v_s * 1000,
                            'h_s': h_s * 1000,
                            'roll_deg': np.rad2deg(roll),
                            'scan_u_range': scan_u_range,
                            'scan_v_range': scan_v_range,
                            'scan_length': np.linalg.norm(
                                sl['scan_pts_B'][-1] - sl['scan_pts_B'][0]) * 1000,
                        })

    print(f"搜索完成: {count} 配置, 找到 {len(golden)} 个黄金位姿")
    return golden


def select_diverse_poses(golden, n_select=8, min_scan_length=20.0):
    """从黄金位姿中挑选多样化的子集

    策略: 按传感器位置 (u_s, v_s, h_s) 聚类/分散 + 扫描线长度过滤
    """
    # 过滤太短的扫描线
    valid = [g for g in golden if g['scan_length'] >= min_scan_length]

    if len(valid) <= n_select:
        return valid

    # 按 h_s 分层, 每层均匀取样
    valid_sorted = sorted(valid, key=lambda g: g['h_s'])
    indices = np.linspace(0, len(valid_sorted) - 1, n_select, dtype=int)
    return [valid_sorted[i] for i in indices]


def golden_poses_to_scene_poses(golden_selected, R_he, t_he):
    """将传感器位姿转为标定场景用的 (R_i, t_i) 法兰位姿

    T_BS = T_BH @ T_HS  →  T_BH = T_BS @ inv(T_HS)
    其中 T_HS 是手眼 (Sensor in Hand)
    R_i = R_BS @ R_he.T
    t_i = t_BS - R_i @ t_he
    """
    X_gt_inv_R = R_he.T

    poses = []
    for g in golden_selected:
        R_i = g['R_BS'] @ X_gt_inv_R
        t_i = g['t_BS'] - R_i @ t_he
        poses.append((R_i, t_i))
    return poses


# ============================================================================
# 验证脚本
# ============================================================================

if __name__ == '__main__':
    from corner_scene import generate_corner_plane
    from reproduction_scene import generate_hand_eye_gt

    rng = np.random.default_rng(42)
    np.random.seed(42)  # 确保 generate_hand_eye_gt 确定性

    # 生成场景
    C, n_B, u_B, v_B, d_1, d_2, pw, ph = generate_corner_plane(rng)
    X_gt = generate_hand_eye_gt()
    R_he, t_he = X_gt[:3, :3], X_gt[:3, 3]

    print("=" * 60)
    print("黄金位姿搜索")
    print("=" * 60)
    print(f"C = ({C[0]*1000:.0f}, {C[1]*1000:.0f}, {C[2]*1000:.0f})mm")
    print(f"平板: {pw*1000:.0f}×{ph*1000:.0f}mm")
    print()

    # 搜索
    golden = search_golden_poses(C, n_B, u_B, v_B, pw, ph,
                                  u_range=(0.02, 0.20), v_range=(0.02, 0.20),
                                  h_range=(0.25, 0.55),
                                  n_u=10, n_v=10, n_h=10, n_roll=36)

    print(f"\n找到 {len(golden)} 个黄金位姿")
    if len(golden) > 0:
        print(f"\n前 5 个示例:")
        for i, g in enumerate(golden[:5]):
            print(f"  [{i}] u={g['u_s']:.0f}mm, v={g['v_s']:.0f}mm, "
                  f"h={g['h_s']:.0f}mm, roll={g['roll_deg']:.0f}°, "
                  f"scan={g['scan_length']:.0f}mm "
                  f"(u:[{g['scan_u_range'][0]:.0f},{g['scan_u_range'][1]:.0f}]mm, "
                  f"v:[{g['scan_v_range'][0]:.0f},{g['scan_v_range'][1]:.0f}]mm)")

    # 挑选多样化位姿
    selected = select_diverse_poses(golden, n_select=8)
    print(f"\n多样化选择: {len(selected)} 个位姿")

    # 转为法兰位姿
    flange_poses = golden_poses_to_scene_poses(selected, R_he, t_he)
    print(f"法兰位姿: {len(flange_poses)} 个")

    # 保存结果
    result = {
        'C': C, 'n_B': n_B, 'u_B': u_B, 'v_B': v_B,
        'R_he': R_he, 't_he': t_he,
        'plate_w': pw, 'plate_h': ph,
        'golden_all': golden,
        'golden_selected': selected,
        'flange_poses': flange_poses,
    }
    import pickle
    with open('golden_poses.pkl', 'wb') as f:
        pickle.dump(result, f)

    print("\n结果保存到 golden_poses.pkl")
