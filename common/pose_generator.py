#!/usr/bin/env python3
"""
pose_generator.py — Num2 auto_grid 候选位姿生成（适配 Sim 仿真）

核心: pitch/yaw 网格搜索 → FOV 扫描线验证两边可见 → 贪心选多样化位姿。
源自 Num2/calibration_framework.py 的 generate_tilted_corner_poses()。

用法:
  from pose_generator import generate_tilted_poses

  cands, sel = generate_tilted_poses(
      C=np.array([0.7, 0., 0.25]), n_B=np.array([0.,0.,1.]),
      u_B=np.array([1.,0.,0.]), v_B=np.array([0.,1.,0.]),
      R_he_nom=R_he, t_he_nom=t_he,
      plate_w=400, plate_h=500,
      fov_corners_S=fov_corners, n_poses=8)
"""

import numpy as np

# Try both import paths (inside Docker vs outside)
try:
    from fov_geometry import so3_exp, compute_fov_plate_scanline
except ImportError:
    import sys, os
    sys.path.insert(0, '/workspace/common')
    from fov_geometry import so3_exp, compute_fov_plate_scanline


def _build_R_edge(pitch_deg, yaw_deg, x_align, n_B, u_B, v_B):
    """构建传感器朝向: x_S=x_align, z_S 由 pitch/yaw 倾斜偏离 -n_B"""
    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    Ku = np.array([[0,-u_B[2],u_B[1]],[u_B[2],0,-u_B[0]],[-u_B[1],u_B[0],0]])
    Kv = np.array([[0,-v_B[2],v_B[1]],[v_B[2],0,-v_B[0]],[-v_B[1],v_B[0],0]])
    Rp = np.eye(3) + np.sin(p)*Ku + (1-np.cos(p))*Ku@Ku
    Ry = np.eye(3) + np.sin(y)*Kv + (1-np.cos(y))*Kv@Kv
    z_S = Ry @ Rp @ (-n_B)
    xd = x_align
    yd = np.cross(z_S, xd)
    nrm = np.linalg.norm(yd)
    if nrm < 1e-8:
        yd = v_B
    else:
        yd /= nrm
    xd = np.cross(yd, z_S)
    return np.column_stack([xd, yd, z_S])


def generate_tilted_poses(C, n_B, u_B, v_B,
                           R_he_nom, t_he_nom,
                           plate_w=400, plate_h=500,
                           fov_corners_S=None,
                           pitch_range=(-30, 30), yaw_range=(-15, 15),
                           n_pitch=8, n_yaw=5,
                           n_poses=8, seed=None,
                           return_candidates=False):
    """Num2 auto_grid 候选位姿生成

    Args:
        C, n_B, u_B, v_B: 平板几何（基座标系）
        R_he_nom, t_he_nom: 名义手眼
        plate_w, plate_h: 板尺寸 [mm]
        fov_corners_S: FOV 角点在传感器系的位置（4×3）
        pitch_range, yaw_range: 搜索范围 [deg]
        n_pitch, n_yaw: 网格分辨率
        n_poses: 输出位姿数
        seed: 随机种子
        return_candidates: 是否返回候选列表

    Returns:
        poses: [(R_i, t_i), ...] 基座标系下机器人法兰位姿
        (若 return_candidates: (poses, selected_dicts))
    """
    rng = np.random.default_rng(seed if seed is not None else 42)

    w_m, h_m = plate_w / 1000.0, plate_h / 1000.0
    pitches = np.linspace(pitch_range[0], pitch_range[1], n_pitch)
    yaws = np.linspace(yaw_range[0], yaw_range[1], n_yaw)
    candidates = []
    MAX_CANDIDATES = 200

    for x_align in [u_B, v_B]:
        for pitch in pitches:
            for yaw in yaws:
                if len(candidates) >= MAX_CANDIDATES:
                    break
                R_i = _build_R_edge(pitch, yaw, x_align, n_B, u_B, v_B)
                R_BS = R_i @ R_he_nom
                z_S = R_BS[:, 2]

                # 位置搜索: 板面内偏移 + standoff
                for u_off in np.linspace(0.01, 0.10, 3):
                    for v_off in np.linspace(0.01, 0.10, 3):
                        target = C + u_off * w_m * u_B + v_off * h_m * v_B
                        for standoff_h in [1.5, 2.2, 3.0]:
                            if len(candidates) >= MAX_CANDIDATES:
                                break
                            standoff_m = 0.35 * standoff_h
                            t_i = target + standoff_m * n_B - R_i @ t_he_nom
                            t_BS = t_i + R_i @ t_he_nom

                            sl = compute_fov_plate_scanline(
                                R_BS, t_BS, C, n_B, u_B, v_B, w_m, h_m,
                                fov_corners_S=fov_corners_S)

                            if not sl['has_intersection'] or len(sl.get('scan_pts_S', [])) < 10:
                                continue

                            eps = [e for e, _ in sl.get('endpoints_S', [])]
                            if 'e1' in eps and 'e2' in eps:
                                z_dev = np.rad2deg(np.arccos(
                                    np.clip(abs(np.dot(z_S, n_B)), 0, 1)))
                                candidates.append({
                                    'R_i': R_i, 't_i': t_i,
                                    'pitch': pitch, 'yaw': yaw,
                                    'z_dev': z_dev,
                                    'n_pts': len(sl['scan_pts_S']),
                                })
                                break  # 跳出 standoff 循环
                        else:
                            continue  # standoff 都没找到 → 换一个 (u_off,v_off)
                        break  # 跳出 v_off 循环
                    else:
                        continue
                    break  # 跳出 u_off 循环

    if len(candidates) == 0:
        if return_candidates:
            return [], []
        return []

    # 贪心选择: 按 z_dev 排序, 逐步选角度最分散的
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


# ── 便捷测试 ──
if __name__ == '__main__':
    # 用 Sim 场景的默认参数测试
    C = np.array([0.7, 0.0, 0.25])
    n_B = np.array([0., 0., 1.])
    u_B = np.array([1., 0., 0.])
    v_B = np.array([0., 1., 0.])

    # 名义手眼 (≈ GT with small perturbation)
    R_he_nom = np.array([[ 0.9996,  0.0263,  0.0028],
                         [-0.0263,  0.9996, -0.0085],
                         [-0.0030,  0.0084,  0.9999]])
    t_he_nom = np.array([-0.011, -0.004, 0.359])

    poses, cands = generate_tilted_poses(
        C, n_B, u_B, v_B,
        R_he_nom, t_he_nom,
        plate_w=400, plate_h=500,
        n_poses=8, seed=42, return_candidates=True)

    print(f"生成 {len(poses)} 个倾斜位姿 (共 {len(cands)} 候选):")
    for i, c in enumerate(cands):
        z_dev = c['z_dev']
        n_pts = c['n_pts']
        print(f"  #{i}: pitch={c['pitch']:+.0f}° yaw={c['yaw']:+.0f}° "
              f"z_dev={z_dev:.1f}° n_pts={n_pts}")

    # 统计
    z_devs = np.array([c['z_dev'] for c in cands])
    print(f"\n统计: z_dev min={z_devs.min():.1f}° max={z_devs.max():.1f}° "
          f"mean={z_devs.mean():.1f}° std={z_devs.std():.1f}°")
    print(f"  → z_dev std={z_devs.std():.1f}° {'✓ 足够' if z_devs.std()>3 else '⚠ 不足'}")
