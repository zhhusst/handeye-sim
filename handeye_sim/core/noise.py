#!/usr/bin/env python3
"""
core/noise.py — 仿真噪声注入模块

物理模型: 关节角误差 → FK 传播 → 法兰位姿误差
  - 真实误差源是 6 个关节角的编码器/零偏/挠度误差
  - 每个关节独立扰动 N(0, σ_joint)
  - 经 FK 传播后同时产生法兰的位置和旋转误差
  - 位置和旋转误差的幅值由机器人构型决定（不是独立参数）

退化模型 (无关节角数据时):
  - 对法兰位姿直接加 task-space 扰动
"""

import numpy as np
from handeye_sim.core.so3 import so3_exp
from handeye_sim.core.kinematics import fanuc_fk


def _mm2m(mm: float) -> float:
    return mm / 1000.0


def _deg2rad(deg: float) -> float:
    return np.deg2rad(deg)


def apply_noise(poses, meas, noise_cfg, rng=None, joint_angles=None):
    """对位姿和测量注入噪声

    Args:
        poses: [(R_i, t_i), ...]  — 干净的机器人法兰位姿
        meas:  [dict, ...]         — 干净的测量
        noise_cfg: dict, 来自 config.yaml scene.noise
            keys: joint_std_deg (主), laser_std_mm
                   robot_trans_std_mm, robot_rot_std_deg (joint_angles 缺失时的 fallback)
        rng: np.random.Generator (可选)
        joint_angles: list of (6,) np.ndarray 或 None
            — 每个位姿对应的 6 维关节角 (弧度, J3_display)
            — 如果提供, 使用关节空间扰动+FK; 否则退化到 task-space 扰动

    Returns:
        (noisy_poses, noisy_meas)
    """
    if not noise_cfg or not noise_cfg.get('enabled', False):
        return poses, meas

    if rng is None:
        rng = np.random.default_rng()

    joint_std_deg = noise_cfg.get('joint_std_deg', 0.01)
    laser_std_m = _mm2m(noise_cfg.get('laser_std_mm', 0.05))

    # ── 1. 扰动机器人位姿 ──

    if joint_angles is not None and len(joint_angles) == len(poses):
        # ★ 关节空间扰动 (物理正确路径)
        joint_std_rad = _deg2rad(joint_std_deg)
        noisy_poses = []
        for joints_i in joint_angles:
            joints_i = np.asarray(joints_i, dtype=float)
            # 每个关节独立扰动
            dq = rng.normal(0, joint_std_rad, 6)
            T_noisy = fanuc_fk(joints_i + dq)
            R_noisy = T_noisy[:3, :3].copy()
            t_noisy = T_noisy[:3, 3].copy()
            noisy_poses.append((R_noisy, t_noisy))
    else:
        # ★ 退化: task-space 扰动 (无关节角数据时)
        trans_std_m = _mm2m(noise_cfg.get('robot_trans_std_mm', 0.3))
        rot_std_deg = noise_cfg.get('robot_rot_std_deg', 0.03)
        rot_std_rad = _deg2rad(rot_std_deg)

        noisy_poses = []
        for R_i, t_i in poses:
            R_i = np.asarray(R_i).copy()
            t_i = np.asarray(t_i).copy()
            axis = rng.normal(0, rot_std_rad / 3.0, 3)
            R_noisy = R_i @ so3_exp(axis)
            t_noisy = t_i + rng.normal(0, trans_std_m, 3)
            noisy_poses.append((R_noisy, t_noisy))

    # ── 2. 扰动激光测量 ──
    noisy_meas = []
    for m in meas:
        nm = dict(m)

        p_S_plane = m.get('p_S_plane', [])
        if len(p_S_plane) > 0:
            pts = np.asarray(p_S_plane)
            pts_noisy = pts + rng.normal(0, laser_std_m, pts.shape)
            nm['p_S_plane'] = pts_noisy.tolist()
        else:
            nm['p_S_plane'] = []

        for key in ['p_S_e1', 'p_S_e2']:
            val = m.get(key)
            if val is not None:
                pt = np.asarray(val)
                nm[key] = (pt + rng.normal(0, laser_std_m / 2.0, 3)).tolist()
            else:
                nm[key] = None

        noisy_meas.append(nm)

    return noisy_poses, noisy_meas


def add_noise_to_calib_data(data, noise_cfg, rng=None, seed=42):
    """对 CalibData 对象注入噪声 (原地修改)

    Args:
        data: CalibData
        noise_cfg: dict
        rng: np.random.Generator (可选)
        seed: 仅在 rng 为 None 时使用

    Returns:
        data (原地修改)
    """
    if not noise_cfg or not noise_cfg.get('enabled', False):
        return data

    if rng is None:
        rng = np.random.default_rng(seed)

    joint_std_deg = noise_cfg.get('joint_std_deg', 0.01)
    joint_std_rad = _deg2rad(joint_std_deg)
    laser_std_m = _mm2m(noise_cfg.get('laser_std_mm', 0.05))

    # 检查是否有关节角数据
    has_joints = all(hasattr(r, 'joints') and r.joints is not None for r in data.records)

    for i, rec in enumerate(data.records):
        if has_joints and rec.joints is not None:
            # 关节空间扰动
            dq = rng.normal(0, joint_std_rad, 6)
            T_noisy = fanuc_fk(np.asarray(rec.joints) + dq)
            rec.pose.R = T_noisy[:3, :3].copy()
            rec.pose.t = T_noisy[:3, 3].copy()
        else:
            # task-space fallback
            trans_std_m = _mm2m(noise_cfg.get('robot_trans_std_mm', 0.3))
            rot_std_rad = _deg2rad(noise_cfg.get('robot_rot_std_deg', 0.03))

            R_i = np.asarray(rec.pose.R).copy()
            t_i = np.asarray(rec.pose.t).copy()
            axis = rng.normal(0, rot_std_rad / 3.0, 3)
            rec.pose.R = R_i @ so3_exp(axis)
            rec.pose.t = t_i + rng.normal(0, trans_std_m, 3)

        # 扰动测量
        meas = rec.meas
        if len(meas.scan_pts_S) > 0:
            pts = np.asarray(meas.scan_pts_S)
            meas.scan_pts_S = (pts + rng.normal(0, laser_std_m, pts.shape)).tolist()

        if meas.p_S_e1 is not None:
            pt = np.asarray(meas.p_S_e1)
            meas.p_S_e1 = (pt + rng.normal(0, laser_std_m / 2.0, 3)).tolist()

        if meas.p_S_e2 is not None:
            pt = np.asarray(meas.p_S_e2)
            meas.p_S_e2 = (pt + rng.normal(0, laser_std_m / 2.0, 3)).tolist()

    return data
