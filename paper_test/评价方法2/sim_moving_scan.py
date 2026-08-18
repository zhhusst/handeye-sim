#!/usr/bin/env python3
"""论文式球移动扫描数据仿真器（预演用，v2）。

每组：传感器（装在法兰上）从球心周围某个方位沿该方位的切向平移，
扫描过程覆盖球面大片弧区。7 组 = 7 个方位（对应论文 7 个板方向）。

核心：对每个方位，构造 flange 姿态使得传感器 z 轴指向球心、x 轴沿
切向；然后 flange 沿切向平移 travel，逐帧采样球面射线交点。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.simulation.scene_truth import (
    HAND_EYE_ROTATION, HAND_EYE_TRANSLATION,
)


def _make_flange_for_view(sphere_center, radius, handeye_R, handeye_t,
                          elevation_deg, azimuth_deg, working_dist_m):
    """构造法兰位姿：传感器 z 轴指向球心，x 轴沿方位切向。"""
    el = np.deg2rad(elevation_deg)
    az = np.deg2rad(azimuth_deg)
    # 传感器 z 轴在基座系：从球心指向传感器原点（球心->外）
    # 用球坐标：z_sensor = -direction_from_center
    direction = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    # 传感器 z 轴（前向）指向球心
    sensor_z_base = -direction
    # x 轴沿方位切向（扫过球面）
    tangent = np.array([-np.sin(az), np.cos(az), 0.0])
    tangent = tangent - sensor_z_base * (tangent @ sensor_z_base)
    tangent /= np.linalg.norm(tangent)
    sensor_y_base = np.cross(sensor_z_base, tangent)
    R_sb = np.column_stack([tangent, sensor_y_base, sensor_z_base])  # R_sensor<-base
    # 传感器原点位置：球心 - direction * working_dist
    sensor_origin = sphere_center + direction * working_dist_m
    # flange = sensor_origin - R_fs @ t_fs  (R_fs 是 R_flange<-sensor)
    flange_R = R_sb @ handeye_R.T  # R_flange<-base = R_sensor<-base @ R_sensor<-flange
    flange_t = sensor_origin - flange_R @ handeye_t
    return flange_R, flange_t, tangent


def generate_moving_scan(scene, flange_R, flange_start, travel_dir, travel_m,
                         frames, rng, noise_m):
    """法兰沿 travel_dir 平移 travel_m，逐帧采样球面交点。"""
    R_fs = scene["handeye_R"]
    t_fs = scene["handeye_t"]
    radius = scene["radius"]
    center = scene["center"]
    half_width = scene["half_width"]
    samples = scene["samples"]

    flange_ts = flange_start[None, :] + np.linspace(
        0.0, travel_m, frames)[:, None] * (travel_dir / np.linalg.norm(travel_dir))
    xs = np.linspace(-half_width, half_width, samples)
    points = np.full((frames, samples, 3), np.nan)

    for fi in range(frames):
        F = flange_ts[fi]
        sensor_origin = F + flange_R @ t_fs
        sensor_z = flange_R @ R_fs @ np.array([0.0, 0.0, 1.0])
        sensor_x = flange_R @ R_fs @ np.array([1.0, 0.0, 0.0])
        for si, x in enumerate(xs):
            ray_origin = sensor_origin + x * sensor_x
            oc = ray_origin - center
            b = 2.0 * float(oc @ sensor_z)
            c = float(oc @ oc) - radius ** 2
            disc = b * b - 4.0 * c
            if disc < 0.0:
                continue
            t = (-b - np.sqrt(disc)) / 2.0
            if t < 0.0:
                continue
            z_sensor = float(t)
            points[fi, si] = [x, 0.0, z_sensor]
            if noise_m > 0.0:
                points[fi, si, 2] += rng.normal(0.0, noise_m)
    return points


def save_dataset_npz(path, groups_points, flange_rots, flange_ts):
    raw_points, raw_offsets, pose_indices = [], [0], []
    for gi, pts in enumerate(groups_points):
        for fi in range(len(pts)):
            raw_points.append(pts[fi])
            raw_offsets.append(raw_offsets[-1] + len(pts[fi]))
            pose_indices.append(gi)
    np.savez_compressed(
        path,
        raw_points_sensor_m=np.vstack(raw_points),
        raw_sample_indices=np.concatenate([np.arange(len(p)) for p in raw_points]),
        raw_frame_offsets=np.asarray(raw_offsets, dtype=np.int64),
        selected_points_sensor_m=np.vstack(raw_points),
        selected_sample_indices=np.concatenate([np.arange(len(p)) for p in raw_points]),
        selected_frame_offsets=np.asarray(raw_offsets, dtype=np.int64),
        frame_pose_indices=np.asarray(pose_indices, dtype=np.int64),
        frame_stamps_ns=np.arange(len(raw_points), dtype=np.int64) * 1_000_000,
        flange_rotations=np.vstack(flange_rots),
        flange_translations_m=np.vstack(flange_ts),
        profile_circle_radii_m=np.full(len(raw_points), 0.0),
        profile_circle_rms_m=np.full(len(raw_points), 0.0),
        profile_chords_m=np.full(len(raw_points), 0.0),
    )


def main() -> None:
    rng = np.random.default_rng(42)
    scene = {
        "handeye_R": HAND_EYE_ROTATION,
        "handeye_t": HAND_EYE_TRANSLATION,
        "center": np.array([1.0, 0.0, 0.15]),
        "radius": 0.010001,
        "half_width": 0.020,
        "samples": 640,
    }
    groups_points, all_rots, all_ts = [], [], []
    for gi in range(7):
        el = 70.0 - gi * 5.0          # 俯仰 70°→40°
        az = gi * 51.428               # 方位 0→360
        flange_R, flange_t, tangent = _make_flange_for_view(
            scene["center"], scene["radius"], scene["handeye_R"],
            scene["handeye_t"], el, az, working_dist_m=0.14,
        )
        # 论文式移动：球沿板法向移动，传感器固定 → 等价于传感器沿
        # 激光平面法向（传感器 Y）平移，每帧切球面不同位置的平行截面，
        # 变换到基座系后点云展开成 3D（非共面），可拟合球。
        travel_dir = flange_R @ scene["handeye_R"] @ np.array([0.0, 1.0, 0.0])
        pts = generate_moving_scan(
            scene, flange_R, flange_t, travel_dir, travel_m=0.040,
            frames=120, rng=rng, noise_m=12e-6,
        )
        groups_points.append(pts)
        all_rots.append(np.repeat(flange_R[None, :, :], 120, axis=0))
        all_ts.append(flange_t[None, :] + np.linspace(0.0, 0.040, 120)[:, None]
                      * travel_dir)

    out = Path("/workspace/tmp_diag/sphere_moving_scan_sim.npz")
    save_dataset_npz(out, groups_points, all_rots, all_ts)
    print(f"saved: {out}")
    npz = np.load(str(out))
    fpi = npz["frame_pose_indices"]
    pts = npz["selected_points_sensor_m"]
    offs = npz["selected_frame_offsets"]
    for p in np.unique(fpi):
        frames = np.where(fpi == p)[0]
        g = np.vstack([pts[offs[f]:offs[f + 1]] for f in frames])
        g = g[np.all(np.isfinite(g), axis=1)]
        s = g.max(axis=0) - g.min(axis=0)
        print(f"pose {p}: {len(frames)}帧 {len(g)}点 "
              f"span X={1000*s[0]:.1f} Y={1000*s[1]:.1f} Z={1000*s[2]:.1f}mm "
              f"(球径20mm)")


if __name__ == "__main__":
    main()
