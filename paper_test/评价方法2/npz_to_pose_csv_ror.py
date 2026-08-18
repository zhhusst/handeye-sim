#!/usr/bin/env python3
"""把移动扫描 npz 转成 sphere_validation_rcim_v2.py 要求的 pose CSV。

含帧级 ROR（radius outlier removal，论文 §7 做法）：
  - 每帧深度 z 中位与组内 z 中位偏差 > z_gate_mm 的帧剔除（杂散点帧）
  - 点数过少的帧剔除
输出: <outdir>/scanner1/pose1.csv ... pose7.csv（基座系 3D 点，mm）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.sphere_validation import transform_profile_to_base


def convert(npz_path: Path, result_path: Path, outdir: Path,
            max_poses: int = 7, z_gate_mm: float = 25.0,
            min_points: int = 50) -> None:
    result = json.load(open(result_path))
    R_fs = np.array(result["handeye"]["rotation"])
    t_fs = np.array(result["handeye"]["translation"])

    npz = np.load(str(npz_path), allow_pickle=True)
    offsets = npz["selected_frame_offsets"]
    points = npz["selected_points_sensor_m"]
    fpi = npz["frame_pose_indices"]
    rots = npz["flange_rotations"]
    ts = npz["flange_translations_m"]

    scanner_dir = outdir / "scanner1"
    scanner_dir.mkdir(parents=True, exist_ok=True)

    pose_indices = np.unique(fpi)[:max_poses]
    for pose_index in pose_indices:
        frames = np.where(fpi == pose_index)[0]
        # ---- 帧级 ROR：先用每帧深度中位找离群帧 ----
        frame_z = []
        for f in frames:
            s = points[offsets[f]:offsets[f + 1]]
            s = s[np.all(np.isfinite(s), axis=1)]
            if len(s) < 5:
                frame_z.append(np.nan)
                continue
            frame_z.append(float(np.median(s[:, 2])))
        frame_z = np.asarray(frame_z)
        good_z = frame_z[np.isfinite(frame_z)]
        if len(good_z) == 0:
            continue
        z_ref = float(np.median(good_z))
        keep_frames = []
        rejected_frames = []
        for f, z in zip(frames, frame_z):
            if np.isnan(z):
                rejected_frames.append(int(f))
                continue
            if abs(z - z_ref) > z_gate_mm * 1e-3:
                rejected_frames.append(int(f))
                continue
            keep_frames.append(f)
        if rejected_frames:
            print(f"pose{int(pose_index)+1}: ROR 剔除 {len(rejected_frames)} 帧 "
                  f"(z偏离>={z_gate_mm}mm): {rejected_frames[:10]}{'...' if len(rejected_frames)>10 else ''}")

        # ---- 点级过滤 + 基座系变换 ----
        base_pts = []
        for f in keep_frames:
            s = points[offsets[f]:offsets[f + 1]]
            s = s[np.all(np.isfinite(s), axis=1)]
            if len(s) < min_points:
                continue
            base_pts.append(transform_profile_to_base(
                s, rots[f], ts[f], R_fs, t_fs
            ))
        pts_mm = np.vstack(base_pts) * 1000.0
        csv_path = scanner_dir / f"pose{int(pose_index) + 1}.csv"
        np.savetxt(csv_path, pts_mm, delimiter=",", header="x,y,z", comments="")
        print(f"pose{int(pose_index)+1}.csv: {len(pts_mm)} points "
              f"(frames {len(keep_frames)})")

    print(f"\nconverted to {scanner_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="npz -> pose CSV（含帧级ROR）")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--outdir", type=Path,
                        default=Path("/workspace/tmp_diag/validation_real_ror"))
    parser.add_argument("--max-poses", type=int, default=7)
    parser.add_argument("--z-gate-mm", type=float, default=25.0,
                        help="帧深度 z 与组中位偏差阈值(mm)，超过剔除")
    parser.add_argument("--min-points", type=int, default=50,
                        help="每帧最少点数，少于剔除")
    args = parser.parse_args()
    convert(args.npz, args.result, args.outdir, max_poses=args.max_poses,
            z_gate_mm=args.z_gate_mm, min_points=args.min_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
