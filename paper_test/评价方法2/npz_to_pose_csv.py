#!/usr/bin/env python3
"""把我们的移动扫描 npz 转成 sphere_validation_rcim_v2.py 要求的 pose CSV。

输入: sphere_moving_scan_sim.npz（或真机采集的相同格式）
      calibration_result.json（手眼）
输出: <outdir>/scanner1/pose1.csv ... pose7.csv（基座系 3D 点，mm，x,y,z）
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
            max_poses: int = 7) -> None:
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
        base_pts = []
        for f in frames:
            pts_s = points[offsets[f]:offsets[f + 1]]
            pts_s = pts_s[np.all(np.isfinite(pts_s), axis=1)]
            if len(pts_s) == 0:
                continue
            base_pts.append(transform_profile_to_base(
                pts_s, rots[f], ts[f], R_fs, t_fs
            ))
        pts_mm = np.vstack(base_pts) * 1000.0
        csv_path = scanner_dir / f"pose{int(pose_index) + 1}.csv"
        np.savetxt(csv_path, pts_mm, delimiter=",", header="x,y,z", comments="")
        print(f"{csv_path}: {len(pts_mm)} points")

    print(f"\nconverted to {scanner_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="npz -> pose CSV 转换器")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("/workspace/tmp_diag/validation"))
    parser.add_argument("--max-poses", type=int, default=7)
    args = parser.parse_args()
    convert(args.npz, args.result, args.outdir, max_poses=args.max_poses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
