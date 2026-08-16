#!/usr/bin/env python3
"""改前(confirm=1,thr=0.8mm) vs 改后(confirm=3,thr=0.4mm) 断点稳定性对比。

用 bag 的 /calibration/endpoints 提供 x_mid，重新跑 plate_edge_from_midpoint，
统计断点 x 轨迹的逐帧跳变（稳定=小跳变，误断=大幅跳动）。
"""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.roi_tracking.segment_detector import plate_edge_from_midpoint


def profile_array(msg):
    return np.asarray(
        point_cloud2.read_points_numpy(msg, field_names=["x", "y", "z"], skip_nans=True),
        dtype=float,
    ).reshape(-1, 3)


def endpoint_array(msg):
    return np.asarray(
        point_cloud2.read_points_numpy(msg, field_names=["x", "y", "z"], skip_nans=True),
        dtype=float,
    ).reshape(-1, 3)


def main():
    path = sys.argv[1]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    ptype = get_message(types["/gocator/profile"])
    etype = get_message(types["/calibration/endpoints"])

    profiles, eps = [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == "/gocator/profile":
            profiles.append(profile_array(deserialize_message(data, ptype)))
        elif topic == "/calibration/endpoints":
            ep = endpoint_array(deserialize_message(data, etype))
            if len(ep) == 2:
                eps.append(ep)

    configs = {
        "old(confirm=1,thr=0.8)": dict(residual_threshold_m=0.0008, confirmation_points=1),
        "new(confirm=3,thr=0.4)": dict(residual_threshold_m=0.0004, confirmation_points=3),
        "new(confirm=3,thr=0.8)": dict(residual_threshold_m=0.0008, confirmation_points=3),
    }
    results = {k: [] for k in configs}
    for i in range(0, min(len(profiles), len(eps)), 4):
        prof, ep = profiles[i], eps[i]
        if len(prof) < 30:
            continue
        x_mid = 0.5 * (np.min(ep[:, 0]) + np.max(ep[:, 0]))
        for name, kw in configs.items():
            out = plate_edge_from_midpoint(prof[:, (0, 2)], x_mid, **kw)
            if out is not None:
                results[name].append((min(out[0][0], out[1][0]),
                                      max(out[0][0], out[1][0])))

    print(f"frames analysed: {len(results['old(confirm=1,thr=0.8)'])}")
    for name, traj in results.items():
        traj = np.asarray(traj)
        if len(traj) < 3:
            print(f"{name:28s}: too few frames")
            continue
        # per-frame max jump of the two endpoints vs previous frame
        jumps = np.max(np.abs(np.diff(traj, axis=0)), axis=1)
        # endpoint x range over the whole trace (stability span)
        span = float(np.max(traj[:, 1] - np.min(traj[:, 1])))
        print(f"{name:28s}: median_jump={np.median(jumps)*1000:.2f}mm  "
              f"p95_jump={np.percentile(jumps,95)*1000:.2f}mm  "
              f"max_jump={np.max(jumps)*1000:.2f}mm  right_span={span*1000:.2f}mm")


if __name__ == "__main__":
    main()
