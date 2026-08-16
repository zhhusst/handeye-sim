#!/usr/bin/env python3
"""诊断真机轮廓"波动点被当成断点"问题。

对 bag 中的每帧 profile：
1. 用 /calibration/endpoints 的 X 范围界定平板表面
2. 计算板面 z 相对线性拟合的残差（波动幅度）
3. 统计波动 RMS/P95/峰值，对比 plate_edge_from_midpoint 的残差阈值 0.8mm
4. 模拟 grow 过程：统计"波动导致的提前 break"（残差>0.8mm 但非真台阶）
"""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2


def profile_array(msg):
    return np.asarray(
        point_cloud2.read_points_numpy(
            msg, field_names=["x", "y", "z"], skip_nans=True
        ),
        dtype=float,
    ).reshape(-1, 3)


def endpoint_array(msg):
    return np.asarray(
        point_cloud2.read_points_numpy(
            msg, field_names=["x", "y", "z"], skip_nans=True
        ),
        dtype=float,
    ).reshape(-1, 3)


def simulate_grow(prof, x_mid_m, residual_threshold_m=0.0008,
                  min_grow_points=8, max_grow_points=400):
    """Return list of (direction, residual_mm, is_first_break) for both sides."""
    pts = prof[np.argsort(prof[:, 0])][:, (0, 2)]
    if len(pts) < 20:
        return None
    i0 = int(np.argmin(np.abs(pts[:, 0] - x_mid_m)))
    xs, zs = pts[:, 0], pts[:, 1]
    out = []
    for direction in (-1, 1):
        idxs = [i0]
        i = i0
        first_break = None
        breaks = []
        while 0 < i + direction < len(pts) - 1 and len(idxs) < max_grow_points:
            nxt = i + direction
            if len(idxs) >= min_grow_points:
                p = np.polyfit(xs[idxs], zs[idxs], 1)
                pred = p[0] * xs[nxt] + p[1]
                resid = abs(zs[nxt] - pred)
                if resid > residual_threshold_m:
                    if first_break is None:
                        first_break = resid
                    breaks.append(resid)
            idxs.append(nxt)
            i = nxt
        out.append((direction, first_break, breaks))
    return out


def main() -> None:
    path = sys.argv[1]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    ptype = get_message(types["/gocator/profile"])
    etype = get_message(types["/calibration/endpoints"])

    profiles, endpoints_list = [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == "/gocator/profile":
            profiles.append(profile_array(deserialize_message(data, ptype)))
        elif topic == "/calibration/endpoints":
            ep = endpoint_array(deserialize_message(data, etype))
            if len(ep) == 2:
                endpoints_list.append(ep)

    print(f"profiles={len(profiles)} endpoints={len(endpoints_list)}")

    rms_list, p95_list, max_list = [], [], []
    over_threshold_frames = 0
    grow_break_frames = 0
    break_residuals = []          # 所有 grow 首次 break 的残差（mm）
    sampled = 0
    for i in range(0, min(len(profiles), len(endpoints_list)), 5):
        prof = profiles[i]
        ep = endpoints_list[i]
        if len(prof) < 30:
            continue
        xmin, xmax = float(np.min(ep[:, 0])), float(np.max(ep[:, 0]))
        mask = (prof[:, 0] >= xmin) & (prof[:, 0] <= xmax)
        plate = prof[mask]
        if len(plate) < 12:
            continue
        xs, zs = plate[:, 0], plate[:, 2]
        p = np.polyfit(xs, zs, 1)
        resid = np.abs(zs - (p[0] * xs + p[1]))
        rms_list.append(float(np.sqrt(np.mean(resid ** 2))))
        p95_list.append(float(np.percentile(resid, 95)))
        max_list.append(float(np.max(resid)))
        if float(np.max(resid)) > 0.0008:
            over_threshold_frames += 1

        # simulate grow from plate midpoint
        x_mid = 0.5 * (xmin + xmax)
        grow = simulate_grow(prof, x_mid)
        if grow is not None:
            sampled += 1
            broke = False
            for direction, first_break, breaks in grow:
                if first_break is not None:
                    broke = True
                    break_residuals.append(1000.0 * first_break)
            if broke:
                grow_break_frames += 1

    n = len(rms_list)
    print(f"\n== 板面波动（线性拟合残差）: {n} 帧 ==")
    print(f"  RMS   [mm]: mean={1000*np.mean(rms_list):.3f}  "
          f"p95={1000*np.percentile(rms_list,95):.3f}  max={1000*np.max(rms_list):.3f}")
    print(f"  P95   [mm]: mean={1000*np.mean(p95_list):.3f}  "
          f"p95={1000*np.percentile(p95_list,95):.3f}  max={1000*np.max(p95_list):.3f}")
    print(f"  峰值 [mm]: mean={1000*np.mean(max_list):.3f}  "
          f"p95={1000*np.percentile(max_list,95):.3f}  max={1000*np.max(max_list):.3f}")
    print(f"  帧峰值>0.8mm: {over_threshold_frames}/{n} "
          f"({100*over_threshold_frames/max(n,1):.0f}%)")

    print(f"\n== grow 模拟（阈值0.8mm）: {sampled} 帧 ==")
    print(f"  至少一侧波动break的帧: {grow_break_frames}/{sampled} "
          f"({100*grow_break_frames/max(sampled,1):.0f}%)")
    if break_residuals:
        br = np.asarray(break_residuals)
        print(f"  break残差 [mm]: mean={np.mean(br):.3f}  "
              f"median={np.median(br):.3f}  p95={np.percentile(br,95):.3f}  "
              f"max={np.max(br):.3f}")


if __name__ == "__main__":
    main()
