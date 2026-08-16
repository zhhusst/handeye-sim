#!/usr/bin/env python3
"""Extract the deployed node's real diagnostics + control timeline from a bag.

Answers, without touching the robot:
  - what the deployed detector actually did frame by frame (state/reason/mode)
  - how fast frames were processed (gap between processed profile stamps)
  - when control commands arrived and whether temporal tracking engaged
  - the acceptance curve over the run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--max-diags", type=int, default=100000)
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {i.name: i.type for i in reader.get_all_topics_and_types()}
    str_type = get_message(types.get("/profile_endpoint_detector/diagnostics", "std_msgs/msg/String"))
    pc2_type = get_message(types.get("/gocator/profile", "sensor_msgs/msg/PointCloud2"))
    joint_type = get_message(types.get("/joint_states", "sensor_msgs/msg/JointState"))

    diags: list[tuple[int, dict]] = []
    controls: list[tuple[int, str]] = []
    profile_stamps: list[int] = []
    joint_stamps: list[int] = []
    t0 = None
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == "/profile_endpoint_detector/diagnostics":
            msg = deserialize_message(data, str_type)
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            diags.append((ts, payload))
            if len(diags) > args.max_diags:
                break
        elif topic == "/calibration/detection_control":
            msg = deserialize_message(data, str_type)
            controls.append((ts, str(msg.data)))
        elif topic == "/gocator/profile":
            profile_stamps.append(ts)
        elif topic == "/joint_states":
            msg = deserialize_message(data, joint_type)
            joint_stamps.append(ts)

    if not diags:
        print("no diagnostics found")
        return
    t0 = diags[0][0]
    ts = [(t - t0) * 1e-9 for t, _ in diags]
    states = [d.get("state", "?") for _, d in diags]
    reasons = [d.get("reason", "") or "" for _, d in diags]
    modes = [d.get("mode", "?") for _, d in diags]
    accepted = [int(d.get("accepted", 0)) for _, d in diags]
    frames = [int(d.get("frames", 0)) for _, d in diags]
    temporal = [bool(d.get("temporal_tracking", False)) for _, d in diags]
    temporal_init = [bool(d.get("temporal_initialized", False)) for _, d in diags]
    temporal_susp = [bool(d.get("temporal_suspended", False)) for _, d in diags]
    seg_len = [d.get("segment_length_mm") for _, d in diags]
    lost_frames = [int(d.get("lost_frames", 0)) for _, d in diags]

    print(f"diags={len(diags)}  span={ts[-1]:.2f}s  avg_fps={len(diags)/max(ts[-1],1e-9):.2f}")
    print(f"profile_msgs={len(profile_stamps)}")
    print("\n-- control commands --")
    for t, cmd in controls:
        print(f"  t={((t-t0)*1e-9):8.3f}s  {cmd}")

    # inter-frame processing gaps (deployed node throughput)
    gaps = np.diff(ts)
    print(f"\n-- processing gap: median={np.median(gaps)*1000:.1f}ms "
          f"p90={np.percentile(gaps,90)*1000:.1f}ms max={np.max(gaps)*1000:.1f}ms --")
    print(f"  (sensor profile rate ~{len(profile_stamps)/max(ts[-1],1e-9):.1f} Hz)")

    # state transition timeline
    print("\n-- state transitions --")
    prev = None
    for t, s, r, m, lf in zip(ts, states, reasons, modes, lost_frames):
        key = (s, m, r)
        if key != prev:
            print(f"  t={t:8.3f}s  state={s:8s} mode={m:16s} reason={r} (lost_frames={lf})")
            prev = key

    # per-reason counts
    print("\n-- rejection reason histogram (non-VALID frames) --")
    from collections import Counter
    cnt = Counter((s, r) for s, r in zip(states, reasons) if s != "VALID")
    for (s, r), n in cnt.most_common(20):
        print(f"  {s:8s} {r:45s} {n}")

    # acceptance curve
    print("\n-- acceptance --")
    for i in (0, len(ts)//4, len(ts)//2, 3*len(ts)//4, len(ts)-1):
        print(f"  t={ts[i]:8.3f}s  accepted={accepted[i]}/{frames[i]} "
              f"({accepted[i]/max(frames[i],1)*100:.1f}%)  temporal={temporal[i]} "
              f"init={temporal_init[i]} susp={temporal_susp[i]}")

    # temporal engagement window
    engaged = [(t, ti) for t, ti in zip(ts, temporal) if ti]
    if engaged:
        print(f"temporal engaged from {engaged[0][0]:.3f}s to {engaged[-1][0]:.3f}s")
    else:
        print("temporal tracking NEVER engaged during this run")

    # segment length over time (VALID only)
    valid_len = [(t, v) for t, v in zip(ts, seg_len) if v is not None]
    if valid_len:
        lens = np.asarray([v for _, v in valid_len])
        print(f"segment length (VALID): min={np.min(lens):.2f} med={np.median(lens):.2f} max={np.max(lens):.2f} mm")


if __name__ == "__main__":
    main()
