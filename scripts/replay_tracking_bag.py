#!/usr/bin/env python3
"""Offline replay of the breakpoint tracking pipeline against recorded bags.

Replays a bag's ``/gocator/profile`` stream through the standalone
``BreakpointTrackingPipeline`` (a faithful copy of the deployed node state
machine) and writes a per-frame diagnostic JSON plus a console summary.

Usage
-----
    python3 replay_tracking_bag.py BAG [BAG ...] [--output out.json] [--stride N]
        [--max-frames N] [--start N] [--scenario auto_seed]

The ``--scenario`` flag mirrors the recorder's ``--passive`` behaviour:
  - ``auto_seed``: replay control commands from the bag's
    ``/calibration/detection_control`` topic (real seed node commands).
  - otherwise: no commands are injected; the pipeline starts in ALIGN and the
    caller must call ``--lock-after N`` or ``--prior`` options manually.

The replay is deterministic and does not touch the deployed node.
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
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
from calibration_pipeline.perception.tracking_pipeline import (
    BreakpointTrackingPipeline,
    TrackingPipelineConfig,
)
from calibration_pipeline.perception.endpoint_detector import EndpointDetectionConfig
from real_config import make_real_config


def _profile_array(message: PointCloud2) -> np.ndarray:
    import sensor_msgs_py.point_cloud2 as point_cloud2

    values = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    if len(values) == 0:
        return np.zeros((0, 3), dtype=float)
    if getattr(values.dtype, "names", None):
        return np.column_stack(
            tuple(np.asarray(values[name], dtype=float) for name in ("x", "y", "z"))
        )
    return np.asarray(values, dtype=float).reshape(-1, 3)


def _prior_array(message: PointCloud2) -> np.ndarray:
    values = _profile_array(message)
    if len(values) < 2:
        return np.empty((0, 3))
    return np.vstack((values[0], values[-1]))


def replay_bag(
    path: Path,
    *,
    output: Path | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    start_frame: int = 0,
    replay_control: bool = False,
    auto_lock_after: int | None = None,
) -> dict:
    # Single parameter source: the real-machine run's detector_parameters.yaml.
    # (P1-5) replay and visualizer must never diverge on ratio / label / gates.
    config = make_real_config()

    pipeline = BreakpointTrackingPipeline(config)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    profile_type = get_message(types.get("/gocator/profile", "sensor_msgs/msg/PointCloud2"))
    control_type = get_message(types.get("/calibration/detection_control", "std_msgs/msg/String"))
    prior_type = get_message(types.get("/calibration/detection_prior", "sensor_msgs/msg/PointCloud2"))
    measured_prior_type = get_message(
        types.get("/calibration/detection_measured_prior", "sensor_msgs/msg/PointCloud2")
    )

    first_stamp = None
    rows: list[dict] = []
    input_frames = 0
    accepted = 0
    frames = 0
    controls: list[dict] = []

    while reader.has_next():
        topic, serialized, stamp = reader.read_next()
        if topic == "/calibration/detection_control" and replay_control:
            msg = deserialize_message(serialized, control_type)
            controls.append({"time_s": None, "command": str(msg.data)})
            pipeline.handle_control(str(msg.data))
            continue
        if topic == "/calibration/detection_prior" and replay_control:
            msg = deserialize_message(serialized, prior_type)
            prior = _prior_array(msg)
            if len(prior) == 2:
                pipeline.handle_prior(prior)
            continue
        if topic == "/calibration/detection_measured_prior" and replay_control:
            msg = deserialize_message(serialized, measured_prior_type)
            prior = _prior_array(msg)
            if len(prior) == 2:
                pipeline.handle_measured_prior(prior)
            continue
        if topic != "/gocator/profile":
            continue

        input_frames += 1
        if input_frames < start_frame:
            continue
        if (input_frames - start_frame) % stride:
            continue
        if first_stamp is None:
            first_stamp = stamp
        message = deserialize_message(serialized, profile_type)
        profile = _profile_array(message)
        timestamp_s = (stamp - first_stamp) * 1.0e-9
        result = pipeline.process_profile(profile, timestamp_s=timestamp_s)
        rows.append(result.to_dict())
        frames += 1
        # Optional: emulate the recorder's non-passive workflow
        # (reset -> wait ALIGN stable -> lock -> SEED_TRACK_START).
        if auto_lock_after is not None and frames == auto_lock_after:
            # P1-6: lock() may fail (alignment not stable yet).  Only start
            # temporal tracking after a real ALIGN->TRACK transition; retry on
            # later frames otherwise.
            if pipeline.lock():
                pipeline.handle_control("SEED_TRACK_START")
            else:
                print(
                    f"[auto_lock] frame {frames}: lock() failed "
                    f"(stable={pipeline.alignment_stable_frames}/"
                    f"{pipeline.config.minimum_lock_frames}), retrying next frame"
                )
        if max_frames is not None and frames >= max_frames:
            break

    valid = [r for r in rows if r["state"] == "VALID"]
    lengths = np.asarray([r["segment_length_mm"] for r in valid if r.get("segment_length_mm") is not None], dtype=float)
    reasons: dict[str, int] = {}
    for r in rows:
        if r["state"] != "VALID":
            reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1

    summary = {
        "bag": str(path),
        "frames": len(rows),
        "accepted": len(valid),
        "acceptance_rate": len(valid) / max(len(rows), 1),
        "replay_control": replay_control,
        "control_commands": controls,
        "length_minimum_mm": float(np.min(lengths)) if len(lengths) else None,
        "length_median_mm": float(np.median(lengths)) if len(lengths) else None,
        "length_maximum_mm": float(np.max(lengths)) if len(lengths) else None,
        "rejection_reasons": reasons,
        "first_valid": valid[0] if valid else None,
        "last_valid": valid[-1] if valid else None,
    }
    if output is not None:
        output.write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="write full per-frame JSON")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=["auto_seed", "passive"],
        default="passive",
        help="auto_seed replays control commands from the bag",
    )
    parser.add_argument(
        "--auto-lock-after",
        type=int,
        help="after this many processed frames, call lock()+SEED_TRACK_START "
        "(emulates recorder non-passive workflow)",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")

    for bag in args.bags:
        summary = replay_bag(
            bag,
            output=args.output,
            stride=args.stride,
            max_frames=args.max_frames,
            start_frame=args.start,
            replay_control=(args.scenario == "auto_seed"),
            auto_lock_after=args.auto_lock_after,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
