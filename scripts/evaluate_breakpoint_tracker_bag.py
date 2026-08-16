#!/usr/bin/env python3
"""Offline, motion-free replay of the breakpoint detector on recorded bags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2

from calibration_pipeline.perception.endpoint_detector import (
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)


def profile_array(message) -> np.ndarray:
    return np.asarray(
        point_cloud2.read_points_numpy(
            message, field_names=["x", "y", "z"], skip_nans=True
        ),
        dtype=float,
    ).reshape(-1, 3)


def replay(
    path: Path, maximum_frames: int | None = None, stride: int = 1
) -> dict:
    detector = ProfileEndpointDetector(
        EndpointDetectionConfig(
            minimum_segment_length_m=0.01,
            maximum_segment_length_m=0.25,
            maximum_residual_rms_m=0.0005,
            smoothing_window=5,
            local_fit_window=12,
            angle_change_threshold_deg=10.0,
            height_jump_threshold_m=0.0002,
            breakpoint_cluster_points=8,
            maximum_abs_surface_midpoint_x_m=0.08,
        )
    )
    angle = np.deg2rad(25.0)
    center = np.array([0.0, 0.0, 0.28])
    offset = 0.04 * np.array([np.cos(angle), 0.0, np.sin(angle)])
    template = np.vstack((center - offset, center + offset))

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    profile_type = get_message(types["/gocator/profile"])

    tracked = None
    reference_length = None
    first_stamp = None
    rows = []
    frames = 0
    input_frames = 0
    consecutive_misses = 0
    maximum_misses = 0
    while reader.has_next():
        topic, serialized, stamp = reader.read_next()
        if topic != "/gocator/profile":
            continue
        input_frames += 1
        if (input_frames - 1) % stride:
            continue
        if first_stamp is None:
            first_stamp = stamp
        message = deserialize_message(serialized, profile_type)
        profile = profile_array(message)
        if tracked is None:
            detection = detector.detect_guided(
                profile,
                template[0],
                template[1],
                normal_gate_m=0.003,
                endpoint_gate_m=0.020,
                maximum_angle_difference_deg=25.0,
                selection_mode="offline_align",
            )
        else:
            detection = detector.detect_temporal_breakpoint_pair(
                profile,
                tracked[0],
                tracked[1],
                endpoint_gate_m=0.025,
                normal_gate_m=0.006,
                maximum_angle_difference_deg=25.0,
                selection_mode="offline_temporal",
            )
        accepted = False
        reason = detector.last_rejection_reason
        if detection is not None:
            measured = np.vstack((detection.first, detection.second))
            if tracked is not None:
                direct = np.linalg.norm(measured - tracked, axis=1)
                swapped = np.linalg.norm(measured[::-1] - tracked, axis=1)
                if float(np.sum(swapped)) < float(np.sum(direct)):
                    measured = measured[::-1]
                step = float(np.max(np.linalg.norm(measured - tracked, axis=1)))
                length = float(np.linalg.norm(measured[1] - measured[0]))
                plausible = (
                    step <= 0.003
                    and reference_length is not None
                    and 0.10 * reference_length <= length <= 4.00 * reference_length
                )
            else:
                plausible = True
                length = float(np.linalg.norm(measured[1] - measured[0]))
            if plausible:
                tracked = measured
                if reference_length is None:
                    reference_length = length
                accepted = True
                reason = ""
        if accepted:
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            maximum_misses = max(maximum_misses, consecutive_misses)
        rows.append(
            {
                "time_s": (stamp - first_stamp) * 1.0e-9,
                "accepted": accepted,
                "reason": reason,
                "length_mm": None
                if tracked is None
                else 1000.0 * float(np.linalg.norm(tracked[1] - tracked[0])),
                "first_x_mm": None if tracked is None else 1000.0 * float(tracked[0, 0]),
                "second_x_mm": None if tracked is None else 1000.0 * float(tracked[1, 0]),
            }
        )
        frames += 1
        if maximum_frames is not None and frames >= maximum_frames:
            break

    valid = [row for row in rows if row["accepted"]]
    lengths = np.asarray([row["length_mm"] for row in valid], dtype=float)
    return {
        "bag": str(path),
        "frames": len(rows),
        "accepted": len(valid),
        "acceptance_rate": len(valid) / max(len(rows), 1),
        "reference_length_mm": None
        if reference_length is None
        else 1000.0 * reference_length,
        "length_minimum_mm": None if not len(lengths) else float(np.min(lengths)),
        "length_median_mm": None if not len(lengths) else float(np.median(lengths)),
        "length_maximum_mm": None if not len(lengths) else float(np.max(lengths)),
        "maximum_consecutive_misses": maximum_misses,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    results = [
        replay(path.resolve(), args.maximum_frames, args.stride)
        for path in args.bags
    ]
    summary = [{k: v for k, v in item.items() if k != "rows"} for item in results]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
