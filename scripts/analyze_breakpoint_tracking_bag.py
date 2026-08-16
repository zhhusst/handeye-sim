#!/usr/bin/env python3
"""Summarize breakpoint tracking and seed-motion state from ROS 2 bags."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DIAGNOSTIC_TOPIC = "/profile_endpoint_detector/diagnostics"
SEED_STATE_TOPIC = "/calibration/seed_motion_state"
CONTROL_TOPIC = "/calibration/detection_control"
JOINT_TOPIC = "/joint_states"


def _open_reader(path: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    return reader, types


def _payload(data: bytes, type_name: str):
    return deserialize_message(data, get_message(type_name))


def _longest_invalid_interval(records):
    longest = None
    active = None
    for timestamp, payload in records:
        valid = payload.get("state") == "VALID"
        if not valid and active is None:
            active = [timestamp, timestamp, payload.get("reason", "")]
        elif not valid:
            active[1] = timestamp
        elif active is not None:
            if longest is None or active[1] - active[0] > longest[1] - longest[0]:
                longest = active
            active = None
    if active is not None and (
        longest is None or active[1] - active[0] > longest[1] - longest[0]
    ):
        longest = active
    return longest


def analyze(path: Path) -> dict:
    reader, types = _open_reader(path)
    diagnostics = []
    seed_states = []
    controls = []
    joints = []
    first_timestamp = None
    last_timestamp = None
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        timestamp = 1.0e-9 * timestamp_ns
        if first_timestamp is None:
            first_timestamp = timestamp
        last_timestamp = timestamp
        if topic not in {DIAGNOSTIC_TOPIC, SEED_STATE_TOPIC, CONTROL_TOPIC, JOINT_TOPIC}:
            continue
        message = _payload(serialized, types[topic])
        if topic == DIAGNOSTIC_TOPIC:
            try:
                diagnostics.append((timestamp, json.loads(message.data)))
            except json.JSONDecodeError:
                pass
        elif topic == SEED_STATE_TOPIC:
            try:
                seed_states.append((timestamp, json.loads(message.data)))
            except json.JSONDecodeError:
                pass
        elif topic == CONTROL_TOPIC:
            controls.append((timestamp, message.data))
        elif topic == JOINT_TOPIC and len(message.position) >= 6:
            joints.append((timestamp, np.asarray(message.position[:6], dtype=float)))

    origin = first_timestamp or 0.0
    states = Counter(item.get("state", "") for _, item in diagnostics)
    reasons = Counter(item.get("reason", "") for _, item in diagnostics)
    valid_lengths = np.asarray(
        [
            item["segment_length_mm"]
            for _, item in diagnostics
            if item.get("state") == "VALID"
            and item.get("segment_length_mm") is not None
        ],
        dtype=float,
    )
    guide_lengths = []
    for _, item in diagnostics:
        first = item.get("guide_first_mm")
        second = item.get("guide_second_mm")
        if first is not None and second is not None:
            guide_lengths.append(
                float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
            )
    guide_lengths = np.asarray(guide_lengths, dtype=float)
    first_invalid = next(
        (
            (timestamp - origin, item.get("state"), item.get("reason", ""))
            for timestamp, item in diagnostics
            if item.get("state") != "VALID"
        ),
        None,
    )
    longest = _longest_invalid_interval(diagnostics)
    if longest is not None:
        longest = {
            "start_s": longest[0] - origin,
            "duration_s": longest[1] - longest[0],
            "first_reason": longest[2],
        }

    seed_state_counts = Counter(
        item.get("state", "") for _, item in seed_states
    )
    seed_phase_counts = Counter(
        item.get("phase", "") for _, item in seed_states
    )
    seed_transitions = []
    previous = None
    for timestamp, item in seed_states:
        signature = (
            item.get("state"),
            item.get("phase"),
            item.get("motion_stage"),
            item.get("target"),
            item.get("failure_reason"),
        )
        if signature != previous:
            seed_transitions.append(
                {
                    "time_s": timestamp - origin,
                    "state": signature[0],
                    "phase": signature[1],
                    "motion_stage": signature[2],
                    "target": signature[3],
                    "failure_reason": signature[4],
                }
            )
            previous = signature

    maximum_joint_speed = 0.0
    total_joint_path = 0.0
    for (time_a, joint_a), (time_b, joint_b) in zip(joints[:-1], joints[1:]):
        dt = time_b - time_a
        distance = float(np.linalg.norm(joint_b - joint_a))
        total_joint_path += distance
        if dt > 1.0e-6:
            maximum_joint_speed = max(maximum_joint_speed, distance / dt)

    def statistics(values):
        if len(values) == 0:
            return None
        return {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "bag": str(path),
        "duration_s": 0.0 if last_timestamp is None else last_timestamp - origin,
        "diagnostic_frames": len(diagnostics),
        "detector_states": dict(states),
        "top_reasons": reasons.most_common(8),
        "first_invalid": first_invalid,
        "longest_invalid": longest,
        "valid_segment_length_mm": statistics(valid_lengths),
        "guide_length_mm": statistics(guide_lengths),
        "controls": [
            {"time_s": timestamp - origin, "command": command}
            for timestamp, command in controls
        ],
        "seed_state_frames": len(seed_states),
        "seed_states": dict(seed_state_counts),
        "seed_phases": dict(seed_phase_counts),
        "seed_transitions": seed_transitions,
        "joint_samples": len(joints),
        "joint_path_rad": total_joint_path,
        "maximum_joint_speed_rad_s": maximum_joint_speed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [analyze(path.resolve()) for path in args.bags]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    for result in results:
        print(f"\n=== {Path(result['bag']).name} ===")
        print(
            f"duration={result['duration_s']:.2f}s, "
            f"diagnostics={result['diagnostic_frames']}, "
            f"states={result['detector_states']}"
        )
        print(f"first_invalid={result['first_invalid']}")
        print(f"longest_invalid={result['longest_invalid']}")
        print(f"length_mm={result['valid_segment_length_mm']}")
        print(
            f"joint_path={result['joint_path_rad']:.4f}rad, "
            f"max_speed={result['maximum_joint_speed_rad_s']:.4f}rad/s"
        )
        print(
            f"seed_states={result['seed_states']}, "
            f"seed_phases={result['seed_phases']}"
        )
        for transition in result["seed_transitions"][:20]:
            print(f"  seed_transition={transition}")


if __name__ == "__main__":
    main()
