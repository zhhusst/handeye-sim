#!/usr/bin/env python3
"""Time-synchronized bag reader shared by all offline tools.

Moved out of visualize_tracking_bag.py so replay / benchmark / visualizer all
read bags through ONE implementation (no duplicated loading code).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _pc2_points(message) -> np.ndarray:
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


class BagTimeline:
    """Load a bag into a time-synchronized frame list."""

    def __init__(self, path: Path, stride: int = 1) -> None:
        self.path = path
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        self.types = {i.name: i.type for i in reader.get_all_topics_and_types()}
        self.profile_type = get_message(
            self.types.get("/gocator/profile", "sensor_msgs/msg/PointCloud2")
        )
        self.joint_type = get_message(
            self.types.get("/joint_states", "sensor_msgs/msg/JointState")
        )
        self.flange_type = get_message(
            self.types.get("/calibration/flange_pose", "geometry_msgs/msg/PoseStamped")
        )
        self.guide_type = get_message(
            self.types.get("/calibration/detection_guide", "sensor_msgs/msg/PointCloud2")
        )
        self.endpoint_type = get_message(
            self.types.get("/calibration/endpoints", "sensor_msgs/msg/PointCloud2")
        )
        self.diag_type = get_message(
            self.types.get(
                "/profile_endpoint_detector/diagnostics", "std_msgs/msg/String"
            )
        )

        # Raw streams (time-stamped)
        self.profiles: list[tuple[int, np.ndarray]] = []
        self.joints: list[tuple[int, np.ndarray]] = []
        self.flanges: list[tuple[int, np.ndarray]] = []
        self.guides: list[tuple[int, np.ndarray]] = []
        self.endpoints: list[tuple[int, np.ndarray]] = []
        self.diags: list[tuple[int, dict]] = []

        while reader.has_next():
            topic, data, ts = reader.read_next()
            if topic == "/gocator/profile":
                self.profiles.append(
                    (ts, _pc2_points(deserialize_message(data, self.profile_type)))
                )
            elif topic == "/joint_states":
                msg = deserialize_message(data, self.joint_type)
                self.joints.append((ts, np.asarray(msg.position, dtype=float)))
            elif topic == "/calibration/flange_pose":
                msg = deserialize_message(data, self.flange_type)
                p = msg.pose.position
                q = msg.pose.orientation
                self.flanges.append(
                    (ts, np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=float))
                )
            elif topic == "/calibration/detection_guide":
                self.guides.append(
                    (ts, _pc2_points(deserialize_message(data, self.guide_type)))
                )
            elif topic == "/calibration/endpoints":
                self.endpoints.append(
                    (ts, _pc2_points(deserialize_message(data, self.endpoint_type)))
                )
            elif topic == "/profile_endpoint_detector/diagnostics":
                msg = deserialize_message(data, self.diag_type)
                try:
                    self.diags.append((ts, json.loads(msg.data)))
                except json.JSONDecodeError:
                    pass

        self.stride = max(1, stride)
        # Build frame list: every stride-th profile, synced with nearest others
        self.frames: list[dict] = []
        for idx, (ts, prof) in enumerate(self.profiles):
            if idx % self.stride:
                continue
            frame = {
                "stamp": ts,
                "profile": prof,
                "joints": self._nearest(self.joints, ts),
                "flange": self._nearest(self.flanges, ts),
                "guide": self._nearest(self.guides, ts),
                "endpoints": self._nearest(self.endpoints, ts),
                "diag": self._nearest(self.diags, ts),
            }
            self.frames.append(frame)
        self.t0 = self.frames[0]["stamp"] if self.frames else 0

    @staticmethod
    def _nearest(stream, ts):
        if not stream:
            return None
        times = np.asarray([s[0] for s in stream], dtype=np.int64)
        i = int(np.searchsorted(times, ts))
        if i >= len(stream):
            i = len(stream) - 1
        return stream[i][1]

    def __len__(self) -> int:
        return len(self.frames)

    def time_s(self, frame: dict) -> float:
        return (frame["stamp"] - self.t0) * 1e-9
