#!/usr/bin/env python3
"""Plot metric profile and recorded endpoint/guide snapshots from one bag."""

from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2


def cloud_xz(message) -> np.ndarray:
    names = {field.name for field in message.fields}
    if not {"x", "z"} <= names:
        return np.empty((0, 2))
    values = point_cloud2.read_points_numpy(
        message, field_names=["x", "z"], skip_nans=True
    )
    return np.asarray(values, dtype=float).reshape(-1, 2)


def nearest(records, stamp):
    times = [item[0] for item in records]
    index = bisect.bisect_left(times, stamp)
    choices = [i for i in (index - 1, index) if 0 <= i < len(records)]
    if not choices:
        return None
    return records[min(choices, key=lambda i: abs(times[i] - stamp))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--times", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--detrend", action="store_true",
        help="plot Z residual after subtracting a robust global TLS line",
    )
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    classes = {name: get_message(kind) for name, kind in types.items()}
    wanted = {
        "/gocator/profile",
        "/calibration/endpoints",
        "/calibration/detection_guide",
    }
    records = {name: [] for name in wanted}
    first_stamp = None
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if first_stamp is None:
            first_stamp = stamp
        if topic not in wanted:
            continue
        message = deserialize_message(data, classes[topic])
        records[topic].append((stamp, cloud_xz(message)))

    rows = int(np.ceil(len(args.times) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(15, 4.5 * rows), squeeze=False)
    for axis, seconds in zip(axes.flat, args.times):
        stamp = first_stamp + int(seconds * 1e9)
        profile_item = nearest(records["/gocator/profile"], stamp)
        endpoint_item = nearest(records["/calibration/endpoints"], stamp)
        guide_item = nearest(records["/calibration/detection_guide"], stamp)
        if profile_item and len(profile_item[1]):
            p = profile_item[1] * 1000.0
            if args.detrend:
                design = np.column_stack((p[:, 0], np.ones(len(p))))
                slope, offset = np.linalg.lstsq(design, p[:, 1], rcond=None)[0]
                p[:, 1] -= slope * p[:, 0] + offset
            axis.plot(p[:, 0], p[:, 1], ".-", color="#d39b00", ms=1, lw=0.35)
        if guide_item and len(guide_item[1]):
            g = guide_item[1] * 1000.0
            if args.detrend and profile_item and len(profile_item[1]):
                g[:, 1] -= slope * g[:, 0] + offset
            axis.plot(g[:, 0], g[:, 1], "o-", color="#8a2be2", lw=1.5, ms=4, label="guide")
        if endpoint_item and len(endpoint_item[1]):
            e = endpoint_item[1] * 1000.0
            if args.detrend and profile_item and len(profile_item[1]):
                e[:, 1] -= slope * e[:, 0] + offset
            axis.plot(e[:, 0], e[:, 1], "rx", ms=9, mew=2, label="output")
        axis.set_title(f"t={seconds:.3f} s")
        axis.set_xlabel("sensor X (mm)")
        axis.set_ylabel("detrended Z (mm)" if args.detrend else "sensor Z (mm)")
        if not args.detrend:
            axis.axis("equal")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    for axis in axes.flat[len(args.times):]:
        axis.set_visible(False)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
