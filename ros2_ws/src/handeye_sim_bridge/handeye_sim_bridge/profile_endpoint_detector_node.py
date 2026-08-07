#!/usr/bin/env python3
"""Extract persistent bilateral endpoints from a metric Gocator profile."""

from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger

from calibration_pipeline.perception import (
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)
from calibration_pipeline.seed_collection.endpoint_tracker import EndpointTracker


ENDPOINT_FIELDS = [
    PointField(
        name="x", offset=0, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="y", offset=4, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="z", offset=8, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="confidence", offset=12, datatype=PointField.FLOAT32, count=1
    ),
    PointField(
        name="sigma", offset=16, datatype=PointField.FLOAT32, count=1
    ),
]


def _profile_array(message: PointCloud2) -> np.ndarray:
    values = point_cloud2.read_points(
        message, field_names=("x", "y", "z"), skip_nans=True
    )
    if len(values) == 0:
        return np.zeros((0, 3), dtype=float)
    if getattr(values.dtype, "names", None):
        return np.column_stack(
            tuple(
                np.asarray(values[name], dtype=float)
                for name in ("x", "y", "z")
            )
        )
    return np.asarray(values, dtype=float).reshape(-1, 3)


class ProfileEndpointDetectorNode(Node):
    """ROS adapter kept independent of simulator truth and robot state."""

    def __init__(self) -> None:
        super().__init__("profile_endpoint_detector")
        self.declare_parameter("endpoint_detection.input_topic", "/gocator/profile")
        self.declare_parameter(
            "endpoint_detection.output_topic", "/calibration/endpoints"
        )
        self.declare_parameter("endpoint_detection.minimum_points", 12)
        self.declare_parameter("endpoint_detection.minimum_segment_points", 10)
        self.declare_parameter(
            "endpoint_detection.minimum_segment_length_m", 0.02
        )
        self.declare_parameter(
            "endpoint_detection.maximum_segment_length_m", 0.35
        )
        self.declare_parameter(
            "endpoint_detection.absolute_neighbor_gap_m", 0.004
        )
        self.declare_parameter(
            "endpoint_detection.neighbor_gap_multiplier", 8.0
        )
        self.declare_parameter(
            "endpoint_detection.residual_mad_multiplier", 3.5
        )
        self.declare_parameter(
            "endpoint_detection.residual_floor_m", 0.00008
        )
        self.declare_parameter(
            "endpoint_detection.maximum_residual_rms_m", 0.0015
        )
        self.declare_parameter(
            "endpoint_detection.endpoint_extension_fraction", 0.5
        )
        self.declare_parameter(
            "endpoint_detection.endpoint_local_fit_points", 24
        )
        self.declare_parameter(
            "endpoint_detection.candidate_ambiguity_ratio", 0.03
        )
        self.declare_parameter(
            "endpoint_detection.identity_ambiguity_ratio", 0.05
        )
        self.declare_parameter(
            "endpoint_detection.initial_first_label", "e1"
        )
        self.declare_parameter(
            "endpoint_detection.minimum_confidence", 0.25
        )

        prefix = "endpoint_detection."
        config = EndpointDetectionConfig(
            minimum_points=int(
                self.get_parameter(prefix + "minimum_points").value
            ),
            minimum_segment_points=int(
                self.get_parameter(prefix + "minimum_segment_points").value
            ),
            minimum_segment_length_m=float(
                self.get_parameter(prefix + "minimum_segment_length_m").value
            ),
            maximum_segment_length_m=float(
                self.get_parameter(prefix + "maximum_segment_length_m").value
            ),
            absolute_neighbor_gap_m=float(
                self.get_parameter(prefix + "absolute_neighbor_gap_m").value
            ),
            neighbor_gap_multiplier=float(
                self.get_parameter(prefix + "neighbor_gap_multiplier").value
            ),
            residual_mad_multiplier=float(
                self.get_parameter(prefix + "residual_mad_multiplier").value
            ),
            residual_floor_m=float(
                self.get_parameter(prefix + "residual_floor_m").value
            ),
            maximum_residual_rms_m=float(
                self.get_parameter(prefix + "maximum_residual_rms_m").value
            ),
            endpoint_extension_fraction=float(
                self.get_parameter(
                    prefix + "endpoint_extension_fraction"
                ).value
            ),
            endpoint_local_fit_points=int(
                self.get_parameter(
                    prefix + "endpoint_local_fit_points"
                ).value
            ),
            candidate_ambiguity_ratio=float(
                self.get_parameter(
                    prefix + "candidate_ambiguity_ratio"
                ).value
            ),
        )
        self.detector = ProfileEndpointDetector(config)
        self.tracker = EndpointTracker(
            ambiguity_ratio=float(
                self.get_parameter(
                    prefix + "identity_ambiguity_ratio"
                ).value
            )
        )
        self.minimum_confidence = float(
            self.get_parameter(prefix + "minimum_confidence").value
        )
        self.initial_first_label = str(
            self.get_parameter(prefix + "initial_first_label").value
        ).strip().lower()
        if self.initial_first_label not in {"e1", "e2"}:
            raise ValueError(
                "endpoint_detection.initial_first_label must be e1 or e2"
            )
        self.identity_initialized = False
        input_topic = str(
            self.get_parameter(prefix + "input_topic").value
        )
        output_topic = str(
            self.get_parameter(prefix + "output_topic").value
        )
        self.publisher = self.create_publisher(PointCloud2, output_topic, 10)
        self.diagnostic_publisher = self.create_publisher(
            String, "~/diagnostics", 10
        )
        self.create_subscription(PointCloud2, input_topic, self._callback, 10)
        self.create_service(Trigger, "~/status", self._status_callback)

        self.frames = 0
        self.accepted = 0
        self.last_status: dict[str, object] = {
            "state": "WAIT_PROFILE",
            "reason": "not_processed",
        }
        self.get_logger().info(
            f"raw-profile endpoint detector ready: {input_topic} -> "
            f"{output_topic}; no simulator truth subscription"
        )

    def _empty_output(self, message: PointCloud2) -> None:
        self.publisher.publish(
            point_cloud2.create_cloud(message.header, ENDPOINT_FIELDS, [])
        )

    def _reject(self, message: PointCloud2, reason: str) -> None:
        self.last_status = {
            "state": "REJECTED",
            "reason": reason,
            "frames": self.frames,
            "accepted": self.accepted,
        }
        self._empty_output(message)
        diagnostics = String()
        diagnostics.data = json.dumps(
            self.last_status, ensure_ascii=False, sort_keys=True
        )
        self.diagnostic_publisher.publish(diagnostics)

    def _callback(self, message: PointCloud2) -> None:
        self.frames += 1
        try:
            profile = _profile_array(message)
            detection = self.detector.detect(profile)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self._reject(message, f"invalid_profile:{error}")
            return
        if detection is None:
            self._reject(message, self.detector.last_rejection_reason)
            return
        if detection.confidence < self.minimum_confidence:
            self._reject(message, "confidence_below_threshold")
            return
        if not self.identity_initialized:
            matched = (
                (detection.first, detection.second)
                if self.initial_first_label == "e1"
                else (detection.second, detection.first)
            )
            self.tracker.reset(*matched)
            self.identity_initialized = True
        else:
            matched = self.tracker.match(detection.first, detection.second)
        if matched is None:
            self._reject(message, "ambiguous_endpoint_identity")
            return

        endpoint_u, endpoint_v = matched
        rows = [
            (
                float(endpoint_u[0]),
                0.0,
                float(endpoint_u[2]),
                float(detection.confidence),
                float(detection.endpoint_sigma_m),
            ),
            (
                float(endpoint_v[0]),
                0.0,
                float(endpoint_v[2]),
                float(detection.confidence),
                float(detection.endpoint_sigma_m),
            ),
        ]
        self.publisher.publish(
            point_cloud2.create_cloud(message.header, ENDPOINT_FIELDS, rows)
        )
        self.accepted += 1
        self.last_status = {
            "state": "VALID",
            "reason": "",
            "frames": self.frames,
            "accepted": self.accepted,
            "acceptance_rate": self.accepted / self.frames,
            "profile_points": int(len(profile)),
            "support_points": detection.support_count,
            "segment_length_mm": 1000.0 * detection.segment_length_m,
            "residual_rms_mm": 1000.0 * detection.residual_rms_m,
            "sample_pitch_mm": 1000.0 * detection.sample_pitch_m,
            "endpoint_sigma_mm": 1000.0 * detection.endpoint_sigma_m,
            "confidence": detection.confidence,
            "initial_first_label": self.initial_first_label,
        }
        diagnostics = String()
        diagnostics.data = json.dumps(
            self.last_status, ensure_ascii=False, sort_keys=True
        )
        self.diagnostic_publisher.publish(diagnostics)

    def _status_callback(self, _request, response):
        response.success = self.last_status.get("state") == "VALID"
        response.message = json.dumps(
            self.last_status, ensure_ascii=False, sort_keys=True
        )
        return response


def main() -> None:
    rclpy.init()
    node = ProfileEndpointDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
