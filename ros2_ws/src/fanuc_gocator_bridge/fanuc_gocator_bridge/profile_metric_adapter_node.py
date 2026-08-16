"""Convert raw Gocator PointCloud2 coordinates from millimetres to metres."""

from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Trigger


def validate_raw_millimetre_coordinates(
    raw_values: dict[str, list[np.ndarray]],
    maximum_abs_coordinate_mm: float,
) -> tuple[float, float]:
    """Validate finite engineering coordinates without assuming a Z origin.

    Gocator Z is relative to its configured measurement datum, so perfectly
    valid millimetre data may be arbitrarily close to zero. Unit ownership is
    established by the ``profile_raw_mm`` topic contract and the driver, not
    inferred from coordinate magnitude.
    """
    z_values = (
        np.concatenate(raw_values["z"])
        if raw_values.get("z")
        else np.empty(0)
    )
    all_values = [
        values for per_axis in raw_values.values() for values in per_axis
    ]
    if z_values.size == 0 or not all_values:
        raise ValueError("raw profile contains no finite Z value")
    median_abs_z = float(np.median(np.abs(z_values)))
    maximum_abs = max(float(np.max(np.abs(values))) for values in all_values)
    if maximum_abs > maximum_abs_coordinate_mm:
        raise ValueError(
            f"raw coordinate magnitude {maximum_abs:.3f} mm is implausible"
        )
    return median_abs_z, maximum_abs


class GocatorMetricAdapterNode(Node):
    """Preserve the entire cloud layout while scaling XYZ in-place on a copy."""

    def __init__(self) -> None:
        super().__init__("gocator_metric_adapter")
        self.declare_parameter("input_topic", "/gocator/profile_raw_mm")
        self.declare_parameter("output_topic", "/gocator/profile")
        self.declare_parameter("output_frame", "gocator_sensor")
        self.declare_parameter("coordinate_scale", 0.001)
        self.declare_parameter("coordinate_axis_sign", [1.0, 1.0, 1.0])
        self.declare_parameter("maximum_abs_coordinate_mm", 10000.0)

        self.output_frame = str(self.get_parameter("output_frame").value)
        self.scale = float(self.get_parameter("coordinate_scale").value)
        self.axis_sign = np.asarray(
            self.get_parameter("coordinate_axis_sign").value, dtype=float
        )
        self.maximum_abs_coordinate_mm = float(
            self.get_parameter("maximum_abs_coordinate_mm").value
        )
        if not 0.0 < self.scale <= 1.0:
            raise ValueError("coordinate_scale must be in (0, 1]")
        if self.axis_sign.shape != (3,) or not np.all(
            np.isin(self.axis_sign, (-1.0, 1.0))
        ):
            raise ValueError("coordinate_axis_sign must contain three +/-1 values")
        if float(np.prod(self.axis_sign)) < 0.0:
            raise ValueError(
                "coordinate_axis_sign must preserve a right-handed frame"
            )

        self.publisher = self.create_publisher(
            PointCloud2, str(self.get_parameter("output_topic").value), 10
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._callback,
            10,
        )
        self.create_service(Trigger, "~/status", self._status_callback)
        self.received_frames = 0
        self.published_frames = 0
        self.rejected_frames = 0
        self.last_error = "waiting_for_raw_profile"
        self.last_abs_z_mm = None

    @staticmethod
    def _float_field(message: PointCloud2, name: str):
        for field in message.fields:
            if field.name == name:
                if (
                    field.datatype != PointField.FLOAT32
                    or field.count != 1
                ):
                    raise ValueError(f"{name} must be one FLOAT32 field")
                return field
        raise ValueError(f"missing PointCloud2 field: {name}")

    def _callback(self, message: PointCloud2) -> None:
        self.received_frames += 1
        try:
            fields = {
                name: self._float_field(message, name)
                for name in ("x", "y", "z")
            }
            data = bytearray(message.data)
            dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
            raw_values = {name: [] for name in fields}
            for row in range(message.height):
                row_offset = row * message.row_step
                for name, field in fields.items():
                    values = np.ndarray(
                        shape=(message.width,),
                        dtype=dtype,
                        buffer=data,
                        offset=row_offset + field.offset,
                        strides=(message.point_step,),
                    )
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        raw_values[name].append(finite.copy())
            median_abs_z, _ = validate_raw_millimetre_coordinates(
                raw_values, self.maximum_abs_coordinate_mm
            )
            for row in range(message.height):
                row_offset = row * message.row_step
                for axis_index, field in enumerate(fields.values()):
                    values = np.ndarray(
                        shape=(message.width,),
                        dtype=dtype,
                        buffer=data,
                        offset=row_offset + field.offset,
                        strides=(message.point_step,),
                    )
                    values *= self.scale * self.axis_sign[axis_index]

            output = PointCloud2()
            output.header = message.header
            output.header.frame_id = self.output_frame
            output.height = message.height
            output.width = message.width
            output.fields = message.fields
            output.is_bigendian = message.is_bigendian
            output.point_step = message.point_step
            output.row_step = message.row_step
            output.data = bytes(data)
            output.is_dense = message.is_dense
            self.publisher.publish(output)
            self.published_frames += 1
            self.last_abs_z_mm = median_abs_z
            self.last_error = ""
        except (ValueError, TypeError, BufferError) as error:
            self.rejected_frames += 1
            self.last_error = str(error)
            self.get_logger().error(self.last_error)

    def _status_callback(self, _request, response):
        payload = {
            "received_frames": self.received_frames,
            "published_frames": self.published_frames,
            "rejected_frames": self.rejected_frames,
            "last_error": self.last_error,
            "last_median_abs_z_mm": self.last_abs_z_mm,
            "coordinate_scale": self.scale,
            "coordinate_axis_sign": self.axis_sign.tolist(),
        }
        response.success = self.published_frames > 0 and not self.last_error
        response.message = json.dumps(payload, ensure_ascii=False)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GocatorMetricAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
