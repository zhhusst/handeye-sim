#!/usr/bin/env python3
"""
profile_viz_node.py — Gocator 2D 轮廓可视化 (独立节点)
  发布: /gocator/profile_viz  (MarkerArray, gocator_sensor 帧)
        /gocator/profile_2d   (Image, 400×600 XZ 平面图)

用法:
  ros2 run handeye_sim_bridge profile_viz
"""

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np


class ProfileVizNode(Node):
    def __init__(self):
        super().__init__('profile_viz_node')

        self.declare_parameter('interfaces.profile_topic', '/gocator/profile')
        self.declare_parameter(
            'interfaces.endpoint_topic', '/calibration/endpoints'
        )
        self.declare_parameter(
            'interfaces.target_surface_topic',
            '/calibration/target_surface_points',
        )
        self.declare_parameter(
            'endpoint_detection.guide_topic',
            '/calibration/detection_guide',
        )
        self.declare_parameter(
            'endpoint_detection.diagnostics_topic',
            '/profile_endpoint_detector/diagnostics',
        )
        self.declare_parameter(
            'interfaces.profile_viz_topic', '/gocator/profile_viz'
        )
        self.declare_parameter(
            'interfaces.profile_image_topic', '/gocator/profile_2d'
        )
        profile_topic = str(
            self.get_parameter('interfaces.profile_topic').value
        )
        endpoint_topic = str(
            self.get_parameter('interfaces.endpoint_topic').value
        )
        target_surface_topic = str(
            self.get_parameter('interfaces.target_surface_topic').value
        )
        guide_topic = str(
            self.get_parameter('endpoint_detection.guide_topic').value
        )
        diagnostics_topic = str(
            self.get_parameter(
                'endpoint_detection.diagnostics_topic'
            ).value
        )
        profile_viz_topic = str(
            self.get_parameter('interfaces.profile_viz_topic').value
        )
        profile_image_topic = str(
            self.get_parameter('interfaces.profile_image_topic').value
        )

        self.create_subscription(PointCloud2, profile_topic,
                                  self._cb, 1)
        self.create_subscription(
            PointCloud2,
            endpoint_topic,
            self._endpoint_cb,
            10,
        )
        self.create_subscription(
            PointCloud2,
            target_surface_topic,
            self._target_surface_cb,
            10,
        )
        self.create_subscription(
            PointCloud2, guide_topic, self._guide_cb, 10
        )
        self.create_subscription(
            String,
            diagnostics_topic,
            self._detector_diagnostics_cb,
            10,
        )

        self._marker_pub = self.create_publisher(
            MarkerArray, profile_viz_topic, 10)
        self._img_pub = self.create_publisher(
            Image, profile_image_topic, 10)

        self._latest = None
        self._endpoints = None
        self._target_surface = None
        self._guide = None
        self._detector_status = {}
        self._endpoints_time = None
        self._target_surface_time = None
        self.declare_parameter('visualization_rate_hz', 3.0)
        self.declare_parameter('visualization_valid_hold_s', 1.5)
        rate = max(
            0.5, float(self.get_parameter('visualization_rate_hz').value)
        )
        self._valid_hold_ns = int(
            1e9 * max(
                0.0,
                float(
                    self.get_parameter(
                        'visualization_valid_hold_s'
                    ).value
                ),
            )
        )
        self._visualization_period_ns = int(1e9 / rate)
        self._last_viz_time = None

        self.get_logger().info(
            f'Profile Viz ready — {profile_viz_topic} + {profile_image_topic}'
        )

    def _cb(self, msg):
        now = self.get_clock().now()
        if (
            self._last_viz_time is not None
            and (now - self._last_viz_time).nanoseconds
            < self._visualization_period_ns
        ):
            return
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            pts = [list(p) for p in read_points(
                msg, field_names=('x','y','z'), skip_nans=True)]
            self._latest = np.array(pts, dtype=np.float64) if pts else None
        except Exception:
            return
        if self._latest is not None and len(self._latest) > 0:
            self._last_viz_time = now
            self._publish_all()

    def _endpoint_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            values = read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
            if len(values) != 2:
                return
            if getattr(values.dtype, 'names', None):
                self._endpoints = np.column_stack(
                    tuple(
                        np.asarray(values[name], dtype=float)
                        for name in ('x', 'y', 'z')
                    )
                )
            else:
                self._endpoints = np.asarray(
                    values, dtype=float
                ).reshape(2, 3)
            self._endpoints_time = self.get_clock().now()
        except Exception:
            return

    def _target_surface_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            values = read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
            if len(values) == 0:
                return
            if getattr(values.dtype, 'names', None):
                target_surface = np.column_stack(
                    tuple(
                        np.asarray(values[name], dtype=float)
                        for name in ('x', 'y', 'z')
                    )
                )
            else:
                target_surface = np.asarray(
                    values, dtype=float
                ).reshape(-1, 3)
            if len(target_surface) == 0:
                return
            self._target_surface = target_surface
            self._target_surface_time = self.get_clock().now()
        except Exception:
            return

    def _guide_cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            values = read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
            if len(values) != 2:
                return
            if getattr(values.dtype, 'names', None):
                self._guide = np.column_stack(
                    tuple(
                        np.asarray(values[name], dtype=float)
                        for name in ('x', 'y', 'z')
                    )
                )
            else:
                self._guide = np.asarray(values, dtype=float).reshape(2, 3)
        except Exception:
            return

    def _detector_diagnostics_cb(self, msg):
        try:
            status = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(status, dict):
            self._detector_status = status

    def _is_fresh(self, received_time) -> bool:
        if received_time is None:
            return False
        return (
            self.get_clock().now() - received_time
        ).nanoseconds <= self._valid_hold_ns

    def _publish_all(self):
        pts = self._latest
        image_stamp = self.get_clock().now().to_msg()
        # These markers are a display-only rendering of the timestamped
        # calibration clouds.  A zero marker stamp asks RViz to use the latest
        # available transform.  This avoids intermittent visualization drops
        # when the real Gocator (about 60 Hz) outruns robot TF publication
        # (about 20 Hz); the original PointCloud2 stamps used by synchronization
        # and calibration remain untouched.
        marker_stamp = Time().to_msg()
        FRAME = 'gocator_sensor'

        # ── MarkerArray ──
        arr = MarkerArray()

        m0 = Marker()
        m0.header.frame_id = FRAME; m0.header.stamp = marker_stamp
        m0.ns = 'profile'; m0.id = 0
        m0.type = Marker.LINE_STRIP; m0.action = Marker.ADD
        m0.scale.x = 0.002
        m0.color.r = 1.0; m0.color.g = 0.8; m0.color.b = 0.0; m0.color.a = 1.0
        for p in pts:
            m0.points.append(Point(x=float(p[0]), y=0.0, z=float(p[2])))
        arr.markers.append(m0)

        selected = Marker()
        selected.header.frame_id = FRAME
        selected.header.stamp = marker_stamp
        selected.ns = 'target_surface'
        selected.id = 20
        if (
            self._target_surface is None
            or not self._is_fresh(self._target_surface_time)
        ):
            selected.action = Marker.DELETE
        else:
            selected.type = Marker.LINE_STRIP
            selected.action = Marker.ADD
            selected.scale.x = 0.0035
            selected.color.r = 0.1
            selected.color.g = 1.0
            selected.color.b = 0.2
            selected.color.a = 1.0
            # The selected samples lie exactly on the yellow raw profile.
            # A tiny display-only offset prevents depth-buffer flicker; the
            # published calibration cloud itself is never modified.
            selected.pose.position.y = -0.001
            selected.points = [
                Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in self._target_surface
            ]
        arr.markers.append(selected)

        guide = Marker()
        guide.header.frame_id = FRAME
        guide.header.stamp = marker_stamp
        guide.ns = 'detection_guide'
        guide.id = 30
        if self._guide is None:
            guide.action = Marker.DELETE
        else:
            guide.type = Marker.LINE_STRIP
            guide.action = Marker.ADD
            guide.scale.x = 0.0025
            guide.color.r = 0.75
            guide.color.g = 0.10
            guide.color.b = 1.0
            guide.color.a = 1.0
            guide.pose.position.y = -0.002
            guide.points = [
                Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in self._guide
            ]
        arr.markers.append(guide)

        corridor = Marker()
        corridor.header.frame_id = FRAME
        corridor.header.stamp = marker_stamp
        corridor.ns = 'detection_guide'
        corridor.id = 31
        if self._guide is None:
            corridor.action = Marker.DELETE
        else:
            first_xz = self._guide[0, (0, 2)]
            second_xz = self._guide[1, (0, 2)]
            vector = second_xz - first_xz
            length = float(np.linalg.norm(vector))
            if length <= 1e-9:
                corridor.action = Marker.DELETE
            else:
                direction = vector / length
                normal = np.array([-direction[1], direction[0]])
                normal_gate = 0.001 * float(
                    self._detector_status.get('guide_normal_gate_mm', 3.0)
                )
                endpoint_gate = 0.001 * float(
                    self._detector_status.get('guide_endpoint_gate_mm', 20.0)
                )
                start = first_xz - endpoint_gate * direction
                stop = second_xz + endpoint_gate * direction
                corners = np.asarray(
                    [
                        start + normal_gate * normal,
                        stop + normal_gate * normal,
                        stop - normal_gate * normal,
                        start - normal_gate * normal,
                        start + normal_gate * normal,
                    ]
                )
                corridor.type = Marker.LINE_STRIP
                corridor.action = Marker.ADD
                corridor.scale.x = 0.001
                corridor.color.r = 0.60
                corridor.color.g = 0.08
                corridor.color.b = 0.90
                corridor.color.a = 0.85
                corridor.pose.position.y = -0.0025
                corridor.points = [
                    Point(x=float(p[0]), y=0.0, z=float(p[1]))
                    for p in corners
                ]
        arr.markers.append(corridor)

        m1 = Marker()
        m1.header.frame_id = FRAME; m1.header.stamp = marker_stamp
        m1.ns = 'axes'; m1.id = 1
        m1.type = Marker.LINE_LIST; m1.action = Marker.ADD
        m1.scale.x = 0.003
        m1.color.r = 1.0; m1.color.g = 0.2; m1.color.b = 0.2; m1.color.a = 1.0
        m1.points = [Point(x=-0.15, y=0.0, z=0.0), Point(x=0.15, y=0.0, z=0.0)]
        arr.markers.append(m1)

        m2 = Marker()
        m2.header.frame_id = FRAME; m2.header.stamp = marker_stamp
        m2.ns = 'axes'; m2.id = 2
        m2.type = Marker.LINE_LIST; m2.action = Marker.ADD
        m2.scale.x = 0.003
        m2.color.r = 0.2; m2.color.g = 0.4; m2.color.b = 1.0; m2.color.a = 1.0
        m2.points = [Point(x=0.0, y=0.0, z=-0.05), Point(x=0.0, y=0.0, z=0.6)]
        arr.markers.append(m2)

        # Tracking ROI boxes (from detector diagnostics: roi_boxes_mm).
        # Red = ROI1, blue = ROI2; same colours as the offline visualizer.
        roi_colors = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        roi_boxes = self._detector_status.get('roi_boxes_mm', [])
        for index in range(2):
            box_marker = Marker()
            box_marker.header.frame_id = FRAME
            box_marker.header.stamp = marker_stamp
            box_marker.ns = 'tracking_roi'
            box_marker.id = 40 + index
            if index >= len(roi_boxes) or roi_boxes[index] is None:
                box_marker.action = Marker.DELETE
            else:
                box = roi_boxes[index]
                xmin = 0.001 * float(box.get('xmin_mm', 0.0))
                zmin = 0.001 * float(box.get('zmin_mm', 0.0))
                xmax = 0.001 * float(box.get('xmax_mm', 0.0))
                zmax = 0.001 * float(box.get('zmax_mm', 0.0))
                corners = [
                    (xmin, zmin), (xmax, zmin),
                    (xmax, zmax), (xmin, zmax), (xmin, zmin),
                ]
                box_marker.type = Marker.LINE_STRIP
                box_marker.action = Marker.ADD
                box_marker.scale.x = 0.002
                red, green, blue = roi_colors[index]
                box_marker.color.r = red
                box_marker.color.g = green
                box_marker.color.b = blue
                box_marker.color.a = 1.0
                box_marker.pose.position.y = -0.0015
                box_marker.points = [
                    Point(x=x, y=0.0, z=z) for x, z in corners
                ]
            arr.markers.append(box_marker)

        endpoint_colors = ((0.1, 0.9, 0.2), (0.2, 0.4, 1.0))
        for index in range(2):
            marker = Marker()
            marker.header.frame_id = FRAME
            marker.header.stamp = marker_stamp
            marker.ns = 'detected_endpoints'
            marker.id = 10 + index
            if (
                self._endpoints is None
                or not self._is_fresh(self._endpoints_time)
            ):
                marker.action = Marker.DELETE
            else:
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = 0.008
                marker.scale.y = 0.008
                marker.scale.z = 0.008
                marker.pose.position = Point(
                    x=float(self._endpoints[index, 0]),
                    y=0.0,
                    z=float(self._endpoints[index, 2]),
                )
                red, green, blue = endpoint_colors[index]
                marker.color.r = red
                marker.color.g = green
                marker.color.b = blue
                marker.color.a = 1.0
            arr.markers.append(marker)

        self._marker_pub.publish(arr)

        # ── 2D Image ──
        img_msg = self._render_2d(pts, image_stamp)
        self._img_pub.publish(img_msg)

    def _render_2d(self, pts, stamp):
        W, H = 400, 600
        X_MIN, X_MAX = -0.10, 0.10
        Z_MIN, Z_MAX = -0.05, 0.60    # -50mm~600mm (负Z=传感器后方, Z=0可见)

        img = np.ones((H, W, 3), dtype=np.uint8) * 240

        def to_px(p):
            col = int((p[0] - X_MIN) / (X_MAX - X_MIN) * (W - 1))
            row = int((Z_MAX - p[1]) / (Z_MAX - Z_MIN) * (H - 1))
            return np.clip(col, 0, W-1), np.clip(row, 0, H-1)
        # 网格线 (每 50mm Z, 每 25mm X)
        for z_mm in range(-50, 601, 50):
            z = z_mm / 1000.0
            _, r = to_px((0, z))
            if 0 <= r < H:
                img[r, :, :] = 200 if z_mm != 0 else 160  # Z=0 稍深
        for x_mm in range(-100, 101, 25):
            x = x_mm / 1000.0
            c, _ = to_px((x, 0))
            if 0 <= c < W:
                img[:, c, :] = 200 if x_mm != 0 else 160

        # Z 轴 (蓝色粗线, -50~600mm)
        c0, r_top = to_px((0, -0.05))
        _, r_bot = to_px((0, 0.60))
        img[r_bot:r_top+1, c0] = [50, 80, 200]
        # X 轴 (红色粗线, Z=0 处)
        c_l, r0 = to_px((-0.1, 0))
        c_r, _ = to_px((0.1, 0))
        img[r0, c_l:c_r+1] = [200, 40, 40]
        # Z=0 加粗标记
        if 0 <= r0 < H:
            img[r0, max(0,c0-2):min(W,c0+3)] = [50, 50, 50]

        for i in range(len(pts) - 1):
            c1, r1 = to_px((pts[i][0], pts[i][2]))
            c2, r2 = to_px((pts[i+1][0], pts[i+1][2]))
            dx = abs(c2 - c1); dy = -abs(r2 - r1)
            sx = 1 if c1 < c2 else -1; sy = 1 if r1 < r2 else -1
            err = dx + dy
            while True:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        rr, cc = r1 + dr, c1 + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            img[rr, cc] = [200, 160, 0]
                if c1 == c2 and r1 == r2:
                    break
                e2 = 2 * err
                if e2 >= dy: err += dy; c1 += sx
                if e2 <= dx: err += dx; r1 += sy

        if self._endpoints is not None:
            colors = ([20, 210, 50], [40, 80, 230])
            for endpoint, color in zip(self._endpoints, colors):
                col, row = to_px((endpoint[0], endpoint[2]))
                for dr in range(-5, 6):
                    for dc in range(-5, 6):
                        if dr * dr + dc * dc > 25:
                            continue
                        rr, cc = row + dr, col + dc
                        if 0 <= rr < H and 0 <= cc < W:
                            img[rr, cc] = color

        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = ''
        img_msg.height = H
        img_msg.width = W
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step = W * 3
        img_msg.data = img.tobytes()
        return img_msg


def main():
    rclpy.init()
    node = ProfileVizNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
