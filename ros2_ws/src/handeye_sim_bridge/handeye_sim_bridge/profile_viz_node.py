#!/usr/bin/env python3
"""
profile_viz_node.py — Gocator 2D 轮廓可视化 (独立节点)
  发布: /gocator/profile_viz  (MarkerArray, gocator_sensor 帧)
        /gocator/profile_2d   (Image, 400×600 XZ 平面图)

用法:
  ros2 run handeye_sim_bridge profile_viz
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np


class ProfileVizNode(Node):
    def __init__(self):
        super().__init__('profile_viz_node')

        self.create_subscription(PointCloud2, '/gocator/profile',
                                  self._cb, 1)

        self._marker_pub = self.create_publisher(
            MarkerArray, '/gocator/profile_viz', 10)
        self._img_pub = self.create_publisher(
            Image, '/gocator/profile_2d', 10)

        self._latest = None

        self.get_logger().info('Profile Viz ready — /gocator/profile_viz + /gocator/profile_2d')

    def _cb(self, msg):
        from sensor_msgs_py.point_cloud2 import read_points
        try:
            pts = [list(p) for p in read_points(
                msg, field_names=('x','y','z'), skip_nans=True)]
            self._latest = np.array(pts, dtype=np.float64) if pts else None
        except Exception:
            return
        if self._latest is not None and len(self._latest) > 0:
            self._publish_all()

    def _publish_all(self):
        pts = self._latest
        stamp = self.get_clock().now().to_msg()
        FRAME = 'gocator_sensor'

        # ── MarkerArray ──
        arr = MarkerArray()

        m0 = Marker()
        m0.header.frame_id = FRAME; m0.header.stamp = stamp
        m0.ns = 'profile'; m0.id = 0
        m0.type = Marker.LINE_STRIP; m0.action = Marker.ADD
        m0.scale.x = 0.002
        m0.color.r = 1.0; m0.color.g = 0.8; m0.color.b = 0.0; m0.color.a = 1.0
        for p in pts:
            m0.points.append(Point(x=float(p[0]), y=0.0, z=float(p[2])))
        arr.markers.append(m0)

        m1 = Marker()
        m1.header.frame_id = FRAME; m1.header.stamp = stamp
        m1.ns = 'axes'; m1.id = 1
        m1.type = Marker.LINE_LIST; m1.action = Marker.ADD
        m1.scale.x = 0.003
        m1.color.r = 1.0; m1.color.g = 0.2; m1.color.b = 0.2; m1.color.a = 1.0
        m1.points = [Point(x=-0.15, y=0.0, z=0.0), Point(x=0.15, y=0.0, z=0.0)]
        arr.markers.append(m1)

        m2 = Marker()
        m2.header.frame_id = FRAME; m2.header.stamp = stamp
        m2.ns = 'axes'; m2.id = 2
        m2.type = Marker.LINE_LIST; m2.action = Marker.ADD
        m2.scale.x = 0.003
        m2.color.r = 0.2; m2.color.g = 0.4; m2.color.b = 1.0; m2.color.a = 1.0
        m2.points = [Point(x=0.0, y=0.0, z=-0.05), Point(x=0.0, y=0.0, z=0.6)]
        arr.markers.append(m2)

        self._marker_pub.publish(arr)

        # ── 2D Image ──
        img_msg = self._render_2d(pts, stamp)
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
