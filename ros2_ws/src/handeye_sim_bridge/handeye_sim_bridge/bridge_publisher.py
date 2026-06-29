#!/usr/bin/env python3
"""
bridge_publisher.py — 手眼标定仿真 发布节点

职责:
  1. 发布 TF: world → plate, world → sensor
  2. 发布 Marker: 平板, FOV, 扫描线, 断点
  3. 接收采集帧数据并可视化
"""

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

from handeye_sim_bridge.fanuc_transforms import matrix_to_quat
from handeye_sim_bridge.calib_visualization import (
    make_plate_marker, make_fov_marker, make_scanline_marker,
    make_edge_marker, make_intersection_line_marker,
    make_sensor_frame_marker,
    make_fov_plane_marker,
)


class CalibPublisher(Node):
    """标定场景发布器 — TF + Marker"""

    def __init__(self):
        super().__init__('calib_publisher')

        # 参数
        self.declare_parameter('half_fov_deg', 15.0)
        self.declare_parameter('max_range', 0.82)
        self.half_fov_deg = self.get_parameter('half_fov_deg').value
        self.max_range = self.get_parameter('max_range').value

        # 发布器
        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, '/calib/markers', 10)

        # 场景参数 (由 runner 设置)
        self.scene = None

        self.get_logger().info('CalibPublisher 已启动')

    def set_scene(self, C, n_B, u_B, v_B, w, h):
        """设置场景参数"""
        self.scene = {
            'C': np.asarray(C),
            'n_B': np.asarray(n_B),
            'u_B': np.asarray(u_B),
            'v_B': np.asarray(v_B),
            'w': float(w),
            'h': float(h),
        }

    def publish_scene_markers(self, stamp, frame_id="world"):
        """发布静态场景 Marker (平板 + 标签)"""
        if self.scene is None:
            return

        C = self.scene['C']
        n_B = self.scene['n_B']
        u_B = self.scene['u_B']
        v_B = self.scene['v_B']
        w = self.scene['w']
        h = self.scene['h']

        markers = MarkerArray()

        # 平板
        plate_arr = make_plate_marker(n_B, u_B, v_B, C, w, h, frame_id)
        for m in plate_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)

        self.marker_pub.publish(markers)

    def publish_all_markers(self, stamp, corners,
                            C, n_B, u_B, v_B, w, h,
                            frame_id="world", sensor_frame="gocator_sensor"):
        """一次性发布所有 Marker（平板 + FOV平面 + TF），避免多次发布导致RViz状态抖动"""
        markers = MarkerArray()

        # 平板
        plate_arr = make_plate_marker(n_B, u_B, v_B, C, w, h, frame_id)
        for m in plate_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)

        # FOV 平面
        plane_arr = make_fov_plane_marker(corners, sensor_frame)
        for m in plane_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)

        # 一次发布
        self.marker_pub.publish(markers)

        # TF — plate 在世界系的位置（单独，不影响 marker topic）
        self.publish_plate_tf(stamp, C, n_B, u_B, v_B)

    def publish_frame_markers(self, stamp, R_BS, t_BS, scan_pts_B,
                               endpoints_B, P0=None, line_dir=None,
                               frame_id="world"):
        """发布单帧可视化 (FOV + 扫描线 + 断点 + 传感器坐标系)"""
        markers = MarkerArray()

        # FOV 三角
        fov_arr = make_fov_marker(R_BS, t_BS, frame_id,
                                   half_fov_deg=self.half_fov_deg,
                                   max_range=self.max_range)
        for m in fov_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)

        # 传感器坐标系
        frame_arr = make_sensor_frame_marker(R_BS, t_BS, frame_id)
        for m in frame_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)

        # 扫描线
        scan_arr = make_scanline_marker(scan_pts_B, frame_id)
        for m in scan_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)

        # 断点
        for i, (etype, pt) in enumerate(endpoints_B):
            m = make_edge_marker(pt, etype, frame_id, marker_id=200 + i)
            m.header.stamp = stamp
            markers.markers.append(m)

        # 交线
        if P0 is not None and line_dir is not None and np.linalg.norm(line_dir) > 0:
            line_arr = make_intersection_line_marker(P0, line_dir, frame_id)
            for m in line_arr.markers:
                m.header.stamp = stamp
                markers.markers.append(m)

        self.marker_pub.publish(markers)

    def publish_tf(self, stamp, parent_frame, child_frame, translation, quat):
        """发布 TF"""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(t)

    def publish_plate_tf(self, stamp, C, n_B, u_B, v_B, frame_id="world"):
        """发布平板 TF (plate → world)"""
        # 平板坐标系: origin=C, x=u_B, y=v_B, z=n_B
        R_plate = np.column_stack([u_B, v_B, n_B])
        q, t = matrix_to_quat(np.eye(4))
        R_plate_full = np.eye(4)
        R_plate_full[:3, :3] = R_plate
        R_plate_full[:3, 3] = C
        q, _ = matrix_to_quat(R_plate_full)
        self.publish_tf(stamp, frame_id, 'plate', C, q)

    def publish_sensor_tf(self, stamp, R_BS, t_BS, frame_id="world"):
        """发布传感器 TF (sensor → world)"""
        T_BS = np.eye(4)
        T_BS[:3, :3] = R_BS
        T_BS[:3, 3] = t_BS
        q, t = matrix_to_quat(T_BS)
        self.publish_tf(stamp, frame_id, 'sensor', t, q)

    def publish_fov_plane(self, stamp, corners, frame_id="world"):
        """发布 FOV 激光平面（4个角点定义的面）"""
        markers = MarkerArray()
        plane_arr = make_fov_plane_marker(corners, frame_id)
        for m in plane_arr.markers:
            m.header.stamp = stamp
            markers.markers.append(m)
        self.marker_pub.publish(markers)
