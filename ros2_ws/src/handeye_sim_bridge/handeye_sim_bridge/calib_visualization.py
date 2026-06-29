"""
calib_visualization.py — RViz 可视化辅助函数

将标定场景中的几何对象转换为 RViz Marker 消息,
用于实时 3D 显示。
"""

import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from handeye_sim_bridge.fanuc_transforms import matrix_to_quat


def _p(x, y, z):
    """创建 Point, 兼容 ROS2 Jazzy 的 API (不接受 *args)"""
    p = Point()
    p.x = float(x); p.y = float(y); p.z = float(z)
    return p


def _pt_arr(pts):
    """将 numpy 数组转成 Point 列表"""
    return [_p(p[0], p[1], p[2]) for p in pts]


def make_plate_marker(n_B, u_B, v_B, C, w, h, frame_id="world",
                       marker_id=100, color=None):
    """平板 Marker (半透明矩形 + 边框)"""
    if color is None:
        color = ColorRGBA(r=float(0.3), g=float(0.6), b=float(0.9), a=float(0.4))

    corners = np.array([
        C,
        C + w * u_B,
        C + w * u_B + h * v_B,
        C + h * v_B,
    ])

    arr = MarkerArray()
    for i, tri_indices in enumerate([(0, 1, 2), (0, 2, 3)]):
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = "plate"
        m.id = marker_id + i
        m.type = Marker.TRIANGLE_LIST
        m.action = Marker.ADD
        m.scale.x = 1.0; m.scale.y = 1.0; m.scale.z = 1.0
        m.color = color
        m.points = _pt_arr(corners[list(tri_indices)])
        arr.markers.append(m)

    # 边框
    edge_m = Marker()
    edge_m.header.frame_id = frame_id
    edge_m.ns = "plate_edge"
    edge_m.id = marker_id + 10
    edge_m.type = Marker.LINE_STRIP
    edge_m.action = Marker.ADD
    edge_m.scale.x = 0.002
    edge_m.color = ColorRGBA(r=float(0.2), g=float(0.5), b=float(0.8), a=float(0.8))
    edge_m.points = _pt_arr(np.vstack([corners, corners[0:1]]))
    arr.markers.append(edge_m)

    return arr


def make_edge_marker(endpoint_B, edge_type, frame_id="world", marker_id=200):
    """边断点标记 (小球)"""
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "endpoints"
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = 0.008; m.scale.y = 0.008; m.scale.z = 0.008
    if edge_type == 'e1':
        m.color = ColorRGBA(r=float(1.0), g=float(0.2), b=float(0.2), a=float(1.0))
    else:
        m.color = ColorRGBA(r=float(0.2), g=float(0.2), b=float(1.0), a=float(1.0))
    m.pose.position.x = float(endpoint_B[0])
    m.pose.position.y = float(endpoint_B[1])
    m.pose.position.z = float(endpoint_B[2])
    return m


def make_fov_marker(R_BS, t_BS, frame_id="world", marker_id=300,
                     half_fov_deg=15.0, max_range=0.82):
    """FOV 三角锥 Marker (半透明面 + 线框)"""
    tip = t_BS
    fov_rad = np.deg2rad(half_fov_deg)
    x_fov = max_range * np.tan(fov_rad)

    base_pts = [
        R_BS @ np.array([-x_fov, 0, max_range]) + t_BS,
        R_BS @ np.array([x_fov, 0, max_range]) + t_BS,
    ]

    arr = MarkerArray()

    # 半透明三角形面 (便于 Publish Point 点击)
    surf = Marker()
    surf.header.frame_id = frame_id
    surf.ns = "fov_surface"
    surf.id = marker_id + 10
    surf.type = Marker.TRIANGLE_LIST
    surf.action = Marker.ADD
    surf.scale.x = 1.0; surf.scale.y = 1.0; surf.scale.z = 1.0
    surf.color = ColorRGBA(r=float(0.2), g=float(1.0), b=float(0.3), a=float(0.12))
    surf.points = _pt_arr(np.array([tip, base_pts[0], base_pts[1]]))
    arr.markers.append(surf)

    # 边框线 (清晰可见)
    for i, (start, end) in enumerate([
        (tip, base_pts[0]), (tip, base_pts[1]), (base_pts[0], base_pts[1])
    ]):
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = "fov"
        m.id = marker_id + i
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.003
        m.color = ColorRGBA(r=float(0.0), g=float(0.9), b=float(0.0), a=float(0.9))
        m.points = _pt_arr(np.array([start, end]))
        arr.markers.append(m)

    return arr


def make_fov_plane_marker(corners_B, frame_id="world", marker_id=800):
    """FOV 激光平面 = 4个角点定义的矩形（半透明面+边框）"""
    arr = MarkerArray()

    # 半透明面 (TRIANGLE_LIST: 两个三角形)
    surf = Marker()
    surf.header.frame_id = frame_id
    surf.ns = "fov_plane_surface"
    surf.id = marker_id
    surf.type = Marker.TRIANGLE_LIST
    surf.action = Marker.ADD
    surf.scale.x = 1.0; surf.scale.y = 1.0; surf.scale.z = 1.0
    surf.color = ColorRGBA(r=float(0.0), g=float(1.0), b=float(0.3), a=float(0.12))
    # 两个三角形: (0,1,2) 和 (0,2,3)
    surf.points = _pt_arr(np.array([
        corners_B[0], corners_B[1], corners_B[2],
        corners_B[0], corners_B[2], corners_B[3],
    ]))
    arr.markers.append(surf)

    # 边框
    edge = Marker()
    edge.header.frame_id = frame_id
    edge.ns = "fov_plane_edge"
    edge.id = marker_id + 10
    edge.type = Marker.LINE_STRIP
    edge.action = Marker.ADD
    edge.scale.x = 0.003
    edge.color = ColorRGBA(r=float(0.0), g=float(1.0), b=float(0.3), a=float(0.9))
    edge.points = _pt_arr(np.array([
        corners_B[0], corners_B[1], corners_B[2],
        corners_B[3], corners_B[0],
    ]))
    arr.markers.append(edge)
    return arr


def make_scanline_marker(scan_pts_B, frame_id="world", marker_id=400):
    """扫描线点云 Marker"""
    if len(scan_pts_B) == 0:
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = "scanline"
        m.id = marker_id
        m.action = Marker.DELETE
        return MarkerArray(markers=[m])

    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "scanline"
    m.id = marker_id
    m.type = Marker.POINTS
    m.action = Marker.ADD
    m.scale.x = 0.003; m.scale.y = 0.003
    m.color = ColorRGBA(r=float(0.0), g=float(1.0), b=float(1.0), a=float(0.8))
    m.points = _pt_arr(scan_pts_B)
    return MarkerArray(markers=[m])


def make_intersection_line_marker(P0, line_dir, frame_id="world", marker_id=500):
    """激光平面∩平板交线 (长虚线)"""
    half_len = 0.3
    p1 = P0 - half_len * line_dir
    p2 = P0 + half_len * line_dir

    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "intersection_line"
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = 0.0015
    m.color = ColorRGBA(r=float(1.0), g=float(0.8), b=float(0.0), a=float(0.4))
    m.points = _pt_arr(np.array([p1, p2]))
    return MarkerArray(markers=[m])


def make_sensor_frame_marker(R_BS, t_BS, frame_id="world", marker_id=600):
    """传感器坐标系三轴标记 (RGB = XYZ)"""
    axis_len = 0.05
    arr = MarkerArray()
    for i, (axis, color_rgb) in enumerate([
        (R_BS[:, 0], (1, 0, 0)),  # X
        (R_BS[:, 1], (0, 1, 0)),  # Y
        (R_BS[:, 2], (0, 0, 1)),  # Z
    ]):
        end = t_BS + axis_len * axis
        m = Marker()
        m.header.frame_id = frame_id
        m.ns = "sensor_frame"
        m.id = marker_id + i
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.002; m.scale.y = 0.004
        m.color = ColorRGBA(r=float(color_rgb[0]), g=float(color_rgb[1]), b=float(color_rgb[2]), a=float(1.0))
        m.points = _pt_arr(np.array([t_BS, end]))
        arr.markers.append(m)
    return arr


def make_text_marker(text, position, frame_id="world", marker_id=700,
                      color=None, scale=0.04):
    """3D 文字 Marker"""
    if color is None:
        color = ColorRGBA(r=float(1.0), g=float(1.0), b=float(1.0), a=float(1.0))
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = "labels"
    m.id = marker_id
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(position[0])
    m.pose.position.y = float(position[1])
    m.pose.position.z = float(position[2])
    m.scale.z = scale
    m.color = color
    m.text = text
    return MarkerArray(markers=[m])
