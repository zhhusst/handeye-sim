"""Pure NumPy laser-plane/board intersection used by the ROS simulation."""

from __future__ import annotations

import numpy as np

from ..geometry import make_transform, rpy_to_matrix, so3_exp, so3_log


def compute_fov_plate_scanline(
    rotation_sensor_base,
    translation_sensor_base,
    corner,
    normal,
    u,
    v,
    width,
    height,
    half_fov_deg=15.0,
    min_range=0.27,
    max_range=0.82,
    n_sample=500,
    half_span=0.8,
    fov_corners_S=None,
):
    """Compute the visible scan segment using the original GitHub algorithm."""
    rotation_sensor_base = np.asarray(rotation_sensor_base, dtype=float)
    translation_sensor_base = np.asarray(translation_sensor_base, dtype=float)
    corner = np.asarray(corner, dtype=float)
    normal = np.asarray(normal, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    laser_normal = rotation_sensor_base[:, 1]
    line_direction = np.cross(laser_normal, normal)
    direction_norm = np.linalg.norm(line_direction)
    if direction_norm < 1e-10:
        return _empty_result()
    line_direction /= direction_norm

    plane_system = np.vstack((laser_normal, normal))
    plane_rhs = np.array(
        [
            np.dot(laser_normal, translation_sensor_base),
            np.dot(normal, corner),
        ]
    )
    try:
        line_origin = np.linalg.lstsq(
            plane_system, plane_rhs, rcond=None
        )[0]
    except np.linalg.LinAlgError:
        return _empty_result()
    line_origin += np.dot(line_direction, corner - line_origin) * line_direction

    rotation_base_sensor = rotation_sensor_base.T
    translation_base_sensor = (
        -rotation_base_sensor @ translation_sensor_base
    )

    if fov_corners_S is not None and len(fov_corners_S) >= 4:
        corners = np.asarray(fov_corners_S, dtype=float)
        if corners.shape != (4, 3):
            raise ValueError("fov_corners_S must have shape (4, 3)")
        tip_x, tip_z = corners[0, 0], corners[0, 2]
        base_z = corners[2, 2]
        left_base_x = corners[3, 0]
        right_base_x = corners[2, 0]
        fov_range_z = base_z - tip_z
        if fov_range_z <= 0.0:
            raise ValueError("calibrated FOV far edge must be beyond its window")

        def in_fov(x_coordinate, z_coordinate):
            if z_coordinate < tip_z or z_coordinate > base_z:
                return False
            fraction = (z_coordinate - tip_z) / fov_range_z
            left = tip_x + (left_base_x - tip_x) * fraction
            right = tip_x + (right_base_x - tip_x) * fraction
            return left - 1e-6 <= x_coordinate <= right + 1e-6
    else:
        tangent = np.tan(np.deg2rad(half_fov_deg))

        def in_fov(x_coordinate, z_coordinate):
            if z_coordinate < min_range or z_coordinate > max_range:
                return False
            return abs(x_coordinate) <= z_coordinate * tangent + 1e-6

    sample_parameters = np.linspace(-half_span, half_span, n_sample)
    valid_indices = []
    for index, parameter in enumerate(sample_parameters):
        point_base = line_origin + parameter * line_direction
        delta = point_base - corner
        board_u = np.dot(delta, u)
        board_v = np.dot(delta, v)
        if (
            board_u < -1e-6
            or board_v < -1e-6
            or board_u > width + 1e-6
            or board_v > height + 1e-6
        ):
            continue
        point_sensor = (
            rotation_base_sensor @ point_base + translation_base_sensor
        )
        if in_fov(point_sensor[0], point_sensor[2]):
            valid_indices.append(index)

    if len(valid_indices) < 3:
        return _empty_result(
            line_origin=line_origin, line_direction=line_direction
        )

    segments = []
    segment_start = valid_indices[0]
    for previous, current in zip(valid_indices, valid_indices[1:]):
        if current - previous > 1:
            segments.append((segment_start, previous))
            segment_start = current
    segments.append((segment_start, valid_indices[-1]))
    segment_start, segment_end = max(
        segments, key=lambda segment: segment[1] - segment[0]
    )

    scan_count = min(200, segment_end - segment_start + 1)
    sample_indices = np.linspace(
        segment_start, segment_end, scan_count, dtype=int
    )
    points_base = np.array(
        [
            line_origin + sample_parameters[index] * line_direction
            for index in sample_indices
        ]
    )
    points_sensor = np.array(
        [
            rotation_base_sensor @ point + translation_base_sensor
            for point in points_base
        ]
    )

    endpoints_base = []
    endpoints_sensor = []
    endpoint_tolerance = 0.005

    def append_endpoint(label, edge_origin, edge_direction, edge_length):
        denominator = np.dot(laser_normal, edge_direction)
        if abs(denominator) <= 1e-12:
            return
        distance = (
            np.dot(laser_normal, translation_sensor_base - edge_origin)
            / denominator
        )
        if not (
            -endpoint_tolerance
            <= distance
            <= edge_length + endpoint_tolerance
        ):
            return
        point_base = edge_origin + distance * edge_direction
        point_sensor = (
            rotation_base_sensor @ point_base + translation_base_sensor
        )
        if in_fov(point_sensor[0], point_sensor[2]):
            endpoints_base.append((label, point_base))
            endpoints_sensor.append((label, point_sensor))

    append_endpoint("e1", corner, u, width)
    append_endpoint("e1", corner + height * v, u, width)
    append_endpoint("e2", corner, v, height)
    append_endpoint("e2", corner + width * u, v, height)

    return {
        "scan_pts_B": points_base,
        "scan_pts_S": points_sensor,
        "endpoints_B": endpoints_base,
        "endpoints_S": endpoints_sensor,
        "has_intersection": True,
        "line_origin_B": line_origin,
        "line_dir": line_direction,
    }


def _empty_result(line_origin=None, line_direction=None):
    return {
        "scan_pts_B": np.zeros((0, 3)),
        "scan_pts_S": np.zeros((0, 3)),
        "endpoints_B": [],
        "endpoints_S": [],
        "has_intersection": False,
        "line_origin_B": (
            np.zeros(3) if line_origin is None else line_origin
        ),
        "line_dir": (
            np.zeros(3) if line_direction is None else line_direction
        ),
    }


def compute_fov_triangle(
    rotation_sensor_base,
    translation_sensor_base,
    half_fov_deg=15.0,
    max_range=0.82,
):
    half_width = max_range * np.tan(np.deg2rad(half_fov_deg))
    tip = np.asarray(translation_sensor_base)
    left = rotation_sensor_base @ np.array([-half_width, 0.0, max_range]) + tip
    right = rotation_sensor_base @ np.array([half_width, 0.0, max_range]) + tip
    return tip, left, right


def build_R_edge(pitch_deg, yaw_deg, x_align, n_B, u_B, v_B):
    del u_B, v_B
    sensor_z = -np.asarray(n_B, dtype=float)
    yaw = so3_exp(sensor_z * np.deg2rad(yaw_deg))
    sensor_x = yaw @ np.asarray(x_align, dtype=float)
    sensor_x -= sensor_z * (sensor_x @ sensor_z)
    sensor_x /= np.linalg.norm(sensor_x)
    sensor_y = np.cross(sensor_z, sensor_x)
    pitched = so3_exp(sensor_x * np.deg2rad(pitch_deg))
    sensor_z = pitched @ sensor_z
    sensor_y = pitched @ sensor_y
    sensor_x = np.cross(sensor_y, sensor_z)
    return np.column_stack((sensor_x, sensor_y, sensor_z))


__all__ = [
    "build_R_edge",
    "compute_fov_plate_scanline",
    "compute_fov_triangle",
    "make_transform",
    "rpy_to_matrix",
    "so3_exp",
    "so3_log",
]
