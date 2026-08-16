import json
from pathlib import Path

import numpy as np

from calibration_pipeline.simulation import compute_fov_plate_scanline


def test_scanline_accepts_scene_publisher_keyword_interface():
    result = compute_fov_plate_scanline(
        rotation_sensor_base=np.eye(3),
        translation_sensor_base=np.zeros(3),
        corner=np.array([-0.5, -0.5, 0.5]),
        normal=np.array([0.0, 0.0, 1.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        width=1.0,
        height=1.0,
    )

    assert set(result) == {
        "scan_pts_B",
        "scan_pts_S",
        "endpoints_B",
        "endpoints_S",
        "has_intersection",
        "line_origin_B",
        "line_dir",
    }


def test_factory_fov_places_sensor_origin_inside_laser_plane():
    path = Path(
        "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"
    )
    corners = np.asarray(
        json.loads(path.read_text(encoding="utf-8"))["fov_corners_S"],
        dtype=float,
    )
    assert np.allclose(corners[:, 1], 0.0)
    assert np.max(corners[:2, 2]) < 0.0
    assert np.min(corners[2:, 2]) > 0.0

    tip = corners[0, (0, 2)]
    left = corners[3, (0, 2)]
    right = corners[2, (0, 2)]
    triangle = np.array([tip, right, left])
    twice_area = np.sum(
        triangle[:, 0] * np.roll(triangle[:, 1], -1)
        - triangle[:, 1] * np.roll(triangle[:, 0], -1)
    )
    orientation = 1.0 if twice_area > 0.0 else -1.0
    origin_margins = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
        edge = end - start
        cross = edge[0] * -start[1] - edge[1] * -start[0]
        origin_margins.append(
            orientation * cross / np.linalg.norm(edge)
        )
    assert min(origin_margins) > 0.0


def test_github_initial_observation_pose_produces_a_scanline():
    corners = json.loads(
        Path(
            "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"
        ).read_text(encoding="utf-8")
    )["fov_corners_S"]
    rotation_sensor_base = np.array(
        [
            [-0.366, 0.817, -0.446],
            [0.815, 0.513, 0.270],
            [0.450, -0.265, -0.853],
        ]
    )
    translation_sensor_base = np.array([0.884, -0.057, 0.520])

    result = compute_fov_plate_scanline(
        rotation_sensor_base=rotation_sensor_base,
        translation_sensor_base=translation_sensor_base,
        corner=np.array([0.7, 0.0, 0.25]),
        normal=np.array([0.0, 0.0, 1.0]),
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        width=0.4,
        height=0.5,
        fov_corners_S=corners,
    )

    assert result["has_intersection"]
    assert len(result["scan_pts_S"]) > 0
    assert {label for label, _ in result["endpoints_S"]} == {"e1", "e2"}
