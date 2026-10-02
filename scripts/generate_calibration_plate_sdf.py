#!/usr/bin/env python3
"""Generate the Gazebo plate from the same board geometry used for profiles.

``board.corner`` is the shared corner on the *top* surface.  Local u/v span
the plate and local n points out of its top surface; thickness extends along
-n.  The source SDF supplies only appearance/material and model structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
DEFAULT_TEMPLATE = WORKSPACE / "ros2_ws/src/handeye_sim_bridge/config/calibration_plate.sdf"


def board_geometry(config_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical box centre, [u, v, thickness] size, and board rotation."""
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    board = parameters["/**"]["ros__parameters"]["board"]
    corner = np.asarray(board["corner"], dtype=float).reshape(3)
    rotation = np.asarray(board["rotation"], dtype=float).reshape(3, 3)
    size = np.asarray(
        [board["length_u_m"], board["length_v_m"], board["thickness_m"]],
        dtype=float,
    )
    if not np.all(np.isfinite(corner)) or not np.all(np.isfinite(rotation)):
        raise ValueError("board.corner and board.rotation must be finite")
    if not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("board lengths and thickness must be finite and positive")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("board.rotation must be an orthonormal rotation matrix")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("board.rotation must have determinant +1")
    center = (
        corner
        + 0.5 * size[0] * rotation[:, 0]
        + 0.5 * size[1] * rotation[:, 1]
        - 0.5 * size[2] * rotation[:, 2]
    )
    return center, size, rotation


def make_plate_sdf(config_path: Path, template_path: Path) -> ET.ElementTree:
    """Preserve template appearance while deriving every physical box value."""
    center, size, rotation = board_geometry(config_path)
    tree = ET.parse(template_path)
    link = tree.find("./model/link")
    if link is None:
        raise ValueError("plate SDF template has no model/link")
    pose = link.find("pose")
    if pose is None:
        raise ValueError("plate SDF template has no link pose")
    rpy = Rotation.from_matrix(rotation).as_euler("xyz")
    pose.text = " ".join(f"{value:.12g}" for value in (*center, *rpy))
    dimensions = " ".join(f"{value:.12g}" for value in size)
    for kind in ("visual", "collision"):
        boxes = link.findall(f"{kind}/geometry/box/size")
        if len(boxes) != 1:
            raise ValueError(f"plate SDF template needs one {kind} box")
        boxes[0].text = dimensions
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output in {args.config.resolve(), args.template.resolve()}:
        parser.error("output must not overwrite the config or SDF template")
    tree = make_plate_sdf(args.config, args.template)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    center, size, _ = board_geometry(args.config)
    print(
        f"Gazebo plate: center={center.tolist()} m, "
        f"size={size.tolist()} m -> {output}"
    )


if __name__ == "__main__":
    main()
