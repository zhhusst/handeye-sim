"""The YAML board geometry drives the Gazebo plate, not a hand-edited SDF."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path("/workspace")
CONFIG = ROOT / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
TEMPLATE = ROOT / "ros2_ws/src/handeye_sim_bridge/config/calibration_plate.sdf"
GENERATOR = ROOT / "scripts/generate_calibration_plate_sdf.py"

spec = importlib.util.spec_from_file_location("generate_calibration_plate_sdf", GENERATOR)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def _assert_sdf_matches_board(tree: ET.ElementTree, board: dict) -> None:
    link = tree.find("./model/link")
    pose = np.fromstring(link.findtext("pose"), sep=" ")
    visual = np.fromstring(link.findtext("visual/geometry/box/size"), sep=" ")
    collision = np.fromstring(link.findtext("collision/geometry/box/size"), sep=" ")
    corner = np.asarray(board["corner"], dtype=float)
    rotation = np.asarray(board["rotation"], dtype=float).reshape(3, 3)
    size = np.asarray(
        [board["length_u_m"], board["length_v_m"], board["thickness_m"]]
    )
    expected_center = corner + rotation @ np.array(
        [size[0] / 2.0, size[1] / 2.0, -size[2] / 2.0]
    )
    np.testing.assert_allclose(pose[:3], expected_center, atol=1e-10)
    np.testing.assert_allclose(Rotation.from_euler("xyz", pose[3:]).as_matrix(), rotation, atol=1e-10)
    np.testing.assert_allclose(visual, size)
    np.testing.assert_allclose(collision, size)


def test_generated_plate_uses_current_yaml_geometry():
    board = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["/**"]["ros__parameters"]["board"]
    _assert_sdf_matches_board(generator.make_plate_sdf(CONFIG, TEMPLATE), board)


def test_position_orientation_dimensions_can_change_only_in_yaml(tmp_path):
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    board = parameters["/**"]["ros__parameters"]["board"]
    board["corner"] = [0.5, -0.12, 0.31]
    board["rotation"] = Rotation.from_euler("xyz", [5.0, -3.0, 20.0], degrees=True).as_matrix().ravel().tolist()
    board["length_u_m"] = 0.30
    board["length_v_m"] = 0.12
    board["thickness_m"] = 0.018
    config = tmp_path / "calibration.yaml"
    config.write_text(yaml.safe_dump(parameters), encoding="utf-8")
    _assert_sdf_matches_board(generator.make_plate_sdf(config, TEMPLATE), board)


def test_launcher_spawns_generated_plate_in_both_modes():
    launcher = (ROOT / "scripts/start_simulation.sh").read_text(encoding="utf-8")
    assert "generate_calibration_plate_sdf.py" in launcher
    assert launcher.count("-file '$PLATE_SDF_PATH'") == 1
    assert launcher.count('-file "$PLATE_SDF_PATH"') == 1
