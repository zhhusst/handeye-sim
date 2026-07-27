import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_pipeline.simulation.synthetic import default_scene


def test_generated_urdf_is_current_and_uses_radians():
    generator_path = Path("urdf/generate_urdf.py")
    specification = importlib.util.spec_from_file_location("generate_urdf", generator_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    generated = module.gen()
    committed = Path("urdf/calib_robot.urdf").read_text(encoding="utf-8")
    assert committed == generated

    root = ET.fromstring(generated)
    limits = [joint.find("limit") for joint in root.findall("joint")]
    limits = [limit for limit in limits if limit is not None]
    assert len(limits) == 6
    assert max(abs(float(limit.attrib["upper"])) for limit in limits) <= 4.0 * 3.1416
    assert root.find("ros2_control") is not None
    plugin = root.find("gazebo/plugin")
    assert plugin is not None
    assert plugin.attrib["filename"] == "gz_ros2_control-system"

    links = {link.attrib["name"]: link for link in root.findall("link")}
    moving_children = {
        joint.find("child").attrib["link"]
        for joint in root.findall("joint")
        if joint.attrib["type"] != "fixed"
    }
    assert moving_children
    for child_name in moving_children:
        assert links[child_name].find("inertial") is not None
        assert links[child_name].find("collision") is not None


def test_simulation_truth_matches_urdf_fixed_handeye_joint():
    root = ET.fromstring(Path("urdf/calib_robot.urdf").read_text(encoding="utf-8"))
    joint = root.find("./joint[@name='fanuc_flange-gocator_sensor_joint']")
    assert joint is not None
    origin = joint.find("origin")
    assert origin is not None
    xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
    rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
    scene = default_scene()
    assert np.allclose(scene.handeye_translation, xyz)
    assert np.allclose(
        scene.handeye_rotation,
        Rotation.from_euler("xyz", rpy).as_matrix(),
    )
