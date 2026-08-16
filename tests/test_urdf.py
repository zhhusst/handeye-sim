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


def test_welding_torch_is_attached_at_validated_tcp():
    root = ET.fromstring(Path("urdf/calib_robot.urdf").read_text(encoding="utf-8"))
    links = {link.attrib["name"]: link for link in root.findall("link")}
    assert "weld_gun" in links
    visual_mesh = links["weld_gun"].find("visual/geometry/mesh")
    assert visual_mesh is not None
    assert visual_mesh.attrib["filename"].endswith("/meshes/weldgun.stl")
    assert visual_mesh.attrib["scale"] == "0.001 0.001 0.001"

    torch_joint = root.find("./joint[@name='fanuc_flange-weld_gun_joint']")
    tool0_joint = root.find("./joint[@name='fanuc_flange-tool0_joint']")
    assert torch_joint is not None
    assert tool0_joint is not None
    assert torch_joint.find("parent").attrib["link"] == "fanuc_flange"
    assert torch_joint.find("child").attrib["link"] == "weld_gun"
    assert tool0_joint.find("child").attrib["link"] == "tool0"

    torch_origin = torch_joint.find("origin")
    tool0_origin = tool0_joint.find("origin")
    assert torch_origin is not None
    assert tool0_origin is not None
    assert np.allclose(
        np.fromstring(torch_origin.attrib["xyz"], sep=" "),
        [-0.046256, -0.000142, 0.375235],
    )
    assert np.allclose(
        np.fromstring(torch_origin.attrib["rpy"], sep=" "),
        [-3.141540, -0.384130, -0.000070],
    )
    assert tool0_origin.attrib == torch_origin.attrib
