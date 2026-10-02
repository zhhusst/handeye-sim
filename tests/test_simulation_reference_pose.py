"""Simulation truth and automatic scene preparation remain YAML-consistent."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from calibration_pipeline.simulation.scene_truth import load_handeye_truth


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
FOV = ROOT / "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"


def load_script(name):
    location = ROOT / "scripts" / name
    specification = importlib.util.spec_from_file_location(location.stem, location)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_reference_candidates_have_dual_edge_guide_and_local_ik():
    planner = load_script("plan_simulation_reference_pose.py")
    result = planner.plan_candidates(CONFIG, FOV)
    assert result["enabled"]
    assert result["candidates"]
    candidate = result["candidates"][0]
    assert np.isclose(candidate["endpoint_spacing_m"], 0.08, atol=1e-5)
    assert abs(candidate["x_mid_m"]) < 1e-5
    assert candidate["local_ik_directions"] >= 3
    assert candidate["domain_margin_m"] >= 0.02
    minimum_cosine = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["/**"]["ros__parameters"][
        "simulation_auto_reference"
    ]["minimum_outward_sensor_y_cosine"]
    assert candidate["outward_sensor_y_cosine"] >= minimum_cosine


def test_changed_board_and_handeye_truth_change_runtime_scene(tmp_path):
    planner = load_script("plan_simulation_reference_pose.py")
    generator = load_script("generate_simulation_robot_urdf.py")
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    simulation = parameters["/**"]["ros__parameters"]
    original_corner = np.asarray(simulation["board"]["corner"], dtype=float)
    simulation["board"]["corner"] = (original_corner + [0.05, -0.04, 0.02]).tolist()
    rotation = Rotation.from_euler("xyz", [2.0, -1.5, 1.0], degrees=True).as_matrix()
    simulation["board"]["rotation"] = rotation.reshape(-1).tolist()
    original_rotation, original_translation = load_handeye_truth(CONFIG)
    changed_rotation = Rotation.from_euler("y", 7.0, degrees=True).as_matrix() @ original_rotation
    changed_translation = original_translation + [0.015, -0.02, 0.01]
    simulation["evaluation"]["handeye_rotation"] = changed_rotation.reshape(-1).tolist()
    simulation["evaluation"]["handeye_translation_m"] = changed_translation.tolist()
    changed_config = tmp_path / "calibration.yaml"
    changed_config.write_text(yaml.safe_dump(parameters), encoding="utf-8")
    runtime_urdf = generator.make_robot_urdf(changed_config, ROOT / "urdf/calib_robot.urdf")
    origin = runtime_urdf.find("./joint[@name='fanuc_flange-gocator_sensor_joint']/origin")
    assert origin is not None
    assert np.allclose(np.fromstring(origin.get("xyz"), sep=" "), changed_translation)
    assert np.allclose(
        Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix(),
        changed_rotation,
    )
    result = planner.plan_candidates(changed_config, FOV)
    assert result["candidates"]


def test_reference_switch_and_infeasible_board_are_explicit(tmp_path):
    planner = load_script("plan_simulation_reference_pose.py")
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    simulation = parameters["/**"]["ros__parameters"]
    simulation["simulation_auto_reference"]["enabled"] = False
    disabled_config = tmp_path / "disabled.yaml"
    disabled_config.write_text(yaml.safe_dump(parameters), encoding="utf-8")
    assert planner.plan_candidates(disabled_config, FOV)["enabled"] is False

    simulation["simulation_auto_reference"]["enabled"] = True
    simulation["board"]["length_u_m"] = 0.02
    simulation["board"]["length_v_m"] = 0.02
    infeasible_config = tmp_path / "infeasible.yaml"
    infeasible_config.write_text(yaml.safe_dump(parameters), encoding="utf-8")
    with pytest.raises(ValueError, match="guide length cannot intersect"):
        planner.plan_candidates(infeasible_config, FOV)
