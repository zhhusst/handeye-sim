"""Regression checks for the current simulator and real-hardware interfaces."""

from pathlib import Path
import json
import importlib.util

import numpy as np
import pytest
import yaml

from calibration_pipeline.geometry import make_transform
from calibration_pipeline.simulation.scanline import compute_fov_plate_scanline
from calibration_pipeline.simulation.scene_truth import (
    HAND_EYE_ROTATION,
    HAND_EYE_TRANSLATION,
)
from calibration_pipeline.roi_tracking import (
    ROIBreakpointPipeline,
    ROIBreakpointPipelineConfig,
)
from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf


ROOT = Path("/workspace")
SIM_CONFIG = ROOT / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
REAL_CONFIG = ROOT / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"


def _parameters(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["/**"]["ros__parameters"]


def _default_board(board: dict) -> bool:
    return bool(
        np.allclose(board["corner"], [0.7, 0.0, 0.25])
        and np.allclose(board["rotation"], np.eye(3).ravel())
        and np.isclose(board["length_u_m"], 0.15)
        and np.isclose(board["length_v_m"], 0.20)
    )


def _reference_joints(sim: dict) -> np.ndarray:
    if sim.get("simulation_auto_reference", {}).get("enabled", False):
        path = ROOT / "scripts/plan_simulation_reference_pose.py"
        specification = importlib.util.spec_from_file_location("plan_simulation_reference_pose", path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        fov_path = ROOT / "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"
        return np.asarray(module.plan_candidates(SIM_CONFIG, fov_path)["candidates"][0]["joints_rad"])
    return np.array([-0.236623, -0.049108, -0.636688, -0.402198, -1.071110, 0.878050])


def test_initialization_settings_match_real_hardware():
    sim = _parameters(SIM_CONFIG)
    real = _parameters(REAL_CONFIG)
    for key in ("handeye_init_rotation_error_deg", "handeye_init_translation_error_mm"):
        assert sim[key] == real[key]
    # Seed motion and servo settings are deliberately varied in simulation
    # experiments; they are not required to equal the real-hardware settings.
    assert sim["solver"]["multistart"]["enabled"] is True
    assert real["solver"]["multistart"]["enabled"] is True


def test_simulation_and_real_use_same_roi_detector_without_extra_endpoint_noise():
    sim = _parameters(SIM_CONFIG)
    real = _parameters(REAL_CONFIG)
    assert sim["endpoint_detection"]["backend"] == "roi"
    assert real["endpoint_detection"]["backend"] == "roi"
    assert sim["endpoint_detection"]["roi_process_every_n_frames"] == 4
    assert real["endpoint_detection"]["roi_process_every_n_frames"] == 4
    assert "simulation_localization_std_m" not in sim["endpoint_detection"]
    assert sim["endpoint_detection"]["plate_growth_residual_threshold_m"] == 0.003
    assert real["endpoint_detection"]["plate_growth_residual_threshold_m"] == 0.0008


def test_default_board_keeps_scripted_initial_dual_edge_view():
    sim = _parameters(SIM_CONFIG)
    board = sim["board"]
    corner = np.asarray(board["corner"], dtype=float)
    if not _default_board(board):
        pytest.skip("board-geometry experiment does not use the scripted default initial pose")

    joints = _reference_joints(sim)
    sensor = forward_kinematics_urdf(joints) @ make_transform(
        HAND_EYE_ROTATION, HAND_EYE_TRANSLATION
    )
    fov_path = ROOT / "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"
    fov = json.loads(fov_path.read_text(encoding="utf-8"))["fov_corners_S"]
    observed = compute_fov_plate_scanline(
        sensor[:3, :3], sensor[:3, 3], corner, np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        board["length_u_m"], board["length_v_m"], fov_corners_S=fov,
    )
    assert observed["has_intersection"]
    assert [name for name, _ in observed["endpoints_B"]] == ["e1", "e2"]
    endpoints = np.asarray([point for _, point in observed["endpoints_S"]])
    assert 0.070 <= np.linalg.norm(endpoints[1] - endpoints[0]) <= 0.090
    assert abs(np.mean(endpoints[:, 0])) <= 0.010


def test_simulated_noisy_profile_can_enter_real_roi_tracking_path():
    sim = _parameters(SIM_CONFIG)
    detection = sim["endpoint_detection"]
    board = sim["board"]
    if not _default_board(board):
        pytest.skip("board-geometry experiment needs a new aligned reference pose")
    joints = _reference_joints(sim)
    sensor = forward_kinematics_urdf(joints) @ make_transform(
        HAND_EYE_ROTATION, HAND_EYE_TRANSLATION
    )
    fov_path = ROOT / "ros2_ws/src/handeye_sim_backend/config/fov_factory_calib.json"
    fov = json.loads(fov_path.read_text(encoding="utf-8"))["fov_corners_S"]
    observed = compute_fov_plate_scanline(
        sensor[:3, :3], sensor[:3, 3], np.asarray(board["corner"]),
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), board["length_u_m"], board["length_v_m"],
        fov_corners_S=fov,
    )
    config = ROIBreakpointPipelineConfig(
        initial_first_label=detection["initial_first_label"],
        alignment_template_center_x_m=detection["alignment_template_center_x_m"],
        alignment_template_center_z_m=detection["alignment_template_center_z_m"],
        alignment_template_length_m=detection["alignment_template_length_m"],
        alignment_template_angle_deg=detection["alignment_template_angle_deg"],
        alignment_normal_gate_m=detection["alignment_normal_gate_m"],
        alignment_endpoint_gate_m=detection["alignment_endpoint_gate_m"],
        minimum_lock_frames=detection["minimum_lock_frames"],
        maximum_segment_length_m=detection["maximum_segment_length_m"],
        plate_residual_threshold_m=detection["plate_growth_residual_threshold_m"],
        surface_residual_threshold_m=detection["roi_surface_residual_threshold_m"],
    )
    pipeline = ROIBreakpointPipeline(config)
    rng = np.random.default_rng(17)
    for index in range(config.minimum_lock_frames):
        profile = np.asarray(observed["scan_pts_S"]).copy()
        profile[:, [0, 2]] += rng.normal(
            0.0, sim["simulation_noise"]["profile_gaussian_std_m"],
            (len(profile), 2),
        )
        result = pipeline.process_profile(profile, 0.05 * index)
        assert result.state == "VALID", result.reason
    assert pipeline.lock()
    for index in range(3):
        profile = np.asarray(observed["scan_pts_S"]).copy()
        profile[:, [0, 2]] += rng.normal(
            0.0, sim["simulation_noise"]["profile_gaussian_std_m"],
            (len(profile), 2),
        )
        profile[:, 0] += 0.0005 * (index + 1)
        result = pipeline.process_profile(profile, 0.05 * (index + config.minimum_lock_frames))
        assert result.state == "VALID", result.reason
