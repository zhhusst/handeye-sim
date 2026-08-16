from pathlib import Path

import numpy as np
import yaml

from fanuc_gocator_bridge.fanuc_eip import decode_cartesian_current_position
from fanuc_gocator_bridge.profile_metric_adapter_node import (
    validate_raw_millimetre_coordinates,
)
from fanuc_m20id25_support.fanuc_kinematic import forward_kinematics_urdf
from handeye_sim_bridge.active_calibration_node import (
    DEFAULT_HAND_EYE_ROTATION,
    DEFAULT_HAND_EYE_TRANSLATION,
)
from handeye_sim_bridge.profile_endpoint_detector_node import (
    ProfileEndpointDetectorNode,
)


WORKSPACE = Path("/workspace")


def _parameters(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["/**"]["ros__parameters"]


def test_core_is_an_independent_ros_package():
    core = WORKSPACE / "ros2_ws/src/handeye_calibration_core"
    assert (core / "package.xml").exists()
    assert (core / "calibration_pipeline/solvers/twelve_dof_v2.py").exists()
    assert not (
        WORKSPACE / "ros2_ws/src/handeye_sim_bridge/calibration_pipeline"
    ).exists()


def test_real_overlay_disables_truth_and_simulated_initial_error():
    real = _parameters(
        WORKSPACE
        / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
    )
    assert real["runtime"]["backend"] == "real"
    assert real["evaluation"]["truth_available"] is False
    assert real["handeye_init_rotation_error_deg"] == 0.0
    assert real["handeye_init_translation_error_mm"] == 0.0
    assert real["sensor"]["safe_trapezoid"][0:2] == [0.07, 0.48]


def test_simulation_and_real_share_the_measurement_contract():
    common = _parameters(
        WORKSPACE / "ros2_ws/src/handeye_sim_bridge/config/calibration.yaml"
    )
    real = _parameters(
        WORKSPACE
        / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
    )
    keys = {
        "joint_state_topic",
        "profile_topic",
        "target_surface_topic",
        "endpoint_topic",
        "detection_prior_topic",
        "detection_control_topic",
        "flange_pose_topic",
        "base_frame",
        "flange_frame",
        "sensor_frame",
        "joint_names",
    }
    assert {
        key: common["interfaces"][key] for key in keys
    } == {key: real["interfaces"][key] for key in keys}


def test_real_joint_output_uses_validated_j23_convention():
    payload = yaml.safe_load(
        (
            WORKSPACE
            / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    parameters = payload["fanuc_joint_state"]["ros__parameters"]
    assert parameters["j23_validated"] is True
    assert parameters["j23_factor"] == 1.0


def test_real_profile_axis_normalization_matches_nominal_handeye_frame():
    payload = yaml.safe_load(
        (
            WORKSPACE
            / "ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    parameters = payload["gocator_metric_adapter"]["ros__parameters"]
    signs = np.asarray(parameters["coordinate_axis_sign"], dtype=float)
    assert np.array_equal(signs, [1.0, -1.0, -1.0])
    assert np.isclose(np.prod(signs), 1.0)


def test_raw_millimetre_profile_may_legitimately_cross_zero_z():
    raw = {
        "x": [np.array([-20.0, 0.0, 20.0])],
        "y": [np.zeros(3)],
        "z": [np.array([-0.4, 0.0, 0.6])],
    }
    median_abs_z, maximum_abs = validate_raw_millimetre_coordinates(
        raw, 10000.0
    )
    assert np.isclose(median_abs_z, 0.4)
    assert np.isclose(maximum_abs, 20.0)


def test_nominal_handeye_and_fk_are_finite():
    rotation = np.asarray(DEFAULT_HAND_EYE_ROTATION).reshape(3, 3)
    translation = np.asarray(DEFAULT_HAND_EYE_TRANSLATION)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9)
    assert translation.shape == (3,)
    transform = forward_kinematics_urdf(np.zeros(6))
    assert transform.shape == (4, 4)
    assert np.all(np.isfinite(transform))


def test_fanuc_curpos_decoder_uses_little_endian_u16_and_float32():
    import struct

    values = list((513).to_bytes(2, "little"))
    values += list((258).to_bytes(2, "little"))
    for value in (100.25, -200.5, 300.75, 10.0, -20.0, 30.0):
        values += list(struct.pack("<f", value))
    decoded = decode_cartesian_current_position(values)
    assert decoded["utool"] == 513
    assert decoded["uframe"] == 258
    assert np.isclose(decoded["x_mm"], 100.25)
    assert np.isclose(decoded["r_deg"], 30.0)


def test_new_source_has_no_runtime_dependency_on_legacy_tree():
    roots = [
        WORKSPACE / "ros2_ws/src/fanuc_gocator_bridge",
        WORKSPACE / "ros2_ws/src/gocator_profile_driver",
        WORKSPACE / "ros2_ws/src/gocator_msgs",
        WORKSPACE / "ros2_ws/src/handeye_calibration_core",
        WORKSPACE / "ros2_ws/src/fanuc_m20id25_support",
    ]
    forbidden = ("/workspace/welding_robopath", "/home/ws/src")
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {
                ".py",
                ".cpp",
                ".hpp",
                ".h",
                ".xml",
                ".yaml",
                ".txt",
            }:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            assert not any(value in content for value in forbidden), path


def _recovery_geometry_node():
    node = object.__new__(ProfileEndpointDetectorNode)
    node.mode = "LOST"
    node.lost_from_mode = "TRACK"
    node.temporal_suspended = True
    node.guide_endpoints = np.array(
        [[-0.030, 0.0, 0.270], [0.030, 0.0, 0.270]], dtype=float
    )
    node.reacquire_maximum_length_change = 0.020
    node.reacquire_maximum_angle_change = 20.0
    return node


def test_local_reacquisition_accepts_continuous_target_geometry():
    node = _recovery_geometry_node()
    measured = np.array(
        [[-0.036, 0.0, 0.273], [0.036, 0.0, 0.273]], dtype=float
    )
    assert node._reacquisition_geometry_is_continuous(measured)


def test_local_reacquisition_rejects_workbench_length_jump():
    node = _recovery_geometry_node()
    # Mirrors the real failure: approximately 58 mm jumped to 101 mm.
    measured = np.array(
        [[-0.0505, 0.0, 0.270], [0.0505, 0.0, 0.270]], dtype=float
    )
    assert not node._reacquisition_geometry_is_continuous(measured)
