import numpy as np

from calibration_pipeline.sphere_validation import (
    SphereArtifact,
    SphereSegmentParameters,
    SphereValidationThresholds,
    select_sphere_profile_segment,
    transform_profile_to_base,
    validate_sphere_views,
)


def _sphere_30() -> SphereArtifact:
    return SphereArtifact(
        "sphere_30mm",
        diameter_m=0.030006,
        roundness_m=0.0000008,
        model="D30GZ",
    )


def test_profile_segmentation_selects_circle_and_rejects_flat_background():
    rng = np.random.default_rng(4)
    background_x = np.linspace(-0.08, -0.03, 100)
    background = np.column_stack(
        (background_x, np.zeros(100), 0.31 + 0.03 * background_x)
    )
    radius = 0.014
    angle = np.linspace(-0.72, 0.72, 90)
    circle = np.column_stack(
        (
            0.015 + radius * np.sin(angle),
            np.zeros_like(angle),
            0.23 - radius * np.cos(angle),
        )
    )
    circle[:, (0, 2)] += rng.normal(0.0, 0.00002, (len(circle), 2))
    points = np.vstack((background, circle))
    indices = np.r_[np.arange(100), np.arange(180, 270)]
    selected = select_sphere_profile_segment(
        points,
        _sphere_30(),
        sample_indices=indices,
        parameters=SphereSegmentParameters(maximum_circle_rms_m=0.0001),
    )
    assert len(selected.points_sensor_m) == len(circle)
    assert np.isclose(selected.circle.radius_m, radius, atol=0.00015)
    assert selected.circle.rms_m < 0.00005


def test_profile_to_base_uses_sensor_to_flange_then_flange_to_base():
    points = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]])
    rotation_bf = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation_bf = np.array([0.7, -0.1, 0.2])
    rotation_fs = np.diag([-1.0, -1.0, 1.0])
    translation_fs = np.array([0.01, 0.02, 0.3])
    expected = (
        rotation_bf
        @ ((rotation_fs @ points.T).T + translation_fs).T
    ).T + translation_bf
    actual = transform_profile_to_base(
        points,
        rotation_bf,
        translation_bf,
        rotation_fs,
        translation_fs,
    )
    assert np.allclose(actual, expected)


def test_multi_view_fixed_radius_report_recovers_sub_tenth_mm_noise():
    rng = np.random.default_rng(20260813)
    artifact = _sphere_30()
    center = np.array([0.83, -0.12, 0.31])
    groups = []
    for _ in range(10):
        directions = rng.normal(size=(160, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radial_noise = rng.normal(0.0, 0.000025, len(directions))
        groups.append(
            center + (artifact.radius_m + radial_noise)[:, None] * directions
        )
    report = validate_sphere_views(
        groups,
        artifact,
        thresholds=SphereValidationThresholds(),
        bootstrap_trials=10,
    )
    assert report["passed"] is True
    assert report["fixed_radius"]["all_points"]["rmse_mm"] < 0.04
    assert abs(report["free_radius_diagnostic"]["diameter_error_mm"]) < 0.02
    assert np.allclose(
        report["fixed_radius"]["center_base_m"], center, atol=0.00002
    )
    assert report["pose_bootstrap"]["successful_trials"] == 10


def test_pass_fail_uses_all_points_instead_of_hiding_a_bad_pose():
    rng = np.random.default_rng(9)
    artifact = _sphere_30()
    center = np.array([0.8, 0.0, 0.3])
    groups = []
    for pose in range(8):
        directions = rng.normal(size=(120, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        offset = 0.00045 if pose == 7 else 0.0
        groups.append(center + (artifact.radius_m + offset) * directions)
    report = validate_sphere_views(groups, artifact, bootstrap_trials=0)
    assert report["passed"] is False
    assert report["checks"]["fixed_radius_rmse"] is False
    assert report["interpretation"]["primary_metric_uses_all_selected_points"] is True


def test_real_config_freezes_both_engraved_sphere_values():
    import yaml
    from pathlib import Path

    path = Path(
        "/workspace/ros2_ws/src/fanuc_gocator_bridge/config/real_calibration.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    artifacts = payload["/**"]["ros__parameters"]["sphere_validation"]["artifacts"]
    assert artifacts["sphere_30mm"]["engraved_diameter_mm"] == 30.0060
    assert artifacts["sphere_30mm"]["engraved_roundness_mm"] == 0.0008
    assert artifacts["sphere_20mm"]["engraved_diameter_mm"] == 20.0020
    assert artifacts["sphere_20mm"]["engraved_roundness_mm"] == 0.0007
