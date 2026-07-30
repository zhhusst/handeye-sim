import numpy as np

from calibration_pipeline.models import SensorROI
from calibration_pipeline.seed_collection import (
    BilateralFeature,
    EndpointTracker,
    adaptive_rotation_plan,
    assess_initial_pose,
    preflight_guided_rotation_plan,
    TranslationServo,
    evaluate_bilateral_feature,
    local_preflight_is_acceptable,
    rotation_diversity,
    seed_feature_is_acceptable,
    star_rotation_plan,
)
from calibration_pipeline.geometry import rotation_x, rotation_y


def test_endpoint_tracker_keeps_physical_identity_after_order_swap():
    tracker = EndpointTracker()
    endpoint_u = np.array([-0.03, 0.0, 0.5])
    endpoint_v = np.array([0.04, 0.0, 0.51])
    tracker.reset(endpoint_u, endpoint_v)
    matched = tracker.match(endpoint_v + 1e-3, endpoint_u + 1e-3)
    assert matched is not None
    assert np.allclose(matched[0], endpoint_u + 1e-3)
    assert np.allclose(matched[1], endpoint_v + 1e-3)


def test_feature_and_translation_servo():
    roi = SensorROI(safe_margin=0.005)
    feature = evaluate_bilateral_feature(
        np.array([-0.02, 0.0, 0.5]), np.array([0.04, 0.0, 0.5]), roi
    )
    assert feature.safe
    assert np.isclose(feature.x_mid, 0.01)
    servo = TranslationServo(gain=1.0, maximum_step=0.02)
    assert servo.choose_axis({0: 0.2, 1: -2.0, 2: 0.1}) == 1
    assert np.isclose(servo.correction(feature.x_mid), 0.005)


def test_star_plan_and_rotation_diversity():
    assert len(star_rotation_plan()) == 5
    result = rotation_diversity(
        [np.eye(3), rotation_x(np.deg2rad(15.0)), rotation_y(np.deg2rad(15.0))]
    )
    assert result["minimum_pairwise_deg"] >= 14.9


def _feature(*, x_mid, z_mid, margin, depth_delta, length=0.115):
    endpoint_u = np.array(
        [x_mid - 0.5 * length, 0.0, z_mid - 0.5 * depth_delta]
    )
    endpoint_v = np.array(
        [x_mid + 0.5 * length, 0.0, z_mid + 0.5 * depth_delta]
    )
    return BilateralFeature(
        endpoint_u=endpoint_u,
        endpoint_v=endpoint_v,
        x_mid=x_mid,
        z_mid=z_mid,
        profile_length=float(np.linalg.norm(endpoint_v - endpoint_u)),
        domain_margin=margin,
        safe=True,
    )


def test_initial_pose_envelope_accepts_reference_and_rejects_observed_failures():
    limits = np.deg2rad(
        np.array(
            [
                [-185, 185],
                [-100, 160],
                [-90, 220],
                [-200, 200],
                [-180, 180],
                [-450, 450],
            ]
        )
    )
    joints = np.array([-0.236, -0.036, -0.633, -0.406, -1.05, 0.879])
    accepted = assess_initial_pose(
        _feature(
            x_mid=0.021,
            z_mid=0.328,
            margin=0.031,
            depth_delta=0.053,
        ),
        joints,
        limits,
        local_ik_directions=3,
    )
    assert accepted.accepted

    insufficient_ik = assess_initial_pose(
        _feature(
            x_mid=0.021,
            z_mid=0.328,
            margin=0.031,
            depth_delta=-0.053,
        ),
        joints,
        limits,
        local_ik_directions=2,
    )
    assert not insufficient_ik.accepted
    assert "local_ik" in insufficient_ik.reasons

    reversed_depth = assess_initial_pose(
        _feature(
            x_mid=0.001,
            z_mid=0.331,
            margin=0.028,
            depth_delta=-0.040,
        ),
        joints,
        limits,
        local_ik_directions=4,
    )
    assert reversed_depth.accepted
    assert np.isclose(reversed_depth.absolute_endpoint_depth_delta_m, 0.040)

    shallow_depth = assess_initial_pose(
        _feature(
            x_mid=-0.004,
            z_mid=0.402,
            margin=0.073,
            depth_delta=-0.010,
        ),
        joints,
        limits,
        local_ik_directions=4,
    )
    assert not shallow_depth.accepted
    assert "absolute_endpoint_depth_delta" in shallow_depth.reasons

    low_reserve = assess_initial_pose(
        _feature(
            x_mid=0.0,
            z_mid=0.304,
            margin=0.0163,
            depth_delta=0.027,
        ),
        joints,
        limits,
        local_ik_directions=4,
    )
    assert not low_reserve.accepted
    assert "domain_margin" in low_reserve.reasons


def test_adaptive_plan_adds_signed_and_reordered_fallback_branches():
    default = star_rotation_plan()
    adaptive = adaptive_rotation_plan()
    assert adaptive[: len(default)] == default
    assert len(adaptive) > len(default)
    stages = {target.stages for target in adaptive}
    assert ((0, 1), (1, -1)) in stages
    assert ((1, -1), (0, 1)) in stages


def test_preflight_guided_plan_prioritizes_measured_safe_directions():
    results = [
        {"axis": 0, "sign": -1, "accepted": True},
        {"axis": 0, "sign": 1, "accepted": False},
        {"axis": 1, "sign": 1, "accepted": True},
    ]
    plan = preflight_guided_rotation_plan(results)
    assert plan[0].stages == ((0, -1),)
    assert plan[1].stages == ((1, 1),)
    assert plan[0].angle_scale == 1.0
    assert plan[1].angle_scale == 1.0
    assert plan[2].stages == ((0, -1),)
    assert plan[2].angle_scale == 0.5
    assert plan[4].stages == ((0, -1), (1, 1))
    assert plan[4].angle_scale == 1.0
    assert ((0, 1),) in {target.stages for target in plan}


def test_partial_seed_requires_centering_and_margin():
    centered = _feature(
        x_mid=0.001,
        z_mid=0.40,
        margin=0.005,
        depth_delta=0.03,
    )
    assert seed_feature_is_acceptable(
        centered,
        maximum_abs_x_mid_m=0.003,
        minimum_domain_margin_m=0.002,
    )
    off_center = _feature(
        x_mid=-0.022,
        z_mid=0.40,
        margin=0.013,
        depth_delta=0.03,
    )
    assert not seed_feature_is_acceptable(
        off_center,
        maximum_abs_x_mid_m=0.003,
        minimum_domain_margin_m=0.002,
    )


def test_dynamic_preflight_requires_three_directions_spanning_both_axes():
    results = [
        {"axis": 0, "accepted": True},
        {"axis": 0, "accepted": True},
        {"axis": 1, "accepted": True},
        {"axis": 1, "accepted": False},
    ]
    assert local_preflight_is_acceptable(results)
    results[2]["accepted"] = False
    assert not local_preflight_is_acceptable(results)
