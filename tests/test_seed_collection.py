import numpy as np

from calibration_pipeline.models import SensorROI
from calibration_pipeline.seed_collection import (
    EndpointTracker,
    TranslationServo,
    evaluate_bilateral_feature,
    rotation_diversity,
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
