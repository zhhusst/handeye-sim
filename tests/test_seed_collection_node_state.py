from types import SimpleNamespace

import numpy as np

from handeye_sim_bridge.seed_collection_node import SeedCollectionNode
from handeye_sim_bridge.profile_endpoint_detector_node import (
    ProfileEndpointDetectorNode,
)
from calibration_pipeline.perception import DualEndpointKalmanTracker
from calibration_pipeline.perception import (
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


def test_automatic_start_keeps_temporal_tracking_off_for_reference_capture():
    node = object.__new__(SeedCollectionNode)
    node.state = "WAIT_MANUAL_INIT"
    node.collection_mode = "automatic"
    node.started = False
    controls = []
    node._publish_detection_control = controls.append
    response = SimpleNamespace(success=False, message="")

    SeedCollectionNode._start_callback(node, None, response)

    assert response.success
    assert node.started
    assert controls == ["SEED_TRACK_STOP"]
    assert "reference before" in response.message


def _reference_completion_node(*, preflight_required):
    node = object.__new__(SeedCollectionNode)
    node.seed_capture_continuation = "REFERENCE"
    node.seed_capture_label = "reference"
    node.seed_capture_frames = [object()]
    node.seed_capture_filter_counts = {"accepted_frames": 18}
    node.seed_capture_last_filter_counts = {}
    node.preflight_was_required = preflight_required
    node.collection_phase = "REFERENCE"
    controls = []
    returns = []
    node._publish_detection_control = controls.append
    node._return_reference = lambda: returns.append(True)
    node._fail = lambda reason: (_ for _ in ()).throw(AssertionError(reason))
    node.get_logger = lambda: _Logger()
    return node, controls, returns


def test_reference_success_enables_tracking_before_optional_preflight():
    node, controls, returns = _reference_completion_node(
        preflight_required=True
    )

    SeedCollectionNode._complete_seed_capture(node, True, "")

    assert controls == ["SEED_TRACK_START"]
    assert node.collection_phase == "PREFLIGHT"
    assert returns == [True]


def test_reference_success_enables_tracking_before_seed_motion():
    node, controls, returns = _reference_completion_node(
        preflight_required=False
    )

    SeedCollectionNode._complete_seed_capture(node, True, "")

    assert controls == ["SEED_TRACK_START"]
    assert node.collection_phase == "COLLECT"
    assert returns == [True]


def test_completed_preflight_starts_motion_without_recapturing_reference():
    node = object.__new__(SeedCollectionNode)
    node.collection_phase = "PREFLIGHT"
    node.preflight_results = [
        {"name": "rx_positive", "axis": 0, "sign": 1, "accepted": True},
        {"name": "ry_negative", "axis": 1, "sign": -1, "accepted": True},
    ]
    node.preflight_plan = (object(), object())
    node.preflight_index = len(node.preflight_plan)
    node.preflight_minimum_directions = 2
    node.records = [object()]
    node.target_count = 6
    node.target_index = 0
    node.latest_joints = np.zeros(6)
    feature = SimpleNamespace(safe=True)
    node._feature = lambda: feature
    node._remember_last_valid = lambda _feature: None
    node._reset_servo = lambda: None
    issued = []
    node._issue_micro_rotation = lambda: issued.append(True)
    node._publish_detection_control = lambda _command: None
    node._finish = lambda: (_ for _ in ()).throw(
        AssertionError("collection finished unexpectedly")
    )
    node._fail = lambda reason: (_ for _ in ()).throw(AssertionError(reason))
    node._begin_seed_capture = lambda *_args: (_ for _ in ()).throw(
        AssertionError("reference batch must not be captured twice")
    )
    node.get_logger = lambda: _Logger()

    SeedCollectionNode._after_return_reference(node)

    assert node.collection_phase == "COLLECT"
    assert issued == [True]


def test_measured_rollback_clears_every_temporal_loss_flag():
    node = object.__new__(ProfileEndpointDetectorNode)
    node.temporal_tracking_enabled = True
    node.temporal_tracker = DualEndpointKalmanTracker()
    node.last_profile_time_s = 3.0
    node.temporal_minimum_search_radius = 0.0015
    node.temporal_prediction_this_frame = True
    node.temporal_last_mahalanobis = (100.0, 100.0)
    node.temporal_search_radius = 0.025
    node.temporal_search_radii = np.array([0.025, 0.025])
    node.temporal_suspended = True
    node.temporal_fallback_reason = "maximum_coast_frames_exceeded"
    measured = np.array(
        [[-0.04, 0.0, 0.28], [0.04, 0.0, 0.31]], dtype=float
    )

    ProfileEndpointDetectorNode._restore_temporal_from_measurement(
        node, measured
    )

    assert node.temporal_tracker.initialized
    np.testing.assert_allclose(node.temporal_tracker.endpoints(), measured)
    np.testing.assert_allclose(node.guide_endpoints, measured)
    assert not node.temporal_suspended
    assert not node.temporal_prediction_this_frame
    assert node.temporal_fallback_reason == ""
    assert node.temporal_tracker.missed_frames == 0
    np.testing.assert_allclose(node.temporal_search_radii, [0.0015, 0.0015])


def _topology_node():
    node = object.__new__(ProfileEndpointDetectorNode)
    node.detector = ProfileEndpointDetector(
        EndpointDetectionConfig(
            minimum_segment_length_m=0.010,
            maximum_segment_length_m=0.250,
        )
    )
    node.tracking_maximum_endpoint_step = 0.003
    node.tracking_minimum_reference_length_ratio = 0.65
    node.tracking_maximum_reference_length_ratio = 1.60
    node.tracking_reference_length = 0.080
    return node


def test_tracking_topology_accepts_small_continuous_endpoint_motion():
    node = _topology_node()
    predicted = np.array(
        [[-0.040, 0.0, 0.280], [0.040, 0.0, 0.310]]
    )
    measured = predicted + np.array(
        [[0.0010, 0.0, -0.0005], [0.0012, 0.0, -0.0004]]
    )

    assert ProfileEndpointDetectorNode._tracking_pair_is_plausible(
        node, measured, predicted
    )


def test_tracking_topology_rejects_gradual_collapse_below_locked_geometry():
    node = _topology_node()
    # This frame is locally smooth (only 1 mm per endpoint), but the filter
    # has already drifted toward a 42 mm internal span. The locked 80 mm pair
    # proves that accepting another inward step would preserve the wrong edge.
    predicted = np.array(
        [[-0.021, 0.0, 0.290], [0.021, 0.0, 0.300]]
    )
    measured = np.array(
        [[-0.020, 0.0, 0.290], [0.020, 0.0, 0.300]]
    )

    assert not ProfileEndpointDetectorNode._tracking_pair_is_plausible(
        node, measured, predicted
    )


def test_tracking_topology_rejects_large_jump_and_identity_crossing():
    node = _topology_node()
    predicted = np.array(
        [[-0.040, 0.0, 0.280], [0.040, 0.0, 0.310]]
    )
    jumped = predicted.copy()
    jumped[0, 0] += 0.010
    crossed = predicted[::-1].copy()

    assert not ProfileEndpointDetectorNode._tracking_pair_is_plausible(
        node, jumped, predicted
    )
    assert not ProfileEndpointDetectorNode._tracking_pair_is_plausible(
        node, crossed, predicted
    )
