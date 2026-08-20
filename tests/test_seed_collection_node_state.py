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
from calibration_pipeline.seed_collection import (
    BroydenDualFeatureServo,
    RotationTarget,
)


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


def _dual_servo_node() -> SeedCollectionNode:
    node = object.__new__(SeedCollectionNode)
    node.servo_controller = "broyden_dual"
    node.collection_phase = "COLLECT"
    node.x_tolerance = 0.010
    node.servo_length_lower = 0.070
    node.servo_length_target = 0.080
    node.servo_length_upper = 0.090
    node.servo_convergence_tolerance = 0.0005
    node.rotation_feedforward_enabled = True
    node.rotation_feedforward_gain = 1.0
    node.rotation_feedforward_maximum_axis_step = 0.05
    node.rotation_feedforward_maximum_norm_step = 0.05
    node.rotation_rate_smoothing = 0.5
    node.rotation_feature_rate = None
    node.rotation_feedforward_samples = 0
    node.rotation_feedforward_last_residual = np.zeros(2)
    node.dual_servo = BroydenDualFeatureServo(
        damping=0.0,
        maximum_axis_step=0.05,
        maximum_norm_step=0.05,
        minimum_singular_value=0.01,
    )
    node.dual_servo.set_jacobian(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    node.get_logger = lambda: _Logger()
    return node


def test_dual_feature_error_activates_to_interior_target_not_band_edge():
    node = _dual_servo_node()
    high = SimpleNamespace(x_mid=0.002, profile_length=0.113)
    inside = SimpleNamespace(x_mid=0.002, profile_length=0.089)
    np.testing.assert_allclose(
        SeedCollectionNode._dual_feature_error(node, high), [0.0, 0.033]
    )
    np.testing.assert_allclose(
        SeedCollectionNode._dual_feature_error(node, inside), [0.0, 0.0]
    )


def test_rotation_feedforward_cancels_learned_length_disturbance():
    node = _dual_servo_node()
    one_degree = np.deg2rad(1.0)
    node.rotation_feature_rate = np.array([0.0, 0.020 / one_degree])
    feature = SimpleNamespace(x_mid=0.0, profile_length=0.080)
    command = SeedCollectionNode._rotation_feedforward_command(
        node, feature, one_degree
    )
    np.testing.assert_allclose(command, [0.0, 0.0, -0.020], atol=1e-12)


def test_rotation_rate_update_separates_commanded_translation_from_rotation():
    node = _dual_servo_node()
    node.pending_rotation = np.deg2rad(1.0)
    node.rotation_pre_feature_vector = np.array([0.0, 0.080])
    node.rotation_pre_measurement_transform = np.eye(4)
    reached = np.eye(4)
    reached[2, 3] = -0.010
    node._measurement_transform = lambda: reached
    node._current_transform = lambda: reached
    feature = SimpleNamespace(x_mid=0.0, profile_length=0.090)

    SeedCollectionNode._update_rotation_feature_rate(node, feature)

    per_degree = node.rotation_feature_rate * np.deg2rad(1.0)
    # Observed length change is +10 mm while the simultaneous translation
    # contributes -10 mm, so the isolated rotation disturbance is +20 mm/deg.
    np.testing.assert_allclose(per_degree, [0.0, 0.020], atol=1e-12)
    assert node.rotation_feedforward_samples == 1


def test_verified_rotation_model_combines_two_degree_rotation_and_translation():
    node = _dual_servo_node()
    node.rotation_step = np.deg2rad(1.0)
    node.rotation_feedforward_accelerated_step = np.deg2rad(2.0)
    node.rotation_feedforward_minimum_verified_steps = 2
    node.rotation_feedforward_samples = 2
    node.failure_count = 0
    node.accumulated_angle = np.deg2rad(2.0)
    node.rotation_target = np.deg2rad(10.0)
    node.target_index = 0
    node.stage_index = 0
    node.plan = (RotationTarget("ry_positive", ((1, 1),)),)
    node.rotation_feature_rate = np.array(
        [0.0, 0.020 / np.deg2rad(1.0)]
    )
    node._current_transform = lambda: np.eye(4)
    node._measurement_transform = lambda: np.eye(4)
    node._feature = lambda: SimpleNamespace(
        x_mid=0.0, profile_length=0.080
    )
    commanded = []
    node._command_transform = (
        lambda transform, stage: commanded.append((transform, stage)) or True
    )

    SeedCollectionNode._issue_micro_rotation(node)

    assert np.isclose(node.pending_rotation, np.deg2rad(2.0))
    assert commanded[0][1] == "MICRO_ROTATION"
    assert commanded[0][0][2, 3] < -0.039


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


def test_pending_pose_history_outlives_high_rate_pose_burst():
    node = object.__new__(SeedCollectionNode)
    node.pending_observation_buffer_size = 2
    node.pending_pose_buffer_size = 4
    node.pending_profiles = {("profile", index): index for index in range(4)}
    node.pending_endpoint_frames = {
        ("endpoint", index): index for index in range(4)
    }
    node.pending_flange_poses = {("pose", index): index for index in range(4)}

    SeedCollectionNode._trim_pending_frames(node)

    assert list(node.pending_profiles) == [("profile", 2), ("profile", 3)]
    assert list(node.pending_endpoint_frames) == [
        ("endpoint", 2),
        ("endpoint", 3),
    ]
    assert list(node.pending_flange_poses) == [
        ("pose", 0),
        ("pose", 1),
        ("pose", 2),
        ("pose", 3),
    ]


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
