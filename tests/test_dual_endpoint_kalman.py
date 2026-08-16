import numpy as np

from calibration_pipeline.perception import (
    DualEndpointKalmanConfig,
    DualEndpointKalmanTracker,
)


def _endpoints(first_xz, second_xz):
    return np.array(
        [
            [first_xz[0], 0.0, first_xz[1]],
            [second_xz[0], 0.0, second_xz[1]],
        ],
        dtype=float,
    )


def test_dual_kalman_tracks_noisy_moving_endpoints_without_identity_switches():
    rng = np.random.default_rng(20260814)
    tracker = DualEndpointKalmanTracker(
        DualEndpointKalmanConfig(
            process_acceleration_std_m_s2=0.25,
            mahalanobis_threshold=13.82,
        )
    )
    initial = _endpoints((-0.04, 0.28), (0.04, 0.31))
    tracker.reset(initial, timestamp_s=0.0)
    velocity = np.array([[0.004, -0.002], [0.003, -0.001]])
    filtered_errors = []

    for frame in range(1, 101):
        timestamp = frame * 0.01
        truth_xz = initial[:, (0, 2)] + timestamp * velocity
        measured_xz = truth_xz + rng.normal(0.0, 0.00010, (2, 2))
        measured = _endpoints(measured_xz[0], measured_xz[1])
        if frame % 2 == 0:
            measured = measured[::-1]
        tracker.predict(timestamp)
        ordered = tracker.order_measurement(
            measured, measurement_sigma_m=0.00010
        )
        assert ordered is not None
        assigned, distances = ordered
        assert max(distances) < tracker.config.mahalanobis_threshold
        assert np.linalg.norm(assigned[0, (0, 2)] - truth_xz[0]) < 0.001
        tracker.update(assigned, measurement_sigma_m=0.00010)
        filtered_errors.append(
            np.linalg.norm(
                tracker.endpoints()[:, (0, 2)] - truth_xz, axis=1
            )
        )

    assert np.max(np.mean(filtered_errors[-20:], axis=0)) < 0.00012


def test_dual_kalman_coasts_over_short_dropout_and_rejects_large_outlier():
    tracker = DualEndpointKalmanTracker()
    initial = _endpoints((-0.03, 0.27), (0.05, 0.30))
    tracker.reset(initial, timestamp_s=0.0)
    initial_gate = tracker.search_radius(
        minimum_m=0.001, maximum_m=0.025, sigma_multiplier=3.0
    )

    for frame in range(1, 5):
        tracker.predict(frame * 0.02)
        tracker.mark_missed()

    expanded_gate = tracker.search_radius(
        minimum_m=0.001, maximum_m=0.025, sigma_multiplier=3.0
    )
    assert tracker.missed_frames == 4
    assert expanded_gate > initial_gate

    outlier = initial.copy()
    outlier[:, 0] += 0.04
    assert (
        tracker.order_measurement(outlier, measurement_sigma_m=0.0001)
        is None
    )


def test_dual_kalman_limits_velocity_created_by_a_noisy_update():
    tracker = DualEndpointKalmanTracker(
        DualEndpointKalmanConfig(maximum_endpoint_speed_m_s=0.10)
    )
    initial = _endpoints((-0.03, 0.27), (0.05, 0.30))
    tracker.reset(initial, timestamp_s=0.0)
    tracker.predict(0.001)
    shifted = initial.copy()
    shifted[:, 0] += 0.02
    tracker.update(shifted, measurement_sigma_m=0.00008)

    velocities = tracker.state[4:].reshape(2, 2)
    assert np.max(np.linalg.norm(velocities, axis=1)) <= 0.10 + 1.0e-12


def test_dual_kalman_does_not_integrate_repeated_or_backward_timestamps():
    tracker = DualEndpointKalmanTracker()
    initial = _endpoints((-0.03, 0.27), (0.05, 0.30))
    tracker.reset(initial, timestamp_s=1.0)
    tracker.state[4:] = np.array([0.1, 0.0, 0.1, 0.0])

    first = tracker.predict(1.0)
    backward = tracker.predict(0.5)

    np.testing.assert_allclose(first, initial)
    np.testing.assert_allclose(backward, initial)
    assert tracker.timestamp_s == 1.0


def test_dual_kalman_updates_one_visible_breakpoint_without_faking_the_other():
    tracker = DualEndpointKalmanTracker(
        DualEndpointKalmanConfig(mahalanobis_threshold=13.82)
    )
    initial = _endpoints((-0.03, 0.27), (0.05, 0.30))
    tracker.reset(initial, timestamp_s=0.0)
    tracker.predict(0.02)
    measured_first = np.array([-0.0297, 0.0, 0.2698])

    tracker.update_partial(
        {0: measured_first}, measurement_sigma_m=0.00010
    )

    assert tracker.missed_frames_by_endpoint.tolist() == [0, 1]
    assert tracker.missed_frames == 1
    assert np.linalg.norm(
        tracker.endpoints()[0, [0, 2]] - measured_first[[0, 2]]
    ) < 0.0001
    # e2 has no measurement and therefore remains a prediction, not a copy of
    # e1 or a made-up pair observation.
    np.testing.assert_allclose(
        tracker.endpoints()[1, [0, 2]], initial[1, [0, 2]], atol=1e-12
    )
    radii = tracker.endpoint_search_radii(
        minimum_m=0.0001, maximum_m=0.025, sigma_multiplier=3.0
    )
    assert radii[1] > radii[0]


def test_dual_kalman_endpoint_candidate_selection_is_local_and_gated():
    tracker = DualEndpointKalmanTracker(
        DualEndpointKalmanConfig(mahalanobis_threshold=13.82)
    )
    initial = _endpoints((-0.03, 0.27), (0.05, 0.30))
    tracker.reset(initial, timestamp_s=0.0)
    candidates = np.array(
        [
            [-0.0298, 0.0, 0.2701],
            [-0.0270, 0.0, 0.2740],
            [0.04, 0.0, 0.40],
        ]
    )

    selected = tracker.select_endpoint_candidate(
        0, candidates, measurement_sigma_m=0.00020
    )

    assert selected is not None
    point, distance = selected
    np.testing.assert_allclose(point, candidates[0])
    assert distance < tracker.config.mahalanobis_threshold
    assert (
        tracker.select_endpoint_candidate(
            1, candidates, measurement_sigma_m=0.00020
        )
        is None
    )
