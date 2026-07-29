import numpy as np
import pytest

from calibration_pipeline.geometry import make_transform
from calibration_pipeline.simulation.noise import (
    JointSnapshotBuffer,
    SimulationNoiseConfig,
    SimulationNoiseModel,
)


def ideal_noise_config(**overrides):
    values = {
        "random_seed": 7,
        "profile_gaussian_std_m": 0.0,
        "endpoint_gaussian_std_m": 0.0,
        "robot_translation_std_m": 0.0,
        "robot_rotation_std_deg": 0.0,
        "board_flatness_rms_m": 0.0,
        "sync_delay_mean_s": 0.0,
        "sync_jitter_std_s": 0.0,
        "point_outlier_probability": 0.0,
        "point_outlier_std_m": 0.0,
        "endpoint_outlier_probability": 0.0,
        "endpoint_outlier_std_m": 0.0,
        "point_dropout_probability": 0.0,
        "frame_dropout_probability": 0.0,
        "endpoint_dropout_probability": 0.0,
    }
    values.update(overrides)
    return SimulationNoiseConfig(**values)


def test_noise_configuration_rejects_invalid_values():
    with pytest.raises(ValueError):
        ideal_noise_config(profile_gaussian_std_m=-1.0)
    with pytest.raises(ValueError):
        ideal_noise_config(point_dropout_probability=1.01)


def test_zero_noise_is_an_exact_identity():
    model = SimulationNoiseModel(ideal_noise_config())
    transform = make_transform(np.eye(3), np.array([0.1, -0.2, 0.3]))
    points = np.array([[0.1, 0.0, 0.4], [-0.2, 0.0, 0.7]])

    assert np.array_equal(model.perturb_flange(transform), transform)
    assert not model.sample_frame_dropout()
    assert model.sample_sync_delay_s() == 0.0
    assert np.array_equal(model.corrupt_profile(points), points)
    endpoint, valid = model.corrupt_endpoint(points[0])
    assert valid
    assert np.array_equal(endpoint, points[0])


def test_seeded_noise_is_reproducible():
    config = ideal_noise_config(
        profile_gaussian_std_m=1e-4,
        endpoint_gaussian_std_m=2e-4,
        robot_translation_std_m=3e-5,
        robot_rotation_std_deg=0.003,
        sync_jitter_std_s=0.001,
        point_outlier_probability=0.2,
        point_outlier_std_m=5e-4,
        endpoint_outlier_probability=0.2,
        endpoint_outlier_std_m=5e-4,
        point_dropout_probability=0.1,
    )
    first = SimulationNoiseModel(config)
    second = SimulationNoiseModel(config)
    points = np.column_stack(
        (np.linspace(-0.1, 0.1, 20), np.zeros(20), np.linspace(0.3, 0.5, 20))
    )
    transform = np.eye(4)

    assert first.sample_sync_delay_s() == second.sample_sync_delay_s()
    assert np.array_equal(
        first.perturb_flange(transform), second.perturb_flange(transform)
    )
    assert np.array_equal(
        first.corrupt_profile(points), second.corrupt_profile(points)
    )
    first_endpoint = first.corrupt_endpoint(points[0])
    second_endpoint = second.corrupt_endpoint(points[0])
    assert first_endpoint[1] == second_endpoint[1]
    assert np.array_equal(first_endpoint[0], second_endpoint[0])


def test_complete_dropout_invalidates_profile_and_endpoints_together():
    model = SimulationNoiseModel(
        ideal_noise_config(frame_dropout_probability=1.0)
    )
    points = np.array([[0.1, 0.0, 0.4], [-0.2, 0.0, 0.7]])
    frame_dropped = model.sample_frame_dropout()

    assert frame_dropped
    assert len(model.corrupt_profile(points, frame_dropped=frame_dropped)) == 0
    _, valid = model.corrupt_endpoint(
        points[0], frame_dropped=frame_dropped
    )
    assert not valid


def test_fixed_flatness_preserves_the_laser_plane():
    model = SimulationNoiseModel(
        ideal_noise_config(board_flatness_rms_m=5e-5)
    )
    points = np.column_stack(
        (np.linspace(0.02, 0.38, 31), np.zeros(31), np.zeros(31))
    )
    laser_normal = np.array([0.0, 1.0, 0.0])
    deformed = model.deform_points_in_laser_plane(
        points,
        laser_normal=laser_normal,
        board_normal=np.array([0.0, 0.0, 1.0]),
        corner=np.zeros(3),
        board_u=np.array([1.0, 0.0, 0.0]),
        board_v=np.array([0.0, 1.0, 0.0]),
        width=0.4,
        height=0.5,
    )

    assert np.max(np.abs(deformed @ laser_normal)) < 1e-12
    assert np.max(np.abs(deformed[:, 2])) > 1e-6
    # It is one spatially fixed surface, not fresh white noise per frame.
    repeated = model.deform_points_in_laser_plane(
        points,
        laser_normal=laser_normal,
        board_normal=np.array([0.0, 0.0, 1.0]),
        corner=np.zeros(3),
        board_u=np.array([1.0, 0.0, 0.0]),
        board_v=np.array([0.0, 1.0, 0.0]),
        width=0.4,
        height=0.5,
    )
    assert np.array_equal(deformed, repeated)


def test_joint_snapshot_buffer_selects_delayed_encoder_state():
    history = JointSnapshotBuffer(maximum_size=4)
    history.append(1_000_000_000, np.array([1.0]))
    history.append(1_010_000_000, np.array([2.0]))
    history.append(1_020_000_000, np.array([3.0]))

    assert history.delayed(1_025_000_000, 0.012)[0] == 2.0
    assert history.delayed(1_025_000_000, 0.100)[0] == 1.0

