import numpy as np

from calibration_pipeline.roi_tracking import (
    ROIBreakpointPipeline,
    ROIBreakpointPipelineConfig,
)
from calibration_pipeline.roi_tracking.types import ROITrackingResult


def _profile() -> np.ndarray:
    x = np.linspace(-0.15, 0.15, 1201)
    z = 0.28 + np.tan(np.deg2rad(25.0)) * x
    on_plate = (x >= -0.04) & (x <= 0.04)
    z[~on_plate] += 0.010
    return np.column_stack((x, np.zeros_like(x), z))


class _ScriptedTracker:
    name = "scripted"

    def __init__(self, successes):
        self.successes = list(successes)
        self.roi = None

    def initialize(self, _frame, roi):
        self.roi = roi

    def update(self, _frame):
        success = self.successes.pop(0) if self.successes else True
        return ROITrackingResult(
            success=success,
            roi=self.roi if success else None,
            reason="" if success else "scripted_failure",
        )

    def reset(self):
        self.roi = None


def _aligned_pipeline(factory):
    pipeline = ROIBreakpointPipeline(
        ROIBreakpointPipelineConfig(
            initial_first_label="e1",
            minimum_lock_frames=3,
            fail_streak_frames=3,
            reacquire_stable_frames=2,
        ),
        tracker_factory=factory,
    )
    profile = _profile()
    for index in range(3):
        result = pipeline.process_profile(profile, 0.01 * index)
        assert result.state == "VALID"
    assert pipeline.lock()
    return pipeline, profile


def test_align_generates_two_metric_rois_and_reference_snapshot():
    pipeline, _profile_value = _aligned_pipeline(
        lambda _name, _rasterizer: _ScriptedTracker([True])
    )

    snapshot = pipeline.snapshot()

    assert pipeline.mode == "TRACK"
    assert snapshot is not None
    assert len(snapshot["rois"]) == 2
    for roi in snapshot["rois"]:
        assert np.isclose(roi["xmax_m"] - roi["xmin_m"], 0.020)
        assert np.isclose(roi["zmax_m"] - roi["zmin_m"], 0.020)


def test_align_ignores_one_periodic_bad_profile():
    pipeline = ROIBreakpointPipeline(
        ROIBreakpointPipelineConfig(
            initial_first_label="e1", minimum_lock_frames=3
        ),
        tracker_factory=lambda _name, _rasterizer: _ScriptedTracker([True]),
    )
    profile = _profile()
    assert pipeline.process_profile(profile, 0.00).state == "VALID"
    assert pipeline.process_profile(profile, 0.01).state == "VALID"

    shifted = profile.copy()
    shifted[:, 2] += 0.08
    assert pipeline.process_profile(shifted, 0.02).state == "REJECTED"
    assert pipeline.alignment_stable_frames == 2

    assert pipeline.process_profile(profile, 0.03).state == "VALID"
    assert pipeline.alignment_stable_frames == 3
    assert pipeline.lock()


def test_one_failed_frame_is_transient_and_next_valid_frame_clears_it():
    sequences = iter(([False, True], [False, True]))
    pipeline, profile = _aligned_pipeline(
        lambda _name, _rasterizer: _ScriptedTracker(next(sequences))
    )

    failed = pipeline.process_profile(profile, 0.10)
    recovered = pipeline.process_profile(profile, 0.11)

    assert failed.state == "REJECTED"
    assert pipeline.mode == "TRACK"
    assert recovered.state == "VALID"
    assert pipeline.fail_streak == 0


def test_three_failures_enter_lost_and_measured_prior_reinitializes_at_rollback():
    pipeline, profile = _aligned_pipeline(
        lambda _name, _rasterizer: _ScriptedTracker([False, False, False])
    )
    trusted = pipeline.last_matched.copy()

    states = [
        pipeline.process_profile(profile, 0.10 + 0.01 * index).state
        for index in range(3)
    ]

    assert states[:2] == ["REJECTED", "REJECTED"]
    assert states[2] == "LOST"
    assert pipeline.mode == "LOST"

    pipeline.handle_measured_prior(trusted)
    pending = pipeline.process_profile(profile, 0.20)
    recovered = pipeline.process_profile(profile, 0.21)

    assert pending.reason == "roi_reacquire_pending"
    assert recovered.state == "VALID"
    assert pipeline.mode == "TRACK"
    np.testing.assert_allclose(recovered.endpoints, trusted, atol=5e-4)


def test_reference_reacquire_uses_locked_reference_not_alignment_template():
    pipeline, profile = _aligned_pipeline(
        lambda _name, _rasterizer: _ScriptedTracker([True])
    )
    reference = pipeline.reference_snapshot["endpoints"].copy()

    pipeline.handle_control("REFERENCE_REACQUIRE")

    assert pipeline.mode == "LOST"
    np.testing.assert_allclose(pipeline.reacquire_anchor, reference)
    pipeline.process_profile(profile, 0.20)
    recovered = pipeline.process_profile(profile, 0.21)
    assert recovered.state == "VALID"
