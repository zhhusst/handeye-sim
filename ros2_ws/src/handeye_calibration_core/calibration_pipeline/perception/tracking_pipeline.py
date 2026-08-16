"""Standalone breakpoint detection + tracking pipeline (zero-ROS diagnostic core).

This module is a faithful, ROS-free extraction of the state machine that the
real node ``profile_endpoint_detector_node.py`` runs per profile frame.  It
exists so the tracking behaviour can be replayed offline against recorded bags,
diagnosed, and improved without touching the deployed ROS node.  Once the
offline pipeline is trustworthy it can be swapped back into the node.

Design goals
------------
1. 1:1 behavioural copy of the deployed node state machine
   (ALIGN / TRACK / PREDICTED_TRACK / LOST / GLOBAL) so offline replay
   reproduces the same accept/reject decisions, reasons and endpoints.
2. Zero ROS dependency.  Inputs are numpy profile arrays, control commands and
   timestamps; outputs are plain dataclasses.
3. Swappable temporal tracker.  ``TemporalTracker`` is a Protocol; the Kalman
   implementation wraps the existing ``DualEndpointKalmanTracker``.  A future
   SPR (structure-preserved registration, Tang & Tomizuka IJRR 2022) tracker
   only needs to implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .endpoint_detector import (
    EndpointDetectionConfig,
    ProfileEndpointDetector,
)
from .dual_endpoint_kalman import (
    DualEndpointKalmanConfig,
    DualEndpointKalmanTracker,
)
from ..seed_collection.endpoint_tracker import EndpointTracker


# ---------------------------------------------------------------------------
# Temporal tracker protocol (SPR extension point)
# ---------------------------------------------------------------------------
@runtime_checkable
class TemporalTracker(Protocol):
    """Interface implemented by the Kalman tracker and, later, an SPR tracker.

    The pipeline only depends on this surface, so a registration-based tracker
    can be substituted without touching the state machine.
    """

    initialized: bool
    missed_frames: int
    missed_frames_by_endpoint: np.ndarray

    def reset(self, endpoints: np.ndarray, timestamp_s: float | None = None) -> None: ...

    def predict(self, timestamp_s: float) -> np.ndarray: ...

    def endpoints(self) -> np.ndarray: ...

    def endpoint_search_radii(
        self, *, minimum_m: float, maximum_m: float, sigma_multiplier: float = 3.0
    ) -> np.ndarray: ...

    def order_measurement(
        self, endpoints: np.ndarray, *, measurement_sigma_m: float
    ) -> tuple[np.ndarray, tuple[float, float]] | None: ...

    def select_endpoint_candidate(
        self,
        endpoint: int,
        candidates: np.ndarray,
        *,
        measurement_sigma_m: float,
    ) -> tuple[np.ndarray, float] | None: ...

    def update(self, endpoints: np.ndarray, *, measurement_sigma_m: float) -> None: ...

    def update_partial(
        self, measurements: dict[int, np.ndarray], *, measurement_sigma_m: float
    ) -> None: ...

    def mark_missed(self, endpoint: int | None = None) -> None: ...


class KalmanTemporalTracker(DualEndpointKalmanTracker):
    """Adapter over the existing Kalman tracker (satisfies TemporalTracker)."""

    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrackingPipelineConfig:
    """All parameters of the deployed node, defaults mirroring the node."""

    # endpoint detector
    detector: EndpointDetectionConfig = field(
        default_factory=EndpointDetectionConfig
    )
    identity_ambiguity_ratio: float = 0.05
    initial_first_label: str = "e1"
    initial_mode: str = "ALIGN"
    minimum_confidence: float = 0.25
    guided_enabled: bool = True

    # alignment
    alignment_template_center_x_m: float = 0.0
    alignment_template_center_z_m: float = 0.42
    alignment_template_length_m: float = 0.10
    alignment_template_angle_deg: float = 25.0
    alignment_normal_gate_m: float = 0.003
    alignment_endpoint_gate_m: float = 0.020
    alignment_maximum_angle_difference_deg: float = 15.0
    alignment_stability_m: float = 0.0015
    minimum_lock_frames: int = 5

    # tracking
    tracking_normal_gate_m: float = 0.006
    tracking_endpoint_gate_m: float = 0.025
    tracking_maximum_angle_difference_deg: float = 25.0
    maximum_lost_frames: int = 5

    # predicted (NBV)
    predicted_normal_gate_m: float = 0.012
    predicted_endpoint_gate_m: float = 0.050
    predicted_maximum_angle_difference_deg: float = 35.0

    # reacquisition
    reacquire_stable_frames: int = 3
    reacquire_maximum_segment_length_change_m: float = 0.020
    reacquire_maximum_segment_angle_change_deg: float = 20.0
    reacquire_stability_m: float = 0.003

    # temporal Kalman
    temporal_tracking_enabled: bool = True
    temporal_initial_position_std_m: float = 0.0005
    temporal_initial_velocity_std_m_s: float = 0.05
    temporal_process_acceleration_std_m_s2: float = 1.0
    temporal_measurement_std_floor_m: float = 0.00008
    temporal_partial_measurement_std_m: float = 0.0005
    temporal_mahalanobis_threshold: float = 13.82
    temporal_maximum_endpoint_speed_m_s: float = 0.25
    temporal_maximum_coast_frames: int = 5
    temporal_minimum_search_radius_m: float = 0.0015
    temporal_search_sigma_multiplier: float = 3.0
    temporal_maximum_local_candidates: int = 6

    # identity / topology
    tracking_maximum_endpoint_step_m: float = 0.003
    tracking_minimum_reference_length_ratio: float = 0.60
    tracking_maximum_reference_length_ratio: float = 1.80

    def __post_init__(self) -> None:
        if self.initial_first_label.strip().lower() not in {"e1", "e2"}:
            raise ValueError("initial_first_label must be e1 or e2")
        if self.initial_mode.strip().upper() not in {"ALIGN", "TRACK"}:
            raise ValueError("initial_mode must be ALIGN or TRACK")
        positive = {
            "alignment_template_length_m": self.alignment_template_length_m,
            "alignment_normal_gate_m": self.alignment_normal_gate_m,
            "alignment_endpoint_gate_m": self.alignment_endpoint_gate_m,
            "alignment_stability_m": self.alignment_stability_m,
            "tracking_normal_gate_m": self.tracking_normal_gate_m,
            "tracking_endpoint_gate_m": self.tracking_endpoint_gate_m,
            "predicted_normal_gate_m": self.predicted_normal_gate_m,
            "predicted_endpoint_gate_m": self.predicted_endpoint_gate_m,
            "reacquire_maximum_segment_length_change_m": (
                self.reacquire_maximum_segment_length_change_m
            ),
            "reacquire_maximum_segment_angle_change_deg": (
                self.reacquire_maximum_segment_angle_change_deg
            ),
            "reacquire_stability_m": self.reacquire_stability_m,
            "temporal_minimum_search_radius_m": self.temporal_minimum_search_radius_m,
            "temporal_search_sigma_multiplier": self.temporal_search_sigma_multiplier,
            "temporal_partial_measurement_std_m": self.temporal_partial_measurement_std_m,
            "temporal_maximum_endpoint_speed_m_s": self.temporal_maximum_endpoint_speed_m_s,
            "tracking_maximum_endpoint_step_m": self.tracking_maximum_endpoint_step_m,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_lock_frames < 1 or self.maximum_lost_frames < 1:
            raise ValueError("frame counts must be positive")
        if self.reacquire_stable_frames < 1:
            raise ValueError("reacquire_stable_frames must be positive")
        if self.temporal_maximum_coast_frames < 1:
            raise ValueError("temporal_maximum_coast_frames must be positive")
        if self.temporal_maximum_local_candidates < 1:
            raise ValueError("temporal_maximum_local_candidates must be positive")
        if self.temporal_minimum_search_radius_m > self.tracking_endpoint_gate_m:
            raise ValueError(
                "temporal minimum search radius must not exceed tracking endpoint gate"
            )
        if not (
            0.0
            < self.tracking_minimum_reference_length_ratio
            <= 1.0
            <= self.tracking_maximum_reference_length_ratio
        ):
            raise ValueError("tracking reference length ratios must bracket one")


# ---------------------------------------------------------------------------
# Per-frame result
# ---------------------------------------------------------------------------
@dataclass
class FrameResult:
    """One pipeline decision, mirroring the node's diagnostics payload."""

    frame_index: int
    timestamp_s: float | None
    state: str                 # VALID / REJECTED / LOST
    reason: str
    mode: str
    endpoints: np.ndarray | None        # (2, 3) matched e1/e2 when VALID
    surface_points: np.ndarray | None   # (N, 3) support inliers
    guide_first_mm: list[float]
    guide_second_mm: list[float]
    guide_normal_gate_mm: float
    guide_endpoint_gate_mm: float
    guide_angle_gate_deg: float
    acceptance_rate: float
    accepted: int
    frames: int
    # detector quality fields (VALID only)
    support_points: int | None = None
    target_surface_points: int | None = None
    segment_length_mm: float | None = None
    residual_rms_mm: float | None = None
    sample_pitch_mm: float | None = None
    endpoint_sigma_mm: float | None = None
    confidence: float | None = None
    breakpoint_count: int | None = None
    selection_mode: str | None = None
    # temporal diagnostics
    temporal_tracking: bool = False
    temporal_initialized: bool = False
    temporal_missed_frames: int = 0
    temporal_missed_frames_by_endpoint: list[int] = field(default_factory=lambda: [0, 0])
    temporal_suspended: bool = False
    temporal_fallback_reason: str = ""
    temporal_search_radius_mm: float = 0.0
    temporal_search_radii_mm: list[float] = field(default_factory=lambda: [0.0, 0.0])
    temporal_mahalanobis: list[float] | None = None
    tracking_reference_length_mm: float | None = None
    # state machine
    locked: bool = False
    alignment_stable_frames: int = 0
    minimum_lock_frames: int = 5
    lost_frames: int = 0
    reacquire_frames: int = 0
    profile_points: int = 0

    def to_dict(self) -> dict:
        d: dict = {
            "state": self.state,
            "reason": self.reason,
            "mode": self.mode,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "frames": self.frames,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "profile_points": self.profile_points,
            "locked": self.locked,
            "alignment_stable_frames": self.alignment_stable_frames,
            "minimum_lock_frames": self.minimum_lock_frames,
            "lost_frames": self.lost_frames,
            "reacquire_frames": self.reacquire_frames,
            "guide_first_mm": self.guide_first_mm,
            "guide_second_mm": self.guide_second_mm,
            "guide_normal_gate_mm": self.guide_normal_gate_mm,
            "guide_endpoint_gate_mm": self.guide_endpoint_gate_mm,
            "guide_angle_gate_deg": self.guide_angle_gate_deg,
            "temporal_tracking": self.temporal_tracking,
            "temporal_initialized": self.temporal_initialized,
            "temporal_missed_frames": self.temporal_missed_frames,
            "temporal_missed_frames_by_endpoint": self.temporal_missed_frames_by_endpoint,
            "temporal_suspended": self.temporal_suspended,
            "temporal_fallback_reason": self.temporal_fallback_reason,
            "temporal_search_radius_mm": self.temporal_search_radius_mm,
            "temporal_search_radii_mm": self.temporal_search_radii_mm,
            "temporal_mahalanobis": self.temporal_mahalanobis,
            "tracking_reference_length_mm": self.tracking_reference_length_mm,
        }
        if self.support_points is not None:
            d["support_points"] = self.support_points
        if self.target_surface_points is not None:
            d["target_surface_points"] = self.target_surface_points
        if self.segment_length_mm is not None:
            d["segment_length_mm"] = self.segment_length_mm
        if self.residual_rms_mm is not None:
            d["residual_rms_mm"] = self.residual_rms_mm
        if self.sample_pitch_mm is not None:
            d["sample_pitch_mm"] = self.sample_pitch_mm
        if self.endpoint_sigma_mm is not None:
            d["endpoint_sigma_mm"] = self.endpoint_sigma_mm
        if self.endpoints is not None:
            d["endpoint_first_mm"] = (1000.0 * self.endpoints[0]).tolist()
            d["endpoint_second_mm"] = (1000.0 * self.endpoints[1]).tolist()
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.breakpoint_count is not None:
            d["breakpoint_count"] = self.breakpoint_count
        if self.selection_mode is not None:
            d["selection_mode"] = self.selection_mode
        return d


# ---------------------------------------------------------------------------
# Pipeline (faithful state-machine copy)
# ---------------------------------------------------------------------------
class BreakpointTrackingPipeline:
    """Zero-ROS state machine equivalent of ``ProfileEndpointDetectorNode``.

    Use :meth:`process_profile` per incoming profile (with a monotonic
    timestamp), :meth:`handle_control` for ``/calibration/detection_control``
    commands, and :meth:`handle_prior` / :meth:`handle_measured_prior` for the
    two prior topics.  :meth:`lock` and :meth:`reset` mirror the ROS services.
    """

    def __init__(self, config: TrackingPipelineConfig | None = None) -> None:
        self.config = config or TrackingPipelineConfig()
        self.detector = ProfileEndpointDetector(self.config.detector)
        self.tracker = EndpointTracker(
            ambiguity_ratio=self.config.identity_ambiguity_ratio
        )
        self.temporal_tracker: TemporalTracker = self._make_temporal_tracker()

        self.frames = 0
        self.accepted = 0
        self.mode = (
            "TRACK"
            if self.config.guided_enabled and self.config.initial_mode == "TRACK"
            else "ALIGN"
            if self.config.guided_enabled
            else "GLOBAL"
        )
        self.template_endpoints = self._make_alignment_template()
        self.guide_endpoints = self.template_endpoints.copy()
        self.tracking_expected: np.ndarray | None = None
        self.predicted_expected: np.ndarray | None = None
        self.prediction_fallback: np.ndarray | None = None
        self.last_matched: np.ndarray | None = None
        self.previous_alignment_match: np.ndarray | None = None
        self.alignment_stable_frames = 0
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.lost_from_mode = "TRACK"
        self.identity_initialized = False
        self.temporal_tracking_requested = False
        self.temporal_prediction_this_frame = False
        self.temporal_search_radius = self.config.temporal_minimum_search_radius_m
        self.temporal_search_radii = np.full(
            2, self.config.temporal_minimum_search_radius_m, dtype=float
        )
        self.temporal_last_mahalanobis: tuple[float, float] | None = None
        self.temporal_suspended = False
        self.temporal_fallback_reason = ""
        self.tracking_reference_length: float | None = None
        self.last_profile_time_s: float | None = None
        self.last_result: FrameResult | None = None

    def _make_temporal_tracker(self) -> TemporalTracker:
        kalman_config = DualEndpointKalmanConfig(
            initial_position_std_m=self.config.temporal_initial_position_std_m,
            initial_velocity_std_m_s=self.config.temporal_initial_velocity_std_m_s,
            process_acceleration_std_m_s2=(
                self.config.temporal_process_acceleration_std_m_s2
            ),
            measurement_std_floor_m=self.config.temporal_measurement_std_floor_m,
            mahalanobis_threshold=self.config.temporal_mahalanobis_threshold,
            maximum_endpoint_speed_m_s=(
                self.config.temporal_maximum_endpoint_speed_m_s
            ),
            assignment_ambiguity_ratio=self.config.identity_ambiguity_ratio,
        )
        return KalmanTemporalTracker(kalman_config)

    # -- public control ----------------------------------------------------
    def process_profile(self, profile: np.ndarray, timestamp_s: float | None = None) -> FrameResult:
        """Run one frame through the exact node state machine."""
        self.frames += 1
        self.last_profile_time_s = timestamp_s
        self._prepare_temporal_prediction(timestamp_s)
        try:
            detection = self._guided_detection(profile)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.last_result = self._reject(f"invalid_profile:{error}", profile_points=len(profile))
            return self.last_result

        if detection is None:
            if self._try_partial_temporal_update(profile):
                return self.last_result  # already set by partial update path
            self.last_result = self._reject(
                self.detector.last_rejection_reason, profile_points=len(profile)
            )
            return self.last_result
        if detection.confidence < self.config.minimum_confidence:
            self.last_result = self._reject(
                "confidence_below_threshold", profile_points=len(profile)
            )
            return self.last_result

        measured = np.vstack((detection.first, detection.second))
        predicted_pair = self.guide_endpoints.copy()
        if self.temporal_prediction_this_frame:
            ordered = self.temporal_tracker.order_measurement(
                measured,
                measurement_sigma_m=detection.endpoint_sigma_m,
            )
            if ordered is None:
                both = (measured.copy(), measured.copy())
                if self._try_partial_temporal_update(
                    profile, candidate_sets=both
                ):
                    return self.last_result
                self.last_result = self._reject(
                    "temporal_endpoint_gate_rejected", profile_points=len(profile)
                )
                return self.last_result
            matched, self.temporal_last_mahalanobis = ordered
        else:
            matched = self._associate(detection)
            if matched is None:
                self.last_result = self._reject(
                    "ambiguous_endpoint_identity", profile_points=len(profile)
                )
                return self.last_result

        tracking_geometry = self.mode == "TRACK" or (
            self.mode == "LOST" and self.lost_from_mode == "TRACK"
        )
        # First TRACK frame after lock has no identity yet: the guide is only
        # an initialisation prior and endpoint_step against it is meaningless.
        if (
            tracking_geometry
            and self.identity_initialized
            and not self._tracking_pair_is_plausible(matched, predicted_pair)
        ):
            if self._try_partial_temporal_update(profile):
                return self.last_result
            self.last_result = self._reject(
                "tracked_endpoint_topology_rejected", profile_points=len(profile)
            )
            return self.last_result

        if not self._reacquisition_geometry_is_continuous(matched):
            self.last_result = self._reject(
                "reacquire_geometry_discontinuity", profile_points=len(profile)
            )
            return self.last_result

        if self.temporal_prediction_this_frame:
            self.temporal_tracker.update(
                matched,
                measurement_sigma_m=detection.endpoint_sigma_m,
            )
            self.guide_endpoints = self.temporal_tracker.endpoints()

        if self.mode == "LOST":
            if self.reacquire_frames > 0 and self.last_matched is not None:
                maximum_change = float(
                    np.max(np.linalg.norm(matched - self.last_matched, axis=1))
                )
                self.reacquire_frames = (
                    self.reacquire_frames + 1
                    if maximum_change <= self.config.reacquire_stability_m
                    else 1
                )
            else:
                self.reacquire_frames = 1
            self.last_matched = matched.copy()
            if self.reacquire_frames < self.config.reacquire_stable_frames:
                self.last_result = self._reject(
                    "local_reacquire_pending", profile_points=len(profile)
                )
                return self.last_result
            self.mode = self.lost_from_mode
            self.reacquire_frames = 0
            self.lost_frames = 0

        if self.mode == "ALIGN":
            if self.previous_alignment_match is None:
                self.alignment_stable_frames = 1
            else:
                maximum_change = float(
                    np.max(
                        np.linalg.norm(
                            matched - self.previous_alignment_match, axis=1
                        )
                    )
                )
                self.alignment_stable_frames = (
                    self.alignment_stable_frames + 1
                    if maximum_change <= self.config.alignment_stability_m
                    else 1
                )
            self.previous_alignment_match = matched.copy()
        elif self.mode == "TRACK":
            self.tracking_expected = matched.copy()
            if not self.temporal_prediction_this_frame:
                self.guide_endpoints = matched.copy()
            self.tracker.reset(*matched)
            if self.tracking_reference_length is None:
                self._set_tracking_reference(matched)
            else:
                # Adaptive reference: follow slow chord-length drift (robot
                # motion sweeping the laser across the plate) so the ratio
                # gate [0.65,1.6]xref never caps a legitimately growing gap.
                # The gate still rejects single-frame collapse onto an inner
                # weak feature because a sudden halving/2x jump exceeds the
                # window around the smoothed reference.  Exponential smoothing
                # keeps a single bad measurement from yanking the reference.
                self.tracking_reference_length = (
                    0.90 * self.tracking_reference_length
                    + 0.10 * float(
                        np.linalg.norm(matched[1] - matched[0])
                    )
                )
            self.identity_initialized = True
            self.lost_frames = 0
        elif self.mode == "PREDICTED_TRACK":
            self.guide_endpoints = matched.copy()
            self.lost_frames = 0
        self.last_matched = matched.copy()
        if (
            self.mode == "TRACK"
            and self.temporal_tracking_requested
            and self.config.temporal_tracking_enabled
            and self.temporal_suspended
        ):
            self._restore_temporal_from_measurement(matched)

        self.last_result = self._valid_result(
            detection=detection,
            matched=matched,
            profile_points=len(profile),
        )
        return self.last_result

    def handle_control(self, command: str) -> None:
        command = str(command).strip().upper()
        if command == "SEED_TRACK_START":
            self.temporal_tracking_requested = self.config.temporal_tracking_enabled
            self.temporal_suspended = False
            self.temporal_fallback_reason = ""
            initial = (
                self.tracking_expected
                if self.tracking_expected is not None
                else self.last_matched
            )
            if self.temporal_tracking_requested and initial is not None:
                self._set_tracking_reference(initial)
                self._restore_temporal_from_measurement(initial)
        elif command == "SEED_TRACK_STOP":
            self.temporal_tracking_requested = False
            self.temporal_prediction_this_frame = False
            self.temporal_last_mahalanobis = None
            self.temporal_suspended = False
            self.temporal_fallback_reason = ""
            trusted = (
                self.tracking_expected
                if self.tracking_expected is not None
                else self.last_matched
            )
            if (
                trusted is not None
                and np.all(np.isfinite(trusted))
                and (
                    self.mode == "TRACK"
                    or (
                        self.mode == "LOST"
                        and self.lost_from_mode == "TRACK"
                    )
                )
            ):
                self.guide_endpoints = trusted.copy()
                self.temporal_tracker.reset(
                    trusted, timestamp_s=self.last_profile_time_s
                )
        elif command == "REFERENCE_REACQUIRE":
            self.prediction_fallback = (
                None
                if self.tracking_expected is None
                else self.tracking_expected.copy()
            )
            self.predicted_expected = self.template_endpoints.copy()
            self.guide_endpoints = self.predicted_expected.copy()
            self.mode = "LOST"
            self.lost_from_mode = "ALIGN"
            self.last_matched = None
            self.lost_frames = 0
            self.reacquire_frames = 0
        elif command == "PREDICTION_COMMIT":
            if (
                self.predicted_expected is None
                and self.mode != "PREDICTED_TRACK"
                and not (
                    self.mode == "LOST"
                    and self.lost_from_mode == "PREDICTED_TRACK"
                )
            ):
                return
            if self.last_matched is not None:
                self.tracking_expected = self.last_matched.copy()
            elif self.predicted_expected is not None:
                self.tracking_expected = self.predicted_expected.copy()
            if self.tracking_expected is not None:
                self.guide_endpoints = self.tracking_expected.copy()
                self.tracker.reset(*self.tracking_expected)
                self.identity_initialized = True
                self.mode = "TRACK"
                self._set_tracking_reference(self.tracking_expected)
                if (
                    self.temporal_tracking_requested
                    and self.config.temporal_tracking_enabled
                ):
                    self._restore_temporal_from_measurement(
                        self.tracking_expected
                    )
            self.predicted_expected = None
            self.prediction_fallback = None
            self.lost_frames = 0
            self.reacquire_frames = 0
        elif command == "PREDICTION_CANCEL":
            if (
                self.predicted_expected is None
                and self.mode != "PREDICTED_TRACK"
                and not (
                    self.mode == "LOST"
                    and self.lost_from_mode == "PREDICTED_TRACK"
                )
            ):
                return
            if self.prediction_fallback is not None:
                self.tracking_expected = self.prediction_fallback.copy()
                self.guide_endpoints = self.tracking_expected.copy()
                self.tracker.reset(*self.tracking_expected)
                self.identity_initialized = True
                self.mode = "TRACK"
                self._set_tracking_reference(self.tracking_expected)
                if (
                    self.temporal_tracking_requested
                    and self.config.temporal_tracking_enabled
                ):
                    self._restore_temporal_from_measurement(
                        self.tracking_expected
                    )
            else:
                self._reset_guidance()
            self.predicted_expected = None
            self.prediction_fallback = None
            self.lost_frames = 0
            self.reacquire_frames = 0

    def handle_prior(self, prior: np.ndarray) -> None:
        """Future-view NBV prior: enter PREDICTED_TRACK."""
        prior = np.asarray(prior, dtype=float).reshape(2, 3)
        if np.linalg.norm(prior[1] - prior[0]) <= 1e-9:
            return
        if self.mode != "PREDICTED_TRACK":
            self.prediction_fallback = (
                None
                if self.tracking_expected is None
                else self.tracking_expected.copy()
            )
        self.predicted_expected = prior.copy()
        self.guide_endpoints = prior.copy()
        self.mode = "PREDICTED_TRACK"
        self.lost_from_mode = "PREDICTED_TRACK"
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.last_matched = None
        self.temporal_tracking_requested = False

    def handle_measured_prior(self, prior: np.ndarray) -> None:
        """Rollback to previously measured geometry: enter LOST (narrow gates)."""
        prior = np.asarray(prior, dtype=float).reshape(2, 3)
        if np.linalg.norm(prior[1] - prior[0]) <= 1e-9:
            return
        self.prediction_fallback = (
            None
            if self.tracking_expected is None
            else self.tracking_expected.copy()
        )
        self.predicted_expected = prior.copy()
        self.guide_endpoints = prior.copy()
        self.mode = "LOST"
        self.lost_from_mode = "TRACK"
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.last_matched = None
        if self.temporal_tracking_requested and self.config.temporal_tracking_enabled:
            self._restore_temporal_from_measurement(prior)

    def lock(self) -> bool:
        """Mirror of the ``~/lock`` service."""
        if not self.config.guided_enabled:
            return False
        if self.mode == "TRACK":
            return True
        if self.mode != "ALIGN":
            return False
        if (
            self.last_matched is None
            or self.alignment_stable_frames < self.config.minimum_lock_frames
        ):
            return False
        self.tracking_expected = self.last_matched.copy()
        self.guide_endpoints = self.tracking_expected.copy()
        self.tracker.reset(*self.tracking_expected)
        self.identity_initialized = True
        self.mode = "TRACK"
        self.lost_frames = 0
        self._set_tracking_reference(self.tracking_expected)
        if self.temporal_tracking_requested and self.config.temporal_tracking_enabled:
            self._restore_temporal_from_measurement(self.tracking_expected)
        return True

    def reset(self) -> None:
        """Mirror of the ``~/reset`` service: full state reset (node 1:1).
        _reset_guidance rebuilds both trackers and clears temporal state;
        here we also zero the frame counters so a fresh experiment starts
        clean (acceptance rate, lost history, Kalman velocity/covariance)."""
        self._reset_guidance()
        self.frames = 0
        self.accepted = 0
        self.last_profile_time_s = None
        self.last_result = None

    # -- internal: detection selection -------------------------------------
    def _guided_detection(self, profile: np.ndarray):
        if not self.config.guided_enabled:
            return self.detector.detect(profile)
        if self.temporal_prediction_this_frame:
            return self.detector.detect_temporal_breakpoint_pair(
                profile,
                self.guide_endpoints[0],
                self.guide_endpoints[1],
                endpoint_gate_m=self.temporal_search_radius,
                normal_gate_m=self.config.tracking_normal_gate_m,
                maximum_angle_difference_deg=self.config.tracking_maximum_angle_difference_deg,
                selection_mode="seed_temporal_track",
            )
        normal_gate, endpoint_gate, angle_gate = self._guide_parameters()
        selection_mode = {
            "ALIGN": "guided_align",
            "TRACK": "guided_track",
            "PREDICTED_TRACK": "predicted_track",
            "LOST": "local_reacquire",
        }.get(self.mode, "guided")
        return self.detector.detect_guided(
            profile,
            self.guide_endpoints[0],
            self.guide_endpoints[1],
            normal_gate_m=normal_gate,
            endpoint_gate_m=endpoint_gate,
            maximum_angle_difference_deg=angle_gate,
            selection_mode=selection_mode,
        )

    def _guide_parameters(self) -> tuple[float, float, float]:
        mode = self.lost_from_mode if self.mode == "LOST" else self.mode
        if (
            self.mode == "LOST"
            and self.lost_from_mode == "TRACK"
            and self.temporal_suspended
        ):
            return (
                self.config.predicted_normal_gate_m,
                self.config.predicted_endpoint_gate_m,
                min(
                    self.config.predicted_maximum_angle_difference_deg,
                    self.config.reacquire_maximum_segment_angle_change_deg,
                ),
            )
        if mode == "ALIGN":
            return (
                self.config.alignment_normal_gate_m,
                self.config.alignment_endpoint_gate_m,
                self.config.alignment_maximum_angle_difference_deg,
            )
        if mode == "PREDICTED_TRACK":
            return (
                self.config.predicted_normal_gate_m,
                self.config.predicted_endpoint_gate_m,
                self.config.predicted_maximum_angle_difference_deg,
            )
        return (
            self.config.tracking_normal_gate_m,
            self.config.tracking_endpoint_gate_m,
            self.config.tracking_maximum_angle_difference_deg,
        )

    def _associate(self, detection) -> np.ndarray | None:
        measured = np.vstack((detection.first, detection.second))
        if self.mode == "GLOBAL":
            if not self.identity_initialized:
                matched = (
                    measured
                    if self.config.initial_first_label == "e1"
                    else measured[::-1]
                ).copy()
                self.tracker.reset(*matched)
                self.identity_initialized = True
                return matched
            matched = self.tracker.match(*measured)
            return None if matched is None else np.vstack(matched)
        if (
            self.mode in {"ALIGN", "TRACK"}
            and not self.identity_initialized
        ):
            matched = (
                measured
                if self.config.initial_first_label == "e1"
                else measured[::-1]
            )
            return matched.copy()
        expected = self.guide_endpoints
        direct = float(np.sum((measured - expected) ** 2))
        swapped = float(np.sum((measured[::-1] - expected) ** 2))
        scale = max(direct, swapped, 1e-12)
        if abs(direct - swapped) / scale < self.config.identity_ambiguity_ratio:
            return None
        return measured.copy() if direct < swapped else measured[::-1].copy()

    # -- internal: temporal ------------------------------------------------
    def _uses_temporal_tracking(self) -> bool:
        return bool(
            self.config.temporal_tracking_enabled
            and self.temporal_tracking_requested
            and not self.temporal_suspended
            and (
                self.mode == "TRACK"
                or (
                    self.mode == "LOST"
                    and self.lost_from_mode == "TRACK"
                )
            )
        )

    def _prepare_temporal_prediction(self, timestamp_s: float | None) -> None:
        self.temporal_prediction_this_frame = False
        self.temporal_last_mahalanobis = None
        if not self._uses_temporal_tracking():
            return
        if not self.temporal_tracker.initialized:
            initial = (
                self.tracking_expected
                if self.tracking_expected is not None
                else self.guide_endpoints
            )
            self.temporal_tracker.reset(initial, timestamp_s=timestamp_s)
        predicted = self.temporal_tracker.predict(timestamp_s)
        trusted = self.tracking_expected
        segment_length = float(np.linalg.norm(predicted[1] - predicted[0]))
        finite = bool(np.all(np.isfinite(predicted)))
        physical_length = bool(
            self.config.detector.minimum_segment_length_m
            <= segment_length
            <= self.config.detector.maximum_segment_length_m
        )
        bounded_from_measurement = bool(
            trusted is not None
            and np.max(np.linalg.norm(predicted - trusted, axis=1))
            <= self.config.tracking_endpoint_gate_m
        )
        if not (finite and physical_length and bounded_from_measurement):
            self._suspend_temporal_tracking("nonphysical_prediction")
            return
        self.guide_endpoints = predicted.copy()
        self.temporal_search_radii = (
            self.temporal_tracker.endpoint_search_radii(
                minimum_m=self.config.temporal_minimum_search_radius_m,
                maximum_m=self.config.tracking_endpoint_gate_m,
                sigma_multiplier=self.config.temporal_search_sigma_multiplier,
            )
        )
        self.temporal_search_radius = float(np.max(self.temporal_search_radii))
        self.temporal_prediction_this_frame = True

    def _suspend_temporal_tracking(self, reason: str) -> None:
        self.temporal_suspended = True
        self.temporal_prediction_this_frame = False
        self.temporal_last_mahalanobis = None
        self.temporal_fallback_reason = str(reason)
        trusted = (
            self.tracking_expected
            if self.tracking_expected is not None
            else self.last_matched
        )
        if trusted is not None and np.all(np.isfinite(trusted)):
            self.guide_endpoints = trusted.copy()
            self.temporal_tracker.reset(
                trusted, timestamp_s=self.last_profile_time_s
            )
        self.temporal_search_radius = self.config.tracking_endpoint_gate_m
        self.temporal_search_radii = np.full(
            2, self.config.tracking_endpoint_gate_m, dtype=float
        )

    def _restore_temporal_from_measurement(self, endpoints: np.ndarray) -> None:
        trusted = np.asarray(endpoints, dtype=float).reshape(2, 3)
        if self.config.temporal_tracking_enabled:
            self.temporal_tracker.reset(
                trusted, timestamp_s=self.last_profile_time_s
            )
        self.guide_endpoints = trusted.copy()
        self.temporal_prediction_this_frame = False
        self.temporal_last_mahalanobis = None
        self.temporal_search_radius = self.config.temporal_minimum_search_radius_m
        self.temporal_search_radii = np.full(
            2, self.config.temporal_minimum_search_radius_m, dtype=float
        )
        self.temporal_suspended = False
        self.temporal_fallback_reason = ""

    def _set_tracking_reference(self, endpoints: np.ndarray) -> None:
        values = np.asarray(endpoints, dtype=float).reshape(2, 3)
        length = float(np.linalg.norm(values[1] - values[0]))
        if np.isfinite(length) and length > 1.0e-9:
            self.tracking_reference_length = length

    # -- internal: topology gates ------------------------------------------
    def _tracking_pair_is_plausible(
        self, measured: np.ndarray, predicted: np.ndarray
    ) -> bool:
        measured = np.asarray(measured, dtype=float).reshape(2, 3)
        predicted = np.asarray(predicted, dtype=float).reshape(2, 3)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(predicted)):
            return False
        measured_vector = measured[1] - measured[0]
        predicted_vector = predicted[1] - predicted[0]
        measured_length = float(np.linalg.norm(measured_vector))
        predicted_length = float(np.linalg.norm(predicted_vector))
        if min(measured_length, predicted_length) <= 1.0e-9:
            return False
        if not (
            self.config.detector.minimum_segment_length_m
            <= measured_length
            <= self.config.detector.maximum_segment_length_m
        ):
            return False
        endpoint_step = np.linalg.norm(measured - predicted, axis=1)
        if float(np.max(endpoint_step)) > self.config.tracking_maximum_endpoint_step_m:
            return False

        predicted_direction = predicted_vector / predicted_length
        predicted_midpoint = np.mean(predicted, axis=0)
        first_coordinate = float(
            (measured[0] - predicted_midpoint) @ predicted_direction
        )
        second_coordinate = float(
            (measured[1] - predicted_midpoint) @ predicted_direction
        )
        if first_coordinate >= second_coordinate:
            return False

        if self.tracking_reference_length is not None:
            minimum = max(
                self.config.detector.minimum_segment_length_m,
                self.config.tracking_minimum_reference_length_ratio
                * self.tracking_reference_length,
            )
            maximum = min(
                self.config.detector.maximum_segment_length_m,
                self.config.tracking_maximum_reference_length_ratio
                * self.tracking_reference_length,
            )
            if not minimum <= measured_length <= maximum:
                return False
        return True

    def _reacquisition_geometry_is_continuous(self, matched: np.ndarray) -> bool:
        if not (
            self.mode == "LOST"
            and self.lost_from_mode == "TRACK"
            and self.temporal_suspended
        ):
            return True
        expected_vector = self.guide_endpoints[1] - self.guide_endpoints[0]
        measured_vector = matched[1] - matched[0]
        expected_length = float(np.linalg.norm(expected_vector))
        measured_length = float(np.linalg.norm(measured_vector))
        if min(expected_length, measured_length) <= 1.0e-9:
            return False
        if (
            abs(measured_length - expected_length)
            > self.config.reacquire_maximum_segment_length_change_m
        ):
            return False
        cosine = float(
            np.clip(
                abs(
                    float(
                        expected_vector @ measured_vector
                        / (expected_length * measured_length)
                    )
                ),
                0.0,
                1.0,
            )
        )
        angle = float(np.rad2deg(np.arccos(cosine)))
        return angle <= self.config.reacquire_maximum_segment_angle_change_deg

    # -- internal: partial update ------------------------------------------
    def _try_partial_temporal_update(
        self,
        profile: np.ndarray,
        *,
        candidate_sets: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        if not self.temporal_prediction_this_frame:
            return False
        if candidate_sets is None:
            candidate_sets = self.detector.temporal_breakpoint_candidates(
                profile,
                self.guide_endpoints,
                endpoint_gate_m=self.temporal_search_radii,
                maximum_candidates_per_endpoint=(
                    self.config.temporal_maximum_local_candidates
                ),
            )

        selected: dict[int, np.ndarray] = {}
        distances: dict[int, float] = {}
        for endpoint, candidates in enumerate(candidate_sets):
            result = self.temporal_tracker.select_endpoint_candidate(
                endpoint,
                candidates,
                measurement_sigma_m=self.config.temporal_partial_measurement_std_m,
            )
            if result is not None:
                selected[endpoint], distances[endpoint] = result
        for endpoint in tuple(selected):
            hypothesis = self.guide_endpoints.copy()
            hypothesis[endpoint] = selected[endpoint]
            if not self._tracking_pair_is_plausible(
                hypothesis, self.guide_endpoints
            ):
                selected.pop(endpoint)
                distances.pop(endpoint)
        if not selected:
            return False

        if len(selected) == 2:
            pair = np.vstack((selected[0], selected[1]))
            vector = pair[1] - pair[0]
            length = float(np.linalg.norm(vector))
            predicted_vector = self.guide_endpoints[1] - self.guide_endpoints[0]
            predicted_length = float(np.linalg.norm(predicted_vector))
            angle = float("inf")
            if length > 1.0e-12 and predicted_length > 1.0e-12:
                cosine = float(
                    np.clip(
                        abs(
                            float(vector @ predicted_vector)
                            / (length * predicted_length)
                        ),
                        0.0,
                        1.0,
                    )
                )
                angle = float(np.rad2deg(np.arccos(cosine)))
            if not (
                self._tracking_pair_is_plausible(
                    pair, self.guide_endpoints
                )
                and angle <= self.config.tracking_maximum_angle_difference_deg
            ):
                rejected = max(distances, key=distances.get)
                selected.pop(rejected)
                distances.pop(rejected)

        self.temporal_tracker.update_partial(
            selected,
            measurement_sigma_m=self.config.temporal_partial_measurement_std_m,
        )
        self.guide_endpoints = self.temporal_tracker.endpoints()
        self.temporal_search_radii = (
            self.temporal_tracker.endpoint_search_radii(
                minimum_m=self.config.temporal_minimum_search_radius_m,
                maximum_m=self.config.tracking_endpoint_gate_m,
                sigma_multiplier=self.config.temporal_search_sigma_multiplier,
            )
        )
        self.temporal_search_radius = float(np.max(self.temporal_search_radii))
        self.temporal_last_mahalanobis = tuple(
            distances.get(endpoint) for endpoint in (0, 1)
        )
        if (
            self.temporal_tracker.missed_frames
            >= self.config.temporal_maximum_coast_frames
        ):
            self._suspend_temporal_tracking("partial_endpoint_coast_exceeded")
            self.temporal_prediction_this_frame = False
            self.last_result = self._reject("partial_endpoint_coast_exceeded")
            return True

        observed = "_".join(f"e{endpoint + 1}" for endpoint in selected)
        # P0-1: a partial measurement is NOT a tracker failure.  The observed
        # endpoints were just update_partial()'d; routing through _reject()
        # would mark_missed() them a second time and drive missed_frames up to
        # an early temporal suspension.  The real node publishes a plain
        # REJECTED status without touching the tracker (node lines ~1458).
        self.last_result = self._frame_result(
            state="REJECTED",
            reason=f"partial_endpoint_update_{observed}",
            endpoints=None,
            surface_points=None,
            profile_points=len(profile),
        )
        return True

    # -- internal: results -------------------------------------------------
    def _reject(self, reason: str, *, profile_points: int = 0) -> FrameResult:
        if self.temporal_prediction_this_frame:
            self.temporal_tracker.mark_missed()
            if (
                self.temporal_tracker.missed_frames
                >= self.config.temporal_maximum_coast_frames
            ):
                self._suspend_temporal_tracking("maximum_coast_frames_exceeded")
        if self.mode == "ALIGN":
            self.alignment_stable_frames = 0
            self.previous_alignment_match = None
        elif self.mode == "LOST":
            self.reacquire_frames = 0
        elif self.config.guided_enabled and self.mode in {
            "TRACK",
            "PREDICTED_TRACK",
        }:
            self.lost_frames += 1
            if self.lost_frames >= self.config.maximum_lost_frames:
                self.lost_from_mode = self.mode
                self.mode = "LOST"
                self.reacquire_frames = 0
        return self._frame_result(
            state="LOST" if self.mode == "LOST" else "REJECTED",
            reason=reason,
            endpoints=None,
            surface_points=None,
            profile_points=profile_points,
        )

    def _valid_result(
        self, detection, matched: np.ndarray, *, profile_points: int
    ) -> FrameResult:
        self.accepted += 1
        return self._frame_result(
            state="VALID",
            reason="",
            endpoints=matched,
            surface_points=detection.surface_points,
            profile_points=profile_points,
            detection=detection,
        )

    def _frame_result(
        self,
        *,
        state: str,
        reason: str,
        endpoints: np.ndarray | None,
        surface_points: np.ndarray | None,
        profile_points: int,
        detection=None,
    ) -> FrameResult:
        normal_gate, endpoint_gate, angle_gate = self._guide_parameters()
        return FrameResult(
            frame_index=self.frames,
            timestamp_s=self.last_profile_time_s,
            state=state,
            reason=reason,
            mode=self.mode,
            endpoints=endpoints,
            surface_points=surface_points,
            guide_first_mm=(1000.0 * self.guide_endpoints[0]).tolist(),
            guide_second_mm=(1000.0 * self.guide_endpoints[1]).tolist(),
            guide_normal_gate_mm=1000.0 * normal_gate,
            guide_endpoint_gate_mm=1000.0 * endpoint_gate,
            guide_angle_gate_deg=angle_gate,
            acceptance_rate=self.accepted / max(self.frames, 1),
            accepted=self.accepted,
            frames=self.frames,
            support_points=(
                int(detection.support_count) if detection is not None else None
            ),
            target_surface_points=(
                int(len(detection.surface_points)) if detection is not None else None
            ),
            segment_length_mm=(
                1000.0 * detection.segment_length_m if detection is not None else None
            ),
            residual_rms_mm=(
                1000.0 * detection.residual_rms_m if detection is not None else None
            ),
            sample_pitch_mm=(
                1000.0 * detection.sample_pitch_m if detection is not None else None
            ),
            endpoint_sigma_mm=(
                1000.0 * detection.endpoint_sigma_m if detection is not None else None
            ),
            confidence=float(detection.confidence) if detection is not None else None,
            breakpoint_count=(
                int(detection.breakpoint_count) if detection is not None else None
            ),
            selection_mode=(
                str(detection.selection_mode) if detection is not None else None
            ),
            temporal_tracking=bool(
                self.temporal_tracking_requested
                and self.config.temporal_tracking_enabled
            ),
            temporal_initialized=self.temporal_tracker.initialized,
            temporal_missed_frames=self.temporal_tracker.missed_frames,
            temporal_missed_frames_by_endpoint=(
                self.temporal_tracker.missed_frames_by_endpoint.tolist()
            ),
            temporal_suspended=self.temporal_suspended,
            temporal_fallback_reason=self.temporal_fallback_reason,
            temporal_search_radius_mm=1000.0 * self.temporal_search_radius,
            temporal_search_radii_mm=(
                1000.0 * self.temporal_search_radii
            ).tolist(),
            temporal_mahalanobis=(
                None
                if self.temporal_last_mahalanobis is None
                else list(self.temporal_last_mahalanobis)
            ),
            tracking_reference_length_mm=(
                None
                if self.tracking_reference_length is None
                else 1000.0 * self.tracking_reference_length
            ),
            locked=self.mode not in {"ALIGN", "GLOBAL"},
            alignment_stable_frames=self.alignment_stable_frames,
            minimum_lock_frames=self.config.minimum_lock_frames,
            lost_frames=self.lost_frames,
            reacquire_frames=self.reacquire_frames,
            profile_points=profile_points,
        )

    def _make_alignment_template(self) -> np.ndarray:
        direction = np.array(
            [
                np.cos(np.deg2rad(self.config.alignment_template_angle_deg)),
                0.0,
                np.sin(np.deg2rad(self.config.alignment_template_angle_deg)),
            ]
        )
        half = 0.5 * self.config.alignment_template_length_m * direction
        center = np.array(
            [
                self.config.alignment_template_center_x_m,
                0.0,
                self.config.alignment_template_center_z_m,
            ]
        )
        return np.vstack((center - half, center + half))

    def _reset_guidance(self) -> None:
        self.mode = (
            "TRACK"
            if self.config.guided_enabled and self.config.initial_mode == "TRACK"
            else "ALIGN"
            if self.config.guided_enabled
            else "GLOBAL"
        )
        self.guide_endpoints = self.template_endpoints.copy()
        self.tracking_expected = None
        self.predicted_expected = None
        self.prediction_fallback = None
        self.last_matched = None
        self.previous_alignment_match = None
        self.alignment_stable_frames = 0
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.lost_from_mode = "TRACK"
        self.identity_initialized = False
        # P0-3: rebuild BOTH trackers and clear all temporal state, exactly
        # like the real node's _reset_guidance.  A stale Kalman velocity /
        # covariance would otherwise pollute the next experiment.
        self.tracker = EndpointTracker(
            ambiguity_ratio=self.config.identity_ambiguity_ratio
        )
        self.temporal_tracker = self._make_temporal_tracker()
        self.temporal_tracking_requested = False
        self.temporal_prediction_this_frame = False
        self.temporal_search_radius = self.config.temporal_minimum_search_radius_m
        self.temporal_search_radii = np.full(
            2, self.config.temporal_minimum_search_radius_m, dtype=float
        )
        self.temporal_last_mahalanobis = None
        self.temporal_suspended = False
        self.temporal_fallback_reason = ""
        self.tracking_reference_length = None
        self.tracking_reference_length = None


__all__ = [
    "BreakpointTrackingPipeline",
    "TrackingPipelineConfig",
    "FrameResult",
    "TemporalTracker",
    "KalmanTemporalTracker",
]
