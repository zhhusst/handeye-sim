"""Online ROI/CSRT bilateral-breakpoint tracking without ROS dependencies.

The operator-assisted ALIGN phase still performs a bounded geometric
detection.  Once locked, two image trackers follow local breakpoint
neighbourhoods while the published endpoints and target surface always come
from the current metric profile.  Tracker boxes are therefore association
priors, never calibration measurements by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time

import numpy as np

from ..perception.tracking_pipeline import FrameResult
from .base import ROITracker
from .factory import create_tracker
from .rasterizer import ProfileRasterizer, RasterizerConfig
from .segment_detector import plate_edge_from_midpoint
from .types import TargetROI, TrackingFrame


@dataclass
class ROIBreakpointPipelineConfig:
    initial_first_label: str = "e1"
    alignment_template_center_x_m: float = 0.0
    alignment_template_center_z_m: float = 0.28
    alignment_template_length_m: float = 0.08
    alignment_template_angle_deg: float = 25.0
    alignment_normal_gate_m: float = 0.003
    alignment_endpoint_gate_m: float = 0.020
    alignment_maximum_angle_difference_deg: float = 15.0
    alignment_stability_m: float = 0.0015
    minimum_lock_frames: int = 5
    minimum_segment_length_m: float = 0.010
    maximum_segment_length_m: float = 0.35
    tracker_name: str = "csrt"
    roi_size_m: float = 0.020
    raster_resolution_m_per_pixel: float = 0.00025
    raster_point_radius_px: int = 2
    raster_margin_m: float = 0.020
    raster_maximum_dimension_px: int = 2400
    core_fraction: float = 0.70
    roi_jump_m: float = 0.030
    breakpoint_jump_m: float = 0.030
    fail_streak_frames: int = 3
    reacquire_stable_frames: int = 3
    reacquire_endpoint_gate_m: float = 0.025
    reacquire_stability_m: float = 0.003
    tracking_minimum_chord_ratio: float = 0.20
    tracking_maximum_chord_ratio: float = 6.00
    plate_residual_threshold_m: float = 0.0008
    surface_residual_threshold_m: float = 0.0012
    minimum_surface_points: int = 8
    endpoint_sigma_floor_m: float = 0.00008

    def __post_init__(self) -> None:
        self.initial_first_label = self.initial_first_label.strip().lower()
        self.tracker_name = self.tracker_name.strip().lower()
        if self.initial_first_label not in {"e1", "e2"}:
            raise ValueError("initial_first_label must be e1 or e2")
        positive = {
            "alignment_template_length_m": self.alignment_template_length_m,
            "alignment_normal_gate_m": self.alignment_normal_gate_m,
            "alignment_endpoint_gate_m": self.alignment_endpoint_gate_m,
            "alignment_stability_m": self.alignment_stability_m,
            "minimum_segment_length_m": self.minimum_segment_length_m,
            "maximum_segment_length_m": self.maximum_segment_length_m,
            "roi_size_m": self.roi_size_m,
            "raster_resolution_m_per_pixel": self.raster_resolution_m_per_pixel,
            "raster_margin_m": self.raster_margin_m,
            "roi_jump_m": self.roi_jump_m,
            "breakpoint_jump_m": self.breakpoint_jump_m,
            "reacquire_endpoint_gate_m": self.reacquire_endpoint_gate_m,
            "reacquire_stability_m": self.reacquire_stability_m,
            "plate_residual_threshold_m": self.plate_residual_threshold_m,
            "surface_residual_threshold_m": self.surface_residual_threshold_m,
            "endpoint_sigma_floor_m": self.endpoint_sigma_floor_m,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_segment_length_m <= self.minimum_segment_length_m:
            raise ValueError("maximum_segment_length_m must exceed the minimum")
        if not 0.0 < self.core_fraction <= 1.0:
            raise ValueError("core_fraction must be in (0, 1]")
        if not 0.0 < self.tracking_minimum_chord_ratio <= 1.0:
            raise ValueError("tracking_minimum_chord_ratio must be in (0, 1]")
        if self.tracking_maximum_chord_ratio < 1.0:
            raise ValueError("tracking_maximum_chord_ratio must be >= 1")
        for name in (
            "minimum_lock_frames",
            "fail_streak_frames",
            "reacquire_stable_frames",
            "raster_point_radius_px",
            "raster_maximum_dimension_px",
            "minimum_surface_points",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


class ROIBreakpointPipeline:
    """ALIGN/TRACK/PREDICTED_TRACK/LOST state machine using two ROI trackers."""

    def __init__(
        self,
        config: ROIBreakpointPipelineConfig | None = None,
        *,
        tracker_factory: Callable[[str, ProfileRasterizer], ROITracker] = create_tracker,
    ) -> None:
        self.config = config or ROIBreakpointPipelineConfig()
        self._tracker_factory = tracker_factory
        self.template_endpoints = self._make_alignment_template()
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.accepted = 0
        self.mode = "ALIGN"
        self.lost_from_mode = "TRACK"
        self.guide_endpoints = self.template_endpoints.copy()
        self.last_matched: np.ndarray | None = None
        self.last_surface_points: np.ndarray | None = None
        self.last_rois: list[TargetROI] | None = None
        self.reference_snapshot: dict | None = None
        self.prediction_fallback: np.ndarray | None = None
        self.predicted_expected: np.ndarray | None = None
        self.reacquire_anchor: np.ndarray | None = None
        self.previous_alignment_match: np.ndarray | None = None
        self.alignment_stable_frames = 0
        self.fail_streak = 0
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.reacquire_previous: np.ndarray | None = None
        self.tracking_reference_length: float | None = None
        self.rasterizer: ProfileRasterizer | None = None
        self.trackers: list[ROITracker] = []
        self.last_profile: np.ndarray | None = None
        self.last_timestamp_s: float | None = None
        self.last_result: FrameResult | None = None
        self.last_tracker_runtime_ms = 0.0
        self.last_failure_details: list[str] = []

    # -- public state-machine surface ---------------------------------
    def process_profile(
        self, profile: np.ndarray, timestamp_s: float | None = None
    ) -> FrameResult:
        self.frames += 1
        self.last_timestamp_s = timestamp_s
        try:
            values = self._finite_profile(profile)
        except (TypeError, ValueError) as error:
            return self._reject(f"invalid_profile:{error}", 0)
        self.last_profile = values.copy()
        if self.mode == "ALIGN":
            return self._process_align(values)
        if self.mode == "LOST" or not self.trackers:
            return self._process_reacquire(values)
        return self._process_track(values)

    def lock(self) -> bool:
        if self.mode == "TRACK" and self.trackers:
            return True
        if (
            self.mode != "ALIGN"
            or self.last_matched is None
            or self.last_profile is None
            or self.last_result is None
            or self.last_result.state != "VALID"
            or self.alignment_stable_frames < self.config.minimum_lock_frames
        ):
            return False
        if not self._initialize_trackers(
            self.last_profile, self.last_matched, self.last_timestamp_s
        ):
            return False
        self.mode = "TRACK"
        self.guide_endpoints = self.last_matched.copy()
        self.fail_streak = 0
        self.lost_frames = 0
        self._set_reference_length(self.last_matched)
        self.reference_snapshot = self.snapshot()
        return True

    def handle_control(self, command: str) -> None:
        command = str(command).strip().upper()
        if command in {"SEED_TRACK_START", "SEED_TRACK_STOP"}:
            return
        if command == "REFERENCE_REACQUIRE":
            if self.reference_snapshot is not None:
                self.restore_snapshot(self.reference_snapshot, from_mode="TRACK")
            return
        if command == "PREDICTION_COMMIT":
            if self.last_matched is not None:
                self.mode = "TRACK"
                self.guide_endpoints = self.last_matched.copy()
                self.predicted_expected = None
                self.prediction_fallback = None
                self.reacquire_anchor = None
                self.fail_streak = 0
                self.lost_frames = 0
            return
        if command == "PREDICTION_CANCEL":
            if self.prediction_fallback is not None:
                self._arm_reacquisition(self.prediction_fallback, "TRACK")
            else:
                self.reset()
            self.predicted_expected = None

    def handle_prior(self, prior: np.ndarray) -> None:
        values = self._valid_endpoint_pair(prior)
        if values is None:
            return
        self.prediction_fallback = (
            None if self.last_matched is None else self.last_matched.copy()
        )
        self.predicted_expected = values.copy()
        self._arm_reacquisition(values, "PREDICTED_TRACK")

    def handle_measured_prior(self, prior: np.ndarray) -> None:
        values = self._valid_endpoint_pair(prior)
        if values is None:
            return
        self.prediction_fallback = (
            None if self.last_matched is None else self.last_matched.copy()
        )
        self._arm_reacquisition(values, "TRACK")

    def snapshot(self) -> dict | None:
        if self.last_matched is None:
            return None
        rois = self.last_rois or self._rois_from_endpoints(self.last_matched)
        return {
            "endpoints": self.last_matched.copy(),
            "rois": [roi.to_dict() for roi in rois],
            "timestamp_s": self.last_timestamp_s,
        }

    def restore_snapshot(self, snapshot: dict, *, from_mode: str = "TRACK") -> bool:
        if not snapshot or "endpoints" not in snapshot:
            return False
        values = self._valid_endpoint_pair(snapshot["endpoints"])
        if values is None:
            return False
        self._arm_reacquisition(values, from_mode)
        return True

    def status_extras(self) -> dict[str, object]:
        return {
            "backend": "roi",
            "tracker": self.config.tracker_name,
            "fail_streak": self.fail_streak,
            "fail_streak_frames": self.config.fail_streak_frames,
            "tracker_runtime_ms": self.last_tracker_runtime_ms,
            "tracker_failures": list(self.last_failure_details),
            "roi_boxes_mm": []
            if not self.last_rois
            else [
                {key.replace("_m", "_mm"): 1000.0 * value for key, value in roi.to_dict().items()}
                for roi in self.last_rois
            ],
            "raster_resolution_mm_per_pixel": None
            if self.rasterizer is None
            else 1000.0 * self.rasterizer.config.resolution_m_per_pixel,
        }

    # -- ALIGN ---------------------------------------------------------
    def _process_align(self, profile: np.ndarray) -> FrameResult:
        candidate = self._detect_plate(profile, self.config.alignment_template_center_x_m)
        if candidate is None:
            self._record_alignment_failure()
            return self._reject("roi_alignment_breakpoints_not_found", len(profile))
        raw_pair, surface, metrics = candidate
        matched = self._initial_order(raw_pair)
        if not self._alignment_pair_is_plausible(matched):
            self._record_alignment_failure()
            return self._reject("roi_alignment_template_gate_rejected", len(profile))
        self.fail_streak = 0
        if self.previous_alignment_match is None:
            self.alignment_stable_frames = 1
        else:
            step = float(
                np.max(np.linalg.norm(matched - self.previous_alignment_match, axis=1))
            )
            self.alignment_stable_frames = (
                self.alignment_stable_frames + 1
                if step <= self.config.alignment_stability_m
                else 1
            )
        self.previous_alignment_match = matched.copy()
        self.last_matched = matched.copy()
        self.last_surface_points = surface.copy()
        self.guide_endpoints = self.template_endpoints.copy()
        return self._valid(matched, surface, metrics, len(profile), "roi_align")

    def _record_alignment_failure(self) -> None:
        """Ignore isolated high-rate bad profiles while the operator aligns.

        The real Gocator stream contains periodic single-frame failures.  The
        offline CSRT validation used stride=4, so resetting a five-frame lock
        counter on every raw bad frame made the online ALIGN service almost
        impossible to call at the right instant.  Three consecutive failures
        still invalidate the accumulated alignment evidence.
        """
        # Keep the last *valid* alignment observation. Invalid Gocator frames
        # carry no contrary geometry; the next valid pair is compared with the
        # previous valid pair and resets the counter if it really moved.
        self.fail_streak += 1

    # -- TRACK ---------------------------------------------------------
    def _process_track(self, profile: np.ndarray) -> FrameResult:
        assert self.rasterizer is not None
        frame = TrackingFrame(
            0.0 if self.last_timestamp_s is None else self.last_timestamp_s,
            profile,
            self.rasterizer.rasterize(profile),
        )
        results = []
        t0 = time.perf_counter()
        for tracker in self.trackers:
            try:
                results.append(tracker.update(frame))
            except Exception as error:  # OpenCV errors must not kill the ROS node.
                results.append(None)
                self.last_failure_details = [f"tracker_exception:{error}"]
        self.last_tracker_runtime_ms = 1000.0 * (time.perf_counter() - t0)
        failures: list[str] = []
        if len(results) != 2 or any(item is None or not item.success or item.roi is None for item in results):
            failures.append("roi_tracker_update_failed")
            rois = None
        else:
            rois = [item.roi for item in results]
        candidate = None
        matched = surface = metrics = None
        if rois is not None:
            x_mid = 0.5 * (rois[0].center[0] + rois[1].center[0])
            candidate = self._detect_plate(profile, x_mid)
            if candidate is None:
                failures.append("no_breakpoint_pair")
            else:
                raw_pair, surface, metrics = candidate
                matched = self._order_to_centers(raw_pair, rois)
                failures.extend(self._validate_tracked_pair(matched, rois))
        if failures:
            self.last_failure_details = failures
            self.fail_streak += 1
            if self.fail_streak >= self.config.fail_streak_frames:
                self._arm_reacquisition(
                    self.last_matched if self.last_matched is not None else self.guide_endpoints,
                    self.mode,
                )
            return self._reject(";".join(failures), len(profile))

        assert matched is not None and surface is not None and metrics is not None
        self.fail_streak = 0
        self.lost_frames = 0
        self.last_failure_details = []
        self.last_matched = matched.copy()
        self.last_surface_points = surface.copy()
        self.last_rois = rois
        self.guide_endpoints = matched.copy()
        return self._valid(matched, surface, metrics, len(profile), "roi_csrt_track")

    # -- LOST / reacquisition -----------------------------------------
    def _arm_reacquisition(self, endpoints: np.ndarray, from_mode: str) -> None:
        self.reacquire_anchor = np.asarray(endpoints, dtype=float).reshape(2, 3).copy()
        self.guide_endpoints = self.reacquire_anchor.copy()
        self.mode = "LOST"
        self.lost_from_mode = from_mode
        self.trackers = []
        self.last_rois = None
        self.reacquire_frames = 0
        self.reacquire_previous = None
        self.lost_frames = max(self.lost_frames, self.fail_streak)

    def _process_reacquire(self, profile: np.ndarray) -> FrameResult:
        anchor = self.reacquire_anchor
        if anchor is None:
            anchor = self.last_matched
        if anchor is None:
            return self._reject("roi_reacquire_without_anchor", len(profile))
        candidate = self._detect_plate(profile, float(np.mean(anchor[:, 0])))
        if candidate is None:
            self.reacquire_frames = 0
            self.reacquire_previous = None
            self.lost_frames += 1
            return self._reject("roi_reacquire_breakpoints_not_found", len(profile))
        raw_pair, surface, metrics = candidate
        matched = self._order_to_endpoints(raw_pair, anchor)
        distance = np.linalg.norm(matched - anchor, axis=1)
        if float(np.max(distance)) > self.config.reacquire_endpoint_gate_m:
            self.reacquire_frames = 0
            self.reacquire_previous = None
            self.lost_frames += 1
            return self._reject("roi_reacquire_endpoint_gate_rejected", len(profile))
        if not self._chord_is_plausible(matched):
            self.reacquire_frames = 0
            self.reacquire_previous = None
            self.lost_frames += 1
            return self._reject("roi_reacquire_chord_rejected", len(profile))
        if self.reacquire_previous is None:
            self.reacquire_frames = 1
        else:
            step = float(
                np.max(np.linalg.norm(matched - self.reacquire_previous, axis=1))
            )
            self.reacquire_frames = (
                self.reacquire_frames + 1
                if step <= self.config.reacquire_stability_m
                else 1
            )
        self.reacquire_previous = matched.copy()
        if self.reacquire_frames < self.config.reacquire_stable_frames:
            return self._reject("roi_reacquire_pending", len(profile))
        if not self._initialize_trackers(profile, matched, self.last_timestamp_s):
            self.reacquire_frames = 0
            return self._reject("roi_reacquire_tracker_init_failed", len(profile))
        self.mode = self.lost_from_mode
        self.fail_streak = 0
        self.lost_frames = 0
        self.reacquire_frames = 0
        self.reacquire_anchor = None
        self.last_matched = matched.copy()
        self.last_surface_points = surface.copy()
        self.guide_endpoints = matched.copy()
        return self._valid(matched, surface, metrics, len(profile), "roi_reacquire")

    # -- geometry / image helpers -------------------------------------
    @staticmethod
    def _finite_profile(profile: np.ndarray) -> np.ndarray:
        values = np.asarray(profile, dtype=float).reshape(-1, 3)
        values = values[np.all(np.isfinite(values[:, (0, 2)]), axis=1)]
        if len(values) < 20:
            raise ValueError("insufficient_finite_points")
        return values

    def _detect_plate(self, profile: np.ndarray, x_mid_m: float):
        pair_xz = plate_edge_from_midpoint(
            profile[:, (0, 2)],
            x_mid_m,
            residual_threshold_m=self.config.plate_residual_threshold_m,
        )
        if pair_xz is None:
            return None
        pair_xz = np.asarray(pair_xz, dtype=float).reshape(2, 2)
        length = float(np.linalg.norm(pair_xz[1] - pair_xz[0]))
        if not self.config.minimum_segment_length_m <= length <= self.config.maximum_segment_length_m:
            return None
        xmin, xmax = sorted(pair_xz[:, 0])
        between = profile[(profile[:, 0] >= xmin) & (profile[:, 0] <= xmax)]
        if len(between) < self.config.minimum_surface_points:
            return None
        slope, intercept = np.polyfit(pair_xz[:, 0], pair_xz[:, 1], 1)
        residual = np.abs(between[:, 2] - (slope * between[:, 0] + intercept))
        surface = between[residual <= self.config.surface_residual_threshold_m]
        if len(surface) < self.config.minimum_surface_points:
            return None
        signed = surface[:, 2] - (slope * surface[:, 0] + intercept)
        rms = float(np.sqrt(np.mean(signed * signed)))
        ordered_x = np.sort(np.unique(surface[:, 0]))
        pitch = float(np.median(np.diff(ordered_x))) if len(ordered_x) > 1 else 0.0
        sigma = max(self.config.endpoint_sigma_floor_m, rms)
        confidence = float(np.clip(1.0 - rms / self.config.surface_residual_threshold_m, 0.0, 1.0))
        pair = np.column_stack((pair_xz[:, 0], np.zeros(2), pair_xz[:, 1]))
        metrics = {
            "segment_length_m": length,
            "residual_rms_m": rms,
            "sample_pitch_m": pitch,
            "endpoint_sigma_m": sigma,
            "confidence": confidence,
        }
        return pair, surface, metrics

    def _initial_order(self, pair: np.ndarray) -> np.ndarray:
        return pair.copy() if self.config.initial_first_label == "e1" else pair[::-1].copy()

    @staticmethod
    def _order_to_endpoints(pair: np.ndarray, expected: np.ndarray) -> np.ndarray:
        direct = float(np.sum((pair - expected) ** 2))
        swapped = float(np.sum((pair[::-1] - expected) ** 2))
        return pair.copy() if direct <= swapped else pair[::-1].copy()

    def _order_to_centers(self, pair: np.ndarray, rois: list[TargetROI]) -> np.ndarray:
        centers = np.array([[roi.center[0], 0.0, roi.center[1]] for roi in rois])
        return self._order_to_endpoints(pair, centers)

    def _alignment_pair_is_plausible(self, pair: np.ndarray) -> bool:
        unordered = self._order_to_endpoints(pair, self.template_endpoints)
        endpoint_error = np.linalg.norm(unordered - self.template_endpoints, axis=1)
        if float(np.max(endpoint_error)) > self.config.alignment_endpoint_gate_m:
            return False
        template_vector = self.template_endpoints[1] - self.template_endpoints[0]
        measured_vector = unordered[1] - unordered[0]
        template_length = float(np.linalg.norm(template_vector))
        measured_length = float(np.linalg.norm(measured_vector))
        if min(template_length, measured_length) <= 1e-9:
            return False
        cosine = float(np.clip(abs(template_vector @ measured_vector) / (template_length * measured_length), 0.0, 1.0))
        angle = float(np.rad2deg(np.arccos(cosine)))
        if angle > self.config.alignment_maximum_angle_difference_deg:
            return False
        direction = template_vector / template_length
        normal = np.array([-direction[2], 0.0, direction[0]])
        midpoint_offset = np.mean(unordered, axis=0) - np.mean(self.template_endpoints, axis=0)
        return abs(float(midpoint_offset @ normal)) <= self.config.alignment_normal_gate_m

    def _validate_tracked_pair(self, pair: np.ndarray, rois: list[TargetROI]) -> list[str]:
        failures: list[str] = []
        for index, roi in enumerate(rois):
            point = pair[index, (0, 2)].reshape(1, 2)
            if not roi.core(self.config.core_fraction).contains(point)[0]:
                failures.append(f"e{index + 1}_outside_roi_core")
        if self.last_rois is not None:
            for index, (current, previous) in enumerate(zip(rois, self.last_rois)):
                if float(np.linalg.norm(current.center - previous.center)) > self.config.roi_jump_m:
                    failures.append(f"roi{index + 1}_jump")
        if self.last_matched is not None:
            steps = np.linalg.norm(pair - self.last_matched, axis=1)
            for index, step in enumerate(steps):
                if float(step) > self.config.breakpoint_jump_m:
                    failures.append(f"e{index + 1}_jump")
        if not self._chord_is_plausible(pair):
            failures.append("chord_length_rejected")
        return failures

    def _chord_is_plausible(self, pair: np.ndarray) -> bool:
        length = float(np.linalg.norm(pair[1] - pair[0]))
        if not self.config.minimum_segment_length_m <= length <= self.config.maximum_segment_length_m:
            return False
        if self.tracking_reference_length is None:
            return True
        return (
            self.config.tracking_minimum_chord_ratio * self.tracking_reference_length
            <= length
            <= self.config.tracking_maximum_chord_ratio * self.tracking_reference_length
        )

    def _initialize_trackers(
        self, profile: np.ndarray, endpoints: np.ndarray, timestamp_s: float | None
    ) -> bool:
        try:
            if self.rasterizer is None or not self._profile_fits_raster(profile):
                self.rasterizer = self._make_rasterizer(profile)
            frame = TrackingFrame(
                0.0 if timestamp_s is None else timestamp_s,
                profile,
                self.rasterizer.rasterize(profile),
            )
            rois = self._rois_from_endpoints(endpoints)
            trackers = [
                self._tracker_factory(self.config.tracker_name, self.rasterizer)
                for _ in range(2)
            ]
            for tracker, roi in zip(trackers, rois):
                tracker.initialize(frame, roi)
        except Exception:
            self.trackers = []
            return False
        self.trackers = trackers
        self.last_rois = rois
        return True

    def _make_rasterizer(self, profile: np.ndarray) -> ProfileRasterizer:
        x = profile[:, 0]
        z = profile[:, 2]
        xmin, xmax = np.quantile(x, [0.001, 0.999])
        zmin, zmax = np.quantile(z, [0.001, 0.999])
        margin = self.config.raster_margin_m
        xmin, xmax = float(xmin - margin), float(xmax + margin)
        zmin, zmax = float(zmin - margin), float(zmax + margin)
        resolution = self.config.raster_resolution_m_per_pixel
        maximum_span = max(xmax - xmin, zmax - zmin)
        resolution = max(
            resolution,
            maximum_span / float(self.config.raster_maximum_dimension_px),
        )
        return ProfileRasterizer(
            RasterizerConfig(
                x_min_m=xmin,
                x_max_m=xmax,
                z_min_m=zmin,
                z_max_m=zmax,
                resolution_m_per_pixel=resolution,
                point_radius_px=self.config.raster_point_radius_px,
            )
        )

    def _profile_fits_raster(self, profile: np.ndarray) -> bool:
        assert self.rasterizer is not None
        cfg = self.rasterizer.config
        xlo, xhi = np.quantile(profile[:, 0], [0.01, 0.99])
        zlo, zhi = np.quantile(profile[:, 2], [0.01, 0.99])
        return bool(
            xlo >= cfg.x_min_m and xhi <= cfg.x_max_m
            and zlo >= cfg.z_min_m and zhi <= cfg.z_max_m
        )

    def _rois_from_endpoints(self, endpoints: np.ndarray) -> list[TargetROI]:
        half = 0.5 * self.config.roi_size_m
        return [
            TargetROI.from_bbox(
                point[0] - half,
                point[2] - half,
                point[0] + half,
                point[2] + half,
            )
            for point in endpoints
        ]

    def _set_reference_length(self, endpoints: np.ndarray) -> None:
        length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if np.isfinite(length) and length > 1e-9:
            self.tracking_reference_length = length

    @staticmethod
    def _valid_endpoint_pair(prior: np.ndarray) -> np.ndarray | None:
        try:
            values = np.asarray(prior, dtype=float).reshape(2, 3)
        except (TypeError, ValueError):
            return None
        if not np.all(np.isfinite(values)) or np.linalg.norm(values[1] - values[0]) <= 1e-9:
            return None
        return values

    def _make_alignment_template(self) -> np.ndarray:
        angle = np.deg2rad(self.config.alignment_template_angle_deg)
        direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
        center = np.array(
            [
                self.config.alignment_template_center_x_m,
                0.0,
                self.config.alignment_template_center_z_m,
            ]
        )
        half = 0.5 * self.config.alignment_template_length_m * direction
        return np.vstack((center - half, center + half))

    # -- result construction ------------------------------------------
    def _valid(
        self,
        endpoints: np.ndarray,
        surface: np.ndarray,
        metrics: dict,
        profile_points: int,
        selection_mode: str,
    ) -> FrameResult:
        self.accepted += 1
        return self._result(
            "VALID",
            "",
            endpoints,
            surface,
            profile_points,
            metrics=metrics,
            selection_mode=selection_mode,
        )

    def _reject(self, reason: str, profile_points: int) -> FrameResult:
        state = "LOST" if self.mode == "LOST" else "REJECTED"
        return self._result(state, reason, None, None, profile_points)

    def _result(
        self,
        state: str,
        reason: str,
        endpoints: np.ndarray | None,
        surface_points: np.ndarray | None,
        profile_points: int,
        *,
        metrics: dict | None = None,
        selection_mode: str | None = None,
    ) -> FrameResult:
        metrics = metrics or {}
        result = FrameResult(
            frame_index=self.frames,
            timestamp_s=self.last_timestamp_s,
            state=state,
            reason=reason,
            mode=self.mode,
            endpoints=endpoints,
            surface_points=surface_points,
            guide_first_mm=(1000.0 * self.guide_endpoints[0]).tolist(),
            guide_second_mm=(1000.0 * self.guide_endpoints[1]).tolist(),
            guide_normal_gate_mm=1000.0 * self.config.alignment_normal_gate_m,
            guide_endpoint_gate_mm=1000.0 * (
                self.config.alignment_endpoint_gate_m
                if self.mode == "ALIGN"
                else self.config.reacquire_endpoint_gate_m
            ),
            guide_angle_gate_deg=self.config.alignment_maximum_angle_difference_deg,
            acceptance_rate=self.accepted / max(self.frames, 1),
            accepted=self.accepted,
            frames=self.frames,
            support_points=None if surface_points is None else int(len(surface_points)),
            target_surface_points=None if surface_points is None else int(len(surface_points)),
            segment_length_mm=None if "segment_length_m" not in metrics else 1000.0 * metrics["segment_length_m"],
            residual_rms_mm=None if "residual_rms_m" not in metrics else 1000.0 * metrics["residual_rms_m"],
            sample_pitch_mm=None if "sample_pitch_m" not in metrics else 1000.0 * metrics["sample_pitch_m"],
            endpoint_sigma_mm=None if "endpoint_sigma_m" not in metrics else 1000.0 * metrics["endpoint_sigma_m"],
            confidence=metrics.get("confidence"),
            breakpoint_count=2 if endpoints is not None else None,
            selection_mode=selection_mode,
            temporal_tracking=False,
            temporal_initialized=bool(self.trackers),
            temporal_missed_frames=self.fail_streak,
            temporal_missed_frames_by_endpoint=[self.fail_streak, self.fail_streak],
            temporal_suspended=self.mode == "LOST",
            temporal_fallback_reason=reason if self.mode == "LOST" else "",
            temporal_search_radius_mm=1000.0 * 0.5 * self.config.roi_size_m,
            temporal_search_radii_mm=[
                1000.0 * 0.5 * self.config.roi_size_m,
                1000.0 * 0.5 * self.config.roi_size_m,
            ],
            temporal_mahalanobis=None,
            tracking_reference_length_mm=None
            if self.tracking_reference_length is None
            else 1000.0 * self.tracking_reference_length,
            locked=self.mode != "ALIGN",
            alignment_stable_frames=self.alignment_stable_frames,
            minimum_lock_frames=self.config.minimum_lock_frames,
            lost_frames=self.lost_frames,
            reacquire_frames=self.reacquire_frames,
            profile_points=profile_points,
        )
        self.last_result = result
        return result


__all__ = ["ROIBreakpointPipeline", "ROIBreakpointPipelineConfig"]
