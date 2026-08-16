"""Robust endpoint extraction from an ordered metric X-Z laser profile.

The detector deliberately has no ROS or simulation dependency.  A real Gocator
driver and the repository's synthetic profile publisher therefore exercise the
same implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EndpointDetectionConfig:
    minimum_points: int = 12
    minimum_segment_points: int = 10
    minimum_segment_length_m: float = 0.01
    maximum_segment_length_m: float = 0.35
    absolute_neighbor_gap_m: float = 0.004
    neighbor_gap_multiplier: float = 8.0
    residual_mad_multiplier: float = 3.5
    residual_floor_m: float = 0.00008
    maximum_residual_rms_m: float = 0.0015
    endpoint_extension_fraction: float = 0.5
    endpoint_local_fit_points: int = 24
    candidate_ambiguity_ratio: float = 0.03
    maximum_fit_iterations: int = 1
    smoothing_window: int = 5
    local_fit_window: int = 12
    angle_change_threshold_deg: float = 10.0
    height_jump_threshold_m: float = 0.0002
    breakpoint_cluster_points: int = 8
    maximum_abs_surface_midpoint_x_m: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_points < 2:
            raise ValueError("minimum_points must be at least two")
        if self.minimum_segment_points < 2:
            raise ValueError("minimum_segment_points must be at least two")
        if self.minimum_segment_points > self.minimum_points:
            raise ValueError(
                "minimum_segment_points cannot exceed minimum_points"
            )
        positive = (
            "minimum_segment_length_m",
            "maximum_segment_length_m",
            "absolute_neighbor_gap_m",
            "neighbor_gap_multiplier",
            "residual_mad_multiplier",
            "residual_floor_m",
            "maximum_residual_rms_m",
            "maximum_fit_iterations",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_segment_length_m <= self.minimum_segment_length_m:
            raise ValueError(
                "maximum_segment_length_m must exceed minimum_segment_length_m"
            )
        if not 0.0 <= self.endpoint_extension_fraction <= 1.0:
            raise ValueError(
                "endpoint_extension_fraction must be in [0, 1]"
            )
        if self.endpoint_local_fit_points < 2:
            raise ValueError("endpoint_local_fit_points must be at least two")
        if not 0.0 <= self.candidate_ambiguity_ratio < 1.0:
            raise ValueError("candidate_ambiguity_ratio must be in [0, 1)")
        if self.smoothing_window < 1 or self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be a positive odd integer")
        if self.local_fit_window < 3:
            raise ValueError("local_fit_window must be at least three")
        if self.angle_change_threshold_deg <= 0.0:
            raise ValueError("angle_change_threshold_deg must be positive")
        if self.height_jump_threshold_m <= 0.0:
            raise ValueError("height_jump_threshold_m must be positive")
        if self.breakpoint_cluster_points < 1:
            raise ValueError("breakpoint_cluster_points must be positive")
        if self.maximum_abs_surface_midpoint_x_m < 0.0:
            raise ValueError(
                "maximum_abs_surface_midpoint_x_m must be non-negative"
            )


@dataclass(frozen=True)
class EndpointDetection:
    first: np.ndarray
    second: np.ndarray
    support_count: int
    raw_segment_count: int
    segment_length_m: float
    residual_rms_m: float
    sample_pitch_m: float
    endpoint_sigma_m: float
    confidence: float
    surface_points: np.ndarray
    breakpoint_count: int
    selection_mode: str

    def __post_init__(self) -> None:
        points = np.asarray(self.surface_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("surface_points must have shape (N, 3)")
        object.__setattr__(self, "surface_points", points)


@dataclass(frozen=True)
class _LineCandidate:
    detection: EndpointDetection
    score: float


class ProfileEndpointDetector:
    """Extract two boundary points from one ordered 2-D laser profile."""

    def __init__(self, config: EndpointDetectionConfig | None = None) -> None:
        self.config = config or EndpointDetectionConfig()
        self.last_rejection_reason = "not_processed"

    def detect(self, profile_points: np.ndarray) -> EndpointDetection | None:
        points = np.asarray(profile_points, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(points[:, (0, 2)]), axis=1)
        points = points[finite]
        if len(points) < self.config.minimum_points:
            self.last_rejection_reason = "insufficient_finite_points"
            return None

        xz = self._remove_isolated_spikes(points[:, (0, 2)])
        if len(xz) < self.config.minimum_points:
            self.last_rejection_reason = "insufficient_filtered_points"
            return None
        segments = self._contiguous_segments(xz)
        candidates: list[_LineCandidate] = []
        for start, stop in segments:
            if stop - start < self.config.minimum_segment_points:
                continue
            component = xz[start:stop]
            breakpoints = self._change_points(component)
            bounds = [0, *breakpoints, len(component)]
            for local_start, local_stop in zip(bounds[:-1], bounds[1:]):
                if local_stop - local_start < self.config.minimum_segment_points:
                    continue
                candidate = self._fit_candidate(
                    component[local_start:local_stop],
                    breakpoint_count=len(breakpoints),
                    selection_mode=(
                        "change_point" if breakpoints else "contiguous"
                    ),
                )
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            self.last_rejection_reason = "no_valid_linear_segment"
            return None

        candidates.sort(key=lambda item: item.score, reverse=True)
        if len(candidates) > 1:
            best, second = candidates[:2]
            scale = max(abs(best.score), abs(second.score), 1e-12)
            if (
                abs(best.score - second.score) / scale
                < self.config.candidate_ambiguity_ratio
            ):
                self.last_rejection_reason = "ambiguous_profile_segments"
                return None
        self.last_rejection_reason = ""
        return candidates[0].detection

    def detect_guided(
        self,
        profile_points: np.ndarray,
        expected_first: np.ndarray,
        expected_second: np.ndarray,
        *,
        normal_gate_m: float,
        endpoint_gate_m: float,
        maximum_angle_difference_deg: float = 25.0,
        selection_mode: str = "guided",
    ) -> EndpointDetection | None:
        """Extract only the finite segment associated with a supplied guide.

        The guide is a data-association prior, never a replacement for a
        measured endpoint.  Raw samples must lie inside its finite corridor;
        the returned line and both endpoints are still fitted exclusively from
        sensor samples.  This deliberately avoids the stateless global choice
        between a target surface, a side wall and a workbench line.
        """
        if normal_gate_m <= 0.0 or endpoint_gate_m <= 0.0:
            raise ValueError("guided detection gates must be positive")
        if maximum_angle_difference_deg <= 0.0:
            raise ValueError(
                "maximum_angle_difference_deg must be positive"
            )
        points = np.asarray(profile_points, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(points[:, (0, 2)]), axis=1)
        points = points[finite]
        if len(points) < self.config.minimum_points:
            self.last_rejection_reason = "insufficient_finite_points"
            return None

        expected = np.asarray(
            [expected_first, expected_second], dtype=float
        ).reshape(2, 3)[:, (0, 2)]
        guide_vector = expected[1] - expected[0]
        guide_length = float(np.linalg.norm(guide_vector))
        if guide_length <= 1e-9:
            raise ValueError("guided segment must have non-zero length")
        guide_direction = guide_vector / guide_length
        guide_normal = np.array(
            [-guide_direction[1], guide_direction[0]]
        )

        xz = points[:, (0, 2)]
        delta = xz - expected[0]
        longitudinal = delta @ guide_direction
        normal_distance = np.abs(delta @ guide_normal)
        selected_indices = np.flatnonzero(
            (longitudinal >= -endpoint_gate_m)
            & (longitudinal <= guide_length + endpoint_gate_m)
            & (normal_distance <= normal_gate_m)
        )
        if len(selected_indices) < self.config.minimum_segment_points:
            self.last_rejection_reason = "guided_roi_has_too_few_points"
            return None

        # Gocator points are acquisition ordered.  Preserve separate runs so
        # two unrelated surfaces inside a broad corridor cannot be merged into
        # one artificial line.  A few rejected samples are tolerated for
        # dropout and isolated outliers.
        maximum_index_gap = max(2, self.config.smoothing_window)
        run_breaks = (
            np.flatnonzero(np.diff(selected_indices) > maximum_index_gap) + 1
        )
        runs = np.split(selected_indices, run_breaks)
        associated: list[tuple[float, EndpointDetection]] = []
        expected_first_xz, expected_second_xz = expected
        for indices in runs:
            if len(indices) < self.config.minimum_segment_points:
                continue
            component = xz[indices]
            breakpoints = self._change_points(component)
            # A wide guide is only a recall corridor.  It can contain the
            # target top surface together with a side wall or workbench.  Fit
            # the whole run (important for a clean, narrow alignment ROI) and
            # also every change-point-delimited physical segment.  Without
            # this second split, widening the recovery gates paradoxically
            # merges several surfaces and makes the known target disappear.
            candidate_components = [component]
            if breakpoints:
                bounds = [0, *breakpoints, len(component)]
                candidate_components.extend(
                    component[start:stop]
                    for start, stop in zip(bounds[:-1], bounds[1:])
                    if stop - start >= self.config.minimum_segment_points
                )
                # Weak height/slope responses inside a real plate top are
                # common in high-resolution Gocator data.  The two physical
                # plate edges are therefore often *not* adjacent entries in
                # ``bounds``.  Restricting candidates to adjacent spans made
                # the ALIGN stage lock a short internal ripple even though
                # both expected physical endpoints were supplied by the ROI.
                #
                # Build a small Cartesian product of measured boundaries near
                # the two guide endpoints.  This crosses internal weak
                # responses without turning the guide itself into a measured
                # endpoint: every span still begins and ends at an observed
                # gap/change boundary and is fitted from raw samples only.
                boundary_points = np.asarray(
                    [
                        component[0]
                        if index == 0
                        else component[-1]
                        if index == len(component)
                        else component[index]
                        for index in bounds
                    ],
                    dtype=float,
                )
                boundary_salience: dict[int, float] = {}
                guided_spans: set[tuple[int, int]] = set()
                maximum_local_boundaries = 8
                for expected_order in (expected, expected[::-1]):
                    local_sets: list[list[int]] = []
                    for endpoint in expected_order:
                        distances = np.linalg.norm(
                            boundary_points - endpoint, axis=1
                        )
                        eligible = np.flatnonzero(
                            distances <= endpoint_gate_m
                        )
                        # Rank by persistent local edge shape first and use
                        # guide distance only as a soft tie-breaker.  Dense
                        # weak responses can otherwise fill a nearest-N list
                        # before the real, slightly misaligned plate edge is
                        # even evaluated.
                        salience = np.asarray(
                            [
                                boundary_salience.setdefault(
                                    int(candidate),
                                    self._sustained_boundary_salience(
                                        component, bounds[int(candidate)]
                                    ),
                                )
                                for candidate in eligible
                            ],
                            dtype=float,
                        )
                        rank_score = (
                            salience
                            - 0.15
                            * distances[eligible]
                            / max(endpoint_gate_m, 1.0e-12)
                        )
                        ordered = eligible[np.argsort(-rank_score)]
                        local_sets.append(
                            [
                                int(bounds[index])
                                for index in ordered[
                                    :maximum_local_boundaries
                                ]
                            ]
                        )
                    for first_boundary in local_sets[0]:
                        for second_boundary in local_sets[1]:
                            start = min(first_boundary, second_boundary)
                            stop = max(first_boundary, second_boundary)
                            if (
                                stop - start
                                >= self.config.minimum_segment_points
                            ):
                                guided_spans.add((start, stop))
                candidate_components.extend(
                    component[start:stop]
                    for start, stop in guided_spans
                )
            for candidate_points in candidate_components:
                candidate = self._fit_candidate(
                    candidate_points,
                    breakpoint_count=len(breakpoints),
                    selection_mode=selection_mode,
                )
                if candidate is None:
                    continue
                detection = candidate.detection
                measured = np.asarray(
                    [detection.first, detection.second], dtype=float
                )[:, (0, 2)]
                direct_errors = np.linalg.norm(measured - expected, axis=1)
                swapped_errors = np.linalg.norm(
                    measured - expected[::-1], axis=1
                )
                if float(np.max(swapped_errors)) < float(
                    np.max(direct_errors)
                ):
                    endpoint_errors = swapped_errors
                else:
                    endpoint_errors = direct_errors
                maximum_endpoint_error = float(np.max(endpoint_errors))
                if maximum_endpoint_error > endpoint_gate_m:
                    continue
                measured_direction = measured[1] - measured[0]
                measured_direction /= max(
                    float(np.linalg.norm(measured_direction)), 1e-15
                )
                angle_error = self._unoriented_angle_deg(
                    measured_direction, guide_direction
                )
                if angle_error > maximum_angle_difference_deg:
                    continue
                measured_midpoint = 0.5 * (measured[0] + measured[1])
                guide_midpoint = 0.5 * (
                    expected_first_xz + expected_second_xz
                )
                midpoint_normal_error = abs(
                    float((measured_midpoint - guide_midpoint) @ guide_normal)
                )
                if midpoint_normal_error > normal_gate_m:
                    continue
                # Association cost, unlike the global quality score, is
                # anchored to the finite expected target.  A longer workbench
                # segment gains no advantage merely because it has more data.
                association_cost = (
                    float(np.mean(endpoint_errors)) / endpoint_gate_m
                    + midpoint_normal_error / normal_gate_m
                    + angle_error / maximum_angle_difference_deg
                )
                associated.append((float(association_cost), detection))

        if not associated:
            self.last_rejection_reason = "guided_target_not_found"
            return None
        associated.sort(key=lambda item: item[0])
        self.last_rejection_reason = ""
        return associated[0][1]

    def detect_temporal_breakpoint_pair(
        self,
        profile_points: np.ndarray,
        predicted_first: np.ndarray,
        predicted_second: np.ndarray,
        *,
        endpoint_gate_m: float,
        normal_gate_m: float,
        maximum_angle_difference_deg: float = 25.0,
        selection_mode: str = "seed_temporal_track",
    ) -> EndpointDetection | None:
        """Measure the physical segment bounded by two tracked breakpoints.

        Unlike :meth:`detect_guided`, this method never uses an arbitrary ROI
        crop as a segment endpoint.  The prediction only gates physical
        acquisition gaps or slope/height change points.  Candidate surface
        points are exactly the ordered samples between two adjacent physical
        boundaries, followed by the normal robust line/inlier validation.
        """
        if endpoint_gate_m <= 0.0 or normal_gate_m <= 0.0:
            raise ValueError("temporal breakpoint gates must be positive")
        if maximum_angle_difference_deg <= 0.0:
            raise ValueError(
                "maximum_angle_difference_deg must be positive"
            )
        points = np.asarray(profile_points, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(points[:, (0, 2)]), axis=1)
        xz = points[finite][:, (0, 2)]
        if len(xz) < self.config.minimum_points:
            self.last_rejection_reason = "insufficient_finite_points"
            return None
        xz = self._remove_isolated_spikes(xz)
        if len(xz) < self.config.minimum_points:
            self.last_rejection_reason = "insufficient_filtered_points"
            return None

        expected = np.asarray(
            [predicted_first, predicted_second], dtype=float
        ).reshape(2, 3)[:, (0, 2)]
        guide_vector = expected[1] - expected[0]
        guide_length = float(np.linalg.norm(guide_vector))
        if guide_length <= 1.0e-9:
            raise ValueError("tracked breakpoint pair must have non-zero length")
        guide_direction = guide_vector / guide_length
        guide_normal = np.array([-guide_direction[1], guide_direction[0]])
        guide_midpoint = np.mean(expected, axis=0)

        associated: list[tuple[float, EndpointDetection]] = []
        for component_start, component_stop in self._contiguous_segments(xz):
            component = xz[component_start:component_stop]
            if len(component) < self.config.minimum_segment_points:
                continue
            changes = self._change_points(component)
            boundaries = [0, *changes, len(component)]
            # Real high-resolution Gocator profiles can raise several weak
            # change responses *inside* a slightly curved/noisy plate surface.
            # Requiring adjacent change points then truncates the target at an
            # internal false response.  The temporal prediction already tells
            # us where both physical boundaries should be, so retain only the
            # few boundary indices nearest each predicted endpoint and fit all
            # unique spans formed by those two local sets.  Span endpoints are
            # still measured change/gap boundaries, never arbitrary ROI cuts.
            boundary_points = np.asarray(
                [
                    component[0]
                    if index == 0
                    else component[-1]
                    if index == len(component)
                    else component[index]
                    for index in boundaries
                ],
                dtype=float,
            )
            boundary_salience: dict[int, float] = {}
            # Three candidates per endpoint cover the clustered response on
            # each physical edge while bounding the expensive robust fits to
            # at most 3 x 3 unique spans per component.
            maximum_local_boundaries = 3
            candidate_spans: set[tuple[int, int]] = set()
            for expected_order in (expected, expected[::-1]):
                local_sets: list[list[int]] = []
                for endpoint in expected_order:
                    distances = np.linalg.norm(
                        boundary_points - endpoint, axis=1
                    )
                    eligible = np.flatnonzero(distances <= endpoint_gate_m)
                    salience = np.asarray(
                        [
                            boundary_salience.setdefault(
                                int(candidate),
                                self._sustained_boundary_salience(
                                    component, boundaries[int(candidate)]
                                ),
                            )
                            for candidate in eligible
                        ],
                        dtype=float,
                    )
                    rank_score = (
                        salience
                        - 0.15
                        * distances[eligible]
                        / max(endpoint_gate_m, 1.0e-12)
                    )
                    ordered = eligible[np.argsort(-rank_score)]
                    local_sets.append(
                        [
                            int(boundaries[index])
                            for index in ordered[:maximum_local_boundaries]
                        ]
                    )
                for first_boundary in local_sets[0]:
                    for second_boundary in local_sets[1]:
                        start = min(first_boundary, second_boundary)
                        stop = max(first_boundary, second_boundary)
                        if (
                            stop - start
                            >= self.config.minimum_segment_points
                        ):
                            candidate_spans.add((start, stop))

            for start, stop in candidate_spans:
                if stop - start < self.config.minimum_segment_points:
                    continue
                candidate = self._fit_candidate(
                    component[start:stop],
                    breakpoint_count=len(changes),
                    selection_mode=selection_mode,
                )
                if candidate is None:
                    continue
                detection = candidate.detection
                measured = np.asarray(
                    [detection.first, detection.second], dtype=float
                )[:, (0, 2)]
                direct_errors = np.linalg.norm(measured - expected, axis=1)
                swapped_errors = np.linalg.norm(
                    measured[::-1] - expected, axis=1
                )
                endpoint_errors = (
                    swapped_errors
                    if float(np.max(swapped_errors))
                    < float(np.max(direct_errors))
                    else direct_errors
                )
                maximum_endpoint_error = float(np.max(endpoint_errors))
                if maximum_endpoint_error > endpoint_gate_m:
                    continue
                measured_direction = measured[1] - measured[0]
                measured_direction /= max(
                    float(np.linalg.norm(measured_direction)), 1.0e-15
                )
                angle_error = self._unoriented_angle_deg(
                    measured_direction, guide_direction
                )
                if angle_error > maximum_angle_difference_deg:
                    continue
                midpoint = np.mean(measured, axis=0)
                midpoint_normal_error = abs(
                    float((midpoint - guide_midpoint) @ guide_normal)
                )
                if midpoint_normal_error > normal_gate_m:
                    continue
                association_cost = (
                    maximum_endpoint_error / endpoint_gate_m
                    + midpoint_normal_error / normal_gate_m
                    + angle_error / maximum_angle_difference_deg
                    + detection.residual_rms_m
                    / self.config.maximum_residual_rms_m
                )
                associated.append((float(association_cost), detection))

        if not associated:
            self.last_rejection_reason = "tracked_breakpoint_pair_not_found"
            return None
        associated.sort(key=lambda item: item[0])
        self.last_rejection_reason = ""
        return associated[0][1]

    def temporal_breakpoint_candidates(
        self,
        profile_points: np.ndarray,
        predicted_endpoints: np.ndarray,
        *,
        endpoint_gate_m: float | tuple[float, float],
        maximum_candidates_per_endpoint: int = 6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return local physical-boundary candidates for each tracked endpoint.

        The full detector intentionally publishes only a fitted two-endpoint
        observation.  During a short occlusion, however, one plate boundary
        can remain visible while the other disappears.  This helper exposes
        nearby acquisition gaps and slope/height change points to the temporal
        filter without manufacturing a complete segment or calibration datum.
        """
        if maximum_candidates_per_endpoint < 1:
            raise ValueError("maximum_candidates_per_endpoint must be positive")
        gates = np.asarray(endpoint_gate_m, dtype=float)
        if gates.ndim == 0:
            gates = np.repeat(gates, 2)
        gates = gates.reshape(2)
        if not np.all(np.isfinite(gates)) or np.any(gates <= 0.0):
            raise ValueError("endpoint candidate gates must be positive")

        points = np.asarray(profile_points, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(points[:, (0, 2)]), axis=1)
        xz = points[finite][:, (0, 2)]
        if len(xz) < self.config.minimum_points:
            return np.empty((0, 3)), np.empty((0, 3))
        xz = self._remove_isolated_spikes(xz)
        if len(xz) < self.config.minimum_points:
            return np.empty((0, 3)), np.empty((0, 3))

        boundary_records: list[tuple[np.ndarray, float]] = []
        for component_start, component_stop in self._contiguous_segments(xz):
            component = xz[component_start:component_stop]
            if len(component) < self.config.minimum_segment_points:
                continue
            changes = self._change_points(component)
            indices = [0, *changes, len(component) - 1]
            boundary_records.extend(
                (
                    component[index].copy(),
                    self._sustained_boundary_salience(component, index),
                )
                for index in indices
            )
        if not boundary_records:
            return np.empty((0, 3)), np.empty((0, 3))

        # Adjacent change windows can describe the same physical edge.  Merge
        # them before association so duplicated candidates do not dominate a
        # local endpoint search.
        candidates: list[np.ndarray] = []
        candidate_salience: list[float] = []
        merge_radius = max(
            2.0 * self.config.residual_floor_m,
            0.25 * self.config.absolute_neighbor_gap_m,
        )
        for point, salience in boundary_records:
            duplicate = next(
                (
                    index
                    for index, existing in enumerate(candidates)
                    if np.linalg.norm(point - existing) <= merge_radius
                ),
                None,
            )
            if duplicate is None:
                candidates.append(point)
                candidate_salience.append(float(salience))
            elif salience > candidate_salience[duplicate]:
                candidates[duplicate] = point
                candidate_salience[duplicate] = float(salience)
        candidate_xz = np.asarray(candidates, dtype=float).reshape(-1, 2)
        salience_values = np.asarray(candidate_salience, dtype=float)
        expected = np.asarray(predicted_endpoints, dtype=float).reshape(2, 3)[
            :, (0, 2)
        ]
        selected: list[np.ndarray] = []
        for endpoint in (0, 1):
            distances = np.linalg.norm(candidate_xz - expected[endpoint], axis=1)
            eligible = np.flatnonzero(distances <= gates[endpoint])
            rank_score = (
                salience_values[eligible]
                - 0.15 * distances[eligible] / gates[endpoint]
            )
            eligible = eligible[np.argsort(-rank_score)]
            local = candidate_xz[eligible[:maximum_candidates_per_endpoint]]
            selected.append(
                np.column_stack(
                    (local[:, 0], np.zeros(len(local)), local[:, 1])
                )
                if len(local)
                else np.empty((0, 3))
            )
        return selected[0], selected[1]

    def _nominal_pitch_and_gap(self, xz: np.ndarray) -> tuple[float, float]:
        neighbor_distance = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        positive = neighbor_distance[
            np.isfinite(neighbor_distance) & (neighbor_distance > 1e-12)
        ]
        if len(positive) == 0:
            return 0.0, self.config.absolute_neighbor_gap_m
        lower_half = positive[positive <= np.median(positive)]
        nominal_pitch = float(
            np.median(lower_half if len(lower_half) else positive)
        )
        threshold = max(
            self.config.absolute_neighbor_gap_m,
            self.config.neighbor_gap_multiplier * nominal_pitch,
        )
        return nominal_pitch, threshold

    def _remove_isolated_spikes(self, xz: np.ndarray) -> np.ndarray:
        if len(xz) < 3:
            return xz
        _, threshold = self._nominal_pitch_and_gap(xz)
        previous_jump = np.linalg.norm(xz[1:-1] - xz[:-2], axis=1)
        next_jump = np.linalg.norm(xz[2:] - xz[1:-1], axis=1)
        bypass_jump = np.linalg.norm(xz[2:] - xz[:-2], axis=1)
        isolated = (
            (previous_jump > threshold)
            & (next_jump > threshold)
            & (bypass_jump <= threshold)
        )
        keep = np.ones(len(xz), dtype=bool)
        keep[1:-1] = ~isolated
        return xz[keep]

    def _contiguous_segments(self, xz: np.ndarray) -> list[tuple[int, int]]:
        if len(xz) < 2:
            return [(0, len(xz))]
        neighbor_distance = np.linalg.norm(np.diff(xz, axis=0), axis=1)
        _, threshold = self._nominal_pitch_and_gap(xz)
        breaks = np.flatnonzero(neighbor_distance > threshold) + 1
        bounds = np.concatenate(([0], breaks, [len(xz)]))
        return [
            (int(start), int(stop))
            for start, stop in zip(bounds[:-1], bounds[1:])
        ]

    def _smooth_for_change_detection(self, xz: np.ndarray) -> np.ndarray:
        """Median-filter X and Z without changing the points used by the fit."""
        radius = self.config.smoothing_window // 2
        if radius == 0 or len(xz) < self.config.smoothing_window:
            return xz.copy()
        padded = np.pad(xz, ((radius, radius), (0, 0)), mode="edge")
        width = self.config.smoothing_window
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, width, axis=0
        )
        return np.median(windows, axis=-1)

    @staticmethod
    def _unoriented_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
        cosine = float(
            np.clip(abs(float(first @ second)), 0.0, 1.0)
        )
        return float(np.rad2deg(np.arccos(cosine)))

    def _change_point_scores(
        self, xz: np.ndarray
    ) -> list[tuple[int, float]]:
        """Find local direction changes and small parallel height steps.

        A pure slope test misses a thin plate lying parallel to a table.  The
        orthogonal separation between the left and right local TLS lines is
        therefore evaluated whenever their directions remain approximately
        parallel.  Thick side walls are detected by the direction term.
        """
        window = self.config.local_fit_window
        if len(xz) < 2 * window + self.config.minimum_segment_points:
            return []
        smoothed = self._smooth_for_change_detection(xz)
        angle_threshold = self.config.angle_change_threshold_deg
        # Shape: (number of windows, window length, XZ).  Calculating all
        # local directions in one NumPy operation avoids thousands of Python
        # calls for a full-resolution Gocator profile.
        local_windows = np.lib.stride_tricks.sliding_window_view(
            smoothed, window, axis=0
        ).transpose(0, 2, 1)
        minimum_separation = max(2, window // 3)
        first, second = np.triu_indices(
            window, k=minimum_separation
        )
        vectors = (
            local_windows[:, second, :] - local_windows[:, first, :]
        )
        norms = np.linalg.norm(vectors, axis=2)
        normalized = np.full_like(vectors, np.nan)
        np.divide(
            vectors,
            norms[:, :, None],
            out=normalized,
            where=norms[:, :, None] > 1e-12,
        )
        with np.errstate(all="ignore"):
            directions = np.nanmedian(normalized, axis=1)
        direction_norms = np.linalg.norm(directions, axis=1)
        invalid = (~np.isfinite(direction_norms)) | (
            direction_norms <= 1e-12
        )
        if np.any(invalid):
            fallback = local_windows[:, -1, :] - local_windows[:, 0, :]
            fallback_norms = np.linalg.norm(fallback, axis=1)
            usable = invalid & (fallback_norms > 1e-12)
            directions[usable] = (
                fallback[usable] / fallback_norms[usable, None]
            )
            direction_norms[usable] = 1.0
        directions = np.divide(
            directions,
            direction_norms[:, None],
            out=np.zeros_like(directions),
            where=direction_norms[:, None] > 1e-12,
        )

        candidate_count = len(smoothed) - 2 * window + 1
        left_windows = local_windows[:candidate_count]
        right_windows = local_windows[
            window : window + candidate_count
        ]
        left_directions = directions[:candidate_count]
        right_directions = directions[
            window : window + candidate_count
        ]
        dot = np.einsum(
            "ni,ni->n", left_directions, right_directions
        )
        angle = np.rad2deg(
            np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
        )
        right_directions = np.where(
            (dot < 0.0)[:, None], -right_directions, right_directions
        )
        average_direction = left_directions + right_directions
        average_norm = np.linalg.norm(average_direction, axis=1)
        average_direction = np.divide(
            average_direction,
            average_norm[:, None],
            out=np.zeros_like(average_direction),
            where=average_norm[:, None] > 1e-12,
        )
        average_normal = np.column_stack(
            (-average_direction[:, 1], average_direction[:, 0])
        )
        left_projection = np.einsum(
            "nwi,ni->nw", left_windows, average_normal
        )
        right_projection = np.einsum(
            "nwi,ni->nw", right_windows, average_normal
        )
        separation = np.abs(
            np.median(right_projection, axis=1)
            - np.median(left_projection, axis=1)
        )
        angle_score = angle / angle_threshold
        # Only interpret the line offset as a height step while the local
        # surfaces are near parallel.  At a real corner, the angle score is
        # the meaningful quantity.
        height_score = np.where(
            angle <= angle_threshold,
            separation / self.config.height_jump_threshold_m,
            0.0,
        )
        scores = np.maximum(angle_score, height_score)
        raw_candidates = [
            (int(offset + window), float(scores[offset]))
            for offset in np.flatnonzero(scores >= 1.0)
        ]
        if not raw_candidates:
            return []

        # One physical edge usually raises the score across several adjacent
        # windows. Non-maximum suppression returns one breakpoint per edge.
        suppression = max(
            self.config.breakpoint_cluster_points,
            self.config.local_fit_window,
        )
        selected: list[tuple[int, float]] = []
        for index, score in sorted(
            raw_candidates, key=lambda item: item[1], reverse=True
        ):
            if all(abs(index - other) > suppression for other, _ in selected):
                selected.append((index, score))
        return sorted(selected, key=lambda item: item[0])

    def _change_points(self, xz: np.ndarray) -> list[int]:
        """Return non-max-suppressed breakpoint indices.

        The scored form is kept separately because temporal data association
        must distinguish a strong physical plate edge from a merely nearby
        weak ripple.  Stateless segment extraction continues to consume only
        the indices and therefore retains its public behaviour.
        """
        return [index for index, _ in self._change_point_scores(xz)]

    def _sustained_boundary_salience(
        self, xz: np.ndarray, index: int
    ) -> float:
        """Score a breakpoint by the shape sustained on both of its sides.

        The short-window change detector is intentionally sensitive and thus
        produces many responses on a noisy real profile.  A physical plate
        edge, unlike a one- or two-sample spike, separates two locally stable
        pieces of surface over several millimetres.  This score measures that
        persistent step/direction change after leaving a small guard band
        around the transition itself.

        It is used only to rank already measured change/gap candidates.  It
        cannot create an endpoint at an arbitrary ROI boundary.
        """
        if not 0 < int(index) < len(xz):
            return 0.0
        local_pitch_samples = xz[
            max(0, int(index) - 64) : min(len(xz), int(index) + 65)
        ]
        nominal_pitch, _ = self._nominal_pitch_and_gap(local_pitch_samples)
        if nominal_pitch <= 1.0e-12:
            return 0.0
        # Four millimetres covers many Gocator samples and rejects narrow
        # speckle spikes, while remaining local relative to the plate chord.
        side_count = int(np.clip(np.ceil(0.004 / nominal_pitch), 24, 96))
        guard_count = int(np.clip(np.ceil(0.0006 / nominal_pitch), 3, 16))
        left_stop = int(index) - guard_count
        right_start = int(index) + guard_count
        if left_stop < 4 or right_start + 4 > len(xz):
            return 0.0
        left = xz[max(0, left_stop - side_count) : left_stop]
        right = xz[right_start : min(len(xz), right_start + side_count)]
        if len(left) < 8 or len(right) < 8:
            return 0.0

        # The long side windows are median-smoothed before TLS.  This is both
        # more stable and substantially cheaper than enumerating every point
        # pair for every candidate at profile rate.
        left = self._smooth_for_change_detection(left)
        right = self._smooth_for_change_detection(right)
        left_center, left_direction = self._fit_tls(left)
        right_center, right_direction = self._fit_tls(right)
        if float((left[-1] - left[0]) @ left_direction) < 0.0:
            left_direction = -left_direction
        if float((right[-1] - right[0]) @ right_direction) < 0.0:
            right_direction = -right_direction
        if float(left_direction @ right_direction) < 0.0:
            right_direction = -right_direction
        average_direction = left_direction + right_direction
        norm = float(np.linalg.norm(average_direction))
        if norm <= 1.0e-12:
            average_direction = left_direction
        else:
            average_direction /= norm
        normal = np.array([-average_direction[1], average_direction[0]])
        step = abs(float(np.median(right @ normal) - np.median(left @ normal)))
        angle = self._unoriented_angle_deg(left_direction, right_direction)

        left_normal = np.array([-left_direction[1], left_direction[0]])
        right_normal = np.array([-right_direction[1], right_direction[0]])
        left_noise = 1.4826 * float(
            np.median(np.abs((left - left_center) @ left_normal))
        )
        right_noise = 1.4826 * float(
            np.median(np.abs((right - right_center) @ right_normal))
        )
        noise = max(
            left_noise + right_noise,
            2.0 * self.config.residual_floor_m,
        )
        step_score = step / max(
            self.config.height_jump_threshold_m,
            2.5 * noise,
        )
        angle_score = angle / self.config.angle_change_threshold_deg
        # A sustained height step is the dominant signature for the thin
        # plate-on-table experiment; the angle term preserves thick side-wall
        # and true-corner support without letting it overwhelm persistence.
        return float(step_score + 0.35 * angle_score)

    @staticmethod
    def _fit_tls(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.mean(xz, axis=0)
        _, _, vh = np.linalg.svd(xz - center, full_matrices=False)
        direction = vh[0]
        direction /= max(float(np.linalg.norm(direction)), 1e-15)
        return center, direction

    def _fit_local_tls_robust(
        self, xz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Median pair-direction line used only by breakpoint detection.

        A least-squares line in a short window has high leverage: one 3 mm
        spike can rotate it enough to imitate a physical corner. Long pair
        directions have a 50% breakdown-style median and remain valid for
        horizontal, sloped and vertical target/side segments.
        """
        minimum_separation = max(2, len(xz) // 3)
        first, second = np.triu_indices(
            len(xz), k=minimum_separation
        )
        vectors = xz[second] - xz[first]
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-12
        if not np.any(valid):
            return self._fit_tls(xz)
        directions = vectors[valid] / norms[valid, None]
        direction = np.median(directions, axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return self._fit_tls(xz)
        return np.median(xz, axis=0), direction / norm

    def _fit_candidate(
        self,
        segment: np.ndarray,
        *,
        breakpoint_count: int = 0,
        selection_mode: str = "contiguous",
    ) -> _LineCandidate | None:
        inliers = np.ones(len(segment), dtype=bool)
        center = np.zeros(2)
        direction = np.array([1.0, 0.0])
        for _ in range(self.config.maximum_fit_iterations):
            if np.count_nonzero(inliers) < self.config.minimum_segment_points:
                return None
            center, direction = self._fit_tls(segment[inliers])
            normal = np.array([-direction[1], direction[0]])
            signed = (segment - center) @ normal
            centered_residual = signed - np.median(signed[inliers])
            mad = float(np.median(np.abs(centered_residual[inliers])))
            sigma = max(1.4826 * mad, self.config.residual_floor_m)
            updated = np.abs(centered_residual) <= (
                self.config.residual_mad_multiplier * sigma
            )
            if np.array_equal(updated, inliers):
                break
            inliers = updated

        support = segment[inliers]
        if len(support) < self.config.minimum_segment_points:
            return None
        center, direction = self._fit_tls(support)
        # Preserve acquisition order.  This is only a candidate order; the
        # temporal tracker assigns the persistent e1/e2 physical identity.
        acquisition_delta = support[-1] - support[0]
        if float(acquisition_delta @ direction) < 0.0:
            direction = -direction
        projection = (support - center) @ direction
        order = np.argsort(projection)
        projection = projection[order]
        ordered_support = support[order]
        if len(projection) < 2:
            return None
        steps = np.diff(projection)
        positive_steps = steps[steps > 1e-12]
        if len(positive_steps) == 0:
            return None
        sample_pitch = float(np.median(positive_steps))
        extension = self.config.endpoint_extension_fraction * sample_pitch
        local_count = min(
            int(self.config.endpoint_local_fit_points), len(ordered_support)
        )

        def local_endpoint(local: np.ndarray, *, first: bool) -> tuple[np.ndarray, float]:
            local_center, local_direction = self._fit_tls(local)
            acquisition_delta = local[-1] - local[0]
            if float(acquisition_delta @ local_direction) < 0.0:
                local_direction = -local_direction
            local_projection = (local - local_center) @ local_direction
            boundary_projection = (
                float(np.min(local_projection)) - extension
                if first
                else float(np.max(local_projection)) + extension
            )
            local_normal = np.array([-local_direction[1], local_direction[0]])
            local_residual = (local - local_center) @ local_normal
            local_rms = float(np.sqrt(np.mean(local_residual**2)))
            return local_center + boundary_projection * local_direction, local_rms

        # Segment selection remains global and robust.  Boundary extrapolation
        # is local so a gently curved physical profile does not bias both
        # endpoints through one global straight-line fit.
        first_xz, first_local_rms = local_endpoint(
            ordered_support[:local_count], first=True
        )
        second_xz, second_local_rms = local_endpoint(
            ordered_support[-local_count:], first=False
        )
        segment_length = float(np.linalg.norm(second_xz - first_xz))
        if not (
            self.config.minimum_segment_length_m
            <= segment_length
            <= self.config.maximum_segment_length_m
        ):
            return None
        midpoint_x = 0.5 * float(first_xz[0] + second_xz[0])
        if (
            self.config.maximum_abs_surface_midpoint_x_m > 0.0
            and abs(midpoint_x)
            > self.config.maximum_abs_surface_midpoint_x_m
        ):
            return None

        normal = np.array([-direction[1], direction[0]])
        residual = (support - center) @ normal
        residual_rms = float(np.sqrt(np.mean(residual**2)))
        if residual_rms > self.config.maximum_residual_rms_m:
            return None
        endpoint_sigma = float(
            np.hypot(
                max(
                    first_local_rms,
                    second_local_rms,
                    self.config.residual_floor_m,
                ),
                sample_pitch / np.sqrt(12.0),
            )
        )
        coverage = len(support) / max(len(segment), 1)
        residual_quality = np.exp(
            -0.5
            * (residual_rms / self.config.maximum_residual_rms_m) ** 2
        )
        support_quality = min(
            1.0, len(support) / max(2.0 * self.config.minimum_segment_points, 1.0)
        )
        confidence = float(
            np.clip(coverage * residual_quality * support_quality, 0.0, 1.0)
        )
        first = np.array([first_xz[0], 0.0, first_xz[1]], dtype=float)
        second = np.array([second_xz[0], 0.0, second_xz[1]], dtype=float)
        detection = EndpointDetection(
            first=first,
            second=second,
            support_count=int(len(support)),
            raw_segment_count=int(len(segment)),
            segment_length_m=segment_length,
            residual_rms_m=residual_rms,
            sample_pitch_m=sample_pitch,
            endpoint_sigma_m=endpoint_sigma,
            confidence=confidence,
            surface_points=np.column_stack(
                (support[:, 0], np.zeros(len(support)), support[:, 1])
            ),
            breakpoint_count=int(breakpoint_count),
            selection_mode=selection_mode,
        )
        # Favor long, densely supported, straight segments without making the
        # score depend on an arbitrary absolute point count alone.
        score = float(
            segment_length
            * np.sqrt(len(support))
            * max(confidence, 1e-6)
            / max(endpoint_sigma, self.config.residual_floor_m)
        )
        return _LineCandidate(detection=detection, score=score)
