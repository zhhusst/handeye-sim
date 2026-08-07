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
    minimum_segment_length_m: float = 0.02
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
            candidate = self._fit_candidate(xz[start:stop])
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

    @staticmethod
    def _fit_tls(xz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.mean(xz, axis=0)
        _, _, vh = np.linalg.svd(xz - center, full_matrices=False)
        direction = vh[0]
        direction /= max(float(np.linalg.norm(direction)), 1e-15)
        return center, direction

    def _fit_candidate(self, segment: np.ndarray) -> _LineCandidate | None:
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
