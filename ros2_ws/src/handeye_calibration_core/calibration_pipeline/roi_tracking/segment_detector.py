#!/usr/bin/env python3
"""Breakpoint detection inside a ROI using line-segment segmentation.

Two classic 2-D line-segmentation methods are provided:

  - split_and_merge(): recursive splitting on max perpendicular distance,
    then an optional merge pass for collinear adjacent segments.
  - iepf(): Iterative End Point Fit, the same recursive split core with
    strict "line through the two endpoints" residuals.

The output is a list of breakpoints (the junctions between segments).
For a step/plate edge the detector returns BOTH corners of the step; the
caller must pick which one is the true plate edge (e.g. the corner closest
to the OTHER ROI's centre, since the plate-surface corner lies toward the
plate interior).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentDetectionConfig:
    max_perp_distance_m: float = 0.0015   # split threshold (1.5 mm)
    min_segment_points: int = 5
    merge_angle_deg: float = 3.0          # merge collinear segments closer than this
    merge_distance_m: float = 0.0015
    eps: float = 1.0e-9


def _perp_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the infinite line through a-b."""
    ab = b - a
    norm = np.hypot(ab[0], ab[1])
    if norm < 1.0e-12:
        return np.hypot(points[:, 0] - a[0], points[:, 1] - a[1])
    abu = ab / norm
    n = np.array([-abu[1], abu[0]])
    return np.abs((points - a) @ n)


def _split(points: np.ndarray, start: int, end: int, cfg: SegmentDetectionConfig,
           segments: list[tuple[int, int]]) -> None:
    """Recursive split of points[start:end+1] (already ordered)."""
    if end - start + 1 < 2:
        return
    a, b = points[start], points[end]
    d = _perp_distance(points[start:end + 1], a, b)
    idx = int(np.argmax(d))
    dmax = float(d[idx])
    if dmax > cfg.max_perp_distance_m and (idx + 1) >= cfg.min_segment_points \
            and (end - start + 1 - idx) >= cfg.min_segment_points:
        _split(points, start, start + idx, cfg, segments)
        _split(points, start + idx, end, cfg, segments)
    else:
        segments.append((start, end))


def _merge(points: np.ndarray, segments: list[tuple[int, int]],
           cfg: SegmentDetectionConfig) -> list[tuple[int, int]]:
    """Greedy merge of adjacent segments that are nearly collinear."""
    if not segments:
        return segments
    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        # vectors of the two segments
        va = points[prev[1]] - points[prev[0]]
        vb = points[seg[1]] - points[seg[0]]
        na = np.hypot(va[0], va[1]); nb = np.hypot(vb[0], vb[1])
        if na < cfg.eps or nb < cfg.eps:
            merged.append(seg)
            continue
        cosang = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
        ang = float(np.rad2deg(np.arccos(cosang)))
        # distance between the two segments' junction points
        gap = float(np.hypot(*(points[prev[1]] - points[seg[0]])))
        if ang <= cfg.merge_angle_deg and gap <= cfg.merge_distance_m:
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)
    return merged


def _breakpoints_from_segments(points: np.ndarray,
                               segments: list[tuple[int, int]]) -> np.ndarray:
    """Junction points between consecutive segments (each segment's last point)."""
    bps = []
    for i in range(len(segments) - 1):
        bps.append(points[segments[i][1]])
    return np.asarray(bps, dtype=float).reshape(-1, 2)


def split_and_merge(points_xz: np.ndarray,
                    cfg: SegmentDetectionConfig | None = None) -> np.ndarray:
    """Split-and-Merge line segmentation; returns breakpoints (N,2) X-Z."""
    cfg = cfg or SegmentDetectionConfig()
    pts = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return np.zeros((0, 2))
    # order by X (profile is scanned along X)
    pts = pts[np.argsort(pts[:, 0])]
    segments: list[tuple[int, int]] = []
    _split(pts, 0, len(pts) - 1, cfg, segments)
    segments = _merge(pts, segments, cfg)
    return _breakpoints_from_segments(pts, segments)


def detect_step_corners(points_xz: np.ndarray,
                         z_diff_threshold_m: float = 0.00012,
                         pad_points: int = 1) -> np.ndarray | None:
    """Locate the step (plate edge) inside the ROI.

    Returns the two corner points (plate end, table end) as (2, 2) X-Z, or
    None when no step is present.  The step zone is found by thresholding the
    smoothed z difference; the corners are padded by ``pad_points`` on each
    side to sit on the flat runs rather than inside the transition.

    The caller must decide which corner is the plate-surface breakpoint --
    typically the corner closest to the OTHER ROI's centre.
    """
    pts = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    if len(pts) < 8:
        return None
    pts = pts[np.argsort(pts[:, 0])]
    z = pts[:, 1]
    # light smoothing before differencing
    win = 3
    zs = np.convolve(z, np.ones(win) / win, mode="valid")
    dz = np.abs(np.diff(zs))
    steep = dz > z_diff_threshold_m
    if not np.any(steep):
        return None
    i0 = int(np.argmax(steep))               # first steep gap (between i0, i0+1)
    i1 = int(len(steep) - 1 - np.argmax(steep[::-1]))
    # pad outward so corners sit on the flat runs
    a = max(0, i0 - pad_points)
    b = min(len(pts) - 1, i1 + 1 + pad_points)
    if b - a < 3:
        return None
    return np.array([pts[a], pts[b]], dtype=float)


def detect_step_intersection(points_xz: np.ndarray,
                             z_diff_threshold_m: float = 0.00012,
                             min_flat_points: int = 5) -> np.ndarray | None:
    """Locate the plate-edge breakpoint precisely.

    Step zone is found by z-difference thresholding (robust against short
    plate runs), then a line is fitted to the flat run on each side of the
    zone and the intersection of the two lines is the breakpoint (sub-mm
    accuracy, stable even when the plate run inside the ROI is short).
    """
    pts = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    if len(pts) < 12:
        return None
    pts = pts[np.argsort(pts[:, 0])]
    z = pts[:, 1]
    win = 3
    zs = np.convolve(z, np.ones(win) / win, mode="valid")
    dz = np.abs(np.diff(zs))
    steep = dz > z_diff_threshold_m
    if not np.any(steep):
        return None
    i0 = int(np.argmax(steep))
    i1 = int(len(steep) - 1 - np.argmax(steep[::-1]))
    # flat runs on both sides
    left = pts[: i0 + 1]
    right = pts[i1 + 1:]
    if len(left) < min_flat_points or len(right) < min_flat_points:
        return None
    a1, b1 = np.polyfit(left[:, 0], left[:, 1], 1)
    a2, b2 = np.polyfit(right[:, 0], right[:, 1], 1)
    denom = a1 - a2
    if abs(denom) < 1.0e-9:
        return None
    xb = (b2 - b1) / denom
    zb = a1 * xb + b1
    return np.array([xb, zb], dtype=float)


def detect_step_precise(points_xz: np.ndarray,
                         z_diff_threshold_m: float = 0.00012,
                         cfg: SegmentDetectionConfig | None = None) -> np.ndarray:
    """Step corners refined by local Split-and-Merge.

    1. z-difference finds the step zone (robust even with a short plate run).
    2. Split-and-Merge runs only on a small window around the step zone, so
       its line fitting yields BOTH corners (plate end + table end) with
       sub-mm accuracy.
    Returns (N, 2) corner candidates; the caller picks the plate-surface one
    (closest to the other ROI's centre).
    """
    cfg = cfg or SegmentDetectionConfig()
    pts = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    if len(pts) < 12:
        return np.zeros((0, 2))
    pts = pts[np.argsort(pts[:, 0])]
    z = pts[:, 1]
    win = 3
    zs = np.convolve(z, np.ones(win) / win, mode="valid")
    dz = np.abs(np.diff(zs))
    steep = dz > z_diff_threshold_m
    if not np.any(steep):
        return np.zeros((0, 2))
    i0 = int(np.argmax(steep))
    i1 = int(len(steep) - 1 - np.argmax(steep[::-1]))
    # local window around the step zone (2 mm of context each side)
    xa = pts[max(0, i0)][0] - 0.002
    xb = pts[min(len(pts) - 1, i1 + 1)][0] + 0.002
    local = pts[(pts[:, 0] >= xa) & (pts[:, 0] <= xb)]
    if len(local) < 10:
        return np.array([pts[i0], pts[i1]], dtype=float).reshape(-1, 2)
    bps = split_and_merge(local, cfg)
    if len(bps) == 0:
        return np.array([pts[i0], pts[i1]], dtype=float).reshape(-1, 2)
    return bps


def plate_edge_from_midpoint(points_xz: np.ndarray,
                               x_mid_m: float,
                               residual_threshold_m: float = 0.0008,
                               min_grow_points: int = 8,
                               max_grow_points: int = 400) -> tuple | None:
    """Method 2: plate-line endpoints from the midpoint.

    The X midpoint of the two ROI centres lies on the plate surface (the
    plate spans both breakpoints).  Starting from that point we grow left and
    right along the X-ordered profile, accepting points while they fit the
    growing plate line (residual < threshold).  The first point that deviates
    (the step) stops growth on that side.  A line is fitted through all
    accepted plate points; E1/E2 = that line's endpoints at the growth
    boundaries.

    Returns (E1, E2) as (2,2) X-Z or None.
    """
    pts = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    if len(pts) < 20:
        return None
    pts = pts[np.argsort(pts[:, 0])]
    # start index nearest to x_mid
    i0 = int(np.argmin(np.abs(pts[:, 0] - x_mid_m)))
    xs = pts[:, 0]
    zs = pts[:, 1]

    def grow(direction: int) -> np.ndarray:
        """Grow from i0 toward direction (+1 right, -1 left); return plate idxs."""
        idxs = [i0]
        i = i0
        while 0 < i + direction < len(pts) - 1 and len(idxs) < max_grow_points:
            nxt = i + direction
            if len(idxs) >= min_grow_points:
                # fit line on the accepted points
                p = np.polyfit(xs[idxs], zs[idxs], 1)
                pred = p[0] * xs[nxt] + p[1]
                if abs(zs[nxt] - pred) > residual_threshold_m:
                    break  # step reached
            idxs.append(nxt)
            i = nxt
        return np.asarray(idxs)

    left = grow(-1)
    right = grow(+1)
    if len(left) + len(right) - 1 < min_grow_points:
        return None
    plate_idx = np.unique(np.concatenate([left, right]))
    p = np.polyfit(xs[plate_idx], zs[plate_idx], 1)
    # endpoints: line evaluated at the growth frontiers (left grew toward -x,
    # so its last index is the leftmost plate point; right[-1] is rightmost)
    xl = xs[left[-1]]
    xr = xs[right[-1]]
    e1 = np.array([xl, p[0] * xl + p[1]])
    e2 = np.array([xr, p[0] * xr + p[1]])
    return e1, e2


def iepf(points_xz: np.ndarray,
         cfg: SegmentDetectionConfig | None = None) -> np.ndarray:
    """Iterative End Point Fit: same recursive core, endpoint-line residuals."""
    cfg = cfg or SegmentDetectionConfig()
    pts = np.asarray(points_xz, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return np.zeros((0, 2))
    pts = pts[np.argsort(pts[:, 0])]
    segments: list[tuple[int, int]] = []
    _split(pts, 0, len(pts) - 1, cfg, segments)
    return _breakpoints_from_segments(pts, segments)
