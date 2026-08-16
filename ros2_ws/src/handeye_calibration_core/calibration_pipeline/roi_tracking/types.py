#!/usr/bin/env python3
"""Shared data structures for ROI tracking.

Coordinate convention
---------------------
The ROI is ALWAYS defined in physical X-Z (metres) via ``polygon_xz_m``.
Pixel bounding boxes exist only transiently inside image-based trackers
(CSRT/KCF) and are converted back to X-Z with the rasterizer.  This keeps
every tracker's output directly comparable in the same physical frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackingFrame:
    """One processed frame handed to every ROITracker."""

    timestamp_s: float
    profile: np.ndarray          # (N, 3) raw Gocator points (x, y, z), metres
    image: np.ndarray            # H x W uint8 grayscale raster (for CSRT/KCF/ECC/Chamfer)


@dataclass
class TargetROI:
    """Target region in physical X-Z coordinates (metres)."""

    polygon_xz_m: np.ndarray     # (4, 2) corner points, metres

    @classmethod
    def from_bbox(cls, xmin_m: float, zmin_m: float,
                  xmax_m: float, zmax_m: float) -> "TargetROI":
        return cls(
            polygon_xz_m=np.array(
                [
                    [xmin_m, zmin_m],
                    [xmax_m, zmin_m],
                    [xmax_m, zmax_m],
                    [xmin_m, zmax_m],
                ],
                dtype=float,
            )
        )

    @property
    def xmin(self) -> float:
        return float(np.min(self.polygon_xz_m[:, 0]))

    @property
    def xmax(self) -> float:
        return float(np.max(self.polygon_xz_m[:, 0]))

    @property
    def zmin(self) -> float:
        return float(np.min(self.polygon_xz_m[:, 1]))

    @property
    def zmax(self) -> float:
        return float(np.max(self.polygon_xz_m[:, 1]))

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.polygon_xz_m, axis=0)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.zmax - self.zmin

    def contains(self, points_xz_m: np.ndarray) -> np.ndarray:
        """Boolean mask: which points (N,2) lie inside the axis-aligned bbox."""
        p = np.asarray(points_xz_m, dtype=float).reshape(-1, 2)
        return (
            (p[:, 0] >= self.xmin)
            & (p[:, 0] <= self.xmax)
            & (p[:, 1] >= self.zmin)
            & (p[:, 1] <= self.zmax)
        )

    def core(self, fraction: float = 0.7) -> "TargetROI":
        """Central ``fraction`` box (same centre, scaled size).  Used as the
        strict containment region: a breakpoint near the ROI border is not a
        trustworthy detection."""
        f = float(fraction)
        cx = 0.5 * (self.xmin + self.xmax)
        cz = 0.5 * (self.zmin + self.zmax)
        hw = 0.5 * (self.xmax - self.xmin) * f
        hh = 0.5 * (self.zmax - self.zmin) * f
        return TargetROI.from_bbox(cx - hw, cz - hh, cx + hw, cz + hh)

    def to_dict(self) -> dict:
        return {
            "xmin_m": self.xmin,
            "zmin_m": self.zmin,
            "xmax_m": self.xmax,
            "zmax_m": self.zmax,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TargetROI":
        return cls.from_bbox(d["xmin_m"], d["zmin_m"], d["xmax_m"], d["zmax_m"])


@dataclass
class ROITrackingResult:
    """Uniform output of every ROITracker."""

    success: bool
    roi: TargetROI | None = None
    confidence: float | None = None
    runtime_ms: float = 0.0
    reason: str = ""
    transform: np.ndarray | None = None   # 2x3 warp for ECC; None otherwise

    def to_dict(self) -> dict:
        d = {
            "success": self.success,
            "runtime_ms": round(self.runtime_ms, 3),
            "reason": self.reason,
        }
        if self.roi is not None:
            d["roi"] = self.roi.to_dict()
        if self.confidence is not None:
            d["confidence"] = round(float(self.confidence), 4)
        return d
