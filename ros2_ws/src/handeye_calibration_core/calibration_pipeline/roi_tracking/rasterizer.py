#!/usr/bin/env python3
"""Fixed-coordinate rasterizer from raw Gocator profiles to grayscale images.

CRITICAL
--------
The X-Z -> pixel mapping is FIXED once (config) and applied to every frame.
Per-frame normalisation (e.g. rescaling to the current profile bounds) would
hide real 10 mm motions from image trackers and destroy the physical meaning.

    u = (x - x_min) / r
    v = (z_max - z) / r        (v grows downward, like image rows)

Profile points are drawn with a 2-3 px radius so CSRT/KCF see a solid line
instead of a 1 px dotted one; an optional light Gaussian blur is applied.
Every tracker must use the SAME rasterizer instance so comparisons are fair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import TargetROI


@dataclass
class RasterizerConfig:
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float
    resolution_m_per_pixel: float = 0.00025   # 0.25 mm/px
    point_radius_px: int = 2
    blur_sigma: float = 0.0                   # 0 = no blur

    @property
    def width_px(self) -> int:
        return max(1, int(np.ceil((self.x_max_m - self.x_min_m) / self.resolution_m_per_pixel)))

    @property
    def height_px(self) -> int:
        return max(1, int(np.ceil((self.z_max_m - self.z_min_m) / self.resolution_m_per_pixel)))


class ProfileRasterizer:
    def __init__(self, config: RasterizerConfig) -> None:
        self.config = config
        self._r = config.resolution_m_per_pixel

    # -- forward / inverse mapping ------------------------------------------
    def metric_to_pixel(self, x_m, z_m) -> tuple[float, float]:
        u = (x_m - self.config.x_min_m) / self._r
        v = (self.config.z_max_m - z_m) / self._r
        return float(u), float(v)

    def pixel_to_metric(self, u_px, v_px) -> tuple[float, float]:
        x = self.config.x_min_m + u_px * self._r
        z = self.config.z_max_m - v_px * self._r
        return float(x), float(z)

    # -- rasterisation ------------------------------------------------------
    def rasterize(self, profile: np.ndarray) -> np.ndarray:
        pts = np.asarray(profile, dtype=float).reshape(-1, 3)
        finite = np.all(np.isfinite(pts[:, (0, 2)]), axis=1)
        pts = pts[finite]
        H = self.config.height_px
        W = self.config.width_px
        img = np.zeros((H, W), dtype=np.uint8)
        if len(pts) == 0:
            return img
        u = (pts[:, 0] - self.config.x_min_m) / self._r
        v = (self.config.z_max_m - pts[:, 2]) / self._r
        # Do not clip out-of-view samples onto the image border.  A profile
        # can legitimately move partly outside the fixed raster during a
        # wrist motion; clipping would turn all of those samples into a bright
        # artificial line which CSRT can lock onto.
        inside = (u >= 0.0) & (u < W) & (v >= 0.0) & (v < H)
        u = u[inside]
        v = v[inside]
        if len(u) == 0:
            return img
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        ui = np.minimum(ui, W - 1)
        vi = np.minimum(vi, H - 1)
        img[vi, ui] = 255
        if self.config.point_radius_px > 1:
            import cv2

            mask = np.zeros_like(img)
            mask[vi, ui] = 255
            kernel = np.ones(
                (self.config.point_radius_px, self.config.point_radius_px), np.uint8
            )
            img = cv2.dilate(mask, kernel, iterations=1)
        if self.config.blur_sigma > 0:
            import cv2

            img = cv2.GaussianBlur(img, (0, 0), self.config.blur_sigma)
        return img

    # -- ROI <-> pixel bbox ------------------------------------------------
    def roi_to_bbox(self, roi: TargetROI) -> tuple[int, int, int, int]:
        """(x, y, w, h) in pixel coords for OpenCV trackers."""
        u0, v0 = self.metric_to_pixel(roi.xmin, roi.zmax)
        u1, v1 = self.metric_to_pixel(roi.xmax, roi.zmin)
        x = int(round(min(u0, u1)))
        y = int(round(min(v0, v1)))
        w = max(1, int(round(abs(u1 - u0))))
        h = max(1, int(round(abs(v1 - v0))))
        # clamp
        x = max(0, min(x, self.config.width_px - 1))
        y = max(0, min(y, self.config.height_px - 1))
        w = min(w, self.config.width_px - x)
        h = min(h, self.config.height_px - y)
        return x, y, w, h

    def bbox_to_roi(self, bbox) -> TargetROI:
        x, y, w, h = (int(v) for v in bbox)
        x0m, z1m = self.pixel_to_metric(x, y)
        x1m, z0m = self.pixel_to_metric(x + w, y + h)
        return TargetROI.from_bbox(
            min(x0m, x1m), min(z0m, z1m), max(x0m, x1m), max(z0m, z1m)
        )
