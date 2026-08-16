#!/usr/bin/env python3
"""ECC image-registration ROI trackers (Euclidean / Affine).

Estimates a geometric warp W from the template to the current frame with
cv2.findTransformECC, then warps the previous ROI corners through W.

Two template modes:
  fixed           : template = init-frame ROI, matched against every frame.
                    No drift, but large shape change can break matching.
  frame_to_frame  : template = previous frame's ROI crop.  Small motion,
                    easier to match, but may accumulate drift.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .base import ROITracker
from .rasterizer import ProfileRasterizer
from .types import TrackingFrame, TargetROI, ROITrackingResult


@dataclass
class ECCConfig:
    motion_type: str = "affine"            # "euclidean" | "affine"
    template_mode: str = "fixed"           # "fixed" | "frame_to_frame"
    search_margin: float = 0.5             # search region = ROI * (1 + margin)
    max_iterations: int = 50
    termination_eps: float = 1.0e-4
    gauss_filt_size: int = 5        # internal ECC smoothing; sparse binary
                                    # lines need it or gradient -> NaN


class ECCTracker(ROITracker):
    def __init__(self, rasterizer: ProfileRasterizer, config: ECCConfig) -> None:
        self.rasterizer = rasterizer
        self.config = config
        self._template = None              # fixed template crop
        self._template_roi_px = None       # ROI corners in template coords
        self._prev_image = None
        self._prev_roi = None
        self._initialized = False

    @property
    def name(self) -> str:
        return f"ecc_{self.config.motion_type}_{self.config.template_mode}"

    def _motion_flag(self) -> int:
        if self.config.motion_type == "euclidean":
            return cv2.MOTION_EUCLIDEAN
        return cv2.MOTION_AFFINE

    def initialize(self, frame: TrackingFrame, roi: TargetROI) -> None:
        self._template = self._crop(frame.image, roi, margin=0.0).astype(np.float32)
        self._template_roi_px = self._roi_corners_px(roi)   # (4,2) in full-image px
        self._prev_image = frame.image.copy()
        self._prev_roi = roi
        self._initialized = True

    def update(self, frame: TrackingFrame) -> ROITrackingResult:
        t0 = time.perf_counter()
        if not self._initialized:
            return ROITrackingResult(False, None, None, 0.0, "not_initialized")
        if self.config.template_mode == "fixed":
            template = self._template
        else:
            template = self._crop(self._prev_image, self._prev_roi, margin=0.0).astype(np.float32)

        search = self._crop(frame.image, self._prev_roi, margin=self.config.search_margin)
        if (search.size == 0 or search.shape[0] < template.shape[0]
                or search.shape[1] < template.shape[1]):
            return ROITrackingResult(False, None, None, (time.perf_counter()-t0)*1000.0,
                                     "search_region_too_small")
        current = search.astype(np.float32)

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.config.max_iterations,
            self.config.termination_eps,
        )
        try:
            rho, warp = cv2.findTransformECC(
                template, current, warp, self._motion_flag(), criteria,
                None, self.config.gauss_filt_size,
            )
        except cv2.error as exc:
            return ROITrackingResult(False, None, None, (time.perf_counter()-t0)*1000.0,
                                     f"ecc_error:{exc}")

        corners = self._template_roi_px if self.config.template_mode == "fixed" \
            else self._roi_corners_px(self._prev_roi)
        (sx0, sy0) = self._search_origin(self._prev_roi, margin=self.config.search_margin)
        rel = corners - np.array([sx0, sy0])
        rel_h = np.hstack([rel, np.ones((len(rel), 1))])
        warped_rel = (warp @ rel_h.T).T
        warped_full = warped_rel + np.array([sx0, sy0])

        xs, zs = [], []
        for u, v in warped_full:
            x, z = self.rasterizer.pixel_to_metric(u, v)
            xs.append(x); zs.append(z)
        roi = TargetROI.from_bbox(min(xs), min(zs), max(xs), max(zs))

        self._prev_image = frame.image.copy()
        self._prev_roi = roi
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return ROITrackingResult(True, roi, float(rho), runtime_ms, "", warp.copy())

    def reset(self) -> None:
        self._template = None
        self._prev_image = None
        self._prev_roi = None
        self._initialized = False

    # -- helpers ------------------------------------------------------------
    def _crop(self, image: np.ndarray, roi: TargetROI, margin: float) -> np.ndarray:
        u0, v0 = self.rasterizer.metric_to_pixel(roi.xmin, roi.zmax)
        u1, v1 = self.rasterizer.metric_to_pixel(roi.xmax, roi.zmin)
        if margin > 0:
            w = abs(u1 - u0); h = abs(v1 - v0)
            u0 -= margin * w; u1 += margin * w
            v0 -= margin * h; v1 += margin * h
        H, W = image.shape[:2]
        x0 = max(0, int(round(min(u0, u1))))
        x1 = min(W, int(round(max(u0, u1))))
        y0 = max(0, int(round(min(v0, v1))))
        y1 = min(H, int(round(max(v0, v1))))
        return image[y0:y1, x0:x1]

    def _search_origin(self, roi: TargetROI, margin: float) -> tuple[int, int]:
        u0, v0 = self.rasterizer.metric_to_pixel(roi.xmin, roi.zmax)
        u1, v1 = self.rasterizer.metric_to_pixel(roi.xmax, roi.zmin)
        if margin > 0:
            w = abs(u1 - u0); h = abs(v1 - v0)
            u0 -= margin * w; v0 -= margin * h
        return int(round(min(u0, u1))), int(round(min(v0, v1)))

    def _roi_corners_px(self, roi: TargetROI) -> np.ndarray:
        corners = []
        for x, z in roi.polygon_xz_m:
            u, v = self.rasterizer.metric_to_pixel(x, z)
            corners.append((u, v))
        return np.asarray(corners, dtype=float)
