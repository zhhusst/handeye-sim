#!/usr/bin/env python3
"""CSRT / KCF ROI trackers (OpenCV bbox tracking API).

Both are thin subclasses of OpenCVBBoxTracker; they differ only in the
tracker factory.  The tracker is created through a namespace-compatible
factory because some OpenCV builds expose the create functions under
``cv2.legacy``.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .base import ROITracker
from .rasterizer import ProfileRasterizer
from .types import TrackingFrame, TargetROI, ROITrackingResult


def create_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not available in this OpenCV build")


def create_kcf():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    raise RuntimeError("KCF tracker not available in this OpenCV build")


class OpenCVBBoxTracker(ROITracker):
    name = "opencv"

    def __init__(self, rasterizer: ProfileRasterizer) -> None:
        self.rasterizer = rasterizer
        self._tracker = None

    # subclass hook
    def _create_tracker(self):
        raise NotImplementedError

    def initialize(self, frame: TrackingFrame, roi: TargetROI) -> None:
        self._tracker = self._create_tracker()
        bbox = self.rasterizer.roi_to_bbox(roi)
        # OpenCV 4.6 Tracker.init() returns None on success and RAISES on
        # failure; treating the None return as False was a bug.
        try:
            self._tracker.init(frame.image, bbox)
        except cv2.error as exc:
            raise RuntimeError(f"{self.name}: tracker.init() failed: {exc}") from exc

    def update(self, frame: TrackingFrame) -> ROITrackingResult:
        t0 = time.perf_counter()
        if self._tracker is None:
            return ROITrackingResult(False, None, None, 0.0, "not_initialized")
        success, bbox = self._tracker.update(frame.image)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        if not success:
            return ROITrackingResult(False, None, None, runtime_ms, "tracker_update_failed")
        # clamp bbox into image
        H, W = frame.image.shape[:2]
        x, y, w, h = (int(v) for v in bbox)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        roi = self.rasterizer.bbox_to_roi((x, y, w, h))
        return ROITrackingResult(True, roi, None, runtime_ms, "")

    def reset(self) -> None:
        self._tracker = None


class CSRTROITracker(OpenCVBBoxTracker):
    name = "csrt"

    def _create_tracker(self):
        return create_csrt()


class KCFROITracker(OpenCVBBoxTracker):
    name = "kcf"

    def _create_tracker(self):
        return create_kcf()
