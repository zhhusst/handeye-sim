#!/usr/bin/env python3
"""Factory for ROI trackers.

    create_tracker("csrt", rasterizer)
    create_tracker("kcf", rasterizer)
    create_tracker("ecc_euclidean", rasterizer)
    create_tracker("ecc_affine_fixed", rasterizer)
    create_tracker("ecc_affine_frame", rasterizer)
    create_tracker("chamfer", rasterizer)
"""
from __future__ import annotations

from .base import ROITracker
from .opencv_tracker import CSRTROITracker, KCFROITracker
from .ecc_tracker import (
    ECCTracker,
    ECCConfig,
)
from .chamfer_tracker import ChamferTracker, ChamferConfig


def create_tracker(name: str, rasterizer) -> ROITracker:
    name = name.strip().lower()
    if name == "csrt":
        return CSRTROITracker(rasterizer)
    if name == "kcf":
        return KCFROITracker(rasterizer)
    if name == "ecc_euclidean":
        return ECCTracker(rasterizer, ECCConfig(motion_type="euclidean", template_mode="fixed"))
    if name == "ecc_affine_fixed":
        return ECCTracker(rasterizer, ECCConfig(motion_type="affine", template_mode="fixed"))
    if name == "ecc_affine_frame":
        return ECCTracker(rasterizer, ECCConfig(motion_type="affine", template_mode="frame_to_frame"))
    if name == "chamfer":
        return ChamferTracker(rasterizer)
    raise ValueError(f"unknown tracker: {name}")


TRACKER_NAMES = [
    "csrt",
    "kcf",
    "ecc_euclidean",
    "ecc_affine_fixed",
    "ecc_affine_frame",
    "chamfer",
]
