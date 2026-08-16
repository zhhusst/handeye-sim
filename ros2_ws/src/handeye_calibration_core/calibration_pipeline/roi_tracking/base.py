#!/usr/bin/env python3
"""Unified tracker interface.

Every ROI tracker (CSRT, KCF, ECC, Chamfer, later SPR) implements this
protocol.  The benchmark loops over trackers and calls ``update(frame)``
without knowing anything about the internals.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import TrackingFrame, TargetROI, ROITrackingResult


@runtime_checkable
class ROITracker(Protocol):
    @property
    def name(self) -> str:
        ...

    def initialize(self, frame: TrackingFrame, roi: TargetROI) -> None:
        """Initialise with the first frame and the target ROI."""
        ...

    def update(self, frame: TrackingFrame) -> ROITrackingResult:
        """Track on a new frame, return the updated ROI (physical X-Z)."""
        ...

    def reset(self) -> None:
        """Return to uninitialised state."""
        ...
