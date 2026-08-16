#!/usr/bin/env python3
"""Chamfer matching ROI tracker.

The target is a 2-D point set (the profile points inside the initial ROI).
Every frame we build a distance transform of the current profile image and
search the similarity transform (translation, rotation, uniform scale) that
minimises the mean distance from the transformed template points to the
current profile.  Coarse-to-fine search; no non-rigid deformation in v1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .base import ROITracker
from .rasterizer import ProfileRasterizer
from .types import TrackingFrame, TargetROI, ROITrackingResult


@dataclass
class ChamferConfig:
    translation_search_m: float = 0.015
    rotation_search_deg: float = 10.0
    scale_min: float = 0.8
    scale_max: float = 1.2
    coarse_steps_t: int = 7          # translations per axis at coarse level
    coarse_steps_r: int = 9          # rotations at coarse level
    coarse_steps_s: int = 5          # scales at coarse level
    refine_steps_t: int = 5
    refine_steps_r: int = 5
    refine_steps_s: int = 3
    max_template_points: int = 200   # downsample template for speed
    max_roi_expand_m: float = 0.02   # extra context when building template


class ChamferTracker(ROITracker):
    name = "chamfer"

    def __init__(self, rasterizer: ProfileRasterizer, config: ChamferConfig | None = None) -> None:
        self.rasterizer = rasterizer
        self.config = config or ChamferConfig()
        self._template_px = None       # (N,2) template points in pixel coords
        self._template_center_px = None
        self._last_roi = None
        self._initialized = False

    def initialize(self, frame: TrackingFrame, roi: TargetROI) -> None:
        # template = profile points inside the ROI (plus small context)
        expand = self.config.max_roi_expand_m
        ctx = TargetROI.from_bbox(
            roi.xmin - expand, roi.zmin - expand,
            roi.xmax + expand, roi.zmax + expand,
        )
        pts = frame.profile
        if len(pts):
            inside = ctx.contains(pts[:, (0, 2)])
            sel = pts[inside]
        else:
            sel = np.zeros((0, 3))
        if len(sel) == 0:
            raise RuntimeError("chamfer: no template points inside ROI")
        px = []
        for x, z in sel[:, (0, 2)]:
            u, v = self.rasterizer.metric_to_pixel(x, z)
            px.append((u, v))
        template = np.asarray(px, dtype=float).reshape(-1, 2)
        if len(template) > self.config.max_template_points:
            idx = np.linspace(0, len(template) - 1, self.config.max_template_points).astype(int)
            template = template[idx]
        self._template_px = template
        self._template_center_px = np.mean(template, axis=0)
        self._last_roi = roi
        self._initialized = True

    def update(self, frame: TrackingFrame) -> ROITrackingResult:
        t0 = time.perf_counter()
        if not self._initialized:
            return ROITrackingResult(False, None, None, 0.0, "not_initialized")

        # distance transform of current profile image
        img = frame.image
        dist = cv2.distanceTransform((255 - img).astype(np.uint8), cv2.DIST_L2, 5)
        # 255-img: profile pixels become 0 -> distanceTransform gives distance
        # from each pixel to the nearest ZERO (= profile) pixel.

        # search window around last ROI (pixel units)
        roi = self._last_roi
        cx_px, cz_px = self.rasterizer.metric_to_pixel(roi.center[0], roi.center[1])
        t_px = self.config.translation_search_m / self.rasterizer.config.resolution_m_per_pixel

        best = self._coarse_to_fine(dist, cx_px, cz_px, t_px)
        if best is None:
            return ROITrackingResult(False, None, None, (time.perf_counter()-t0)*1000.0,
                                     "chamfer_no_match")

        cost, (dtx, dtz, dtheta_deg, scale) = best

        # build current ROI: transform last ROI corners through the similarity
        corners = self._last_roi.polygon_xz_m
        new_corners = []
        for x, z in corners:
            u, v = self.rasterizer.metric_to_pixel(x, z)
            u2, v2 = self._apply_sim(u, v, dtx, dtz, dtheta_deg, scale, cx_px, cz_px)
            nx, nz = self.rasterizer.pixel_to_metric(u2, v2)
            new_corners.append((nx, nz))
        new_corners = np.asarray(new_corners, dtype=float)
        new_roi = TargetROI.from_bbox(
            float(np.min(new_corners[:, 0])), float(np.min(new_corners[:, 1])),
            float(np.max(new_corners[:, 0])), float(np.max(new_corners[:, 1])),
        )
        self._last_roi = new_roi
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        # convert cost (mean distance in px) to mm for a readable confidence
        conf_mm = float(cost * self.rasterizer.config.resolution_m_per_pixel * 1000.0)
        return ROITrackingResult(True, new_roi, max(0.0, 5.0 - conf_mm), runtime_ms,
                                 f"cost_mm={conf_mm:.2f}")

    def reset(self) -> None:
        self._template_px = None
        self._last_roi = None
        self._initialized = False

    # -- search -------------------------------------------------------------
    def _apply_sim(self, u, v, dtx, dtz, theta_deg, scale, cx, cz):
        th = np.deg2rad(theta_deg)
        c, s = np.cos(th), np.sin(th)
        du, dv = u - cx, v - cz
        ru = c * du - s * dv
        rv = s * du + c * dv
        return cx + scale * ru + dtx, cz + scale * rv + dtz

    def _sample_cost(self, dist, dtx, dtz, theta_deg, scale, cx, cz):
        tp = self._template_px
        th = np.deg2rad(theta_deg)
        c, s = np.cos(th), np.sin(th)
        du = tp[:, 0] - cx
        dv = tp[:, 1] - cz
        ru = c * du - s * dv
        rv = s * du + c * dv
        u2 = cx + scale * ru + dtx
        v2 = cz + scale * rv + dtz
        ui = np.clip(np.round(u2).astype(int), 0, dist.shape[1] - 1)
        vi = np.clip(np.round(v2).astype(int), 0, dist.shape[0] - 1)
        return float(np.mean(dist[vi, ui]))

    def _coarse_to_fine(self, dist, cx, cz, t_px):
        cfg = self.config
        # coarse grid
        best = None
        best_cost = float("inf")
        for dt in np.linspace(-t_px, t_px, cfg.coarse_steps_t):
            for dz in np.linspace(-t_px, t_px, cfg.coarse_steps_t):
                for dr in np.linspace(-cfg.rotation_search_deg, cfg.rotation_search_deg,
                                      cfg.coarse_steps_r):
                    for ds in np.linspace(cfg.scale_min, cfg.scale_max, cfg.coarse_steps_s):
                        cst = self._sample_cost(dist, dt, dz, dr, ds, cx, cz)
                        if cst < best_cost:
                            best_cost = cst
                            best = (dt, dz, dr, ds)
        if best is None:
            return None
        # refine around coarse best
        dt0, dz0, dr0, ds0 = best
        dt_step = (2 * t_px) / max(1, cfg.coarse_steps_t - 1)
        dz_step = dt_step
        dr_step = (2 * cfg.rotation_search_deg) / max(1, cfg.coarse_steps_r - 1)
        ds_step = (cfg.scale_max - cfg.scale_min) / max(1, cfg.coarse_steps_s - 1)
        for dt in np.linspace(dt0 - dt_step, dt0 + dt_step, cfg.refine_steps_t):
            for dz in np.linspace(dz0 - dz_step, dz0 + dz_step, cfg.refine_steps_t):
                for dr in np.linspace(dr0 - dr_step, dr0 + dr_step, cfg.refine_steps_r):
                    for ds in np.linspace(ds0 - ds_step, ds0 + ds_step, cfg.refine_steps_s):
                        cst = self._sample_cost(dist, dt, dz, dr, ds, cx, cz)
                        if cst < best_cost:
                            best_cost = cst
                            best = (dt, dz, dr, ds)
        return best_cost, best


def create_chamfer(rasterizer: ProfileRasterizer) -> ChamferTracker:
    return ChamferTracker(rasterizer)
