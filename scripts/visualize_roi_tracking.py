#!/usr/bin/env python3
"""Interactive ROI tracking visualizer.

Complete workflow in ONE window:
    1. pick the first frame (slider / arrows)
    2. drag to draw the target ROI (red dashed box)
    3. click "Start" -> chosen trackers initialise on this frame+ROI
    4. play / scrub and watch each tracker's ROI follow the target

Colors: red=CSRT blue=KCF green=ECC-euclidean purple=ECC-affine-fixed
        orange=ECC-affine-frame cyan=Chamfer  gold *=bag endpoints

Usage
-----
    python3 visualize_roi_tracking.py BAG.mcap [--trackers csrt] [--stride 4]
                                              [--fail-streak 3]

The breakpoint validation (E1/E2 core containment, no_bp, jumps) marks a
frame failed when ANY check fails.  Single noisy frames from laser-profile
outliers are filtered out: a failure is only confirmed after
``--fail-streak`` CONSECUTIVE failed frames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

sys.path.insert(0, "/workspace/ros2_ws/src/handeye_calibration_core")
sys.path.insert(0, "/workspace/scripts")
from calibration_pipeline.tools.bag_timeline import BagTimeline
from calibration_pipeline.roi_tracking.factory import create_tracker, TRACKER_NAMES
from calibration_pipeline.roi_tracking.rasterizer import (
    ProfileRasterizer,
    RasterizerConfig,
)
from calibration_pipeline.roi_tracking.types import TrackingFrame, TargetROI
from calibration_pipeline.roi_tracking.segment_detector import (
    plate_edge_from_midpoint,
)

COLORS = {
    "csrt": "red",
    "kcf": "blue",
    "ecc_euclidean_fixed": "green",
    "ecc_affine_fixed": "purple",
    "ecc_affine_frame_to_frame": "orange",
    "chamfer": "cyan",
}


class ROIViewer:
    def __init__(self, bag: Path, tracker_names, stride: int = 4,
                 fail_streak: int = 3):
        self.tl = BagTimeline(bag, stride=stride)
        self.names = tracker_names
        self.stride = stride

        xs, zs = [], []
        for f in self.tl.frames:
            if len(f["profile"]):
                xs += [f["profile"][:, 0].min(), f["profile"][:, 0].max()]
                zs += [f["profile"][:, 2].min(), f["profile"][:, 2].max()]
        self.rz = ProfileRasterizer(RasterizerConfig(
            x_min_m=min(xs) - 0.02, x_max_m=max(xs) + 0.02,
            z_min_m=min(zs) - 0.02, z_max_m=max(zs) + 0.02,
            resolution_m_per_pixel=0.00025, point_radius_px=2,
        ))

        self.idx = 0
        self.rois: list = []            # user-drawn ROIs (mm, x0 z0 x1 z1), max 2
        self.drag_start = None
        self.roi_patch = None
        self.roi_colors = ["red", "blue"]
        self.trackers = {}              # name -> tracker
        self.traj = {}                  # name -> {frame: (roi|None, success, reason)}
        self.max_processed = -1         # highest frame processed
        self.playing = False
        # detection-failure thresholds (sequential validation)
        self.core_fraction = 0.7        # BP must be in central 70% of ROI
        self.roi_jump_m = 0.015         # ROI centre jump > 15 mm -> fail
        self.bp_jump_m = 0.015          # breakpoint jump > 15 mm -> fail
        # a failure is only confirmed after this many CONSECUTIVE failed
        # frames; single noisy frames (laser-profile outliers) are ignored
        self.fail_streak_threshold = max(1, int(fail_streak))
        self._fail_streak = 0           # running consecutive-failed count
        self.frame_bp = {}              # i -> {"e1","e2","failed","fail_streak","confirmed"}
        self._prev_roi_centers = None
        self._prev_bp = None

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.subplots_adjust(bottom=0.26)
        ax_slider = self.fig.add_axes([0.12, 0.12, 0.72, 0.03])
        self.slider = Slider(ax_slider, "frame", 0, len(self.tl) - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)
        ax_start = self.fig.add_axes([0.12, 0.05, 0.14, 0.04])
        self.btn_start = Button(ax_start, "Start Tracking")
        self.btn_start.on_clicked(self._start)
        ax_play = self.fig.add_axes([0.28, 0.05, 0.10, 0.04])
        self.btn_play = Button(ax_play, "Play")
        self.btn_play.on_clicked(self._toggle_play)
        ax_reset = self.fig.add_axes([0.40, 0.05, 0.10, 0.04])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self._reset)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self._tick)

        self._render()

    # -- frame / image helpers ---------------------------------------------
    # -- breakpoint detection inside a tracked ROI ----------------------
    def _detect_breakpoints(self, roi: TargetROI, profile: np.ndarray,
                            other_center: np.ndarray):
        """Locate the step (plate edge) inside the tracked ROI; return the
        corner closest to the other ROI's centre (the plate-surface corner).
        detect_step_precise = z-diff step zone + local split-and-merge,
        stable 60/60 frames in offline tests."""
        if profile is None or len(profile) == 0:
            return None
        mask = roi.contains(profile[:, (0, 2)])
        pts = profile[mask][:, (0, 2)]
        if len(pts) < 12:
            return None
        bps = detect_step_precise(pts)
        if len(bps) == 0:
            return None
        dists = np.hypot(bps[:, 0] - other_center[0],
                         bps[:, 1] - other_center[1])
        return bps[int(np.argmin(dists))]

    def _make_frame(self, i):
        f = self.tl.frames[i]
        return TrackingFrame(self.tl.time_s(f), f["profile"],
                             self.rz.rasterize(f["profile"]))

    def _ensure_processed(self, up_to):
        """Sequentially run trackers up to frame ``up_to`` and cache ROIs."""
        if not self.trackers or up_to <= self.max_processed:
            return
        for i in range(self.max_processed + 1, min(up_to, len(self.tl)) + 1):
            fr = self._make_frame(i)
            for name, tr in self.trackers.items():
                try:
                    res = tr.update(fr)
                    self.traj[name][i] = (res.roi, res.success, res.reason)
                except Exception as exc:
                    self.traj[name][i] = (None, False, f"exception:{exc}")
            self._compute_bp_validation(i)
            self.max_processed = i

    def _compute_bp_validation(self, i: int) -> None:
        """Breakpoint detection + failure validation for frame ``i`` (called
        sequentially so previous-frame state is the true previous frame)."""
        names = list(self.trackers.keys())
        rois_now = {}
        for name in names:
            entry = self.traj[name].get(i)
            if entry is not None and entry[0] is not None:
                rois_now[name] = entry[0]
        failed: list[str] = []
        e1 = e2 = None
        if len(rois_now) >= 2:
            r0 = rois_now[names[0]]
            r1 = rois_now[names[1]]
            prof = self.tl.frames[i]["profile"]
            if prof is not None and len(prof):
                x_mid = 0.5 * (r0.center[0] + r1.center[0])
                res = plate_edge_from_midpoint(prof[:, (0, 2)], x_mid)
                if res is not None:
                    e1, e2 = res
                    if not r0.core(self.core_fraction).contains(e1.reshape(1, 2))[0]:
                        failed.append("E1_out_core")
                    if not r1.core(self.core_fraction).contains(e2.reshape(1, 2))[0]:
                        failed.append("E2_out_core")
                else:
                    failed.append("no_bp")
            else:
                failed.append("no_profile")
        else:
            failed.append("roi_missing")
        # sequential jumps vs the previous frame
        if self._prev_roi_centers is not None:
            for name in names:
                if name in rois_now and name in self._prev_roi_centers:
                    d = float(np.hypot(*(rois_now[name].center
                                         - self._prev_roi_centers[name])))
                    if d > self.roi_jump_m:
                        failed.append(f"ROI_jump_{name}({d*1000:.0f}mm)")
        if self._prev_bp is not None and e1 is not None and e2 is not None:
            d1 = float(np.hypot(*(e1 - self._prev_bp[0])))
            d2 = float(np.hypot(*(e2 - self._prev_bp[1])))
            if d1 > self.bp_jump_m:
                failed.append(f"E1_jump({d1*1000:.0f}mm)")
            if d2 > self.bp_jump_m:
                failed.append(f"E2_jump({d2*1000:.0f}mm)")
        # update sequential state
        self._prev_roi_centers = {n: rois_now[n].center for n in rois_now}
        if e1 is not None and e2 is not None:
            self._prev_bp = (e1, e2)
        # a single noisy frame is NOT a failure (laser-profile outliers).
        # Only a streak of failed frames >= fail_streak_threshold is
        # confirmed as a real failure; one good frame resets the counter.
        if failed:
            self._fail_streak += 1
        else:
            self._fail_streak = 0
        self.frame_bp[i] = {
            "e1": e1, "e2": e2,
            "failed": ", ".join(failed) if failed else None,
            "fail_streak": self._fail_streak,
            "confirmed": self._fail_streak >= self.fail_streak_threshold,
        }

    # -- rendering ---------------------------------------------------------
    def _render(self):
        self.ax.cla()
        f = self.tl.frames[self.idx]
        prof = f["profile"]
        if len(prof):
            self.ax.scatter(prof[:, 0] * 1000, prof[:, 2] * 1000, s=1.2, c="0.5")
        ep = f["endpoints"]
        if ep is not None and len(ep) >= 2:
            self.ax.scatter(ep[:, 0] * 1000, ep[:, 2] * 1000, c="gold", s=50,
                            marker="*", label="bag endpoints")
        # user ROIs (before start): two dashed boxes, red then blue
        self.roi_patch = None
        if not self.trackers:
            for k, rbox in enumerate(self.rois):
                x0, z0, x1, z1 = rbox
                col = self.roi_colors[k % len(self.roi_colors)]
                self.ax.add_patch(Rectangle(
                    (min(x0, x1), min(z0, z1)), abs(x1 - x0), abs(z1 - z0),
                    fill=False, edgecolor=col, linewidth=2, linestyle="--", zorder=8,
                ))
                self.ax.text(min(x0, x1), min(z0, z1) - 3, f"ROI{k+1}",
                             fontsize=8, color=col)
        # in-progress drag box
        if getattr(self, "_tmp_roi", None) is not None and not self.trackers:
            x0, z0, x1, z1 = self._tmp_roi
            k = len(self.rois)
            col = self.roi_colors[k % len(self.roi_colors)]
            self.ax.add_patch(Rectangle(
                (min(x0, x1), min(z0, z1)), abs(x1 - x0), abs(z1 - z0),
                fill=False, edgecolor=col, linewidth=2, linestyle="--", zorder=8,
            ))

        # tracker ROIs at current frame
        if self.trackers:
            self._ensure_processed(self.idx)
            for name, tr in self.trackers.items():
                entry = self.traj[name].get(self.idx)
                base = name.split("#")[0]
                col = COLORS.get(base, "black")
                if entry is None:
                    continue
                r, ok, reason = entry
                if ok and r is not None:
                    self.ax.plot(
                        [r.xmin * 1000, r.xmax * 1000, r.xmax * 1000,
                         r.xmin * 1000, r.xmin * 1000],
                        [r.zmin * 1000, r.zmin * 1000, r.zmax * 1000,
                         r.zmax * 1000, r.zmin * 1000],
                        color=col, linewidth=1.5, label=name)
                else:
                    self.ax.scatter([], [], color=col)  # keep legend
                    self.ax.text(0.02, 0.96 - 0.035 * len(self.trackers),
                                 f"{name}: FAIL({reason})", transform=self.ax.transAxes,
                                 fontsize=7, color=col)
        # breakpoint detection result (computed sequentially + validated)
        if self.trackers:
            bp = self.frame_bp.get(self.idx)
            if bp is not None:
                confirmed = bp.get("confirmed", True)
                if bp["e1"] is not None:
                    e1, e2 = bp["e1"], bp["e2"]
                    self.ax.scatter(e1[0] * 1000, e1[1] * 1000, s=110,
                                    marker="o", facecolors="none",
                                    edgecolors=self.roi_colors[0],
                                    linewidths=2.5, zorder=9, label="E1")
                    self.ax.scatter(e2[0] * 1000, e2[1] * 1000, s=110,
                                    marker="o", facecolors="none",
                                    edgecolors=self.roi_colors[1],
                                    linewidths=2.5, zorder=9, label="E2")
                if bp["failed"] is not None:
                    if confirmed:
                        self.ax.text(
                            0.02, 0.90,
                            f"BP FAILED (streak {bp['fail_streak']}/"
                            f"{self.fail_streak_threshold}): {bp['failed']}",
                            transform=self.ax.transAxes, fontsize=8,
                            color="red", fontweight="bold")
                    else:
                        self.ax.text(
                            0.02, 0.90,
                            f"transient ({bp['fail_streak']}/"
                            f"{self.fail_streak_threshold}): {bp['failed']}",
                            transform=self.ax.transAxes, fontsize=8,
                            color="orange")

        state = "tracking" if self.trackers else "select ROI"
        self.ax.set_xlabel("X [mm]"); self.ax.set_ylabel("Z [mm]")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(
            f"frame {self.idx}/{len(self.tl)-1} t={self.tl.time_s(f):.1f}s | {state} | "
            f"drag=ROI1/ROI2  t=start  space=play  r=reset")
        if self.trackers:
            self.ax.legend(fontsize=7, loc="lower right")
        self.fig.canvas.draw_idle()

    # -- interactions ------------------------------------------------------
    def _on_slider(self, val):
        self.idx = int(round(val))
        self._render()

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1 or self.trackers:
            return
        if len(self.rois) >= 2:
            print("already 2 ROIs; Reset to redraw")
            return
        self.drag_start = (event.xdata, event.ydata)
        self._tmp_roi = (event.xdata, event.ydata, event.xdata, event.ydata)
        self._render()

    def _on_motion(self, event):
        if self.drag_start is None or event.inaxes is not self.ax or self.trackers:
            return
        x0, z0 = self.drag_start
        self._tmp_roi = (x0, z0, event.xdata, event.ydata)
        self._render()

    def _on_release(self, event):
        if self.drag_start is None or self.trackers:
            return
        if event.inaxes is not self.ax:
            return
        x0, z0 = self.drag_start
        self._tmp_roi = (x0, z0, event.xdata, event.ydata)
        self.drag_start = None
        if abs(event.xdata - x0) > 2 and abs(event.ydata - z0) > 2:
            self.rois.append(self._tmp_roi)
            print(f"ROI{len(self.rois)} set: "
                  f"({min(x0, event.xdata):.0f},{min(z0, event.ydata):.0f}).."
                  f"({max(x0, event.xdata):.0f},{max(z0, event.ydata):.0f}) mm; "
                  f"draw ROI{len(self.rois)+1} or Start")
        self._tmp_roi = None
        self._render()

    def _on_key(self, event):
        if event.key == "left":
            self.idx = max(0, self.idx - 1)
            self.slider.set_val(self.idx)
        elif event.key == "right":
            self.idx = min(len(self.tl) - 1, self.idx + 1)
            self.slider.set_val(self.idx)
        elif event.key == "t":
            self._start()
        elif event.key == " ":
            self._toggle_play()
        elif event.key == "r":
            self._reset()

    def _start(self, _event=None):
        if not self.rois:
            print("draw at least one ROI first")
            return
        self.playing = False
        self.btn_play.label.set_text("Play")
        self.trackers = {}
        self.traj = {}
        start = self.idx
        if not self.rois:
            print("draw at least one ROI first")
            return
        fr = self._make_frame(start)
        for k, rbox in enumerate(self.rois):
            x0, z0, x1, z1 = rbox
            roi = TargetROI.from_bbox(
                min(x0, x1) / 1000.0, min(z0, z1) / 1000.0,
                max(x0, x1) / 1000.0, max(z0, z1) / 1000.0)
            for base in self.names:
                tr = create_tracker(base, self.rz)
                key = f"{tr.name}#{k}"
                try:
                    tr.initialize(fr, roi)
                except Exception as exc:
                    print(f"[{key}] init FAILED: {exc}")
                    continue
                self.trackers[key] = tr
                self.traj[key] = {start: (roi, True, "init")}
        self.max_processed = start
        self.frame_bp = {}
        self._prev_roi_centers = None
        self._prev_bp = None
        self._fail_streak = 0
        if self.trackers:
            self._compute_bp_validation(start)
        print(f"started tracking from frame {start} with "
              f"{list(self.trackers.keys())}")
        self._render()

    def _toggle_play(self, _event=None):
        if not self.trackers:
            print("start tracking first")
            return
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        if self.idx < len(self.tl) - 1:
            self.idx += 1
            self.slider.set_val(self.idx)
        else:
            self.playing = False
            self.btn_play.label.set_text("Play")
            self.timer.stop()

    def _reset(self, _event=None):
        self.playing = False
        self.btn_play.label.set_text("Play")
        self.timer.stop()
        self.trackers = {}
        self.traj = {}
        self.max_processed = -1
        self.rois = []
        self._tmp_roi = None
        self._fail_streak = 0
        self.idx = 0
        self.slider.set_val(0)
        print("reset; draw up to two ROIs")

    def show(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", type=Path)
    ap.add_argument("--trackers", default="csrt",
                    help="comma-separated tracker names (default csrt)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fail-streak", type=int, default=3,
                    help="consecutive failed frames before a breakpoint "
                         "failure is confirmed (default 3; isolated noisy "
                         "frames are ignored)")
    args = ap.parse_args()
    names = [t.strip().lower() for t in args.trackers.split(",") if t.strip()]
    for n in names:
        if n not in TRACKER_NAMES:
            print(f"unknown tracker {n}; options: {TRACKER_NAMES}")
            return
    v = ROIViewer(args.bag.resolve(), names, stride=args.stride,
                  fail_streak=args.fail_streak)
    print("workflow: pick frame -> drag ROI -> Start (or t) -> Play (space)")
    v.show()


if __name__ == "__main__":
    main()
